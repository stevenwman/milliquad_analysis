"""Debug rough terrain: single run with --visualize to inspect in GUI.

Usage:
    uv run python debug_rough.py --scene scene4 --freq 30
    uv run python debug_rough.py --scene scene1 --freq 10 --spawn-z 0.003
"""
import eval_rough_terrain as ert
import simulation_fast_new as sim_module
from config_new import MJCF_PATHS, SIM_DURATION, SIMULATION_TIMEOUT, sim_params_from_point
import pathlib, tempfile, numpy as np
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="scene4")
parser.add_argument("--freq", type=float, default=30.0)
parser.add_argument("--height-std", type=float, default=0.00025, help="height std in meters (default 0.25mm)")
parser.add_argument("--y-offset", type=float, default=0.0, help="Y offset in meters")
parser.add_argument("--spawn-z", type=float, default=0.003, help="spawn height above hfield in meters (default 3mm)")
parser.add_argument("--spawn-x", type=float, default=0.03, help="spawn X offset in meters (default 30mm)")
parser.add_argument("--record", type=str, default=None, help="path to save video (omit for GUI)")
args = parser.parse_args()

ert.TERRAIN_HEIGHT_MEAN = args.height_std * 2
ert.TERRAIN_HEIGHT_STD = args.height_std
ert._y_offset = args.y_offset

# Override spawn pose with configurable height/x
_ORIG_POSE = ert._ORIGINAL_INIT_POSE
def _custom_init_pose(data, init_yaw_jitter_deg=0.0, rng=None):
    _ORIG_POSE(data, init_yaw_jitter_deg=init_yaw_jitter_deg, rng=rng)
    data.qpos[0] += args.spawn_x
    data.qpos[1] += ert._y_offset
    data.qpos[2] += args.spawn_z
sim_module._initialize_pose = _custom_init_pose

run_dir = pathlib.Path("results/20260225T225248_step_argmin_progress")
point = ert.load_best_point(run_dir)
sim_params = sim_params_from_point(point)
sp = dict(sim_params)
sp["drive_freq"] = args.freq

print(f"Scene: {args.scene}, freq: {args.freq}Hz")
print(f"Terrain: mean={ert.TERRAIN_HEIGHT_MEAN*1000:.2f}mm, std={ert.TERRAIN_HEIGHT_STD*1000:.2f}mm")
print(f"Spawn: x={args.spawn_x*1000:.0f}mm, z=+{args.spawn_z*1000:.1f}mm, y_offset={args.y_offset*1000:.1f}mm")

with tempfile.TemporaryDirectory(prefix="eval_rough_") as tmp_dir:
    mjcf = ert.inject_tiled_rough(MJCF_PATHS[args.scene], tmp_dir)
    try:
        kwargs = dict(
            sim_duration=SIM_DURATION, wall_timeout=300,
            ignore_stuck_detection=True,
        )
        if args.record:
            kwargs["record_path"] = args.record
            print(f"Recording to {args.record} ...")
        else:
            kwargs["visualize"] = True
            print("Launching GUI viewer...")

        traj = sim_module.run_simulation(sp, mjcf_path=mjcf, **kwargs)
        if traj is None:
            print("RESULT: None (unstable)")
        else:
            print(f"RESULT: {len(traj)} steps, final x={traj[-1]['pos'][0]*1000:.1f}mm")
    finally:
        ert.cleanup_temp_xmls()
