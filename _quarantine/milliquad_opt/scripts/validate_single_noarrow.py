#!/usr/bin/env python3
"""Run a single reference condition with best params, NO arrow overlays.

Identical to validate_single.py except it imports simulation_camera_noarrow,
which omits the per-leg magnet/goal direction arrows in the viewer.

Usage:
    # Visualize a single-leg flat run without arrows
    uv run python validate_single_noarrow.py results/20260303T192801_flat_tg \
        --ref-id scene1_f10 --visualize
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys
from datetime import datetime

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# Ensure parent dir on path for config imports
_PARENT = str(pathlib.Path(__file__).resolve().parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from analysis._common import load_best_point
from config import sim_params_from_point, reference_rows

_CONFIG_DIR = pathlib.Path(__file__).resolve().parent


def _detect_terrain_from_dir(run_dir: pathlib.Path) -> str:
    """Auto-detect config module name from results dir name.

    Strips timestamp prefix, then tries progressively shorter suffixes
    until a matching config_<suffix>.py is found. Also tries stripping
    trailing alpha chars from the last token (e.g. '065gate' → '065').

    E.g. '20260303T151416_step_065gate' tries:
      config_step_065gate.py → config_step_065.py → config_step.py
    """
    import re

    name = run_dir.name
    # Strip timestamp prefix (YYYYMMDDTHHMMSS_)
    parts = name.split("_", 1)
    suffix = parts[1] if len(parts) > 1 else name

    tokens = suffix.split("_")
    for end in range(len(tokens), 0, -1):
        candidate = "_".join(tokens[:end])
        if (_CONFIG_DIR / f"config_{candidate}.py").exists():
            return candidate
        # Try stripping trailing alpha from last token (e.g. 065gate → 065)
        trimmed = re.sub(r'[a-zA-Z]+$', '', tokens[end - 1])
        if trimmed and trimmed != tokens[end - 1]:
            candidate2 = "_".join(tokens[:end - 1] + [trimmed])
            if (_CONFIG_DIR / f"config_{candidate2}.py").exists():
                return candidate2

    return "flat"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=pathlib.Path,
                        help="Results directory with optimization_bests.csv")
    parser.add_argument("--ref-id", required=True,
                        help="Reference ID to run (e.g. scene1_f10, scene_wheel_f30)")
    parser.add_argument("--terrain", type=str, default=None,
                        help="Override auto-detected terrain (e.g. 'step_065', 'flat_tg')")
    parser.add_argument("--visualize", action="store_true",
                        help="Launch interactive viewer")
    parser.add_argument("--record", action="store_true",
                        help="Record video to run_dir/")
    parser.add_argument("--seed", type=int, default=99999,
                        help="RNG seed for yaw/position jitter (default: 99999)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Sim duration in seconds (default: from config)")

    # Camera
    parser.add_argument("--cam-azimuth", type=float, default=None)
    parser.add_argument("--cam-elevation", type=float, default=None)
    parser.add_argument("--cam-distance", type=float, default=None)
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--no-tracking", action="store_true",
                        help="Use fixed camera instead of tracking body")
    parser.add_argument("--slow-mo", type=float, default=1.0,
                        help="Slow-motion multiplier (2.0 = half speed)")

    args = parser.parse_args()

    # --- Load config and params ---
    terrain = args.terrain or _detect_terrain_from_dir(args.run_dir)
    config_mod = importlib.import_module(f"config_{terrain}")

    point = load_best_point(args.run_dir)
    sim_params = sim_params_from_point(point)

    import simulation_camera_noarrow as sim_module

    # --- Find matching ref ---
    MJCF_PATHS = dict(config_mod.MJCF_PATHS)
    SIM_DURATION = args.duration if args.duration is not None else config_mod.SIM_DURATION
    ref_rows = list(reference_rows(config_mod.REFERENCE_DATA))

    match = None
    for ref_row in ref_rows:
        rid = ref_row.get("id", f"{ref_row['scene']}_f{ref_row.get('ctrl_freq', 10.0):.0f}")
        if rid == args.ref_id:
            match = ref_row
            break

    if match is None:
        available = [r.get("id", f"{r['scene']}_f{r.get('ctrl_freq', 10.0):.0f}") for r in ref_rows]
        sys.exit(f"ERROR: ref-id '{args.ref_id}' not found.\nAvailable: {', '.join(available)}")

    scene = match["scene"]
    freq = match.get("ctrl_freq", 10.0)
    target = match["speed"]

    if scene not in MJCF_PATHS:
        sys.exit(f"ERROR: scene '{scene}' has no MJCF path in config_{terrain}")

    # --- Jitter setup ---
    is_rough = terrain.startswith("rough")
    sp = dict(sim_params)
    sp["drive_freq"] = freq

    extra_kw: dict = {}
    if is_rough:
        y_jitter = config_mod.Y_JITTER
        spawn_x = config_mod.SPAWN_X
        spawn_z = config_mod.SPAWN_Z_RAISE
        rng = np.random.default_rng(args.seed)
        y_offset = rng.uniform(-y_jitter, y_jitter)
        extra_kw["spawn_offset"] = (spawn_x, y_offset, spawn_z)
    else:
        yaw_jitter_deg = config_mod.INIT_YAW_JITTER_DEG
        extra_kw["init_yaw_jitter_deg"] = yaw_jitter_deg
        extra_kw["rng_seed"] = args.seed

    # --- Camera kwargs ---
    cam_kw: dict = {
        "cam_tracking": not args.no_tracking,
    }
    if args.cam_azimuth is not None:
        cam_kw["cam_azimuth"] = args.cam_azimuth
    if args.cam_elevation is not None:
        cam_kw["cam_elevation"] = args.cam_elevation
    if args.cam_distance is not None:
        cam_kw["cam_distance"] = args.cam_distance
    if args.cam_lookat is not None:
        cam_kw["cam_lookat"] = tuple(args.cam_lookat)
    if args.slow_mo != 1.0:
        cam_kw["slow_mo"] = args.slow_mo

    # --- Determine output path ---
    record_path = None
    if args.record:
        video_dir = args.run_dir / "video_custom"
        video_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        video_name = f"{ts}_{args.ref_id}_s{args.seed}.mp4"
        record_path = str(video_dir / video_name)

    # --- Run ---
    print(f"Ref: {args.ref_id}  |  scene={scene}  freq={freq}Hz  target={target*100:.1f}cm/s")
    print(f"Terrain: {terrain}  |  Seed: {args.seed}")
    if record_path:
        print(f"Recording to: {record_path}")
    if args.visualize:
        print("Launching viewer...")

    traj = sim_module.run_simulation(
        sp,
        mjcf_path=MJCF_PATHS[scene],
        sim_duration=SIM_DURATION,
        visualize=args.visualize and not args.record,
        record_path=record_path,
        ignore_stuck_detection=True,
        progress=True,
        **extra_kw,
        **cam_kw,
    )

    if traj is None:
        print("CRASH: simulation returned None (unstable)")
        return

    # --- Quick summary ---
    t0_idx = next((i for i, s in enumerate(traj) if s["time"] >= 0.1), 0)
    final = traj[-1]
    start = traj[t0_idx]
    dt = final["time"] - start["time"]
    if dt > 1e-6:
        vx = (final["pos"][0] - start["pos"][0]) / dt
        lat = abs(final["pos"][1] - start["pos"][1])
        print(f"\nResult: vx={vx*100:.2f} cm/s  |  target={target*100:.1f} cm/s  |  "
              f"err={abs(vx - target)/max(target, 1e-6)*100:.1f}%  |  lateral={lat*1000:.1f}mm")
    else:
        print("\nResult: trajectory too short to compute velocity")

    print(f"Duration: {final['time']:.2f}s  |  Steps: {len(traj)}")


if __name__ == "__main__":
    main()
