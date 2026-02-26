#!/usr/bin/env python3
"""Evaluate robot on tiled rough terrain (3x identical patches).

Terrain is placed to the side (+X) of the robot spawn. Robot walks
from flat ground onto the terrain. No spawn-height hacks needed.

Usage:
    uv run python eval_rough_terrain.py results/XXXXX --scenes scene4 --freqs 30 --record
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import tempfile
import xml.etree.ElementTree as ET

import imageio.v3 as iio
import numpy as np

from config_new import (
    MJCF_PATHS,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    sim_params_from_point,
    space,
)
from optimizer_new import calculate_cost

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "utils"))
from terrain_mesh import generate_heightmap

import simulation_fast_new as sim_module

PARAM_NAMES = [dim.name for dim in space]

_ORIGINAL_INIT_POSE = sim_module._initialize_pose

# Module-level Y offset, set per-trial before calling run_simulation
_y_offset: float = 0.0

def _raised_init_pose(data, init_yaw_jitter_deg=0.0, rng=None):
    _ORIGINAL_INIT_POSE(data, init_yaw_jitter_deg=init_yaw_jitter_deg, rng=rng)
    data.qpos[0] += 0.03  # 3cm forward
    data.qpos[1] += _y_offset
    data.qpos[2] += 0.01  # 1cm up

sim_module._initialize_pose = _raised_init_pose

# ---------------------------------------------------------------------------
# Terrain params — match the OBJ generation command:
#   python terrain_mesh.py --nX 10 --nY 6 --sL 0.005 \
#     --height-mean 0.002 --height-std 0.001 --z-safe 0.00025 --seed 42
# ---------------------------------------------------------------------------
TERRAIN_NX = 10
TERRAIN_NY = 6
TERRAIN_SL = 0.005       # 5mm tile side
TERRAIN_HEIGHT_MEAN = 0.002
TERRAIN_HEIGHT_STD = 0.001
TERRAIN_Z_SAFE = 0.00025
TERRAIN_SEED = 42

N_TILES = 3              # how many copies to tile along +X
FLAT_LEAD = 0.025        # 25mm flat ground before terrain starts
PIXELS_PER_SQUARE = 8    # 8x8 pixels per logical tile (20 caused arena overflow, 1 oversmooths)

DEFAULT_FREQS = [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# Param loading
# ---------------------------------------------------------------------------

def load_best_point(run_dir: pathlib.Path) -> list[float]:
    bests_csv = run_dir / "optimization_bests.csv"
    bests_rows = list(csv.DictReader(open(bests_csv)))
    if not bests_rows:
        print(f"ERROR: no rows in {bests_csv}")
        sys.exit(1)
    best_id = bests_rows[-1]["id"]

    multi_csv = run_dir / "multi_optimization_results.csv"
    with open(multi_csv) as f:
        for row in csv.DictReader(f):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    raise ValueError(f"id {best_id!r} not found in {multi_csv}")


# ---------------------------------------------------------------------------
# MJCF terrain injection
# ---------------------------------------------------------------------------

_temp_xml_files: list[str] = []


def inject_tiled_rough(xml_path: str, tmp_dir: str) -> str:
    """Generate blocky hfield tiled N_TILES times and inject into MJCF."""
    logical_heights = generate_heightmap(
        nX=TERRAIN_NX, nY=TERRAIN_NY,
        height_mean=TERRAIN_HEIGHT_MEAN, height_std=TERRAIN_HEIGHT_STD,
        z_safe=TERRAIN_Z_SAFE, seed=TERRAIN_SEED,
    )  # shape: (TERRAIN_NX, TERRAIN_NY), all positive, min = z_safe

    # Transpose to (NY, NX) — MuJoCo hfield: image rows=Y, columns=X
    logical_heights = logical_heights.T

    # Tile N_TILES times along X (axis=1 = columns)
    tiled_heights = np.tile(logical_heights, (1, N_TILES))

    # Upsample for blocky appearance
    hires = np.kron(tiled_heights, np.ones((PIXELS_PER_SQUARE, PIXELS_PER_SQUARE)))

    # hfield size
    total_nx = TERRAIN_NX * N_TILES
    x_half = total_nx * TERRAIN_SL / 2.0
    y_half = TERRAIN_NY * TERRAIN_SL / 2.0

    z_top = float(tiled_heights.max())
    z_bottom = 0.001

    # Normalize to [0, 1]
    normalized = np.clip(hires / z_top, 0.0, 1.0)
    img = (normalized * np.iinfo(np.uint16).max).astype(np.uint16)

    hfield_png = str(pathlib.Path(tmp_dir) / "rough_terrain.png")
    iio.imwrite(hfield_png, img)
    hfield_png_abs = str(pathlib.Path(hfield_png).resolve())

    # Parse MJCF
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Add hfield asset
    hf = ET.SubElement(asset, "hfield")
    hf.set("name", "rough_terrain")
    hf.set("file", hfield_png_abs)
    hf.set("size", f"{x_half} {y_half} {z_top} {z_bottom}")

    # Place terrain ahead of robot in +X
    pos_x = FLAT_LEAD + x_half
    geom = ET.SubElement(worldbody, "geom")
    geom.set("name", "rough_terrain_geom")
    geom.set("type", "hfield")
    geom.set("hfield", "rough_terrain")
    geom.set("pos", f"{pos_x} 0.0 0.0")
    geom.set("rgba", "0.6 0.55 0.5 1")

    # Write alongside original for correct relative path resolution
    src_dir = pathlib.Path(xml_path).parent
    stem = pathlib.Path(xml_path).stem
    out_xml = str(src_dir / f"{stem}_rough_tmp.xml")
    _temp_xml_files.append(out_xml)
    tree.write(out_xml)
    return out_xml


def cleanup_temp_xmls():
    for path in _temp_xml_files:
        try:
            pathlib.Path(path).unlink(missing_ok=True)
        except OSError:
            pass
    _temp_xml_files.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate robot on tiled rough terrain")
    parser.add_argument("run_dir", type=str, help="Results directory with optimization CSVs")
    parser.add_argument("--scenes", nargs="+", type=str, default=None)
    parser.add_argument("--freqs", nargs="+", type=float, default=None)
    parser.add_argument("--duration", type=float, default=SIM_DURATION)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    scenes = list(MJCF_PATHS.keys())
    if args.scenes:
        scenes = [s for s in args.scenes if s in MJCF_PATHS]

    freqs = args.freqs if args.freqs else DEFAULT_FREQS
    configs = [{"id": f"{s}_f{int(f)}", "scene": s, "freq": f} for s in scenes for f in freqs]

    print(f"Loading best params from {run_dir} ...")
    point = load_best_point(run_dir)
    sim_params = sim_params_from_point(point)

    total_x_mm = TERRAIN_NX * TERRAIN_SL * 1000 * N_TILES
    print(f"Terrain: {N_TILES}x tiles, total {total_x_mm:.0f}mm, "
          f"flat lead {FLAT_LEAD*1000:.0f}mm, height std={TERRAIN_HEIGHT_STD*1000:.1f}mm")
    print(f"  {len(configs)} configs, duration={args.duration}s")
    print()

    try:
        with tempfile.TemporaryDirectory(prefix="eval_rough_") as tmp_dir:
            # --- Flat baseline (skip with --visualize for quick checks) ---
            flat_results = {}
            if not args.visualize:
                print("Running flat baseline ...")
                for cfg in configs:
                    sp = dict(sim_params)
                    sp["drive_freq"] = cfg["freq"]
                    traj = sim_module.run_simulation(
                        sp, mjcf_path=MJCF_PATHS[cfg["scene"]],
                        sim_duration=args.duration, wall_timeout=SIMULATION_TIMEOUT,
                    )
                    if traj is None:
                        flat_results[cfg["id"]] = None
                        print(f"  {cfg['id']:<20} FAIL")
                    else:
                        cd = calculate_cost(traj, target_velocity=1.0, verbose=False)
                        flat_results[cfg["id"]] = cd
                        print(f"  {cfg['id']:<20} {cd['avg_forward_velocity']*1000:>7.1f} mm/s")
                print()

            # --- Rough terrain (robot spawns at origin, walks onto terrain) ---
            print("Running rough terrain ...")
            terrain_results = {}
            for cfg in configs:
                mjcf = inject_tiled_rough(MJCF_PATHS[cfg["scene"]], tmp_dir)
                sp = dict(sim_params)
                sp["drive_freq"] = cfg["freq"]
                traj = sim_module.run_simulation(
                    sp, mjcf_path=mjcf,
                    sim_duration=args.duration, wall_timeout=SIMULATION_TIMEOUT,
                    visualize=args.visualize, ignore_stuck_detection=True,
                )
                if traj is None:
                    terrain_results[cfg["id"]] = None
                    print(f"  {cfg['id']:<20} FAIL")
                else:
                    cd = calculate_cost(traj, target_velocity=1.0, verbose=False)
                    terrain_results[cfg["id"]] = cd
                    print(f"  {cfg['id']:<20} {cd['avg_forward_velocity']*1000:>7.1f} mm/s")
                cleanup_temp_xmls()
            print()

            # --- Record videos ---
            if args.record:
                vid_dir = run_dir / "rough_terrain_videos"
                vid_dir.mkdir(exist_ok=True)
                print(f"Recording videos to {vid_dir} ...")
                for cfg in configs:
                    if terrain_results.get(cfg["id"]) is None:
                        continue
                    mjcf = inject_tiled_rough(MJCF_PATHS[cfg["scene"]], tmp_dir)
                    sp = dict(sim_params)
                    sp["drive_freq"] = cfg["freq"]
                    vid_path = str(vid_dir / f"{cfg['id']}_rough.mp4")
                    sim_module.run_simulation(
                        sp, mjcf_path=mjcf,
                        sim_duration=args.duration, wall_timeout=SIMULATION_TIMEOUT,
                        record_path=vid_path, ignore_stuck_detection=True,
                    )
                    cleanup_temp_xmls()
                    print(f"  Saved: {vid_path}")

    finally:
        cleanup_temp_xmls()

    # --- Results table ---
    print()
    hdr = f"  {'config':<20} {'rough(mm/s)':>11} {'flat(mm/s)':>10} {'ratio':>6} {'tumble':>7} {'lat(cm)':>7} {'yaw':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for cfg in configs:
        rid = cfg["id"]
        tr = terrain_results.get(rid)
        fr = flat_results.get(rid)

        if tr is None:
            print(f"  {rid:<20} {'FAIL':>11}")
            continue

        rough_mm = tr["avg_forward_velocity"] * 1000
        tumble_str = "Y" if tr["tumble_penalty"] > 0 else "N"
        lat_cm = tr["lateral_displacement"] * 100
        yaw_deg = tr["yaw_deviation_deg"]

        if fr is not None and fr["avg_forward_velocity"] > 1e-6:
            flat_mm = fr["avg_forward_velocity"] * 1000
            ratio = tr["avg_forward_velocity"] / fr["avg_forward_velocity"]
            print(f"  {rid:<20} {rough_mm:>10.1f} {flat_mm:>10.1f} {ratio:>6.2f}"
                  f"    {tumble_str:>3} {lat_cm:>6.2f} {yaw_deg:>5.1f}")
        else:
            flat_str = "FAIL" if fr is None else f"{fr['avg_forward_velocity']*1000:.1f}"
            print(f"  {rid:<20} {rough_mm:>10.1f} {flat_str:>10} {'--':>6}"
                  f"    {tumble_str:>3} {lat_cm:>6.2f} {yaw_deg:>5.1f}")
    print()


if __name__ == "__main__":
    main()
