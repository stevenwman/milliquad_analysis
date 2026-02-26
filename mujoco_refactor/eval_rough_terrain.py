#!/usr/bin/env python3
"""Evaluate robot on tiled rough terrain (3× identical patches).

Generates a blocky hfield (matching the physical OBJ terrain) and tiles it
N times along +X, then runs sim evals for all scene/freq combos.

Usage:
    uv run python eval_rough_terrain.py results/XXXXX
    uv run python eval_rough_terrain.py results/XXXXX --scenes scene1 scene4
    uv run python eval_rough_terrain.py results/XXXXX --freqs 10 30
    uv run python eval_rough_terrain.py results/XXXXX --record
    uv run python eval_rough_terrain.py results/XXXXX --visualize --scenes scene4 --freqs 30
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

# Add utils/ to path for terrain_mesh import
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "utils"))
from terrain_mesh import generate_heightmap

import simulation_fast_new as sim_module

PARAM_NAMES = [dim.name for dim in space]
_ORIGINAL_Z_HEIGHT = sim_module.INITIAL_Z_HEIGHT

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
FLAT_LEAD = -0.03        # terrain shifted back so robot spawns 3cm into it
PIXELS_PER_SQUARE = 20   # upsampling for blocky appearance

DEFAULT_FREQS = [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# Param loading
# ---------------------------------------------------------------------------

def load_best_point(run_dir: pathlib.Path) -> list[float]:
    """Load full-precision params for the final best from optimization CSVs."""
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
    """Generate blocky hfield tiled N_TILES times and inject into MJCF.

    Follows the same approach as mujoco/visualize_rollout_rough.py:
    1. Generate logical heights per tile (same seed as physical OBJ)
    2. Tile the heightmap N_TILES times in X
    3. Upsample with np.kron for blocky flat-topped tiles
    4. Save as PNG, inject hfield asset + geom into MJCF
    """
    # Generate logical heights (same as terrain_mesh.py with these params)
    logical_heights = generate_heightmap(
        nX=TERRAIN_NX, nY=TERRAIN_NY,
        height_mean=TERRAIN_HEIGHT_MEAN, height_std=TERRAIN_HEIGHT_STD,
        z_safe=TERRAIN_Z_SAFE, seed=TERRAIN_SEED,
    )  # shape: (TERRAIN_NX, TERRAIN_NY), all positive, min = z_safe

    # Transpose to (NY, NX) — MuJoCo hfield: image rows=Y, columns=X
    logical_heights = logical_heights.T  # (NY, NX)

    # Tile N_TILES times along X (axis=1 = columns)
    tiled_heights = np.tile(logical_heights, (1, N_TILES))  # (NY, N_TILES*NX)

    # Upsample for blocky appearance (each tile → PIXELS_PER_SQUARE pixels)
    hires = np.kron(tiled_heights, np.ones((PIXELS_PER_SQUARE, PIXELS_PER_SQUARE)))

    # hfield size
    total_nx = TERRAIN_NX * N_TILES
    x_half = total_nx * TERRAIN_SL / 2.0
    y_half = TERRAIN_NY * TERRAIN_SL / 2.0

    z_top = float(tiled_heights.max())
    z_bottom = 0.001  # small positive base (MuJoCo requires > 0)

    # MuJoCo hfield: elevation = normalized_data * z_top
    # So normalize heights to [0, 1] by dividing by z_top
    normalized = hires / z_top
    normalized = np.clip(normalized, 0.0, 1.0)
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

    # Floor left intact — acts as safety net under terrain dips

    # Add hfield asset
    hf = ET.SubElement(asset, "hfield")
    hf.set("name", "rough_terrain")
    hf.set("file", hfield_png_abs)
    hf.set("size", f"{x_half} {y_half} {z_top} {z_bottom}")

    # Lower terrain so min surface ≈ floor level (z=0) for smooth transition.
    # Surface ranges from ~0 to z_top - z_safe above floor.
    pos_x = FLAT_LEAD + x_half
    pos_z = 0.0  # terrain sits on floor, robot drops onto it

    geom = ET.SubElement(worldbody, "geom")
    geom.set("name", "rough_terrain_geom")
    geom.set("type", "hfield")
    geom.set("hfield", "rough_terrain")
    geom.set("pos", f"{pos_x} 0.0 {pos_z}")
    geom.set("rgba", "0.6 0.55 0.5 1")

    # Write alongside original for correct relative path resolution
    src_dir = pathlib.Path(xml_path).parent
    stem = pathlib.Path(xml_path).stem
    out_xml = str(src_dir / f"{stem}_rough_tmp.xml")
    _temp_xml_files.append(out_xml)
    tree.write(out_xml)
    return out_xml, z_top


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
    parser = argparse.ArgumentParser(
        description="Evaluate robot on tiled rough terrain",
    )
    parser.add_argument("run_dir", type=str,
                        help="Results directory with optimization CSVs")
    parser.add_argument("--scenes", nargs="+", type=str, default=None,
                        help="Filter morphologies (e.g. scene1 scene4)")
    parser.add_argument("--freqs", nargs="+", type=float, default=None,
                        help=f"Frequencies to test (default: {DEFAULT_FREQS})")
    parser.add_argument("--duration", type=float, default=SIM_DURATION,
                        help=f"Sim duration in seconds (default: {SIM_DURATION})")
    parser.add_argument("--record", action="store_true",
                        help="Save mp4 videos")
    parser.add_argument("--visualize", action="store_true",
                        help="Open MuJoCo viewer (runs one config at a time)")

    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    scenes = list(MJCF_PATHS.keys())
    if args.scenes:
        scenes = [s for s in args.scenes if s in MJCF_PATHS]
        if not scenes:
            print(f"ERROR: no valid scenes in {args.scenes}")
            print("Available:", ", ".join(MJCF_PATHS.keys()))
            sys.exit(1)

    freqs = args.freqs if args.freqs else DEFAULT_FREQS

    # Build config list
    configs = []
    for scene in scenes:
        for freq in freqs:
            configs.append({"id": f"{scene}_f{int(freq)}", "scene": scene, "freq": freq})

    # Load best params
    print(f"Loading best params from {run_dir} ...")
    point = load_best_point(run_dir)
    sim_params = sim_params_from_point(point)

    # Terrain geometry summary
    tile_x_mm = TERRAIN_NX * TERRAIN_SL * 1000
    tile_y_mm = TERRAIN_NY * TERRAIN_SL * 1000
    total_x_mm = tile_x_mm * N_TILES
    print(f"Terrain: {N_TILES}x tiles, each {tile_x_mm:.0f}x{tile_y_mm:.0f}mm, "
          f"total {total_x_mm:.0f}mm after {FLAT_LEAD*1000:.0f}mm flat lead")
    print(f"  height: mean={TERRAIN_HEIGHT_MEAN*1000:.1f}mm, "
          f"std={TERRAIN_HEIGHT_STD*1000:.1f}mm, z_safe={TERRAIN_Z_SAFE*1000:.2f}mm, seed={TERRAIN_SEED}")
    print(f"  {len(configs)} configs, duration={args.duration}s")
    print()

    try:
        with tempfile.TemporaryDirectory(prefix="eval_rough_") as tmp_dir:
            # --- Run flat baseline ---
            print("Running flat baseline ...")
            flat_results = {}
            for cfg in configs:
                sp = dict(sim_params)
                sp["drive_freq"] = cfg["freq"]
                traj = sim_module.run_simulation(
                    sp,
                    mjcf_path=MJCF_PATHS[cfg["scene"]],
                    sim_duration=args.duration,
                    wall_timeout=SIMULATION_TIMEOUT,
                )
                if traj is None:
                    flat_results[cfg["id"]] = None
                    print(f"  {cfg['id']:<20} FAIL")
                else:
                    cd = calculate_cost(traj, target_velocity=1.0, verbose=False)
                    flat_results[cfg["id"]] = cd
                    print(f"  {cfg['id']:<20} {cd['avg_forward_velocity']*1000:>7.1f} mm/s")
            print()

            # --- Run rough terrain ---
            print("Running rough terrain ...")
            terrain_results = {}
            terrain_z_top = None
            for cfg in configs:
                mjcf, terrain_z_top = inject_tiled_rough(MJCF_PATHS[cfg["scene"]], tmp_dir)
                sp = dict(sim_params)
                sp["drive_freq"] = cfg["freq"]

                # Spawn robot above the terrain
                sim_module.INITIAL_Z_HEIGHT = terrain_z_top + 0.002
                traj = sim_module.run_simulation(
                    sp,
                    mjcf_path=mjcf,
                    sim_duration=args.duration,
                    wall_timeout=SIMULATION_TIMEOUT,
                    visualize=args.visualize,
                )
                if traj is None:
                    terrain_results[cfg["id"]] = None
                    print(f"  {cfg['id']:<20} FAIL")
                else:
                    cd = calculate_cost(traj, target_velocity=1.0, verbose=False)
                    terrain_results[cfg["id"]] = cd
                    print(f"  {cfg['id']:<20} {cd['avg_forward_velocity']*1000:>7.1f} mm/s")

                # Clean up temp XML after each config to avoid stale files
                cleanup_temp_xmls()

            # Restore original Z height
            sim_module.INITIAL_Z_HEIGHT = _ORIGINAL_Z_HEIGHT

            # --- Record videos ---
            if args.record:
                vid_dir = run_dir / "rough_terrain_videos"
                vid_dir.mkdir(exist_ok=True)
                print(f"\nRecording videos to {vid_dir} ...")
                sim_module.INITIAL_Z_HEIGHT = terrain_z_top + 0.002
                for cfg in configs:
                    if terrain_results.get(cfg["id"]) is None:
                        continue
                    mjcf, _ = inject_tiled_rough(MJCF_PATHS[cfg["scene"]], tmp_dir)
                    sp = dict(sim_params)
                    sp["drive_freq"] = cfg["freq"]
                    vid_path = str(vid_dir / f"{cfg['id']}_rough.mp4")
                    sim_module.run_simulation(
                        sp,
                        mjcf_path=mjcf,
                        sim_duration=args.duration,
                        wall_timeout=SIMULATION_TIMEOUT,
                        record_path=vid_path,
                    )
                    cleanup_temp_xmls()
                    print(f"  Saved: {vid_path}")

    finally:
        cleanup_temp_xmls()

    # --- Print results table ---
    print()
    print(f"  Rough terrain: {N_TILES}x tiles ({total_x_mm:.0f}mm), "
          f"std={TERRAIN_HEIGHT_STD*1000:.1f}mm")
    print()
    hdr = f"  {'config':<20} {'rough(mm/s)':>11} {'flat(mm/s)':>10} {'ratio':>6} {'tumble':>7} {'lat(cm)':>7} {'yaw':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for cfg in configs:
        ref_id = cfg["id"]
        tr = terrain_results.get(ref_id)
        fr = flat_results.get(ref_id)

        if tr is None:
            print(f"  {ref_id:<20} {'FAIL':>11}")
            continue

        rough_mm = tr["avg_forward_velocity"] * 1000
        tumble_str = "Y" if tr["tumble_penalty"] > 0 else "N"
        lat_cm = tr["lateral_displacement"] * 100
        yaw_deg = tr["yaw_deviation_deg"]

        if fr is not None and fr["avg_forward_velocity"] > 1e-6:
            flat_mm = fr["avg_forward_velocity"] * 1000
            ratio = tr["avg_forward_velocity"] / fr["avg_forward_velocity"]
            print(f"  {ref_id:<20} {rough_mm:>10.1f} {flat_mm:>10.1f} {ratio:>6.2f}"
                  f"    {tumble_str:>3} {lat_cm:>6.2f} {yaw_deg:>5.1f}")
        else:
            flat_str = "FAIL" if fr is None else f"{fr['avg_forward_velocity']*1000:.1f}"
            print(f"  {ref_id:<20} {rough_mm:>10.1f} {flat_str:>10} {'--':>6}"
                  f"    {tumble_str:>3} {lat_cm:>6.2f} {yaw_deg:>5.1f}")

    print()


if __name__ == "__main__":
    main()
