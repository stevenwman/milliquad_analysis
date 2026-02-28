"""Energy analysis (Euler vs RK4) on rough terrain with zzz_rough_v2 params.

Computes W_ext, W_int, W_xfrc, W_joint, dE, Dissipation for each reference.
Also records RK4 videos.

Usage:
    uv run python energy_analysis_rough.py results/zzz_rough_v2
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from config_rough import (
    FLAT_LEAD,
    MJCF_PATHS,
    N_TILES,
    PIXELS_PER_SQUARE,
    REFERENCE_DATA,
    SETTLE_TIME,
    SIM_DURATION,
    TERRAIN_HEIGHT_MEAN,
    TERRAIN_HEIGHT_STD,
    TERRAIN_NX,
    TERRAIN_NY,
    TERRAIN_SEED,
    TERRAIN_SL,
    TERRAIN_Z_SAFE,
    sim_params_from_point,
    space,
)
from config_new import REFERENCE_DATA as FLAT_REFS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "utils"))
from terrain_mesh import generate_heightmap

PARAM_NAMES = [dim.name for dim in space]


def load_best_point(run_dir: pathlib.Path) -> list[float]:
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    if not rows:
        sys.exit(f"ERROR: no rows in {bests_csv}")
    best = rows[-1]
    multi_csv = run_dir / "multi_optimization_results.csv"
    if multi_csv.exists():
        best_id = best["id"]
        for row in csv.DictReader(open(multi_csv)):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    return [float(best[name]) for name in PARAM_NAMES]


def inject_rough_terrain(xml_path: str, out_xml: str, integrator: str | None = None) -> str:
    """Create rough terrain MJCF with optional integrator and energy flag."""
    import imageio.v3 as iio

    logical_heights = generate_heightmap(
        nX=TERRAIN_NX, nY=TERRAIN_NY,
        height_mean=TERRAIN_HEIGHT_MEAN, height_std=TERRAIN_HEIGHT_STD,
        z_safe=TERRAIN_Z_SAFE, seed=TERRAIN_SEED,
    ).T
    tiled_heights = np.tile(logical_heights, (1, N_TILES))
    hires = np.kron(tiled_heights, np.ones((PIXELS_PER_SQUARE, PIXELS_PER_SQUARE)))

    x_half = TERRAIN_NX * N_TILES * TERRAIN_SL / 2.0
    y_half = TERRAIN_NY * TERRAIN_SL / 2.0
    z_top = float(tiled_heights.max())
    z_bottom = 0.001

    img = (np.clip(hires / z_top, 0.0, 1.0) * np.iinfo(np.uint16).max).astype(np.uint16)
    png_path = str(pathlib.Path(out_xml).with_suffix(".png"))
    iio.imwrite(png_path, img)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # Arena memory for noslip + heightfield
    size_elem = root.find("size")
    if size_elem is None:
        size_elem = ET.SubElement(root, "size")
    size_elem.set("memory", "128M")

    # Option: integrator + energy flag
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    if integrator:
        option.set("integrator", integrator)
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("energy", "enable")

    # Heightfield asset
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    hf = ET.SubElement(asset, "hfield")
    hf.set("name", "rough_terrain")
    hf.set("file", str(pathlib.Path(png_path).resolve()))
    hf.set("size", f"{x_half} {y_half} {z_top} {z_bottom}")

    # Terrain geom
    geom = ET.SubElement(worldbody, "geom")
    geom.set("name", "rough_terrain_geom")
    geom.set("type", "hfield")
    geom.set("hfield", "rough_terrain")
    geom.set("pos", f"{FLAT_LEAD + x_half} 0.0 0.0")

    tree.write(out_xml)
    return out_xml


def _joint_axis_world(leg_xquat: np.ndarray) -> np.ndarray:
    """Joint axis [0,0,1] rotated into world frame by leg body quaternion."""
    w = leg_xquat[:, 0]
    x = leg_xquat[:, 1]
    y = leg_xquat[:, 2]
    z = leg_xquat[:, 3]
    return np.column_stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ])


def compute_energy_breakdown(traj: list[dict], settle_time: float) -> dict | None:
    """Compute W_ext, W_int, W_xfrc, W_joint, dE, Dissipation from trajectory."""
    start_idx = 0
    for i, s in enumerate(traj):
        if s["time"] >= settle_time:
            start_idx = i
            break
    active = traj[start_idx:]
    if len(active) < 2:
        return None
    if "tau_ext" not in active[0] or "omega" not in active[0]:
        return None

    n = len(active) - 1
    dt = np.empty(n)
    p_ext = np.empty(n)
    p_int = np.empty(n)
    p_joint = np.empty(n)

    for i in range(n):
        s = active[i]
        dt[i] = active[i + 1]["time"] - s["time"]

        tau_ext = s["tau_ext"]    # (4, 3)
        tau_int = s["tau_int"]    # (4, 3)
        omega = s["omega"]        # (4, 3) — full body angular velocity, pre-step
        axis = _joint_axis_world(s["leg_xquat"])  # (4, 3)
        jvel = s["joint_vel"]     # (4,)

        # Naive: P = Σ τ · ω
        p_ext[i] = np.sum(tau_ext * omega)
        p_int[i] = np.sum(tau_int * omega)

        # Joint-projected: P = Σ (τ · â) * q̇
        p_joint[i] = sum(np.dot(tau_ext[j], axis[j]) * jvel[j] for j in range(4))

    W_ext = np.sum(p_ext * dt) * 1e6    # J -> uJ
    W_int = np.sum(p_int * dt) * 1e6
    W_xfrc = W_ext + W_int
    W_joint = np.sum(p_joint * dt) * 1e6

    # dE from MuJoCo's data.energy (if available)
    if "energy" in active[0] and "energy" in active[-1]:
        E_start = sum(active[0]["energy"])
        E_end = sum(active[-1]["energy"])
        dE = (E_end - E_start) * 1e6
    else:
        dE = float("nan")

    dissip = W_xfrc - dE

    return {
        "W_ext": W_ext,
        "W_int": W_int,
        "W_xfrc": W_xfrc,
        "W_joint": W_joint,
        "dE": dE,
        "Dissip": dissip,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--record-rk4", action="store_true", default=True,
                        help="Record RK4 videos (default: True)")
    args = parser.parse_args()

    import simulation_fast_rough as sim

    # Monkey-patch _record_state to capture data.energy
    _orig_record = sim._record_state
    def _patched_record(trajectory, data, step_cache=None):
        _orig_record(trajectory, data, step_cache)
        trajectory[-1]["energy"] = data.energy.copy()
    sim._record_state = _patched_record

    point = load_best_point(args.run_dir)
    sim_params_base = sim_params_from_point(point)
    print(f"Loaded best params from {args.run_dir}")

    # Build references: all 15 flat combos, with rough targets where available
    rough_targets = {}
    for r in REFERENCE_DATA:
        rough_targets[(r["scene"], int(r["ctrl_freq"]))] = r["speed"]
    refs = []
    for row in FLAT_REFS:
        scene = row["scene"]
        freq = int(row["ctrl_freq"])
        refs.append({
            "scene": scene, "freq": freq,
            "target": rough_targets.get((scene, freq)),  # None if no rough target
            "id": f"{scene}_f{freq}",
        })

    x_half = TERRAIN_NX * N_TILES * TERRAIN_SL / 2.0
    spawn_offset = (FLAT_LEAD + x_half, 0.0, 0.01)

    # Generate terrain XMLs for each integrator
    euler_paths = {}
    rk4_paths = {}
    tmp_files = []
    for scene, xml_path in MJCF_PATHS.items():
        euler_xml = str(pathlib.Path(xml_path).parent / f"{scene}_rough_energy_euler_tmp.xml")
        rk4_xml = str(pathlib.Path(xml_path).parent / f"{scene}_rough_energy_rk4_tmp.xml")
        inject_rough_terrain(xml_path, euler_xml, integrator=None)
        inject_rough_terrain(xml_path, rk4_xml, integrator="RK4")
        euler_paths[scene] = euler_xml
        rk4_paths[scene] = rk4_xml
        tmp_files.extend([euler_xml, rk4_xml,
                          euler_xml.replace(".xml", ".png"),
                          rk4_xml.replace(".xml", ".png")])

    video_dir = args.run_dir / "rk4_videos"
    video_dir.mkdir(exist_ok=True)

    try:
        for integrator_name, paths in [("Euler", euler_paths), ("RK4", rk4_paths)]:
            print(f"\n{'='*70}")
            print(f"  {integrator_name} INTEGRATOR — Rough Terrain")
            print(f"{'='*70}")
            print(f"| Ref | vx (mm/s) | target | err% | W_ext | W_int | W_xfrc | W_joint | dE | Dissip |")
            print(f"|-----|-----------|--------|------|-------|-------|--------|---------|-----|--------|")

            for ref in refs:
                sp = dict(sim_params_base)
                sp["drive_freq"] = float(ref["freq"])
                mjcf = paths[ref["scene"]]

                record_path = None
                if integrator_name == "RK4" and args.record_rk4:
                    record_path = str(video_dir / f"{ref['id']}_rough_rk4.mp4")

                try:
                    traj = sim.run_simulation(
                        sp, mjcf_path=mjcf, sim_duration=SIM_DURATION,
                        visualize=False, progress=False,
                        spawn_offset=spawn_offset,
                        ignore_stuck_detection=True,
                        record_path=record_path,
                    )
                except Exception as e:
                    print(f"| {ref['id']} | CRASH: {e} |")
                    continue
                if traj is None:
                    print(f"| {ref['id']} | FAILED |")
                    continue

                # Velocity
                settle_idx = next((i for i, s in enumerate(traj) if s["time"] >= SETTLE_TIME), 0)
                dt_total = traj[-1]["time"] - traj[settle_idx]["time"]
                vx = (traj[-1]["pos"][0] - traj[settle_idx]["pos"][0]) / dt_total if dt_total > 1e-6 else 0.0
                if ref["target"] is not None and ref["target"] > 1e-9:
                    err_str = f"{abs(vx - ref['target']) / ref['target'] * 100:.1f}%"
                    target_str = f"{ref['target']*1000:.1f}"
                else:
                    err_str = "—"
                    target_str = "—"

                # Energy
                eb = compute_energy_breakdown(traj, SETTLE_TIME)
                if eb is None:
                    print(f"| {ref['id']} | {vx*1000:.1f} | {target_str} | {err_str} | NO ENERGY DATA |")
                    continue

                print(f"| {ref['id']} | {vx*1000:.1f} | {target_str} | {err_str} "
                      f"| {eb['W_ext']:+.1f} | {eb['W_int']:+.1f} | {eb['W_xfrc']:+.1f} "
                      f"| {eb['W_joint']:+.1f} | {eb['dE']:+.1f} | {eb['Dissip']:+.1f} |")

    finally:
        for p in tmp_files:
            path = pathlib.Path(p)
            if path.exists():
                path.unlink()
        print("\nCleaned up temp terrain XMLs")


if __name__ == "__main__":
    main()
