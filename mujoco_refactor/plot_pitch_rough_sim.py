"""Pitch amplitude (RMS) and COT vs frequency on rough terrain — simulation.

Pitch: computed from FL and BL leg body positions (atan2 of dz/dx),
same as eval_sim_experimental.py. Detrend + RMS in steady state.

COT: reuses eval_cot_v2.py's joint-projected power formula.

Usage:
    uv run python plot_pitch_rough_sim.py results/zzz_rough_v2
    uv run python plot_pitch_rough_sim.py results/zzz_rough_v2 --flat  # overlay flat terrain
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 14
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from config_rough import (
    FLAT_LEAD,
    MJCF_PATHS,
    N_TILES,
    PIXELS_PER_SQUARE,
    SETTLE_TIME,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
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
from eval_cot_v2 import compute_locomotion_metrics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "utils"))
from terrain_mesh import generate_heightmap

PARAM_NAMES = [dim.name for dim in space]

SCENE_MORPH = {"scene1": "leg", "scene2": "2leg", "scene4": "4leg", "scene_wheel": "wheel"}
COLORS = {"leg": "#1E88E5", "2leg": "#FFC107", "4leg": "#007561", "wheel": "#D81B60"}
LABELS = {"leg": "L1", "2leg": "L2", "4leg": "L4", "wheel": "WR"}
FREQS = [10, 30, 50]

FL_IDX = 1  # front-left leg body in leg_xpos
BL_IDX = 3  # back-left leg body in leg_xpos


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


def inject_rough_terrain(xml_path: str, out_xml: str) -> str:
    import xml.etree.ElementTree as ET
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

    size_elem = root.find("size")
    if size_elem is None:
        size_elem = ET.SubElement(root, "size")
    size_elem.set("memory", "128M")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    hf = ET.SubElement(asset, "hfield")
    hf.set("name", "rough_terrain")
    hf.set("file", str(pathlib.Path(png_path).resolve()))
    hf.set("size", f"{x_half} {y_half} {z_top} {z_bottom}")

    geom = ET.SubElement(worldbody, "geom")
    geom.set("name", "rough_terrain_geom")
    geom.set("type", "hfield")
    geom.set("hfield", "rough_terrain")
    geom.set("pos", f"{FLAT_LEAD + x_half} 0.0 0.0")

    tree.write(out_xml)
    return out_xml


def compute_pitch_rms(traj: list[dict]) -> float:
    """Pitch amplitude RMS (degrees) from FL/BL positions, matching eval_sim_experimental.py."""
    fl_pos = np.array([s["leg_xpos"][FL_IDX] for s in traj])
    bl_pos = np.array([s["leg_xpos"][BL_IDX] for s in traj])
    t = np.array([s["time"] for s in traj])

    dx = fl_pos[:, 0] - bl_pos[:, 0]
    dz = fl_pos[:, 2] - bl_pos[:, 2]
    theta = np.degrees(np.unwrap(np.arctan2(dz, dx)))

    mask = t >= SETTLE_TIME
    if mask.sum() < 10:
        return 0.0
    theta_ss = theta[mask]
    return float(np.std(theta_ss))  # std = RMS of detrended signal


def main():
    parser = argparse.ArgumentParser(description="Pitch RMS + COT vs frequency on rough terrain (sim)")
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--flat", action="store_true", help="Also run on flat terrain for comparison")
    args = parser.parse_args()

    import simulation_fast_rough as sim

    point = load_best_point(args.run_dir)
    sim_params_base = sim_params_from_point(point)
    print(f"Loaded best params from {args.run_dir}")

    # Get robot masses for COT
    robot_masses: dict[str, float] = {}
    for scene, xml_path in MJCF_PATHS.items():
        model = mujoco.MjModel.from_xml_path(xml_path)
        robot_masses[scene] = float(sum(model.body_mass))

    # Generate rough terrain XMLs
    rough_paths = {}
    for scene, xml_path in MJCF_PATHS.items():
        out = str(pathlib.Path(xml_path).parent / f"{scene}_rough_pitch_tmp.xml")
        inject_rough_terrain(xml_path, out)
        rough_paths[scene] = out

    x_half = TERRAIN_NX * N_TILES * TERRAIN_SL / 2.0
    spawn_offset = (FLAT_LEAD + x_half, 0.0, 0.01)

    terrains = [("Rough Terrain", rough_paths, spawn_offset)]
    if args.flat:
        terrains.append(("Flat Terrain", dict(MJCF_PATHS), None))

    all_data = {}
    for terrain_label, paths, offset in terrains:
        print(f"\n=== {terrain_label} ===")
        data = {m: {"freqs": [], "pitches": [], "cots": []} for m in COLORS}

        for scene, morph in SCENE_MORPH.items():
            mjcf = paths[scene]
            mass = robot_masses[scene]
            for freq in FREQS:
                sp = dict(sim_params_base)
                sp["drive_freq"] = float(freq)
                print(f"  {scene} f{freq} ({morph})...", end="", flush=True)
                try:
                    traj = sim.run_simulation(
                        sp, mjcf_path=mjcf, sim_duration=SIM_DURATION,
                        visualize=False, progress=False,
                        spawn_offset=offset, ignore_stuck_detection=True,
                    )
                except Exception as e:
                    print(f" CRASHED: {e}")
                    continue
                if traj is None:
                    print(" failed")
                    continue

                pitch_rms = compute_pitch_rms(traj)
                metrics = compute_locomotion_metrics(traj, mass)
                cot = metrics["cot"] if metrics else float("nan")

                data[morph]["freqs"].append(freq)
                data[morph]["pitches"].append(pitch_rms)
                data[morph]["cots"].append(cot)
                print(f" pitch={pitch_rms:.2f}\u00b0  COT={cot:.1f}")

        all_data[terrain_label] = data

    # --- Print table ---
    for terrain_label, data in all_data.items():
        print(f"\n{terrain_label}:")
        print(f"  {'Morph':<8} {'Freq':<6} {'Pitch (\u00b0)':<10} {'COT':<8}")
        print("  " + "-" * 35)
        for morph in ("leg", "2leg", "4leg", "wheel"):
            for f, p, c in zip(data[morph]["freqs"], data[morph]["pitches"], data[morph]["cots"]):
                print(f"  {morph:<8} {f:<6} {p:<10.2f} {c:<8.1f}")

    # --- Plot: 2 rows (pitch, COT) × N terrain columns ---
    n_terrains = len(all_data)
    fig, axes = plt.subplots(2, n_terrains, figsize=(7 * n_terrains, 9), squeeze=False)

    for col, (terrain_label, data) in enumerate(all_data.items()):
        ax_pitch = axes[0, col]
        ax_cot = axes[1, col]

        for morph in ("leg", "2leg", "4leg", "wheel"):
            d = data[morph]
            if not d["freqs"]:
                continue
            ax_pitch.plot(d["freqs"], d["pitches"], "o-", color=COLORS[morph],
                          label=LABELS[morph], markersize=8, linewidth=2)
            ax_cot.plot(d["freqs"], d["cots"], "o-", color=COLORS[morph],
                        label=LABELS[morph], markersize=8, linewidth=2)

        ax_pitch.set_ylabel("Pitch Amplitude RMS (\u00b0)")
        ax_pitch.set_title(f"{terrain_label}")
        ax_pitch.set_xticks(FREQS)
        ax_pitch.set_xlim(5, 55)
        ax_pitch.legend(loc="upper left", fontsize=12)
        ax_pitch.grid(True, alpha=0.3)

        ax_cot.set_xlabel("Frequency (Hz)")
        ax_cot.set_ylabel("Cost of Transport")
        ax_cot.set_xticks(FREQS)
        ax_cot.set_xlim(5, 55)
        ax_cot.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = str(args.run_dir / "pitch_cot_rough_sim.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()

    # Cleanup
    for xml_path in rough_paths.values():
        for ext in (".xml", ".png"):
            p = pathlib.Path(xml_path).with_suffix(ext)
            if p.exists():
                p.unlink()
    print("Cleaned up terrain XMLs")


if __name__ == "__main__":
    main()
