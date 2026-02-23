"""
Compare COT across top-N results for all reference conditions.

By default, runs each (scene, frequency) pair from REFERENCE_DATA.
Use --scene and --freq to narrow down.

Usage:
    uv run python compare_cot.py
    uv run python compare_cot.py --top 5
    uv run python compare_cot.py --scene scene4 --freq 30
    uv run python compare_cot.py --freq 10 --freq 50
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco

from config import CSV_PATH, MJCF_PATHS, reference_rows
from visualize_rollout import _sim_params_from_csv_row, compute_locomotion_metrics
import importlib

# ---- Simulation module selector ----
# Switch between vectorized (4.68x faster) and original (bit-exact) simulation.
# Hot-swap: change to "simulation" to use original implementation.
SIM_MODULE = "simulation_fast"
_sim = importlib.import_module(SIM_MODULE)


def main():
    parser = argparse.ArgumentParser(description="Compare COT for top-N results across reference conditions.")
    parser.add_argument("--top", type=int, default=3, help="Number of top results to compare.")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to results CSV.")
    parser.add_argument("--duration", type=float, default=5.0, help="Sim duration (seconds).")
    parser.add_argument("--scene", type=str, default=None, choices=list(MJCF_PATHS.keys()),
                        help="Filter to a single scene (default: all).")
    parser.add_argument("--freq", type=float, action="append", default=None,
                        help="Filter to specific drive frequency/frequencies in Hz (repeatable, default: all).")
    args = parser.parse_args()

    try:
        with open(args.csv, "r") as f:
            results = sorted(csv.DictReader(f), key=lambda r: float(r["cost"]))
    except FileNotFoundError:
        print(f"Error: {args.csv} not found.")
        return

    n = min(args.top, len(results))
    if n == 0:
        print("No results in CSV.")
        return

    # Build list of (ref_id, scene, freq) conditions to evaluate
    ref_rows = reference_rows()
    if args.scene:
        ref_rows = [r for r in ref_rows if r["scene"] == args.scene]
    if args.freq:
        ref_rows = [r for r in ref_rows if r["ctrl_freq"] in args.freq]

    if not ref_rows:
        print("No reference rows match the given --scene/--freq filters.")
        return

    # Load robot mass per scene (only for scenes we'll actually use)
    needed_scenes = {r["scene"] for r in ref_rows}
    scene_mass = {}
    for scene in needed_scenes:
        model = mujoco.MjModel.from_xml_path(MJCF_PATHS[scene])
        scene_mass[scene] = sum(model.body_mass)

    # Header
    print(f"\n{'Rank':<6} {'ID':<10} {'Cost':<10}", end="")
    for ref in ref_rows:
        label = ref["id"]
        print(f" {'Vel_'+label:<18} {'COT_'+label:<14} {'Dist(mm)':<12} {'AvgPwr(W)':<14}", end="")
    print()
    print("-" * (26 + 58 * len(ref_rows)))

    total_sims = n * len(ref_rows)
    sim_num = 0
    for rank_idx in range(n):
        row = results[rank_idx]
        rank = rank_idx + 1

        try:
            sim_params = _sim_params_from_csv_row(row)
        except (KeyError, ValueError) as e:
            print(f"  #{rank}: skipped — {e}")
            continue

        ref_data = {}
        for ref in ref_rows:
            sim_num += 1
            ref_id = ref["id"]
            scene = ref["scene"]
            freq = ref["ctrl_freq"]
            print(f"\r  Running sim {sim_num}/{total_sims} (rank {rank}, {ref_id})...", end="", flush=True)
            mjcf_path = MJCF_PATHS[scene]
            sp = dict(sim_params)
            sp["drive_freq"] = freq
            traj = _sim.run_simulation(
                sp,
                mjcf_path=mjcf_path,
                sim_duration=args.duration,
                visualize=False,
                ignore_stuck_detection=True,
                progress=True,
            )
            if traj:
                ref_data[ref_id] = compute_locomotion_metrics(traj, scene_mass[scene])
            else:
                ref_data[ref_id] = None

        # Clear progress line, print result row
        print(f"\r{rank:<6} {row['id']:<10} {float(row['cost']):<10.4f}", end="")
        for ref in ref_rows:
            ref_id = ref["id"]
            vel_key = f"velocity_{ref_id}"
            csv_vel = float(row[vel_key]) if vel_key in row else float("nan")
            m = ref_data[ref_id]
            if m:
                print(f" {csv_vel:<18.4f} {m['cot']:<14.4f} {m['total_distance']*1e3:<12.1f} {m['avg_power_ext']:<14.4e}", end="")
            else:
                print(f" {csv_vel:<18.4f} {'FAIL':<14} {'FAIL':<12} {'FAIL':<14}", end="")
        print()

    print()


if __name__ == "__main__":
    main()
