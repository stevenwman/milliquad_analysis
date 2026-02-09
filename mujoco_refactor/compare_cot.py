"""
Compare COT across top-N results for all scenes.

Usage:
    uv run python compare_cot.py
    uv run python compare_cot.py --top 5 --csv path/to/results.csv
"""

import argparse
import csv

import mujoco

from config import CSV_PATH, MJCF_PATHS
from visualize_rollout import _sim_params_from_csv_row, compute_locomotion_metrics
import simulation


def main():
    parser = argparse.ArgumentParser(description="Compare COT for top-N results across scenes.")
    parser.add_argument("--top", type=int, default=3, help="Number of top results to compare.")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to results CSV.")
    parser.add_argument("--duration", type=float, default=5.0, help="Sim duration (seconds).")
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

    # Load robot mass per scene
    scene_mass = {}
    for scene, path in MJCF_PATHS.items():
        model = mujoco.MjModel.from_xml_path(path)
        scene_mass[scene] = sum(model.body_mass)

    # Header
    scenes = list(MJCF_PATHS.keys())
    print(f"\n{'Rank':<6} {'ID':<10} {'Cost':<10}", end="")
    for s in scenes:
        print(f" {'Vel_'+s+'(m/s)':<14} {'COT_'+s:<12} {'Dist_'+s+'(mm)':<14} {'AvgPwr_'+s+'(W)':<16}", end="")
    print()
    print("-" * (36 + 56 * len(scenes)))

    total_sims = n * len(scenes)
    sim_num = 0
    for rank_idx in range(n):
        row = results[rank_idx]
        rank = rank_idx + 1

        try:
            sim_params = _sim_params_from_csv_row(row)
        except (KeyError, ValueError) as e:
            print(f"  #{rank}: skipped — {e}")
            continue

        row_data = {}
        for scene in scenes:
            sim_num += 1
            print(f"\r  Running sim {sim_num}/{total_sims} (rank {rank}, {scene})...", end="", flush=True)
            mjcf_path = MJCF_PATHS[scene]
            traj = simulation.run_simulation(
                sim_params,
                mjcf_path=mjcf_path,
                sim_duration=args.duration,
                visualize=False,
                ignore_stuck_detection=True,
                progress=True,
            )
            if traj:
                row_data[scene] = compute_locomotion_metrics(traj, scene_mass[scene])
            else:
                row_data[scene] = None

        # Clear progress line, print result row
        print(f"\r{rank:<6} {row['id']:<10} {float(row['cost']):<10.4f}", end="")
        for scene in scenes:
            vel_key = f"velocity_{scene}"
            csv_vel = float(row[vel_key]) if vel_key in row else float("nan")
            m = row_data[scene]
            if m:
                print(f" {csv_vel:<14.4f} {m['cot']:<12.4f} {m['total_distance']*1e3:<14.1f} {m['avg_power_ext']:<16.4e}", end="")
            else:
                print(f" {csv_vel:<14.4f} {'FAIL':<12} {'FAIL':<14} {'FAIL':<16}", end="")
        print()

    print()


if __name__ == "__main__":
    main()
