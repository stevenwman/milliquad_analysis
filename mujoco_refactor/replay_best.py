#!/usr/bin/env python3
"""Replay the exact 33 simulations (11 refs × 3 jitter trials) that produced a
specific best-cost result, using the same params and deterministic seeds.

Usage:
    uv run python replay_best.py results/20260219T142207_loose_fudge [--row 2236]

If --row is omitted, uses the last row of optimization_bests.csv and searches
multi_optimization_results.csv for the matching id to determine the global point
index (needed for exact seed reproduction).
"""

import argparse
import csv
import importlib
import pathlib
import sys
import time

import numpy as np

from config import (
    DEFAULT_CTRL_FREQ,
    INIT_JITTER_SEED,
    INIT_JITTER_TRIALS,
    INIT_YAW_JITTER_DEG,
    MJCF_PATHS,
    PITCH_RMS_TARGET_DEG,
    PITCH_RMS_WEIGHT,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    reference_rows,
    sim_params_from_point,
    space,
)
from optimizer import calculate_cost

SIM_MODULE = "simulation_fast"

REF_ROWS = reference_rows()
REF_INDEX_BY_ID = {row["id"]: i for i, row in enumerate(REF_ROWS)}
PARAM_NAMES = [dim.name for dim in space]


def load_params_from_csv_row(row: dict) -> list[float]:
    """Extract the optimizer point (list[float]) from a CSV row."""
    return [float(row[name]) for name in PARAM_NAMES]


def find_global_point_index(run_dir: pathlib.Path, target_id: str) -> int:
    """Find the 0-indexed row number of target_id in multi_optimization_results.csv."""
    multi_csv = run_dir / "multi_optimization_results.csv"
    with open(multi_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if row["id"] == target_id:
                return i
    raise ValueError(f"id {target_id!r} not found in {multi_csv}")


def main():
    parser = argparse.ArgumentParser(description="Replay best-cost simulations with exact seeds")
    parser.add_argument("run_dir", type=str, help="Results directory to replay from")
    parser.add_argument("--row", type=int, default=None,
                        help="Global point index (0-indexed row in multi CSV). "
                             "If omitted, auto-detected from last bests row.")
    parser.add_argument("--record", action="store_true",
                        help="Record videos (saved to run_dir/replay_*.mp4)")
    parser.add_argument("--visualize", action="store_true",
                        help="Open interactive viewer for each sim (serial)")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    # Find the best id from optimization_bests.csv, then load full-precision
    # params from multi_optimization_results.csv (bests CSV truncates to ~8 digits,
    # which causes butterfly-effect divergence in chaotic sims).
    bests_csv = run_dir / "optimization_bests.csv"
    bests_rows = list(csv.DictReader(open(bests_csv)))
    if not bests_rows:
        print(f"ERROR: no rows in {bests_csv}")
        sys.exit(1)
    best_id = bests_rows[-1]["id"]
    best_cost = float(bests_rows[-1]["cost"])

    # Determine global_point_index and load full-precision params from multi CSV
    if args.row is not None:
        global_idx = args.row
        multi_csv = run_dir / "multi_optimization_results.csv"
        multi_rows = list(csv.DictReader(open(multi_csv)))
        best = multi_rows[global_idx]
    else:
        global_idx = find_global_point_index(run_dir, best_id)
        multi_csv = run_dir / "multi_optimization_results.csv"
        multi_rows = list(csv.DictReader(open(multi_csv)))
        best = multi_rows[global_idx]
    point = load_params_from_csv_row(best)
    print(f"Replaying id={best_id}, cost={best_cost:.6f}, global_point_index={global_idx}")

    sim_params = sim_params_from_point(point)
    sim_module = importlib.import_module(SIM_MODULE)

    n_refs = len(REF_ROWS)
    n_trials = max(1, INIT_JITTER_TRIALS)

    print(f"\nRunning {n_refs} refs × {n_trials} trials = {n_refs * n_trials} sims\n")
    print(f"{'ref_id':<20} {'trial':>5} {'seed':>10} {'velocity':>9} {'cost':>10} {'lateral':>8} {'yaw':>5} {'time':>6}")
    print("-" * 82)

    all_ref_results = {}  # ref_id -> list of (cost, velocity, lateral, yaw)

    for ref_row in REF_ROWS:
        ref_id = ref_row["id"]
        ref_idx = REF_INDEX_BY_ID[ref_id]
        scene = ref_row["scene"]
        mjcf_path = MJCF_PATHS[scene]
        target_vel = ref_row["speed"]
        speed_std = ref_row.get("speed_std", 0.0)
        pitch_target = ref_row.get("pitch_amp_deg", PITCH_RMS_TARGET_DEG)
        pitch_weight = ref_row.get("pitch_weight", PITCH_RMS_WEIGHT)

        trial_results = []

        for trial_idx in range(n_trials):
            seed = INIT_JITTER_SEED + (global_idx * n_refs + ref_idx) * n_trials + trial_idx

            sim_params_scene = dict(sim_params)
            sim_params_scene["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)

            record_path = None
            if args.record:
                record_path = str(run_dir / f"replay_{ref_id}_t{trial_idx}.mp4")

            t0 = time.perf_counter()
            trajectory = sim_module.run_simulation(
                sim_params_scene,
                mjcf_path=mjcf_path,
                sim_duration=SIM_DURATION,
                visualize=args.visualize and not args.record,
                record_path=record_path,
                wall_timeout=SIMULATION_TIMEOUT,
                init_yaw_jitter_deg=INIT_YAW_JITTER_DEG,
                rng_seed=seed,
            )
            elapsed = time.perf_counter() - t0

            if trajectory is None:
                print(f"  {ref_id:<20} {trial_idx:>5} {seed:>10} {'FAILED':>9} {'1e6':>10} {'--':>8} {'--':>5} {elapsed:>5.1f}s")
                trial_results.append((1e6, 0.0, 0.0, 0.0))
            else:
                cost_data = calculate_cost(
                    trajectory, target_vel,
                    speed_std=speed_std,
                    pitch_target_deg=pitch_target,
                    pitch_weight=pitch_weight,
                    verbose=False,
                )
                vel = cost_data["avg_forward_velocity"]
                lat = cost_data.get("lateral_displacement", 0.0)
                yaw = cost_data.get("yaw_deviation_deg", 0.0)
                tc = cost_data["total_cost"]
                trial_results.append((tc, vel, lat, yaw))
                print(f"  {ref_id:<20} {trial_idx:>5} {seed:>10} {vel:>8.4f}  {tc:>9.6f} {lat*100:>6.1f}cm {yaw:>4.0f}° {elapsed:>5.1f}s")

        all_ref_results[ref_id] = trial_results

    # Print median summary (matches optimizer aggregation)
    print(f"\n{'='*82}")
    print(f"MEDIAN SUMMARY (matching optimizer aggregation)")
    print(f"{'ref_id':<20} {'target':>7} {'median_vel':>10} {'Δ%':>5} {'median_cost':>12} {'lat':>7} {'yaw':>5}")
    print("-" * 72)

    total_weighted_cost = 0.0
    vel_errors = []
    for ref_row in REF_ROWS:
        ref_id = ref_row["id"]
        target = ref_row["speed"]
        weight = ref_row.get("weight", 1.0)
        trials = all_ref_results[ref_id]
        costs = [t[0] for t in trials]
        vels = [t[1] for t in trials]
        lats = [t[2] for t in trials]
        yaws = [t[3] for t in trials]
        med_cost = float(np.median(costs))
        med_vel = float(np.median(vels))
        med_lat = float(np.median(lats))
        med_yaw = float(np.median(yaws))
        dpct = (med_vel - target) / target * 100 if target != 0 else 0
        total_weighted_cost += weight * med_cost
        vel_errors.append((med_vel - target) / target if target != 0 else 0)
        print(f"  {ref_id:<20} {target:>6.3f} {med_vel:>9.4f}  {dpct:>+4.0f}% {med_cost:>11.6f} {med_lat*100:>5.1f}cm {med_yaw:>4.0f}°")

    # Variance term
    rel_errors = [abs(e) for e in vel_errors]
    variance = float(np.var(rel_errors)) if len(rel_errors) > 1 else 0.0

    print(f"\n  Sum weighted ref costs: {total_weighted_cost:.6f}")
    print(f"  Velocity variance term: {variance:.6f}")
    print(f"  (Original reported cost: {best_cost:.6f})")


if __name__ == "__main__":
    main()
