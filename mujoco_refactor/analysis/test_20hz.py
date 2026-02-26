#!/usr/bin/env python3
"""Test best-fit params against 20Hz experimental data (unseen during training).

Also reports training-set (10/30/50 Hz) fits for context.
Runs 3 jitter trials per config (±2° yaw), aggregated via median — same as optimizer.

Usage:
    uv run python test_20hz.py results/20260219T142207_loose_fudge
"""

import argparse
import csv
import importlib
import pathlib
import sys

import numpy as np

from config import (
    DEFAULT_CTRL_FREQ,
    INIT_JITTER_TRIALS,
    INIT_YAW_JITTER_DEG,
    MJCF_PATHS,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    reference_rows,
    sim_params_from_point,
    space,
)
from optimizer import calculate_cost

SIM_MODULE = "simulation_fast"
PARAM_NAMES = [dim.name for dim in space]
JITTER_BASE_SEED = 99999  # fixed seed for reproducibility (not tied to optimizer point index)

# Corrected 20Hz experimental velocities (last-50% steady state, ddof=1).
# See experimental_data/docs/VELOCITY_SUMMARY_FLAT.md "20Hz Row Validation" section.
EXPERIMENTAL_20HZ = [
    {"id": "scene1_f20",      "scene": "scene1",      "ctrl_freq": 20.0, "speed": 0.1264, "speed_std": 0.0047},
    {"id": "scene2_f20",      "scene": "scene2",      "ctrl_freq": 20.0, "speed": 0.1131, "speed_std": 0.0420},
    {"id": "scene4_f20",      "scene": "scene4",      "ctrl_freq": 20.0, "speed": 0.1841, "speed_std": 0.0156},
    {"id": "scene_wheel_f20", "scene": "scene_wheel",  "ctrl_freq": 20.0, "speed": 0.3058, "speed_std": 0.0068},
]


def load_best_point(run_dir: pathlib.Path) -> list[float]:
    """Load full-precision params for the final best from multi CSV."""
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


def run_jitter_trials(sim_module, sim_params: dict, scene: str, freq: float,
                      ref_idx: int) -> dict | None:
    """Run N jitter trials, return median cost_data or None if all fail."""
    sp = dict(sim_params)
    sp["drive_freq"] = freq
    n_trials = max(1, INIT_JITTER_TRIALS)
    trial_results = []

    for trial_idx in range(n_trials):
        seed = JITTER_BASE_SEED + ref_idx * n_trials + trial_idx
        traj = sim_module.run_simulation(
            sp,
            mjcf_path=MJCF_PATHS[scene],
            sim_duration=SIM_DURATION,
            wall_timeout=SIMULATION_TIMEOUT,
            init_yaw_jitter_deg=INIT_YAW_JITTER_DEG,
            rng_seed=seed,
        )
        if traj is not None:
            trial_results.append(traj)

    if not trial_results:
        return None

    # Pick the trial whose forward velocity is the median (same as optimizer)
    vels = []
    cost_datas = []
    for traj in trial_results:
        cd = calculate_cost(traj, 1.0, verbose=False)  # target irrelevant, just extracting vel
        vels.append(cd["avg_forward_velocity"])
        cost_datas.append(cd)

    median_idx = int(np.argsort(vels)[len(vels) // 2])
    return cost_datas[median_idx]


def print_header():
    print(f"  {'id':<20} {'exp':>7} {'sim':>7} {'err%':>6} {'1σ?':>4}"
          f"  {'lat(cm)':>7} {'yaw°':>5} {'pitch°':>6} {'tumble':>7}")
    print("  " + "-" * 82)


def print_row(ref_id: str, target: float, std: float, cost_data: dict | None):
    if cost_data is None:
        print(f"  {ref_id:<20} {target*1000:>6.1f}  {'FAIL':>7} {'--':>6} {'--':>4}"
              f"  {'--':>7} {'--':>5} {'--':>6} {'--':>7}")
        return
    vel = cost_data["avg_forward_velocity"]
    lat = cost_data["lateral_displacement"]
    yaw = cost_data["yaw_deviation_deg"]
    pitch = cost_data["pitch_rms_deg"]
    tumble = cost_data["tumble_penalty"]
    err_pct = (vel - target) / target * 100
    within = abs(vel - target) <= std if std > 0 else False
    print(f"  {ref_id:<20} {target*1000:>6.1f}  {vel*1000:>6.1f}  {err_pct:>+5.1f}% {'Y' if within else 'N':>3}"
          f"  {lat*100:>6.2f}  {yaw:>5.1f}  {pitch:>5.2f}  {tumble:>7.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test 20Hz prediction vs experiment")
    parser.add_argument("run_dir", type=str, help="Results directory")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    point = load_best_point(run_dir)
    sim_params = sim_params_from_point(point)
    sim_module = importlib.import_module(SIM_MODULE)

    n_trials = max(1, INIT_JITTER_TRIALS)
    print(f"Jitter: ±{INIT_YAW_JITTER_DEG}° yaw, {n_trials} trials/config, median aggregation")
    print()

    # -- Training set (10/30/50 Hz) --
    ref_rows = reference_rows()
    print("=" * 90)
    print("TRAINING SET (10/30/50 Hz) — fitted during optimization")
    print_header()

    for ref_idx, ref in enumerate(ref_rows):
        ref_id = ref["id"]
        target = ref["speed"]
        std = ref.get("speed_std", 0.0)
        cost_data = run_jitter_trials(
            sim_module, sim_params, ref["scene"],
            ref.get("ctrl_freq", DEFAULT_CTRL_FREQ), ref_idx)
        print_row(ref_id, target, std, cost_data)

    # -- 20Hz holdout --
    print()
    print("=" * 90)
    print("20Hz HOLDOUT — unseen during training")
    print_header()

    offset = len(ref_rows)  # avoid seed collision with training set
    for i, ref in enumerate(EXPERIMENTAL_20HZ):
        ref_id = ref["id"]
        target = ref["speed"]
        std = ref["speed_std"]
        cost_data = run_jitter_trials(
            sim_module, sim_params, ref["scene"], ref["ctrl_freq"], offset + i)
        print_row(ref_id, target, std, cost_data)

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
