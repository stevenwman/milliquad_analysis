#!/usr/bin/env python3
"""Test 16-dim optimized params on step terrain with jitter."""

import csv
import sys
from pathlib import Path
import numpy as np

from config_new import sim_params_from_point
from config_step import STEP_START_X, REFERENCE_DATA
from simulation_fast_new import run_simulation

# Step terrain suffix
STEP_SUFFIX = "_step_8x1mm_4.5L_50lead"

# Jitter config
N_JITTER_TRIALS = 5
JITTER_YAW_DEG = 5.0
BASE_SEED = 12345

# Load best params from 16-dim optimization
BEST_RUN = "results/20260223T193537_16dim_cold"
best_csv = Path(BEST_RUN) / "optimization_bests.csv"

if not best_csv.exists():
    print(f"Error: {best_csv} not found")
    sys.exit(1)

rows = list(csv.DictReader(open(best_csv)))
best = rows[-1]  # Last row = current best

print(f"=" * 80)
print(f"STEP TERRAIN VALIDATION: 16-dim optimized params")
print(f"=" * 80)
print(f"Source: {BEST_RUN}")
print(f"  Cost: {best['cost']}")
print(f"  N_eval: {best['n_eval']}")
print(f"  Elapsed: {best['elapsed_min']} min")
print()
print(f"Jitter: {N_JITTER_TRIALS} trials, ±{JITTER_YAW_DEG}° yaw variation")
print(f"Aggregation: BEST (min cost) trial")
print()

# Extract params (16 dimensions)
param_names = [
    'sliding_friction', 'torsional_friction', 'rolling_friction',
    'solref_timeconst', 'solref_dampratio', 'solimp_dmin', 'solimp_delta_d',
    'solimp_width', 'solimp_midpoint', 'solimp_power',
    'magnetic_moment_fudge', 'magnetic_field_fudge', 'dof_damping',
    'noslip_iterations', 'noslip_tolerance', 'margin'
]

# Handle numpy-wrapped strings
def parse_value(s):
    s = str(s).strip()
    if s.startswith('np.float64('):
        return float(s[len('np.float64('):-1])
    return float(s)

point = np.array([parse_value(best[p]) for p in param_names])
params_base = sim_params_from_point(point)

# Reference velocities
ref_velocities = {}
for r in REFERENCE_DATA:
    scene = r['scene']
    freq = int(r['ctrl_freq'])
    ref_id = f"{scene}_f{freq}"
    ref_velocities[ref_id] = r['speed']

# Test conditions (step terrain)
TEST_CONDITIONS = [
    ("scene_1" + STEP_SUFFIX, 10, "multi_milli_quad"),
    ("scene_1" + STEP_SUFFIX, 20, "multi_milli_quad"),
    ("scene_1" + STEP_SUFFIX, 30, "multi_milli_quad"),
    ("scene_2" + STEP_SUFFIX, 10, "multi_milli_quad"),
    ("scene_2" + STEP_SUFFIX, 20, "multi_milli_quad"),
    ("scene_2" + STEP_SUFFIX, 30, "multi_milli_quad"),
    ("scene_4" + STEP_SUFFIX, 10, "multi_milli_quad"),
    ("scene_4" + STEP_SUFFIX, 20, "multi_milli_quad"),
    ("scene_4" + STEP_SUFFIX, 30, "multi_milli_quad"),
    ("scene_wheel" + STEP_SUFFIX, 10, "wheel_milli_quad"),
    ("scene_wheel" + STEP_SUFFIX, 20, "wheel_milli_quad"),
    ("scene_wheel" + STEP_SUFFIX, 30, "wheel_milli_quad"),
]

def compute_velocity_step_aware(trajectory, step_start_x=STEP_START_X):
    """Compute forward velocity only after robot passes step_start_x."""
    if not trajectory:
        return 0.0

    # Find first state after step_start_x
    start_state = None
    for state in trajectory:
        if state["pos"][0] >= step_start_x:
            start_state = state
            break

    if start_state is None:
        return 0.0

    final_state = trajectory[-1]
    active_duration = final_state["time"] - start_state["time"]

    if active_duration < 1e-6:
        return 0.0

    forward_displacement = final_state["pos"][0] - start_state["pos"][0]
    return forward_displacement / active_duration

def compute_cost_components(trajectory, target_vel, step_start_x=STEP_START_X):
    """Compute cost components for step terrain."""
    if not trajectory:
        return {"velocity": 1e6, "lateral": 1e6, "tumble": 1e6}

    vel = compute_velocity_step_aware(trajectory, step_start_x)
    if target_vel is not None and target_vel > 0:
        vel_cost = abs(vel - target_vel) / target_vel
    else:
        vel_cost = 0.0  # No reference, just use zero cost for sorting

    # Lateral displacement
    lateral = abs(trajectory[-1]["pos"][1]) if trajectory else 0.0

    # Tumble (pitch/roll RMS after step start)
    tumbles = []
    for state in trajectory:
        if state["pos"][0] >= step_start_x:
            quat = state["quat"]
            # Convert quat to euler (simplified - just check for large rotations)
            pitch = 2 * np.arcsin(np.clip(quat[2], -1, 1))
            roll = 2 * np.arctan2(quat[1], quat[0])
            tumbles.append(pitch**2 + roll**2)

    tumble = np.sqrt(np.mean(tumbles)) if tumbles else 0.0

    return {"velocity": vel_cost, "lateral": lateral, "tumble": tumble, "vel_sim": vel}

# Video output directory
video_dir = Path("step_videos_16dim")
video_dir.mkdir(exist_ok=True)

print(f"{'Condition':<20} {'Ref':>10} {'Best':>10} {'Worst':>10} {'Spread':>8} {'RefErr%':>8}")
print("-" * 80)

results = []
for scene, freq, robot_dir in TEST_CONDITIONS:
    params = params_base.copy()
    params['drive_freq'] = freq

    mjcf_path = f"{robot_dir}/{scene}.xml"
    scene_short = scene.replace(STEP_SUFFIX, "")

    # Fix naming for reference lookup
    if scene_short.startswith("scene_") and scene_short[6:].isdigit():
        scene_for_ref = scene_short.replace("_", "")
    else:
        scene_for_ref = scene_short

    ref_id = f"{scene_for_ref}_f{freq}"
    ref_vel = ref_velocities.get(ref_id, None)  # None if no reference data

    # Run jittered trials
    trial_results = []
    for trial_idx in range(N_JITTER_TRIALS):
        rng_seed = BASE_SEED + trial_idx
        try:
            trajectory = run_simulation(
                params=params,
                mjcf_path=mjcf_path,
                sim_duration=5.0,
                rng_seed=rng_seed,
                init_yaw_jitter_deg=JITTER_YAW_DEG,
            )
            cost_components = compute_cost_components(trajectory, ref_vel)
            total_cost = (5.0 * cost_components["velocity"] +
                         1.0 * cost_components["lateral"] +
                         1.0 * cost_components["tumble"])
            trial_results.append({
                "cost": total_cost,
                "vel": cost_components["vel_sim"],
                "lateral": cost_components["lateral"],
                "tumble": cost_components["tumble"],
            })
        except Exception as e:
            print(f"  {scene_short}_f{freq:<14} ERROR trial {trial_idx}: {e}")
            continue

    if not trial_results:
        continue

    # Pick best (min cost) trial
    best_trial_idx = min(range(len(trial_results)), key=lambda i: trial_results[i]["cost"])
    best_trial = trial_results[best_trial_idx]
    worst_trial = max(trial_results, key=lambda x: x["cost"])

    vel_best = best_trial["vel"]
    vel_worst = worst_trial["vel"]
    vel_spread = abs(vel_best - vel_worst)

    if ref_vel is not None:
        ref_err_pct = (vel_best - ref_vel) / ref_vel * 100 if ref_vel > 0 else 0
        ref_str = f"{ref_vel*100:>9.2f}"
        err_str = f"{ref_err_pct:>+7.1f}%"
    else:
        ref_err_pct = None
        ref_str = "N/A"
        err_str = "N/A"

    scene_id = f"{scene_short}_f{freq}"
    print(f"{scene_id:<20} {ref_str:>10}  {vel_best*100:>9.2f}  {vel_worst*100:>9.2f}  {vel_spread*100:>7.2f}  {err_str:>8}")

    # Re-run best trial with video recording (skip if video already exists)
    best_rng_seed = BASE_SEED + best_trial_idx
    video_path = str(video_dir / f"{scene_id}.mp4")
    if not Path(video_path).exists():
        try:
            run_simulation(
                params=params,
                mjcf_path=mjcf_path,
                sim_duration=5.0,
                rng_seed=best_rng_seed,
                init_yaw_jitter_deg=JITTER_YAW_DEG,
                record_path=video_path,
            )
        except Exception as e:
            print(f"  WARNING: Video recording failed for {scene_id}: {e}")

    results.append({
        "condition": scene_id,
        "ref_vel": ref_vel,
        "vel_best": vel_best,
        "vel_worst": vel_worst,
        "spread": vel_spread,
        "ref_err_pct": ref_err_pct,
    })

print("-" * 80)

if results:
    # Calculate stats only for conditions with reference data
    results_with_ref = [r for r in results if r["ref_err_pct"] is not None]

    if results_with_ref:
        mean_err = np.mean([abs(r["ref_err_pct"]) for r in results_with_ref])
        max_err = np.max([abs(r["ref_err_pct"]) for r in results_with_ref])
        print(f"\nSUMMARY:")
        print(f"  Mean absolute ref error: {mean_err:.1f}% ({len(results_with_ref)} conditions with reference)")
        print(f"  Max absolute ref error:  {max_err:.1f}%")
    else:
        print(f"\nSUMMARY:")
        print(f"  No reference data for comparison")

    mean_spread = np.mean([r["spread"] * 100 for r in results])
    print(f"  Mean jitter spread:      {mean_spread:.2f} cm/s")
    print(f"\nJitter spread = |best - worst| velocity across {N_JITTER_TRIALS} trials")
    print(f"\nVideos of best trials saved to: {video_dir}/")
else:
    print("\nNo successful tests completed.")
