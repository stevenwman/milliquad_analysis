#!/usr/bin/env python3
"""Compare velocities between original and body-flipped robot XMLs."""

import csv
import sys
from pathlib import Path
import numpy as np

from config import sim_params_from_point
from simulation_fast import run_simulation

# Load best params from specified run
BEST_RUN = "results/20260219T142207_loose_fudge"
best_csv = Path(BEST_RUN) / "optimization_bests.csv"

if not best_csv.exists():
    print(f"Error: {best_csv} not found")
    sys.exit(1)

rows = list(csv.DictReader(open(best_csv)))
best = rows[-1]  # Last row = best params

print(f"Loading best params from: {BEST_RUN}")
print(f"  Cost: {best['cost']}")
print(f"  N_eval: {best['n_eval']}")
print()

# Extract param point from CSV
param_names = [
    'sliding_friction', 'torsional_friction', 'rolling_friction',
    'solref_timeconst', 'solref_dampratio', 'solimp_dmin', 'solimp_delta_d',
    'solimp_width', 'solimp_midpoint', 'solimp_power',
    'magnetic_moment_fudge', 'magnetic_field_fudge', 'dof_damping'
]
point = np.array([float(best[p]) for p in param_names])

# Convert to sim params (returns dict with ground_friction, solref, solimp, etc.)
params_base = sim_params_from_point(point)

# Test conditions: diverse morphologies and frequencies
TEST_CONDITIONS = [
    ("scene1", 30, "multi_milli_quad"),
    ("scene2", 20, "multi_milli_quad"),
    ("scene4", 30, "multi_milli_quad"),
    ("scene_wheel", 30, "wheel_milli_quad"),
]

# MJCF paths
def mjcf_path(scene, robot_dir, flipped=False):
    if flipped:
        return f"robot_bodyflip_test/{robot_dir}/{scene}.xml"
    else:
        return f"{robot_dir}/{scene}.xml"

def compute_velocity(trajectory, settle_time=0.5):
    """Compute forward velocity from trajectory."""
    if not trajectory:
        return 0.0

    final_state = trajectory[-1]
    start_state = trajectory[0]

    # Skip settle time
    for state in trajectory:
        if state["time"] >= settle_time:
            start_state = state
            break

    active_duration = final_state["time"] - start_state["time"]
    if active_duration < 1e-6:
        return 0.0

    forward_displacement = final_state["pos"][0] - start_state["pos"][0]
    return forward_displacement / active_duration

print("=" * 80)
print("VELOCITY COMPARISON: Original vs Body-Flipped")
print("=" * 80)
print(f"{'Condition':<20} {'Original':>12} {'Flipped':>12} {'Δ (cm/s)':>12} {'Δ%':>8}")
print("-" * 80)

results = []
for scene, freq, robot_dir in TEST_CONDITIONS:
    # Add drive_freq to params
    params = params_base.copy()
    params['drive_freq'] = freq

    # Run original
    mjcf_orig = mjcf_path(scene, robot_dir, flipped=False)
    try:
        trajectory_orig = run_simulation(
            params=params,
            mjcf_path=mjcf_orig,
            sim_duration=3.0,
            rng_seed=42
        )
        vel_orig = compute_velocity(trajectory_orig, settle_time=0.5)
    except Exception as e:
        print(f"  {scene}_f{freq:<14} ERROR (original): {e}")
        continue

    # Run flipped
    mjcf_flip = mjcf_path(scene, robot_dir, flipped=True)
    try:
        trajectory_flip = run_simulation(
            params=params,
            mjcf_path=mjcf_flip,
            sim_duration=3.0,
            rng_seed=42
        )
        vel_flip = compute_velocity(trajectory_flip, settle_time=0.5)
    except Exception as e:
        print(f"  {scene}_f{freq:<14} ERROR (flipped): {e}")
        continue

    # Compute difference
    delta_cms = (vel_flip - vel_orig) * 100  # m/s to cm/s
    delta_pct = (vel_flip - vel_orig) / vel_orig * 100 if vel_orig != 0 else 0

    print(f"  {scene}_f{freq:<14} {vel_orig*100:>10.2f}  {vel_flip*100:>10.2f}  {delta_cms:>+10.2f}  {delta_pct:>+6.1f}%")

    results.append({
        'condition': f"{scene}_f{freq}",
        'vel_orig': vel_orig,
        'vel_flip': vel_flip,
        'delta_cms': delta_cms,
        'delta_pct': delta_pct
    })

print("-" * 80)

if results:
    # Summary statistics
    deltas_cms = [r['delta_cms'] for r in results]
    deltas_pct = [abs(r['delta_pct']) for r in results]

    print(f"\nSUMMARY:")
    print(f"  Mean absolute velocity change: {np.mean(np.abs(deltas_cms)):.2f} cm/s")
    print(f"  Max absolute velocity change:  {np.max(np.abs(deltas_cms)):.2f} cm/s")
    print(f"  Mean absolute % change:        {np.mean(deltas_pct):.1f}%")
    print(f"  Max absolute % change:         {np.max(deltas_pct):.1f}%")

    # Interpretation
    max_delta_pct = np.max(deltas_pct)
    print(f"\nINTERPRETATION:")
    if max_delta_pct < 5:
        print(f"  ✓ Body flip has NEGLIGIBLE effect (<5% change)")
        print(f"    → Safe to ignore or fix purely for visual correctness")
    elif max_delta_pct < 15:
        print(f"  ⚠ Body flip has MODERATE effect (5-15% change)")
        print(f"    → May need to investigate which orientation matches real robot")
    else:
        print(f"  ✗ Body flip has SIGNIFICANT effect (>15% change)")
        print(f"    → MUST correct orientation or refit params")
else:
    print("\nNo successful comparisons completed.")
