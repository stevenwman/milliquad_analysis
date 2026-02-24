#!/usr/bin/env python3
"""Compare velocities on STEP terrain between original and body-flipped robot XMLs."""

import csv
import sys
from pathlib import Path
import numpy as np

from config import sim_params_from_point
from config_step import STEP_START_X, REFERENCE_DATA
from simulation_fast import run_simulation

# Step terrain XML suffix
STEP_SUFFIX = "_step_8x1mm_4.5L_50lead"

# Load reference data for step terrain
# Convert to lookup dict: {scene_freq: speed}
ref_velocities = {}
for r in REFERENCE_DATA:
    scene = r['scene']
    freq = int(r['ctrl_freq'])
    ref_id = f"{scene}_f{freq}"
    ref_velocities[ref_id] = r['speed']

# Load best params from specified run (flat terrain optimizer)
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

# Convert to sim params
params_base = sim_params_from_point(point)

# Test conditions: step terrain variants (using config_step infrastructure)
TEST_CONDITIONS = [
    ("scene_1" + STEP_SUFFIX, 30, "multi_milli_quad"),
    ("scene_2" + STEP_SUFFIX, 20, "multi_milli_quad"),
    ("scene_4" + STEP_SUFFIX, 30, "multi_milli_quad"),
    ("scene_wheel" + STEP_SUFFIX, 30, "wheel_milli_quad"),
]

# MJCF paths
def mjcf_path(scene, robot_dir, flipped=False):
    if flipped:
        return f"robot_bodyflip_test/{robot_dir}/{scene}.xml"
    else:
        return f"{robot_dir}/{scene}.xml"

def compute_velocity_step_aware(trajectory, step_start_x=STEP_START_X):
    """Compute forward velocity only after robot passes step_start_x (step-aware).

    Uses STEP_START_X from config_step.py (flat_lead length).
    """
    if not trajectory:
        return 0.0

    # Find first state after step_start_x
    start_state = None
    for state in trajectory:
        if state["pos"][0] >= step_start_x:
            start_state = state
            break

    if start_state is None:
        # Robot never reached steps
        return 0.0

    final_state = trajectory[-1]
    active_duration = final_state["time"] - start_state["time"]

    if active_duration < 1e-6:
        return 0.0

    forward_displacement = final_state["pos"][0] - start_state["pos"][0]
    return forward_displacement / active_duration

print("=" * 95)
print("VELOCITY COMPARISON (STEP TERRAIN): Original vs Body-Flipped vs Reference")
print("=" * 95)
print(f"{'Condition':<20} {'Ref':>10} {'Original':>10} {'Flipped':>10} {'Δflip':>10} {'Δflip%':>7} {'RefErr%':>8}")
print("-" * 95)

# Create video output directory
video_dir = Path("bodyflip_comparison_videos")
video_dir.mkdir(exist_ok=True)

results = []
for scene, freq, robot_dir in TEST_CONDITIONS:
    # Add drive_freq to params
    params = params_base.copy()
    params['drive_freq'] = freq

    # Run original
    mjcf_orig = mjcf_path(scene, robot_dir, flipped=False)
    scene_name = scene.replace(STEP_SUFFIX, "")
    video_orig = str(video_dir / f"{scene_name}_f{freq}_ORIGINAL.mp4")
    try:
        trajectory_orig = run_simulation(
            params=params,
            mjcf_path=mjcf_orig,
            sim_duration=5.0,  # Longer for step terrain
            rng_seed=42,
            record_path=video_orig
        )
        vel_orig = compute_velocity_step_aware(trajectory_orig)
    except Exception as e:
        print(f"  {scene[:20]+'...' if len(scene)>20 else scene:<30} ERROR (original): {e}")
        continue

    # Run flipped
    mjcf_flip = mjcf_path(scene, robot_dir, flipped=True)
    video_flip = str(video_dir / f"{scene_name}_f{freq}_FLIPPED.mp4")
    try:
        trajectory_flip = run_simulation(
            params=params,
            mjcf_path=mjcf_flip,
            sim_duration=5.0,
            rng_seed=42,
            record_path=video_flip
        )
        vel_flip = compute_velocity_step_aware(trajectory_flip)
    except Exception as e:
        print(f"  {scene[:20]+'...' if len(scene)>20 else scene:<30} ERROR (flipped): {e}")
        continue

    # Compute differences
    delta_cms = (vel_flip - vel_orig) * 100  # m/s to cm/s
    delta_pct = (vel_flip - vel_orig) / vel_orig * 100 if vel_orig != 0 else 0

    # Get reference velocity
    scene_short = scene.replace(STEP_SUFFIX, "")
    # Fix naming: scene_1 -> scene1, scene_2 -> scene2, but keep scene_wheel
    if scene_short.startswith("scene_") and scene_short[6:].isdigit():
        scene_for_ref = scene_short.replace("_", "")  # scene_1 -> scene1
    else:
        scene_for_ref = scene_short  # scene_wheel stays scene_wheel
    scene_id = f"{scene_short}_f{freq}"
    ref_id = f"{scene_for_ref}_f{freq}"
    ref_vel = ref_velocities.get(ref_id, None)

    if ref_vel is not None:
        ref_err_pct = (vel_flip - ref_vel) / ref_vel * 100 if ref_vel != 0 else 0
        print(f"  {scene_id:<20} {ref_vel*100:>10.2f}  {vel_orig*100:>10.2f}  {vel_flip*100:>10.2f}  {delta_cms:>+9.2f}  {delta_pct:>+6.1f}%  {ref_err_pct:>+6.1f}%")
    else:
        # No reference data
        print(f"  {scene_id:<20} {'N/A':>10}  {vel_orig*100:>10.2f}  {vel_flip*100:>10.2f}  {delta_cms:>+9.2f}  {delta_pct:>+6.1f}%  {'N/A':>8}")

    results.append({
        'condition': scene_short,
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
    print(f"\nINTERPRETATION (STEP TERRAIN):")
    if max_delta_pct < 5:
        print(f"  ✓ Body flip has NEGLIGIBLE effect (<5% change)")
        print(f"    → Safe to ignore or fix purely for visual correctness")
    elif max_delta_pct < 15:
        print(f"  ⚠ Body flip has MODERATE effect (5-15% change)")
        print(f"    → May need to investigate which orientation matches real robot")
    else:
        print(f"  ✗ Body flip has SIGNIFICANT effect (>15% change)")
        print(f"    → MUST correct orientation or refit params")

    print(f"\nVIDEOS SAVED:")
    print(f"  Directory: {video_dir}/")
    print(f"  Files: {len(results)*2} videos (original + flipped for each condition)")
else:
    print("\nNo successful comparisons completed.")
