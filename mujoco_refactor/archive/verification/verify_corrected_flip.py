#!/usr/bin/env python3
"""Sanity check: verify corrected body flip with videos."""

from pathlib import Path
from config import sim_params_from_point, reference_rows
from simulation_fast import run_simulation
import numpy as np
import csv

# Load best params from flat optimizer
BEST_RUN = "results/20260219T142207_loose_fudge"
best_csv = Path(BEST_RUN) / "optimization_bests.csv"
rows = list(csv.DictReader(open(best_csv)))
best = rows[-1]

# Extract params
param_names = [
    'sliding_friction', 'torsional_friction', 'rolling_friction',
    'solref_timeconst', 'solref_dampratio', 'solimp_dmin', 'solimp_delta_d',
    'solimp_width', 'solimp_midpoint', 'solimp_power',
    'magnetic_moment_fudge', 'magnetic_field_fudge', 'dof_damping'
]
point = np.array([float(best[p]) for p in param_names])
params_base = sim_params_from_point(point)

# Test conditions: one per morphology
TEST_CONDITIONS = [
    ("scene_1", 30, "multi_milli_quad"),
    ("scene_2", 20, "multi_milli_quad"),
    ("scene_4", 30, "multi_milli_quad"),
    ("scene_wheel", 30, "wheel_milli_quad"),
]

# Video output
video_dir = Path("corrected_flip_videos")
video_dir.mkdir(exist_ok=True)

print("=" * 70)
print("SANITY CHECK: Corrected body flip (Y-axis 180° + Z offset)")
print("=" * 70)
print(f"Using params from: {BEST_RUN}")
print(f"Videos will be saved to: {video_dir}/\n")

ref_rows = reference_rows()
ref_velocities = {r["id"]: r["speed"] for r in ref_rows}

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

print(f"{'Scene':<20} {'Freq':>6} {'Velocity':>10} {'Ref':>10} {'Error%':>8}")
print("-" * 70)

for scene, freq, robot_dir in TEST_CONDITIONS:
    params = params_base.copy()
    params['drive_freq'] = freq

    mjcf_path = f"{robot_dir}/{scene}.xml"
    video_path = str(video_dir / f"{scene}_f{freq}.mp4")

    try:
        trajectory = run_simulation(
            params=params,
            mjcf_path=mjcf_path,
            sim_duration=3.0,
            rng_seed=42,
            record_path=video_path
        )
        vel = compute_velocity(trajectory, settle_time=0.5)

        # Get reference
        scene_for_ref = scene.replace("_", "") if scene.startswith("scene_") and scene[6:].isdigit() else scene
        ref_id = f"{scene_for_ref}_f{freq}"
        ref_vel = ref_velocities.get(ref_id, None)

        if ref_vel is not None:
            err_pct = (vel - ref_vel) / ref_vel * 100 if ref_vel != 0 else 0
            print(f"{scene:<20} {freq:>6} {vel*100:>9.2f} cm/s {ref_vel*100:>9.2f} {err_pct:>+7.1f}%")
        else:
            print(f"{scene:<20} {freq:>6} {vel*100:>9.2f} cm/s {'N/A':>10} {'N/A':>8}")

    except Exception as e:
        print(f"{scene:<20} {freq:>6} ERROR: {e}")

print("-" * 70)
print(f"\nVideos saved to: {video_dir}/")
print("Check videos to verify:")
print("  - Body geom is flipped 180° (front↔back)")
print("  - Legs/joints move correctly (not twisted)")
print("  - Robots locomote properly")
