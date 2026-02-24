#!/usr/bin/env python3
"""Record videos of 16-dim params on step terrain to visualize failures."""

import csv
from pathlib import Path
import numpy as np

from config_new import sim_params_from_point
from simulation_fast_new import run_simulation

# Load best params from 16-dim optimization
BEST_RUN = "results/20260223T193537_16dim_cold"
best_csv = Path(BEST_RUN) / "optimization_bests.csv"

rows = list(csv.DictReader(open(best_csv)))
best = rows[-1]

# Extract params (16 dimensions)
param_names = [
    'sliding_friction', 'torsional_friction', 'rolling_friction',
    'solref_timeconst', 'solref_dampratio', 'solimp_dmin', 'solimp_delta_d',
    'solimp_width', 'solimp_midpoint', 'solimp_power',
    'magnetic_moment_fudge', 'magnetic_field_fudge', 'dof_damping',
    'noslip_iterations', 'noslip_tolerance', 'margin'
]

def parse_value(s):
    s = str(s).strip()
    if s.startswith('np.float64('):
        return float(s[len('np.float64('):-1])
    return float(s)

point = np.array([parse_value(best[p]) for p in param_names])
params_base = sim_params_from_point(point)

# Video output directory
video_dir = Path("step_videos_16dim")
video_dir.mkdir(exist_ok=True)

# Key test cases to record
STEP_SUFFIX = "_step_8x1mm_4.5L_50lead"
TEST_CASES = [
    # Failure cases
    ("scene_1" + STEP_SUFFIX, 30, "multi_milli_quad", "barely_moves"),
    ("scene_4" + STEP_SUFFIX, 30, "multi_milli_quad", "overshoots_3x"),
    ("scene_wheel" + STEP_SUFFIX, 30, "wheel_milli_quad", "catastrophic_5x"),
    # Mixed performance
    ("scene_2" + STEP_SUFFIX, 20, "multi_milli_quad", "okay_minus9pct"),
]

print("=" * 80)
print("RECORDING VIDEOS: 16-dim params on step terrain")
print("=" * 80)
print(f"Source: {BEST_RUN}")
print(f"Output: {video_dir}/")
print()

for scene, freq, robot_dir, label in TEST_CASES:
    params = params_base.copy()
    params['drive_freq'] = freq

    mjcf_path = f"{robot_dir}/{scene}.xml"
    scene_short = scene.replace(STEP_SUFFIX, "")
    video_path = str(video_dir / f"{scene_short}_f{freq}_{label}.mp4")

    print(f"Recording: {scene_short} @ {freq}Hz ({label})...", end=" ", flush=True)

    try:
        trajectory = run_simulation(
            params=params,
            mjcf_path=mjcf_path,
            sim_duration=5.0,
            rng_seed=42,
            record_path=video_path,
        )
        print(f"✓ saved to {video_path}")
    except Exception as e:
        print(f"✗ ERROR: {e}")

print()
print("=" * 80)
print(f"Videos saved to: {video_dir}/")
print()
print("Expected observations:")
print("  - scene_1_f30: Robot barely moves, struggles on steps")
print("  - scene_4_f30: Robot goes way too fast, overshoots wildly")
print("  - scene_wheel_f30: Catastrophic - spins out of control at 5.4× target speed")
print("  - scene_2_f20: Somewhat reasonable locomotion (only -9% error)")
