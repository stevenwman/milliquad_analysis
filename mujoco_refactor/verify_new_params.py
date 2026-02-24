#!/usr/bin/env python3
"""Verify that new solver parameters actually affect simulation trajectories."""

import csv
import sys
from pathlib import Path
import numpy as np

from config_new import sim_params_from_point, space
from simulation_fast_new import run_simulation

# Baseline params from flat optimizer
BASELINE_RUN = "results/20260222T181114_with_20hz_no-deadzone"
baseline_csv = Path(BASELINE_RUN) / "optimization_bests.csv"

if not baseline_csv.exists():
    print(f"Error: {baseline_csv} not found")
    sys.exit(1)

rows = list(csv.DictReader(open(baseline_csv)))
best = rows[-1]

# Extract baseline point (13 params)
old_param_names = [
    'sliding_friction', 'torsional_friction', 'rolling_friction',
    'solref_timeconst', 'solref_dampratio', 'solimp_dmin', 'solimp_delta_d',
    'solimp_width', 'solimp_midpoint', 'solimp_power',
    'magnetic_moment_fudge', 'magnetic_field_fudge', 'dof_damping'
]
# Handle numpy-wrapped strings like 'np.float64(0.123)'
def parse_value(s):
    s = str(s).strip()
    if s.startswith('np.float64('):
        return float(s[len('np.float64('):-1])
    return float(s)

point_baseline = [parse_value(best[p]) for p in old_param_names]

# Add default values for new params (middle of range)
new_param_defaults = {
    "noslip_iterations": 50,
    "noslip_tolerance": 1e-4,
    "margin": 0.0025,
}

# Build full 16-param point
point_full = point_baseline + [new_param_defaults[dim.name] for dim in space[13:]]

# Test scene
MJCF_PATH = "multi_milli_quad/scene_4.xml"
CTRL_FREQ = 30
SIM_DURATION = 3.0

def run_test(params_dict, label):
    """Run simulation and return final position."""
    params_dict['drive_freq'] = CTRL_FREQ
    try:
        trajectory = run_simulation(
            params=params_dict,
            mjcf_path=MJCF_PATH,
            sim_duration=SIM_DURATION,
            rng_seed=42,
        )
        if trajectory:
            return trajectory[-1]['pos'].copy()
        else:
            return None
    except Exception as e:
        print(f"  {label} FAILED: {e}")
        return None

print("=" * 80)
print("PARAMETER EFFECT VERIFICATION (New Solver Params)")
print("=" * 80)
print(f"Baseline: {BASELINE_RUN}")
print(f"Test scene: {MJCF_PATH}, freq={CTRL_FREQ}Hz, duration={SIM_DURATION}s")
print()

# Test each new param dimension
new_param_indices = list(range(13, 16))  # Indices 13-15 for the 3 new param dimensions

results = []

for idx in new_param_indices:
    dim = space[idx]
    param_name = dim.name
    lo, hi = dim.low, dim.high

    # 3 test values: 10th, 50th, 90th percentile
    if dim.prior == "log-uniform":
        vals = np.logspace(np.log10(lo), np.log10(hi), 10)[[1, 5, 9]]
    else:
        vals = np.linspace(lo, hi, 10)[[1, 5, 9]]

    positions = []
    for val in vals:
        test_point = point_full.copy()
        test_point[idx] = val
        params = sim_params_from_point(test_point)
        pos = run_test(params, f"{param_name}={val:.4g}")
        positions.append(pos)

    # Check if all 3 positions are identical (dead parameter)
    if all(p is not None for p in positions):
        diffs = [np.linalg.norm(positions[i] - positions[0]) for i in range(1, 3)]
        max_diff = max(diffs)
        is_dead = max_diff < 1e-6

        status = "DEAD" if is_dead else "ACTIVE"
        print(f"{param_name:<30} {status:>8}  max_diff={max_diff:.2e}")

        results.append({
            "param": param_name,
            "status": status,
            "max_diff": max_diff,
        })
    else:
        print(f"{param_name:<30} {'ERROR':>8}  (simulation failed)")

print()
print("=" * 80)
active = [r for r in results if r["status"] == "ACTIVE"]
dead = [r for r in results if r["status"] == "DEAD"]

print(f"SUMMARY: {len(active)} active, {len(dead)} dead out of {len(results)} tested")
if dead:
    print(f"Dead params: {', '.join([r['param'] for r in dead])}")
else:
    print("All params have measurable effect on trajectories!")
