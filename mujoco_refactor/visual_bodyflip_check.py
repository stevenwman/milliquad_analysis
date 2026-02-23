#!/usr/bin/env python3
"""Visual check: run sim with zero field to see body orientation."""

import numpy as np
from simulation_fast import run_simulation

# Zero-field sim params (robot just sits under gravity)
params = {
    'ground_friction': [0.5, 0.0001, 0.00001],  # [sliding, torsional, rolling]
    'solref': [0.002, 1.0],  # [timeconst, dampratio]
    'solimp': [0.9, 0.99, 0.001, 0.5, 2.0],  # [dmin, dmax, width, midpoint, power]
    'dof_damping': 1e-9,
    'kp_mag': 0.0,  # ZERO FIELD
    'drive_freq': 10,  # Doesn't matter, field is zero
    'mag_params': {'m_mag': 0.0},  # ZERO FIELD
}

# Test case: 4-leg robot (easy to see orientation)
scene = "scene_4"
robot_dir = "multi_milli_quad"

print("=" * 60)
print("VISUAL CHECK: Body orientation with zero magnetic field")
print("=" * 60)
print("\nRobot will just sit/settle under gravity - check body orientation")
print()

# Run original
print("Recording ORIGINAL (non-flipped)...")
mjcf_orig = f"{robot_dir}/{scene}.xml"
run_simulation(
    params=params,
    mjcf_path=mjcf_orig,
    sim_duration=1.0,
    record_path="visual_check_ORIGINAL.mp4",
    rng_seed=42
)
print("  → Saved: visual_check_ORIGINAL.mp4")

# Run flipped
print("\nRecording FLIPPED (body inverted)...")
mjcf_flip = f"robot_bodyflip_test/{robot_dir}/{scene}.xml"
run_simulation(
    params=params,
    mjcf_path=mjcf_flip,
    sim_duration=1.0,
    record_path="visual_check_FLIPPED.mp4",
    rng_seed=42
)
print("  → Saved: visual_check_FLIPPED.mp4")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("  1. Watch both videos side-by-side")
print("  2. Check if body geom is flipped but legs/magnets unchanged")
print("  3. If joints look twisted/broken → abort, flip broke kinematics")
print("  4. If looks good → run: uv run python compare_bodyflip.py")
print("=" * 60)
