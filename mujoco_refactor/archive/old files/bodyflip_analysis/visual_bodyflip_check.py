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
print("\nShowing FLIPPED robot (body rotated 180° yaw)")
print("Robot will just sit/settle under gravity")
print("Close viewer window when done\n")

# Run flipped only
mjcf_flip = f"robot_bodyflip_test/{robot_dir}/{scene}.xml"
run_simulation(
    params=params,
    mjcf_path=mjcf_flip,
    sim_duration=5.0,
    visualize=True,
    rng_seed=42
)

print("\n" + "=" * 60)
print("WHAT TO CHECK:")
print("  - Body geom should be rotated 180° (front↔back)")
print("  - Legs/magnets should look unchanged")
print("  - If joints look twisted/broken → wrong rotation axis")
print("\nNEXT STEP:")
print("  If looks good → run: uv run python compare_bodyflip.py")
print("=" * 60)
