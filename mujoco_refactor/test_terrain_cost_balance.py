#!/usr/bin/env python3
"""Test warm-start params on both terrains to check cost magnitude balance."""

import numpy as np
from config_multi_terrain import (
    CMAES_X0,
    reference_rows,
    sim_params_from_point,
    space,
    STEP_START_X,
    INIT_JITTER_SEED,
    INIT_YAW_JITTER_DEG,
    SIM_DURATION,
)
from simulation_fast import run_simulation
from optimizer_multi_terrain import calculate_cost_flat, calculate_cost_step

# Convert CMAES_X0 dict to point array
point = np.array([CMAES_X0[dim.name] for dim in space])
params_base = sim_params_from_point(point)

print("=" * 80)
print("TERRAIN COST BALANCE TEST")
print("=" * 80)
print(f"Testing warm-start params (cost=0.380 on flat-only optimizer)")
print(f"Measuring cost magnitudes on flat vs step terrain...")
print()

# Separate references by terrain
flat_refs = [r for r in reference_rows() if r["terrain"] == "flat"]
step_refs = [r for r in reference_rows() if r["terrain"] == "step"]

print(f"Flat refs: {len(flat_refs)}, Step refs: {len(step_refs)}")
print()

# Sample a few references from each terrain (not all 19 to save time)
flat_sample = flat_refs[:3]  # First 3 flat refs
step_sample = step_refs[:3]  # First 3 step refs

flat_costs = []
step_costs = []

print("Running flat terrain samples...")
for ref in flat_sample:
    params = params_base.copy()
    params['drive_freq'] = ref['ctrl_freq']

    scene = ref['scene']
    # Fix scene naming: scene1 -> scene_1, scene2 -> scene_2, etc.
    if scene.startswith("scene") and scene != "scene_wheel":
        scene_file = scene.replace("scene", "scene_")
    else:
        scene_file = scene

    if scene == "scene_wheel":
        mjcf_path = f"wheel_milli_quad/{scene_file}.xml"
    else:
        mjcf_path = f"multi_milli_quad/{scene_file}.xml"

    # Single trial (no jitter for speed)
    trajectory = run_simulation(
        params=params,
        mjcf_path=mjcf_path,
        sim_duration=SIM_DURATION,
        visualize=False,
        rng_seed=INIT_JITTER_SEED,
        init_yaw_jitter_deg=0.0,  # No jitter for faster test
    )

    cost_data = calculate_cost_flat(
        trajectory,
        target_velocity=ref['speed'],
        speed_std=ref.get('speed_std', 0.0),
        verbose=False,
    )

    flat_costs.append(cost_data['total_cost'])
    print(f"  {ref['id']:<20} cost={cost_data['total_cost']:.4f}")

print()
print("Running step terrain samples...")
for ref in step_sample:
    params = params_base.copy()
    params['drive_freq'] = ref['ctrl_freq']

    scene = ref['scene']
    # Fix scene naming: scene1 -> scene_1, scene2 -> scene_2, etc.
    if scene.startswith("scene") and scene != "scene_wheel":
        scene_file = scene.replace("scene", "scene_")
    else:
        scene_file = scene

    step_suffix = "_step_8x1mm_4.5L_50lead"
    if scene == "scene_wheel":
        mjcf_path = f"wheel_milli_quad/{scene_file}{step_suffix}.xml"
    else:
        mjcf_path = f"multi_milli_quad/{scene_file}{step_suffix}.xml"

    # Single trial (no jitter for speed)
    trajectory = run_simulation(
        params=params,
        mjcf_path=mjcf_path,
        sim_duration=SIM_DURATION,
        visualize=False,
        rng_seed=INIT_JITTER_SEED,
        init_yaw_jitter_deg=0.0,
    )

    cost_data = calculate_cost_step(
        trajectory,
        target_velocity=ref['speed'],
        speed_std=ref.get('speed_std', 0.0),
        step_start_x=STEP_START_X,
        verbose=False,
    )

    step_costs.append(cost_data['total_cost'])
    print(f"  {ref['id']:<20} cost={cost_data['total_cost']:.4f}")

print()
print("=" * 80)
print("COST MAGNITUDE COMPARISON")
print("=" * 80)

flat_avg = np.mean(flat_costs)
step_avg = np.mean(step_costs)
ratio = flat_avg / step_avg if step_avg > 0 else np.inf

print(f"Flat terrain average cost: {flat_avg:.4f} (sample of {len(flat_costs)})")
print(f"Step terrain average cost: {step_avg:.4f} (sample of {len(step_costs)})")
print(f"Ratio (flat/step): {ratio:.2f}x")
print()

if ratio > 1.5:
    print(f"⚠️  IMBALANCE DETECTED: Flat costs are {ratio:.2f}x higher than step")
    print(f"   Recommendation: Set STEP_TERRAIN_WEIGHT = {ratio:.2f} to balance")
    print(f"   Or set FLAT_TERRAIN_WEIGHT = {1/ratio:.2f}")
elif ratio < 0.67:
    print(f"⚠️  IMBALANCE DETECTED: Step costs are {1/ratio:.2f}x higher than flat")
    print(f"   Recommendation: Set FLAT_TERRAIN_WEIGHT = {1/ratio:.2f} to balance")
    print(f"   Or set STEP_TERRAIN_WEIGHT = {ratio:.2f}")
else:
    print(f"✓ Costs are reasonably balanced (ratio {ratio:.2f}x is close to 1.0)")
    print(f"  Current weights (both 1.0) are appropriate")

print()
print("=" * 80)
