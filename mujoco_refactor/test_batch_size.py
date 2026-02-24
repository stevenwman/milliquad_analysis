#!/usr/bin/env python3
"""Test different batch sizes to find optimal parallelization."""

import time
import multiprocessing as mp
import numpy as np

from config_multi_terrain import (
    reference_rows,
    sim_params_from_point,
    space,
    CMAES_X0,
    INIT_JITTER_SEED,
    INIT_YAW_JITTER_DEG,
    SIM_DURATION,
)
from simulation_fast import run_simulation

# Convert CMAES_X0 to point
point = np.array([CMAES_X0[dim.name] for dim in space])
params_base = sim_params_from_point(point)

# Test subset: 4 references (2 flat, 2 step)
test_refs = [
    ("scene1", 10, "flat", "multi_milli_quad/scene_1.xml"),
    ("scene2", 30, "flat", "multi_milli_quad/scene_2.xml"),
    ("scene1", 10, "step", "multi_milli_quad/scene_1_step_8x1mm_4.5L_50lead.xml"),
    ("scene4", 30, "step", "multi_milli_quad/scene_4_step_8x1mm_4.5L_50lead.xml"),
]

# 2 jitter trials per ref = 8 total sims
N_JITTER = 2

def run_single_sim(args):
    """Worker function for parallel execution."""
    scene, freq, terrain, mjcf_path, jitter_idx = args
    params = params_base.copy()
    params['drive_freq'] = freq

    try:
        trajectory = run_simulation(
            params=params,
            mjcf_path=mjcf_path,
            sim_duration=SIM_DURATION,
            rng_seed=INIT_JITTER_SEED + jitter_idx,
            init_yaw_jitter_deg=INIT_YAW_JITTER_DEG,
            visualize=False,
        )
        return len(trajectory) if trajectory else 0
    except Exception as e:
        return -1

# Build task list
tasks = []
for scene, freq, terrain, mjcf_path in test_refs:
    for jitter_idx in range(N_JITTER):
        tasks.append((scene, freq, terrain, mjcf_path, jitter_idx))

print("=" * 80)
print("BATCH SIZE OPTIMIZATION TEST")
print("=" * 80)
print(f"System info:")
print(f"  CPU cores (logical):  {mp.cpu_count()}")
try:
    physical_cores = len(set([c.id for c in mp.cpu_count() if hasattr(c, 'id')]))
except:
    physical_cores = mp.cpu_count() // 2  # Rough estimate assuming hyperthreading
print(f"  CPU cores (physical): ~{physical_cores} (estimate)")
print(f"  Test tasks: {len(tasks)} simulations")
print()

# Test different batch sizes
batch_sizes = [1, 2, 4, 6, 8, 12, 16, 24]
# Filter to reasonable range based on CPU count
max_reasonable = mp.cpu_count() * 2
batch_sizes = [b for b in batch_sizes if b <= max_reasonable]

results = []

for batch_size in batch_sizes:
    print(f"Testing batch_size={batch_size:2d}...", end=" ", flush=True)

    start_time = time.time()

    with mp.Pool(processes=batch_size) as pool:
        outputs = pool.map(run_single_sim, tasks)

    elapsed = time.time() - start_time
    successes = sum(1 for o in outputs if o > 0)

    speedup = results[0][1] / elapsed if results else 1.0
    efficiency = speedup / batch_size if batch_size > 1 else 1.0

    print(f"time={elapsed:5.1f}s  speedup={speedup:4.2f}x  efficiency={efficiency*100:5.1f}%  ({successes}/{len(tasks)} ok)")

    results.append((batch_size, elapsed, speedup, efficiency))

print()
print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"{'Batch':>6} {'Time(s)':>8} {'Speedup':>8} {'Efficiency':>10} {'Recommendation'}")
print("-" * 80)

best_idx = min(range(len(results)), key=lambda i: results[i][1])

for i, (bs, elapsed, speedup, efficiency) in enumerate(results):
    marker = "  ← FASTEST" if i == best_idx else ""
    if efficiency > 0.85 and i < best_idx:
        marker = "  ← BEST (good efficiency)"
    print(f"{bs:6d} {elapsed:8.1f} {speedup:8.2f}x {efficiency*100:9.1f}% {marker}")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Find batch size with >85% efficiency that's close to fastest
good_efficiency = [(bs, t, sp, eff) for bs, t, sp, eff in results if eff > 0.85]
if good_efficiency:
    # Pick fastest among high-efficiency options
    best = min(good_efficiency, key=lambda x: x[1])
    print(f"Recommended batch_size: {best[0]}")
    print(f"  - Achieves {best[3]*100:.1f}% parallelization efficiency")
    print(f"  - {best[2]:.2f}× speedup over serial execution")
    print(f"  - Estimated full run time: {600 * best[1] / len(tasks) / 60:.1f} minutes")
else:
    print(f"Recommended batch_size: {results[best_idx][0]}")
    print(f"  - Fastest absolute time (efficiency may be suboptimal)")

print()
print(f"Current config uses batch_size=8")
optimal = best[0] if good_efficiency else results[best_idx][0]
if optimal != 8:
    print(f"→ Consider updating to batch_size={optimal} in config_multi_terrain.py")
else:
    print(f"→ Current setting is optimal!")
print()
