# Agent Transfer Briefing (2026-02-26)

## What This Project Does

MuJoCo simulation and system identification for LEGO-based "milliquad" robots (~6mm). Robots move via external rotating magnetic field that spins embedded magnets in each leg. We optimize MuJoCo contact/solver parameters to match experimental velocity data across 4 robot morphologies and multiple frequencies.

## Current Architecture (mujoco_refactor/)

### Two parallel optimization systems (both 16-dim):

| System | Config | Optimizer | Sim Engine | Purpose |
|--------|--------|-----------|------------|---------|
| Flat terrain | `config_new.py` | `optimizer_new.py` | `simulation_fast_new.py` | Match flat ground velocity data |
| Step terrain | `config_step.py` | `optimizer_step.py` | `simulation_fast_new.py` (shared) | Match staircase velocity data |

### 16-Dim Search Space (shared by both systems)

Original 13 dims (contact/friction/magnetics) + 3 new MuJoCo solver params:
- `noslip_iterations` (int 0-60): friction solver iterations
- `noslip_tolerance` (float 1e-6 to 1e-3): convergence threshold
- `margin` (float 0.0 to 0.005): contact detection distance

### Key Scripts
- `show_bests.py` / `show_bests_step.py` — pretty-print optimization results
- `eval_best_trial.py` — validation: N jitter trials per ref, pick closest to target, record video
- `eval_rough_terrain.py` — rough terrain evaluation with tiled hfield
- `characterize_params.py` — parameter sensitivity analysis

### Legacy systems (DO NOT MODIFY, kept for reference):
- `config.py` + `simulation_fast.py` + `optimizer.py` — original 13-dim flat system

## Robot Morphologies (4 scenes)

| Scene | Description | MJCF |
|-------|-------------|------|
| scene1 | Single leg | `mulit_milli_quad/scene_1.xml` |
| scene2 | Double leg | `mulit_milli_quad/scene_2.xml` |
| scene4 | Quad leg | `mulit_milli_quad/scene_4.xml` |
| scene_wheel | Wheel (no legs) | `mulit_milli_quad/scene_wheel.xml` |

## Reference Data

### Flat terrain (15 refs in config_new.py)
- scene1/2/4 x f10/f20/f30/f50 + scene_wheel x f10/f20/f30
- Velocities: 51-449 mm/s (forward-only vx, after 0.1s settle time)
- Jitter: 3 trials, aggregated via MEDIAN

### Step terrain (12 refs in config_step.py)
- scene1/2/4 x f10/f20/f30 + scene_wheel x f10/f20/f30
- scene_wheel f10/f20 are FAILURE MODES (target=0.0 mm/s — robot doesn't move)
- Velocities measured in step region only (x >= 50mm to 90% of STEP_END_X)
- Jitter: 3 trials, aggregated via BEST (argmin cost) — not median
- Progress penalty: `(1 - progress_fraction)^2`, weight=2.0

### Step geometry
```
flat_lead (50mm) | 7 steps (4.5mm each) | final step (20mm platform)
x=0              x=50mm                   x=81.5mm                x=101.5mm
                 STEP_START_X                                      STEP_END_X
Step height: 1mm, width: 100mm
```

## Best Results

### Flat 16-dim: `results/20260225T003517_flat_16dim_corrected_warm/`
- **Cost: 0.377** (converged at eval 1976 of 4800)
- Key params: sliding_friction=0.49, magnetic_moment_fudge=0.65, noslip_iterations~0, margin~0
- 11/15 refs within 1-sigma of experimental data

### Step 16-dim: `results/20260225T225248_step_argmin_progress/`
- **Cost: 0.210** (converged at eval 2496 of 4800)
- Key params: sliding_friction=0.50, magnetic_moment_fudge=0.92, noslip_iterations=31
- Zero tumble across all refs, most within +/-10% velocity error
- NOTE: magnetic_moment_fudge diverges significantly between flat (0.65) and step (0.92)

## Critical Lessons Learned

### 1. Warm-start for expanded search space
When adding new dimensions, initialize them at **MuJoCo defaults** (not space midpoints). Midpoints push the optimizer away from the 13-dim optimum. This was validated empirically — midpoint warm-start gave cost=0.445 vs defaults giving cost=0.377.

### 2. Single-terrain overfitting
Flat-optimized params fail catastrophically on steps (92.8% mean velocity error). Step-optimized params likely also degrade on flat. A multi-terrain approach was designed (`config_multi_terrain.py`) but not fully validated.

### 3. Wheel morphology is chaotic on steps
scene_wheel on step terrain shows bimodal velocity distribution regardless of initial yaw angle. ~10% of trials match target velocity; outcome is purely stochastic. See `WHEEL_STEP_CHAOS_ANALYSIS.md` for full sweep data.

### 4. Cliff-fall artifact
Robots fall off the last step, corrupting yaw/lateral/tumble metrics. Mitigated by: extended final step platform (20mm), yaw_cost_weight=0, and velocity measured only up to 90% of step region.

### 5. Cost function structure
- Velocity error dominates (~92% of flat cost)
- Dead-zone disabled (VELOCITY_DEADZONE=False) — plain quadratic cost
- Tumble penalty normalized per-step
- Pitch RMS disabled (pitch_weight=0.0)

## CLI Patterns

```bash
# Run flat optimizer (16-dim)
cd mujoco_refactor
uv run python optimizer_new.py --n-calls 600 --suffix my_test

# Run step optimizer (16-dim)
uv run python optimizer_step.py --n-calls 600 --suffix my_test

# View results
uv run python show_bests.py results/XXXXX/   # flat
uv run python show_bests_step.py results/XXXXX/   # step

# Validate best params (jitter trials + record videos)
uv run python eval_best_trial.py results/XXXXX/ --record

# Filter to specific scenes/freqs
uv run python optimizer_new.py --scenes scene4 --freqs 10 30 --n-calls 300

# Warm-start from previous run
uv run python optimizer_new.py --warm-start-from results/XXXXX/ --suffix warm_v2

# Rough terrain evaluation
uv run python eval_rough_terrain.py results/XXXXX/ --scenes scene4 --freqs 30 --record
```

## Open Problems

1. **Multi-terrain optimization**: Need params that work on BOTH flat and step terrain. Design exists in `config_multi_terrain.py` (19 refs, hierarchical cost) but not fully validated.
2. **Rough terrain evaluation**: `eval_rough_terrain.py` has "contact shenanigans" at FLAT_LEAD=0.025 — needs debugging.
3. **Parameter divergence**: flat vs step optimized params disagree on magnetic_moment_fudge (0.65 vs 0.92) and noslip_iterations (0 vs 31). This suggests the contact model may be fundamentally underdetermined for multi-terrain generalization.
