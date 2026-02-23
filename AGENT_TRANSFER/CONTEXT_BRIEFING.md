# Context Briefing for New Agent

Read this document first. It contains everything you need to work on this project.

## Project: LEGO Milliquad MuJoCo Simulation

A MuJoCo simulation + system identification pipeline for a miniature (~6mm) magnetically-actuated robot. The robot has legs with embedded magnets; an external rotating magnetic field drives locomotion. We optimize 13 contact/friction/magnetic parameters via CMA-ES to match simulated velocities to experimental measurements.

## Repository Structure

```
mujoco_refactor/           # Active codebase (all work happens here)
  config.py                # Flat-terrain config: search space, reference data, constants
  config_step.py           # Step-terrain config: thin overlay on config.py
  simulation_fast.py       # Core MuJoCo engine (shared by both optimizers)
  optimizer.py             # Flat-terrain CMA-ES optimizer
  optimizer_step.py        # Step-terrain CMA-ES optimizer
  show_bests.py            # Pretty-print flat optimizer results
  show_bests_step.py       # Pretty-print step optimizer results
  terrain_test.py          # Validate fitted params on terrain (step/rough)
  terrain_config.py        # Terrain preset definitions
  analysis/                # Data analysis scripts
  results/                 # Optimization run outputs (timestamped dirs)
  multi_milli_quad/        # MJCF robot model files

experimental_data/csv/     # Experimental velocity measurements
  flat/                    # Flat terrain CSVs
  steps/                   # Step terrain CSVs
```

## Two Optimizers (Independent)

### Flat Optimizer (`optimizer.py` + `config.py`)
- Fits 13 params to 11-15 flat-terrain velocity targets
- Best run: `results/20260219T142207_loose_fudge`
- Jitter aggregation: **median** of 3 trials
- Cost: velocity_error + lateral + tumble + yaw + variance

### Step Optimizer (`optimizer_step.py` + `config_step.py`)
- Fits same 13 params to 10 step-terrain velocity targets
- `config_step.py` imports `space`, `sim_params_from_point` from `config.py` (no duplication)
- Jitter aggregation: **best (argmin cost)** — stores `best_trial_{rid}` in CSVs
- Key differences from flat:
  - Step-aware velocity: measured only for `pos[0] >= 0.05` (after 50mm flat lead)
  - No settle time gate
  - YAW_COST_WEIGHT = 0 (cliff-fall artifact makes yaw useless)
  - LATERAL_COST_WEIGHT = 1 (reduced from flat's 5)
  - No pitch RMS, no velocity deadzone
  - SIM_DURATION = 5.0s (vs flat's 3.0s)
- Step XMLs generated at startup: `scene_X_step_8x1mm_4.5L_50lead.xml`
- XMLs must be alongside originals (MuJoCo relative path resolution)
- `mjcf_path` passed in task tuple (multiprocessing spawn empties module-level dicts)
- Warm-started from flat best params

## 13 Search Space Parameters

Defined in `config.py`, shared by both optimizers:
1. sliding_friction (log-uniform)
2. torsional_friction (log-uniform)
3. rolling_friction (log-uniform)
4. solref_timeconst (log-uniform)
5. solref_dampratio (uniform)
6. solimp_dmin (uniform)
7. solimp_delta_d (uniform) — reparameterized: dmax = dmin + delta_d * (0.9999 - dmin)
8. solimp_width (log-uniform)
9. solimp_midpoint (uniform)
10. solimp_power (uniform)
11. magnetic_moment_fudge (uniform)
12. magnetic_field_fudge (uniform)
13. dof_damping (log-uniform)

## Experimental Data

### Step terrain targets (in config_step.py, q75-300 window):
| config | speed (m/s) | std (m/s) |
|---|---:|---:|
| scene1_f10 | 0.0199 | 0.0018 |
| scene1_f20 | 0.0473 | 0.0106 |
| scene1_f30 | 0.0331 | 0.0066 |
| scene2_f10 | 0.0542 | 0.0105 |
| scene2_f20 | 0.0894 | 0.0275 |
| scene2_f30 | 0.1335 | 0.0129 |
| scene4_f10 | 0.0716 | 0.0074 |
| scene4_f20 | 0.1038 | 0.0120 |
| scene4_f30 | 0.0898 | 0.0202 |
| wheel_f30  | 0.0938 | 0.0097 |

Velocity extraction: forward-only vx, q75-300 window (75% index +/- 150 timesteps, clamped to recording bounds). The student's published Table 1-1 used total |v| — we deliberately use forward-only vx because the sim measures pos[0] displacement.

### Experimental CSV filenames
- `s{freq}leg` = scene1 (1-leg), e.g. `s10leg1-1.csv`
- `s{freq}2leg` = scene2 (2-leg), e.g. `s102leg1-1.csv`
- `s{freq}4leg` = scene4 (4-leg), e.g. `s104leg1-1.csv`
- `s{freq}w` = wheel, e.g. `s30w1-1.csv`

## Key Lessons Learned

1. **CMA-ES warm-start is critical**: Cold-start with sliding_friction log-midpoint < 0.05 causes sigma collapse. Always warm-start from known-good params.
2. **magnetic_moment_fudge converges to ~0.80** across all runs.
3. **Cliff-fall artifact**: Robot falls off last step, corrupting yaw/lateral/tumble at trajectory end. Mitigated by 20mm final platform + yaw weight=0.
4. **Step XMLs must be in same dir as originals**: MuJoCo resolves relative paths from XML location.
5. **Multiprocessing with spawn**: Module-level dicts are empty in workers. Pass data through task tuple.
6. **solimp_dmin/width/midpoint/solref_timeconst diverge** across morphologies — these are the hardest params to fit globally.

## Current State (as of 2026-02-23)

- Step optimizer is fully built and smoke-tested
- Ready to run: `cd mujoco_refactor && uv run python optimizer_step.py --n-calls 1200 --suffix step_v1`
- No step optimization results yet (only smoke test with 16 evals)
- Flat optimizer best: `results/20260219T142207_loose_fudge` (cost ~0.38)
- Flat-fit params have 62% mean velocity error on step terrain (expected — flat training can't constrain step-climbing physics)

## Documentation

- `mujoco_refactor/STEP_OPTIMIZER_PLAN.md` — full design rationale for step optimizer
- `mujoco_refactor/STEP_TERRAIN_VALIDATION.md` — experimental data analysis, method comparison, sim vs experiment gaps
