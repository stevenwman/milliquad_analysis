# Project Memory

## Architecture (mujoco_refactor/)
- `config.py`: Single source of truth for all constants, search space, and param conversion
- `simulation.py` / `simulation_fast.py`: Core MuJoCo engine with condim=6 (torsional+rolling friction)
- `optimizer.py`: CMA-ES optimization loop. Cost = velocity_error + lateral + tumble + yaw + variance
- `REFERENCE_DATA` in config.py is the sole source for optimization targets

## CLI Args (optimizer.py)
- `--scenes scene1 scene4`: filter REFERENCE_DATA to specific scenes
- `--freqs 10 30`: filter to specific control frequencies
- `--n-calls 600`: override N_CALLS from config
- `--suffix tag`: append to results folder name (e.g. `results/20260217T..._tag/`)
- `--warm-start-from <dir>`: load best params from optimization_bests.csv in given dir as CMAES_X0
- Filters update `_REF_ROWS` globally; all printouts/CSVs respect filtered set

## Cost Function
- Velocity error dominates ~92% of total cost; lateral ~2%, variance ~6%
- Dead-zone (`speed_std`) stops velocity penalty inside 1-sigma; variance uses raw relative error
- Tumble penalty normalized per-step; pitch RMS disabled (pitch_weight=0.0)
- Jitter trials aggregated via **median** — tolerates 1 chaotic outlier
- `VELOCITY_COST_WEIGHT=5`, `LATERAL_COST_WEIGHT=5`, `TUMBLE_COST_WEIGHT=1`, `VELOCITY_VARIANCE_WEIGHT=2`

## Search Space (tightened 2026-02-18)
- `solimp_delta_d` reparameterization: `dmax = dmin + delta_d * (0.9999 - dmin)` — guarantees dmax > dmin
- Bounds tightened from per-morphology/per-frequency sweep analysis:
  - sliding_friction [0.01, 2.0], moment_fudge [0.75, 0.90], dof_damping [1e-10, 1e-8]
  - solref_dampratio [1.0, 10.0], solimp_power [2.0, 7.0]
  - Divergent params (solimp_dmin/width/midpoint, solref_timeconst) left wide
- `magnetic_moment_fudge` pins at ~0.80 across all runs
- Jitter seeds: collision-free index-based mapping

## CMA-ES Warm-Start Lessons
- **Cold-start stagnation**: If sliding_friction log-midpoint < 0.05, CMA-ES sigma collapses before finding locomotion. Scene2 and f30 both failed this way.
- **Fix**: warm-start from known-good params or use `--warm-start-from`
- Current CMAES_X0 in config.py: consensus of per-morphology sweep bests

## Per-Morphology/Frequency Analysis (2026-02-18, with warm-start fixes)
- Per-scene costs: 0.0001–0.026; per-freq costs: 0.0003–0.028
- **CONVERGE both**: sliding_friction (12%), magnetic_moment_fudge (8%), dof_damping (9%)
- **CONVERGE freq-only**: solref_dampratio (2.4%), solimp_power (14%)
- **DIVERGE both**: solimp_dmin, solimp_width, solimp_midpoint, solref_timeconst
- Combined run (all 11 refs): cost 0.734 — hard problem, needs warm-start + 2400 evals
- Scripts: `run_per_morphology.sh`, `compare_morphology_params.py`

## Terrain Validation (terrain_test.py, 2026-02-22)
- `terrain_config.py`: presets (flat, step_default/tall/many, rough_mild/harsh), separate from optimizer config
- `terrain_test.py`: loads best params from run dir, runs flat/step/rough terrain, reports velocity ratio + tumble/lateral/yaw
- MJCF editing: edited XMLs written **alongside originals** (same dir) for correct relative path resolution; `_terrain_tmp.xml` suffix, cleaned up in `try/finally`
- MuJoCo hfield `z_bottom` must be > 0; `generate_terrain_hfield()` returns 0.0, clamped to 0.001
- Friction/contact: `mjENBL_OVERRIDE` + `geom_condim[:]=6` applies fitted params globally to all geoms including terrain — no special handling needed
- Step terrain tumble artifact: robots fall off cliff at end of staircase → yaw~180°. Increase `final_step_length` to avoid
- `rough_mild` (0.5mm std) barely affects robots; use `rough_harsh` (1mm std) for differentiation
- `VELOCITY_DEADZONE = False` in config.py: plain quadratic cost, configurable flag

## Step Terrain Optimizer (2026-02-23)
- **Separate** from flat optimizer — `config_step.py` + `optimizer_step.py` + `show_bests_step.py`
- `config_step.py` is a thin overlay: imports `space`, `sim_params_from_point`, `MJCF_PATHS` from `config.py`
- 10 reference conditions: scene1/2/4 × f10/f20/f30 + scene_wheel × f30
- Targets: **vx q75-300** (75% index +/- 150 timesteps, clamped to bounds, forward velocity only)
- Step geometry: 8 steps, 1mm high, 4.5mm long, 50mm flat lead, 20mm final platform, 100mm wide
- Step XMLs generated at startup: `scene_X_step_8x1mm_4.5L_50lead.xml` alongside originals
- **Step-aware cost**: velocity measured only for `pos[0] >= STEP_START_X` (50mm), no settle time
- Cost weights: velocity=5, tumble=1, lateral=1, yaw=0 (cliff-fall artifact), variance=2
- No pitch RMS, no velocity deadzone
- Warm-start X0 from `20260222T181114_with_20hz_no-deadzone` best params
- Multiprocessing: `mjcf_path` passed in task tuple (spawn re-imports module, empties global dict)
- Jitter trials aggregated via **best (argmin cost)**, not median — `best_trial_{rid}` stored in both CSVs
- Cliff-fall artifact: robot falls off last step → corrupts yaw/lateral/tumble at end of trajectory
- See `STEP_OPTIMIZER_PLAN.md` and `STEP_TERRAIN_VALIDATION.md` for detailed design + data analysis
- Experimental velocity: student's Table 1-1 used total |v|, we use forward-only vx (matches sim)

## Code Style Preferences
- User prefers no fallback logic / magic number defaults — make params explicit and required
- Keep things clean: prune dead code paths when simplifying
- `show_bests.py` prints params with bound flags (<< LO, >> HI) on final best

## CSV Output
- CSVs write directly into `results/<timestamp_suffix>/`
- `show_bests.py` defaults to latest results dir; accepts explicit path as argv[1]
- Video recording: rank 1 only
