# Plan: Extract Sim Data in Experimental CSV Format

## Context
We want to run flat-ground simulations and output time-series CSVs that mimic the experimental data format (`experimental_data/csv/flat/f10leg1-1.csv`). This lets us feed sim data into the existing plotting pipeline (`flat_pipeline.py`) and eventually compute cost of transport from sim energy data. For now: velocity and pitch time-series, no post-processing in the CSV.

## What we're building

A new script `mujoco_refactor/eval_sim_experimental.py` that:
1. Loads best params from a results dir (same as `eval_best_trial.py`)
2. Runs flat sim trials with jitter, picks closest to target (same selection logic)
3. Extracts FL and BL leg body positions as proxies for front/back magnets
4. Writes per-ref CSVs in experimental format + summary .md + HTML plots

Run for both `flat_10_30_50` and `step_argmin_progress` param sets, all 15 flat refs.

## Files to modify

### 1. `mujoco_refactor/simulation_fast_new.py` — 1-line addition

Add `"leg_xpos"` to `_record_state()` (line ~332):
```python
"leg_xpos": data.xpos[_LEG_BODY_SLICE].copy(),  # (4, 3) FR/FL/BR/BL world pos
```

**Backward compatibility**: This is purely additive — it adds a new key to the trajectory dict. All existing consumers (`optimizer_new.py`, `optimizer_step.py`, `eval_best_trial.py`, etc.) access trajectory entries by key name (e.g., `s["pos"]`, `s["time"]`). No existing code iterates over all keys or destructures entries. Adding `"leg_xpos"` cannot break anything — it's simply ignored by code that doesn't look for it.

No changes to function signatures, return types, or any other behavior.

### 2. New file: `mujoco_refactor/eval_sim_experimental.py`

**CLI**:
```
uv run python eval_sim_experimental.py results/20260225T122342_flat_10_30_50/
uv run python eval_sim_experimental.py results/20260225T225248_step_argmin_progress/
```
Optional: `--scenes`, `--freqs`, `--n-trials` (default 3)

**Per-ref workflow**:
1. Run N jitter trials via `run_simulation()` (reuse `load_best_point` from `eval_best_trial.py`)
2. Extract flat velocity (same as `extract_flat_velocity`) to pick best trial
3. From best trial trajectory, extract:
   - **mass_A** = FL leg body = `leg_xpos[1]` (index within slice: FR=0, FL=1, BR=2, BL=3)
   - **mass_C** = BL leg body = `leg_xpos[3]`
4. Downsample from 2kHz → 1kHz (every other timestep) to match experimental sampling
5. Write CSV

**Coordinate mapping** (sim → experimental CSV convention):
| CSV column | Source | Notes |
|---|---|---|
| col 0: t | `traj["time"]` | seconds |
| col 1: mass_A x | `-leg_xpos[1][0]` | negate: sim +x=forward, CSV -x=forward |
| col 2: mass_A y | `leg_xpos[1][2]` | sim z (up) → CSV y (up) |
| col 3: mass_A vx | `diff(col1)/dt` | finite difference, negated naturally |
| col 4: mass_A vy | `diff(col2)/dt` | finite difference |
| col 5-8: mass_C | same but `leg_xpos[3]` | BL body |
| col 9: mass_B θ | `0.0` | placeholder (unused by pipeline) |
| col 10: mass_B ω | `0.0` | placeholder (unused by pipeline) |
| col 11: mass_C θ | `atan2(dz, dx)` of FL-BL | degrees, relative to t=0, unwrapped |

Sign negation on x/vx ensures `flat_pipeline.py` works unmodified:
```python
# flat_pipeline.py line 25 — double-negative gives correct forward velocity
vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))
```

**Theta computation** (geometric, from marker positions — NOT body quaternion):
```python
dx = xpos_FL[0] - xpos_BL[0]   # forward separation (sim x)
dz = xpos_FL[2] - xpos_BL[2]   # height difference (sim z)
theta = np.degrees(np.arctan2(dz, dx))  # pitch angle in degrees
```
- This matches what a camera would measure: angle of the line between front and back markers
- Subtract theta at t=0: `theta -= theta[0]`
- Unwrap: `theta = np.degrees(np.unwrap(np.radians(theta)))` to handle ±180 discontinuities

**CSV header** (matches experimental format exactly):
```
,mass_A,,,,mass_C,,,,mass_B,,mass_C
t,x,y,vx,vy,x,y,vx,vy,θ,ω,θ
```

**File naming**: `sim_<scene>_f<freq>.csv` in `<run_dir>/sim_csvs/`

**Summary .md**: `<run_dir>/sim_experimental_summary.md` with table:
```
| Ref | Target (mm/s) | Sim (mm/s) | Err% | Mean Pitch (deg) | Seed |
```

### 3. HTML plotting (inline in eval_sim_experimental.py)

After all CSVs are written, generate per-(freq × morphology) HTML plots using plotly directly — same 4-panel layout as experimental HTMLs (speed, height, theta, omega). Omega panel will be zeros (placeholder).

- Read back each sim CSV with `np.genfromtxt(..., skip_header=2)`
- Apply same extraction as `flat_pipeline._extract_flat`: `vx = 0.5 * ((-col3 * 1000) + (-col7 * 1000))`, etc.
- Group by (freq, morphology): each HTML = 1 condition = 1 sim trial curve
- Output to `experimental_data/plots/fake_exp/<run_label>/` (e.g., `fake_exp/flat_10_30_50/10hz_leg.html`)
- Naming matches experimental HTMLs: `<freq>hz_<morph>.html`

Morph name mapping: `scene1→leg`, `scene2→2legged`, `scene4→4legged`, `scene_wheel→wheel`

No modifications to existing experimental pipeline code.

## Key reuse from existing code
- `eval_best_trial.py:load_best_point()` — param loading pattern
- `eval_best_trial.py:extract_flat_velocity()` — velocity for trial selection
- `config_new.py`: `REFERENCE_DATA`, `MJCF_PATHS`, `sim_params_from_point`, `space`, `SETTLE_TIME`, `SIM_DURATION`, `SIMULATION_TIMEOUT`
- `simulation_fast_new.py:run_simulation()` — sim execution

## Verification
1. Run on `flat_10_30_50`: `uv run python eval_sim_experimental.py results/20260225T122342_flat_10_30_50/`
2. Run on `step_argmin_progress`: `uv run python eval_sim_experimental.py results/20260225T225248_step_argmin_progress/`
3. Sanity check: summary table velocities should match FLAT_VALIDATION_COMPARISON.md (within jitter variation)
4. Check a CSV can be loaded by experimental pipeline: `np.genfromtxt(csv_path, delimiter=",", skip_header=2)` should give (N, 12) array
5. Spot-check: velocity from CSV cols 3,7 should match summary table values
