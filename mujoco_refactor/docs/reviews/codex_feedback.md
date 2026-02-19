# Codex Review: `mujoco_refactor`

## 0) Full Understanding of the Codebase (End-to-End)

This section is intentionally explicit so another reviewer (Claude) can challenge each assumption.

### 0.1 Primary Goal
The codebase calibrates MuJoCo simulation parameters so simulated locomotion matches experimental speed targets across multiple robot setups.

The optimization objective combines:
- Forward-speed matching per reference condition.
- Stability/behavior penalties (tumble, lateral drift, yaw spin, optional pitch RMS).
- Aggregation across scenes/frequencies with per-reference weights.

### 0.2 Control/Data Flow
1. `config.py` defines shared constants, search space, and reference rows.
2. `optimizer.py` proposes candidate points in the 13D space.
3. Each candidate is mapped to simulation parameters via `sim_params_from_point()`.
4. For each reference row `(scene, ctrl_freq, target speed, etc.)`, a MuJoCo rollout is run.
5. Rollout trajectory is scored by `calculate_cost()`.
6. Per-reference scores are averaged across jitter trials, then aggregated into per-point total cost.
7. Optimizer `tell()` updates model with those costs.
8. Results are written to timestamped CSVs under `results/<run_tag>/`.

### 0.3 What `config.py` Controls
File: `mujoco_refactor/config.py`

- Simulation constants:
  - Timestep: 2 kHz (`SIM_TIMESTEP = 1/2000`).
  - Settle time before actuation costing/drive motion.
  - Stuck detection thresholds.
- Scene selection:
  - `scene1`, `scene2`, `scene4`, `scene_wheel` -> MJCF paths.
- Experimental references:
  - `REFERENCE_DATA` contains scene/frequency/speed targets, optional uncertainty (`speed_std`) and weights.
- Optimization setup:
  - 13D search space (`space`) including contact params + magnetic fudges + dof damping.
  - Backend selection (`skopt` or `cmaes`), batch size, eval budget.
- Canonical mapping:
  - `point_to_params()` and `sim_params_from_point()` convert optimizer vector -> named physical params.

Important: `sim_params_from_point()` is the canonical parameter map; this is central for reproducibility.

### 0.4 What the Simulation Engine Actually Does
Files: `mujoco_refactor/simulation.py`, `mujoco_refactor/simulation_fast.py`

At each step:
1. Compute drive angle from time and `drive_freq` after settle delay.
2. Compute external magnetic torque on each leg: magnet north vs rotating goal direction.
3. Compute inter-joint magnetic coupling via dipole field summation.
4. Apply resulting world-frame torques to leg bodies through `data.xfrc_applied[:,3:6]`.
5. `mj_step()` advances physics.
6. Check instability conditions and optional stuck criteria.
7. Record trajectory state (`time`, `pos`, `vel`, `quat`) and torque data (`tau_ext`, `tau_int`, `omega`).

Both modules expose same API (`run_simulation(...)`).

### 0.5 How `simulation_fast.py` Differs Structurally
`simulation_fast.py` replaces scalar/scipy-heavy internals with vectorized numpy paths:
- Batched quaternion-vector rotation.
- Batched magnet-state computation.
- Fully vectorized dipole interaction tensor for all leg pairs.

Intended behavior: same force model, lower overhead.

### 0.6 What `optimizer.py` Does Precisely
File: `mujoco_refactor/optimizer.py`

- Worker function `_evaluate_one_scene(...)`:
  - Builds sim params from point.
  - Injects scene-specific frequency.
  - Runs one rollout (with deterministic init-yaw jitter seed).
  - Converts trajectory to scalar cost.
- `calculate_cost(...)` terms:
  - Velocity error with optional 1-sigma dead-zone.
  - Lateral displacement penalty.
  - Tumble penalty from body-z alignment.
  - Yaw deviation threshold penalty.
  - Optional pitch RMS term.
- `_aggregate_scene_results(...)`:
  - Averages trials per reference.
  - Applies per-reference weights.
  - Builds scene-level summaries and total point cost.
  - Adds cross-reference velocity variance penalty.
- Main loop `_run_batch_optimization(...)`:
  - `ask` points.
  - Evaluate `(point, ref, trial)` tasks in multiprocessing pool.
  - Aggregate and `tell` costs.
  - Append per-point results to CSV.
  - Track/append new global bests.

### 0.7 Utility Scripts
- `replay.py`: load one CSV row, reconstruct params, run viewer per reference condition.
- `visualize_rollout.py`: pick rank from CSV, run sim, compute locomotion metrics/COT.
- `compare_cot.py`: run top-N results across selected reference rows, tabulate COT and distance.
- `verify_sim.py`: attempts trajectory equivalence check between `simulation` and `simulation_fast`.
- `show_bests.py`: pretty-print latest or specified `optimization_bests.csv`.

### 0.8 Explicit Model/Layout Assumptions (Verified)
I validated MuJoCo model structure via runtime introspection for all 4 scenes.
Current assumptions that hold now:
- Body IDs: `world=0`, main body `1`, legs `2..5`.
- Hinge leg DoFs occupy last 4 velocity DoFs (`6..9` with freejoint first).
- Ground geom is named `floor`.

These assumptions are currently true, but are still implicit contracts between XML and Python.

---

## 1) What I Verified Empirically

### 1.1 Syntax/Import viability
- `python -m compileall mujoco_refactor` succeeds.

### 1.2 Force-level parity (`simulation` vs `simulation_fast`)
I compared `_apply_magnetic_forces()` outputs on the exact same MuJoCo state over 2000 steps.

Observed maxima:
- `max_tau_ext_diff ~= 5.08e-21`
- `max_tau_int_diff ~= 4.13e-21`
- `max_xfrc_diff ~= 7.41e-21`

Interpretation:
- Force computation is effectively equivalent numerically.
- Fast-sim math path appears logically consistent with base-sim at force level.

### 1.3 Trajectory/cost divergence still occurs
Even with force-level near-equality, full rollouts diverge materially in some runs/conditions.
Control checks showed:
- `m_mag = 0` (no inter-joint coupling): exact/near-exact parity.
- With full coupling: trajectory differences can grow and cost ranking can shift.

Interpretation:
- This is consistent with chaotic contact dynamics amplifying tiny numeric perturbations.
- Not necessarily a formula bug, but trajectory-level interchangeability is not guaranteed.

### 1.4 CMA-ES ask/tell behavior nuance
I checked pycma behavior directly:
- `es.ask(number=k)` does support `k`.
- `es.tell(...)` requires at least `mu` solutions.
- With `popsize=8` (default `mu=4`), `tell` works for `k>=4`, fails for `k<4`.

Why this matters here:
- Current wrapper in `optimizer.py` uses `es.ask()` without passing `n_this`, so it always returns full population.
- This hides `N_CALLS` remainder handling logic and can over-evaluate when budgets are not aligned.

### 1.5 Jitter-seed collision evidence
Current seed component for refs is:
- `sum(ord(c) for ref_id) % 1000`

Current collisions:
- `scene2_f30` collides with `scene4_f10`
- `scene2_f50` collides with `scene4_f30`

Meaning:
- For same `(global_point_index, trial_index)`, those ref pairs receive identical seeds.
- This is independent of parallelism; it is deterministic hash collision.

---

## 2) Updated Findings (Prioritized)

1. **High: CMA remainder handling is under-specified and wrapper currently ignores `n_this`.**
File refs: `mujoco_refactor/optimizer.py:684`, `mujoco_refactor/optimizer.py:726`

Current CMA ask wrapper:
- Calls `es.ask()` with no count argument.
- Therefore always returns full popsize (`BATCH_SIZE`), even when `n_this` is smaller.

Impact:
- If `N_CALLS` not aligned with batch/population behavior, evaluation budget accounting is wrong.

2. **Medium: `simulation_fast` is force-equivalent but not guaranteed trajectory-identical in chaotic regimes.**
File refs: `mujoco_refactor/simulation.py`, `mujoco_refactor/simulation_fast.py`, `mujoco_refactor/verify_sim.py`

Updated position after deeper testing:
- I no longer claim a clear formula mismatch.
- I do claim trajectory-level parity claims like “bit-exact/equivalent” are too strong operationally under contact chaos.

3. **Medium: Reference-seed collisions reduce cross-reference jitter independence.**
File ref: `mujoco_refactor/optimizer.py:264`

Given your preference, this should be changed to collision-free mapping.

4. **Low-Medium: `compute_locomotion_metrics()` key validation misses `tau_int`.**
File refs: `mujoco_refactor/visualize_rollout.py:109`, `mujoco_refactor/visualize_rollout.py:124`

It checks `tau_ext` and `omega`, then unconditionally reads `tau_int`.

5. **Low: Duplicate mapping logic risk in `visualize_rollout.py`.**
File refs: `mujoco_refactor/config.py:331`, `mujoco_refactor/visualize_rollout.py:33`

Canonical conversion exists in config, but CSV reconstruction reimplements mapping.

6. **Low: `_SCENE_TARGETS` is currently unused.**
File refs: `mujoco_refactor/optimizer.py:84`, `mujoco_refactor/optimizer.py:91`

---

## 3) Concrete Guidance You Asked For

### 3.1 If `N_CALLS` is not divisible by batch size: what is the correct path?

For CMA-ES (`popsize=8`, `mu=4` typical):
- Remainder `0`: straightforward.
- Remainder `4..7`: feasible exact-budget final partial generation (`ask(number=remainder)`, `tell` accepted).
- Remainder `1..3`: not feasible for `tell` because `< mu`.

Practical policies (choose one explicitly):
1. **Strict/simple policy (recommended):** enforce CMA runs where final remainder is 0 or `>= mu`; otherwise fail fast with clear message.
2. **Overrun policy:** evaluate extra points to reach `mu` or full popsize and accept budget overrun.
3. **Backend-switch policy:** for exact arbitrary budgets, use `skopt` backend.

### 3.2 Collision-free ref index map (requested)

Current ref hash should be replaced by deterministic unique indexing.

Collision-free seed construction pattern:
- `ref_index_by_id = {row['id']: i for i, row in enumerate(_REF_ROWS)}`
- `n_refs = len(_REF_ROWS)`
- `n_trials = max(1, INIT_JITTER_TRIALS)`
- `ref_idx = ref_index_by_id[ref_row['id']]`
- `unique = ((global_point_index * n_refs) + ref_idx) * n_trials + trial_index`
- `seed = INIT_JITTER_SEED + unique`

Properties:
- Unique seed for each `(point, ref, trial)` tuple.
- Stable across runs as long as `_REF_ROWS` ordering is stable.
- Not dependent on multiprocessing order.

---

## 4) Clarifications Aligned with Your Preferences

- You explicitly want replay/analysis defaults to error unless path is provided.
  - I accept that as intentional, not a bug.
- You consider `simulation_fast` canonical and parity differences plausibly chaotic.
  - My updated conclusion agrees at force-level parity.
- You want collision-free cross-ref jitter seeds.
  - This remains a valid improvement and is actionable.

---

## 5) Challenge Checklist for Claude (to audit my analysis)

Please have Claude verify these specific claims:
1. In current `optimizer.py`, does CMA wrapper pass `n_this` into `es.ask(number=...)`? (I observed it does not.)
2. Does pycma `tell` in your installed version require `len(solutions) >= mu`? (I observed yes.)
3. On same MuJoCo state, do `simulation` and `simulation_fast` force outputs match to machine precision? (I measured ~1e-21 max deltas.)
4. Are trajectory-level divergences with full coupling reproducible despite force-level parity? (I observed yes.)
5. Do current ref IDs collide under `sum(ord(c)) % 1000` hashing? (I found two collisions in current dataset.)

If Claude disproves any of these with repro steps, I’ll treat those as higher-confidence corrections.

---

## 6) Net Recommendation Summary

Highest-value immediate changes:
1. Make CMA remainder policy explicit and implement it in code path.
2. Replace ref hash jitter seed with collision-free index-based seed.
3. Tone down/qualify “equivalent/bit-exact” language for fast sim unless criterion is force-level only.
4. Add `tau_int` presence check in locomotion metrics guard.

