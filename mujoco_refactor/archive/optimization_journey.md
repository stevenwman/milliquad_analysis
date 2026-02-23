# Run Lineage for `20260219T075624`

How we arrived at the current best parameters, focused on reproducibility.

## Run Chain

```
20260218T100611_FULL_CORRECT_FRICTION/   (per-morphology/frequency sweep)
  ├── 10 sub-runs (scene1,2,4,wheel × solo; f10,30,50 × solo; combined)
  ├── scene2 & f30 failed cold-start → reran warm-started
  └── convergence analysis (compare_morphology_params.py)
        ↓ tightened bounds + consensus X0
20260218T223844                          (combined, warm-started, cost=0.014)
        ↓ replay_cmaes_state.py → cmaes_state.pkl
20260219T075624                          (resumed via --resume-from)
```

## Step 1: Per-Morphology/Frequency Sweep

**Run:** `results/20260218T100611_FULL_CORRECT_FRICTION/`

**Prereq:** condim=6 friction fix (commits `a331032`, `0f7ab37`) — torsional and
rolling friction were previously no-ops in the simulation.

**What:** Ran `run_per_morphology.sh`, which launches 7 solo CMA-ES optimizations
(one per scene, one per frequency) plus a combined run. Each solo run optimizes
the same 13D space but only evaluates a subset of the 11 references.

**Config:** sigma0=0.5, seed=69420, BATCH_SIZE=8, cold-start (X0=None, midpoint).

**Sub-run results:**

| Sub-run | Evals | Best Cost | Status |
|---------|-------|-----------|--------|
| solo_scene1 | 1160 | 0.0258 | OK |
| solo_scene2 | 144 | 12.77 | **Failed** (cold-start stagnation) |
| solo_scene4 | 1152 | 0.0003 | OK |
| solo_scene_wheel | 912 | 0.0001 | OK |
| solo_f10 | 680 | 0.0003 | OK |
| solo_f30 | 216 | 13.33 | **Failed** (cold-start stagnation) |
| solo_f50 | 1168 | 0.0284 | OK |
| combined | 840 | 0.734 | Bad local min |

**Reruns of failed sub-runs** (warm-started from a prior sweep's best):
- `20260218T213230_solo_scene2` → cost 0.0001 at 784 evals
- `20260218T215525_solo_f30` → cost 0.0006 at 720 evals

## Step 2: Convergence Analysis

**Script:** `compare_morphology_params.py` on the 10 solo results (8 original + 2
reruns).

**Finding:** Some params converge to similar values across all morphologies/freqs
(true physics), others diverge (compensating for model mismatch).

Converged → narrowed bounds. Divergent → left as-is.

The sweep ran with "maximally permissive" bounds (saved in
`results/20260218T100611_FULL_CORRECT_FRICTION/*/config.py`). Only params that
converged across sweeps got tightened for the combined run:

| Parameter | Sweep bounds | Converged? | New bounds | What changed |
|-----------|-------------|-----------|------------|--------------|
| sliding_friction | [1e-6, 10.0] | Yes (12%) | [0.01, 2.0] | lower raised 4 OOM, upper from 10→2 |
| magnetic_moment_fudge | [0.8, 1.2] | Yes (8%) | [0.75, 0.90] | narrowed around ~0.80 pin |
| dof_damping | [1e-14, 1e-6] | Yes (9%) | [1e-10, 1e-8] | 8 OOM range → 2 OOM |
| solref_dampratio | [0.01, 10.0] | Freq-only (2.4%) | [1.0, 10.0] | lower from 0.01→1.0 |
| solimp_power | [1.0, 10.0] | Freq-only (14%) | [2.0, 7.0] | both ends narrowed |
| solimp_dmin | [0.001, 0.999] | No | [0.001, 0.999] | unchanged |
| solimp_delta_d | [0.01, 0.99] | No | [0.01, 0.99] | unchanged |
| solimp_width | [1e-7, 1] | No | [1e-7, 1] | unchanged |
| solimp_midpoint | [0.01, 0.99] | No | [0.01, 0.99] | unchanged |
| solref_timeconst | [1e-5, 1.0] | No | [1e-5, 1.0] | unchanged |
| torsional_friction | [1e-6, 10.0] | No | [1e-6, 10.0] | unchanged |
| rolling_friction | [1e-6, 1e-3] | No | [1e-6, 1e-3] | unchanged |
| magnetic_field_fudge | [0.8, 1.2] | No | [0.8, 1.2] | unchanged |

**Warm-start X0:** `compare_morphology_params.py` calls `find_latest("solo_*")`
to load the best params from each of the 7 (or 9, with reruns) solo sub-runs.
The X0 was set manually by eyeballing the comparison table output:
- Converged params: geometric mean across all solo bests
- Divergent params: scene4 values (middle-of-the-road morphology)

```python
CMAES_X0 = {
    "sliding_friction": 0.17,     "torsional_friction": 0.00011,
    "rolling_friction": 5e-6,     "solref_timeconst": 0.003,
    "solref_dampratio": 6.0,      "solimp_dmin": 0.30,
    "solimp_delta_d": 0.35,       "solimp_width": 6.6e-5,
    "solimp_midpoint": 0.41,      "solimp_power": 5.3,
    "magnetic_moment_fudge": 0.81, "magnetic_field_fudge": 0.93,
    "dof_damping": 8e-10,
}
```

## Step 3: Combined Optimization

**Run:** `results/20260218T223844/`

**Command:** `uv run python optimizer.py` (no CLI flags — config.py had the
tightened bounds and X0 baked in)

**Config:** sigma0=0.5, seed=69420, BATCH_SIZE=8, N_CALLS=2400, warm-start from
consensus X0 above, tightened bounds from step 2. All 11 references active.

**Result:** cost 1.08 → 0.014 over 2272 evals (~2.5 hours). Final sigma=0.01975.

## Step 4: Resume

**Problem:** The run completed its 2400-eval budget. We wanted to continue but
warm-starting from the best params resets sigma to 0.5 (way too wide for a
refined solution). Confirmed by the failed run `20260219T064143_couldnt_recreate`
which only reached cost=0.130.

**Fix:** Reconstructed full CMA-ES state from CSV history:

```bash
cd mujoco_refactor
uv run python replay_cmaes_state.py results/20260218T223844
```

This replays every ask/tell batch through a fresh CMA-ES (same seed/config),
producing `cmaes_state.pkl` with sigma=0.01975, learned covariance, and evolution
paths.

**Run:** `results/20260219T075624/`

```bash
uv run python optimizer.py --resume-from results/20260218T223844 --n-calls 2400
```

**Config:** Identical config.py to T223844 (validated by bounds check in pickle).
CMA-ES continues from exactly where T223844 left off.

## Reproducing from Scratch

The config.py snapshot in `results/20260218T223844/config.py` is the exact config
used. To reproduce the full chain:

```bash
cd mujoco_refactor

# 1. Ensure you're on the right code (condim=6 friction fix)
git checkout e4bb5f9  # or later

# 2. Run per-morphology sweep (uses config.py as-is with sigma0=0.5, X0=consensus)
#    Note: the sweep script ran with an EARLIER config (cold-start, wider bounds).
#    The sweep results informed the config changes in step 2 above.
bash run_per_morphology.sh

# 3. Analyze convergence
uv run python compare_morphology_params.py

# 4. Apply tightened bounds + X0 to config.py (see Step 2 above)

# 5. Run combined
uv run python optimizer.py --n-calls 2400

# 6. Resume if needed
uv run python replay_cmaes_state.py results/<timestamp>
uv run python optimizer.py --resume-from results/<timestamp> --n-calls 2400
```
