# Optimization Journey: How We Got to cost=0.014

## Timeline Summary

| Date | Run | Best Cost | Key Change |
|------|-----|-----------|------------|
| Feb 15 | Various combined | 0.94 → 0.61 | Iterating cost function: added lateral, yaw, tried CMA-ES sigma |
| Feb 16 | `wider_param_C_BEST-SO-FAR` | 0.128 | dmax reparameterization + wider bounds |
| Feb 17 | `20260217T115204_FULL` | per-scene 0.001–0.008 | First per-morphology/frequency sweep |
| Feb 17 | `20260217T172330_FULL` | per-scene 0.0000–0.003 | Second sweep (better) |
| Feb 18 | `20260218T100611_FULL_CORRECT_FRICTION` | per-scene 0.0001–0.026 | **Third sweep after fixing condim=6 friction** |
| Feb 18 | ↳ scene2 & f30 sub-runs | 12.77, 13.33 | **Failed** — cold-start stagnation |
| Feb 18 | ↳ rerun scene2 & f30 (warm-started) | 0.0001, 0.0006 | Fixed with warm-start from known-good params |
| Feb 18 | ↳ combined sub-run | 0.734 | Bad local minimum (sliding_friction=3.66) |
| Feb 18 | Convergence analysis | — | Identified converged vs divergent params across sweeps |
| Feb 18 | `20260218T223844` | **0.014** | Tightened bounds + warm-start X0 from sweep consensus |
| Feb 19 | `20260219T064143_couldnt_recreate` | 0.130 | Warm-start from T223844 best, sigma=0.5 too wide |
| Feb 19 | `20260219T075624` | 0.012 (ongoing) | Resumed from T223844 CMA-ES state (sigma=0.02) |

---

## Phase 1: Building the Cost Function (Feb 14–16)

Starting from a basic velocity-matching objective, we iteratively added penalty
terms to handle failure modes:

- **Lateral penalty** (Feb 15): robots drifting sideways were getting good
  forward velocity scores. Added squared lateral displacement penalty.
- **Yaw penalty** (Feb 15): single-leg robots spinning out at high frequencies.
  Added heading deviation penalty beyond 60 deg.
- **Tumble penalty**: robots flipping over. Per-frame penalty when uprightness
  drops below horizontal.
- **Velocity variance** (Feb 16): uneven fit across references. Added penalty on
  variance of relative velocity errors.
- **Dead-zone** (Feb 16): using experimental speed_std as dead-zone so velocity
  error inside 1-sigma of measurement uncertainty costs zero.

CMA-ES replaced skopt (Bayesian/GP) around Feb 15 — GP was struggling in 13D
(boundary-seeking, few improvements). CMA-ES immediately performed better.

Key sigma0 insight: sigma=0.15 stagnated at cost 0.692; sigma=0.5 broke through
to 0.612. Wide initial exploration is critical.

Best combined cost by end of Phase 1: **0.128** (run `wider_param_C_BEST-SO-FAR`,
after dmax reparameterization and wider bounds).

## Phase 2: Per-Morphology/Frequency Sweeps (Feb 17–18)

Instead of optimizing all 11 references at once, we ran per-scene and
per-frequency sweeps to understand which params are shared vs morphology-specific.

### Sweep structure

Each sweep ran 7 solo optimizations (scene1, scene2, scene4, scene_wheel, f10,
f30, f50) plus one combined run. Three full sweeps were run:

1. `20260217T115204_FULL` — first sweep
2. `20260217T172330_FULL` — second sweep
3. `20260218T100611_FULL_CORRECT_FRICTION` — **after fixing condim=6 friction**

The condim=6 fix (commit `a331032`, `0f7ab37`) was a critical bug: torsional and
rolling friction were previously having no effect on the simulation. Once fixed,
friction parameters became meaningful and the sweep results changed substantially.

### Failures in the third sweep

Two sub-runs failed with costs >12:
- **scene2** (cost=12.77 at 144 evals): cold-start, CMA-ES sigma collapsed
  before finding locomotion
- **f30** (cost=13.33 at 216 evals): same failure mode

Root cause: the log-midpoint of sliding_friction was below 0.05, which produces
no locomotion. CMA-ES starting from there with sigma=0.5 couldn't escape fast
enough.

Fix: re-ran both with `--warm-start-from` a known-good run, which placed the
initial mean in a locomotion-producing region. Results: scene2=0.0001, f30=0.0006.

### Convergence analysis

Using `compare_morphology_params.py`, we analyzed which params converged across
sweeps:

**Converged** (similar across morphologies/frequencies — these are true physical
parameters):
- `sliding_friction`: 0.05–0.36 (12% relative spread)
- `magnetic_moment_fudge`: pins at ~0.80 across all runs (8%)
- `dof_damping`: 5e-10 to 3e-9 (9%)

**Converged by frequency only** (not morphology):
- `solref_dampratio`: 5.8–6.9 (2.4%)
- `solimp_power`: 4.3–5.9 (14%)

**Divergent** (different across morphologies — likely compensating for model
mismatch):
- `solimp_dmin`, `solimp_width`, `solimp_midpoint`, `solref_timeconst`

## Phase 3: Informed Combined Optimization (Feb 18)

Armed with convergence analysis, we made two key changes to config.py:

### 1. Tightened search space bounds

Narrowed bounds on converged params to focus CMA-ES search:
- `sliding_friction`: [0.01, 2.0] (was [1e-5, 0.8])
- `magnetic_moment_fudge`: [0.75, 0.90] (was [0.5, 1.5])
- `dof_damping`: [1e-10, 1e-8] (was [7e-12, 7e-9])
- `solref_dampratio`: [1.0, 10.0] (was [0.1, 2.0])
- `solimp_power`: [2.0, 7.0] (was [1.0, 6.0])

Divergent params kept wide to give the optimizer freedom.

### 2. Warm-start X0 from sweep consensus

Instead of cold-starting from the space midpoint, we set CMAES_X0 to the
geometric mean of per-morphology sweep bests (for converged params) and scene4
values (for divergent params):

```python
CMAES_X0 = {
    "sliding_friction": 0.17,
    "torsional_friction": 0.00011,
    "rolling_friction": 5e-6,
    "solref_timeconst": 0.003,
    "solref_dampratio": 6.0,
    "solimp_dmin": 0.30,
    "solimp_delta_d": 0.35,
    "solimp_width": 6.6e-5,
    "solimp_midpoint": 0.41,
    "solimp_power": 5.3,
    "magnetic_moment_fudge": 0.81,
    "magnetic_field_fudge": 0.93,
    "dof_damping": 8e-10,
}
```

### Result: `20260218T223844`

Cost trajectory: 1.08 → 0.54 → 0.22 → 0.075 → **0.014** over 2272 evals
(~2.5 hours).

Average |delta%| velocity across all 11 references: **4.8%**. Most references
within 5% of experimental measurement. Zero tumble, low lateral drift, no yaw
blowups.

## Phase 4: Continuing the Run (Feb 19)

### Failed attempt: warm-start continuation

Tried warm-starting a new run from T223844's best params (`--warm-start-from`).
This only transfers the mean vector (X0) — CMA-ES resets sigma to 0.5 and
covariance to identity. The old run had refined sigma down to ~0.02, so sigma=0.5
scattered the first generation far from the optimum. Result: cost=0.130, worse
than where we started.

### Solution: full state resume

Built `replay_cmaes_state.py` to reconstruct the CMA-ES internal state (sigma,
covariance matrix, evolution paths) from the CSV history by replaying the
ask/tell sequence. Added `--resume-from` to optimizer.py to load the pickled
state. The resumed run (`20260219T075624`) continues from exactly where T223844
left off.

## Best Parameters (as of `20260218T223844`, cost=0.014)

| Parameter | Value | Note |
|-----------|-------|------|
| sliding_friction | 0.504 | |
| torsional_friction | 5.9e-4 | |
| rolling_friction | 2.7e-6 | |
| solref_timeconst | 0.0025 | |
| solref_dampratio | 2.82 | |
| solimp_dmin | 0.495 | |
| solimp_delta_d | 0.700 | → dmax = 0.848 |
| solimp_width | 2.7e-5 | |
| solimp_midpoint | 0.673 | |
| solimp_power | 5.51 | |
| magnetic_moment_fudge | 0.752 | near lower bound |
| magnetic_field_fudge | 0.976 | |
| dof_damping | 5.9e-10 | |
