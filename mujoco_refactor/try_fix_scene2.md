# Scene2 Failure Analysis (condim=6 sweep, 2026-02-18)

## Summary

Scene2 (2-leg morphology) produced **zero locomotion** across all 1200 CMA-ES evals.
Best cost: 12.77 (vs scene1: 0.026, scene4: 0.0008). Max velocity achieved: 0.009 m/s (targets: 0.08-0.26 m/s).

**Root cause: CMA-ES stagnation in flat zero-velocity plateau, NOT a condim=6 incompatibility.**

Scene2 does produce locomotion at condim=6 — scene4-best params give 0.144 m/s on scene2 (target 0.1796).

## Evidence

### 1. Optimizer never found the viable parameter region

The working params (scene4-best applied to scene2) have `sliding_friction = 0.153`.
The scene2 optimizer's 95th percentile only reached `sliding_friction = 0.004` — 40x too low.

7 of 13 params were OUTSIDE the scene2 optimizer's explored range:

| Param | Scene2 optimizer median | Working value | Gap |
|-------|------------------------|---------------|-----|
| sliding_friction | 0.0015 | 0.153 | 100x |
| rolling_friction | 0.000198 | 4.8e-6 | 40x (opposite dir) |
| solimp_delta_d | 0.64 | 0.056 | 11x |
| solimp_width | 0.0018 | 3.0e-5 | 60x |
| dof_damping | 2.6e-10 | 9.4e-10 | 3.6x |

### 2. CMA-ES sigma collapsed before finding improvement

| Evals | sf spread (decades) | Best cost | What happened |
|-------|-------------------|-----------|---------------|
| 1-80 | 2.15 | 12.87 | Initial exploration — nothing moves |
| 81-160 | 1.35 | 12.77 | Sigma shrinking, no gradient signal |
| 161-240 | 0.96 | 12.91 | Sigma collapsed — can't escape plateau |
| 241-320 | 0.88 | 12.89 | Stuck permanently |

Compare to scene4 (which worked):

| Evals | sf spread (decades) | Best cost | What happened |
|-------|-------------------|-----------|---------------|
| 1-80 | 2.32 | 12.23 | Same flat start |
| 161-240 | 1.69 | **1.40** | Found movement, CMA-ES shifts population |
| 241-320 | 1.55 | **0.02** | Population moves to sf ~ 0.01-1.1 |

### 3. Why scene2 specifically?

Scene2's 2-leg geometry needs higher sliding friction (~0.05+) to generate ground traction at condim=6.
Scene1 and scene4 got lucky — some initial samples landed in the viable region, giving CMA-ES a gradient.
Scene2's viable basin is narrower/shifted, and none of the first 80 random samples hit it.

### 4. Confirmation test

```
scene4_best params @ scene2 condim=6: vel = 0.144 m/s (target 0.1796)  -- WORKS
scene1_best params @ scene2 condim=6: vel = 0.100 m/s (target 0.1796)  -- WORKS
```

## TODO — Fixes to try

1. **Warm-start scene2 from scene4-best params** — set CMAES_X0 to scene4's best point before re-running scene2 solo. This puts the optimizer directly in the viable basin.

2. **Warm-start all scenes from a shared good starting point** — instead of cold-start (space midpoint), use a known-good point as X0 for all per-morphology runs. Prevents any scene from starting in a dead plateau.

3. **Narrower sliding_friction lower bound** — current [1e-6, 10] has log-midpoint at 0.003. If no morphology works below 0.01, raising the lower bound to 1e-3 or 1e-2 shifts the CMA-ES midpoint into the viable region ([1e-2, 10] midpoint = 0.316).

4. **Larger sigma0 or restarts** — sigma0=0.5 wasn't enough. Could try sigma0=0.7, or implement CMA-ES with restarts (IPOP-CMA-ES) to re-explore after sigma collapse.

5. **Two-phase optimization** — Phase 1: coarse grid search or large-sigma CMA-ES to find any point with nonzero velocity per scene. Phase 2: fine-tune from that point with normal CMA-ES.
