# Validation Results (2026-02-26)

## Step-Optimized Params on Step Terrain

**Run**: `20260225T225248_step_argmin_progress`
**Cost**: 0.210 at eval 2496

### Per-Reference Velocity Match

All refs zero tumble. Velocity errors mostly within +/-10%.

Key observations:
- scene1/2/4 at f10/f20/f30: good velocity matches
- scene_wheel f10/f20: correctly produces near-zero velocity (failure mode validated)
- scene_wheel f30: chaotic — only ~10% of trials match target (see WHEEL_STEP_CHAOS_ANALYSIS.md)

### Wheel Morphology Chaos Analysis

See `mujoco_refactor/WHEEL_STEP_CHAOS_ANALYSIS.md` for full data.

**Summary table:**

| Frequency | Target (mm/s) | Mean (mm/s) | Max (mm/s) | >10 mm/s rate | Behavior |
|-----------|---------------|-------------|------------|---------------|----------|
| f10 | 0.0 | 4.2 | 13.6 | 10% | Strong failure — nearly always stuck |
| f20 | 0.0 | 8.9 | 34.7 | 30% | Soft failure — occasional partial traversal |
| f30 | 93.8 | 30.5 | 104.4 | 17% (<20%err) | Chaotic success — bimodal, ~10% good matches |

## Cross-Terrain Generalization (NOT YET DONE)

Step-optimized params have NOT been systematically validated on flat terrain yet.
Flat-optimized params were tested on steps: 92.8% mean velocity error (catastrophic failure).

## Rough Terrain (INCOMPLETE)

`eval_rough_terrain.py` exists but has contact issues at FLAT_LEAD=0.025. Needs debugging.
