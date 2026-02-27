# PARAMETER TRIFECTA ANALYSIS — FLAT VS STEP VS ROUGH

**Date**: 2026-02-27

## Source Runs

| Label | Run Dir | Terrain | Cost | Dims | Refs |
|-------|---------|---------|------|------|------|
| flat | `20260225T122342_flat_10_30_50` | Flat (f10/f30/f50) | 0.128 | 16 | 9 |
| step | `20260225T225248_step_argmin_progress` | Step 8×1mm | 0.210 | 16 | 10 |
| rough | `zzz_rough_v2` | Rough seed42 1mm std | 0.395 | 16 | 7 |

## Parameter Comparison

```
param                    flat       step      rough     notes
─────────────────────────────────────────────────────────────────────
sliding_friction         0.407      0.499     0.626     ↑ with terrain difficulty
torsional_friction       1.60e-4    7.85e-3   1.58e-4   step 50× outlier
rolling_friction         4.59e-6    2.01e-4   1.70e-6   step 50× outlier
solref_timeconst         0.00205    0.00258   0.000845  rough 2.5× stiffer
solref_dampratio         3.82       1.50      4.08      step much lower
solimp_dmin              0.435      0.213     0.329     all different
solimp_dmax              0.994      0.890     0.733     ↓ with terrain difficulty
solimp_width             4.69e-5    2.02e-4   2.10e-5   step 4× wider
solimp_midpoint          0.647      0.830     0.877     ↑ with terrain difficulty
solimp_power             5.04       4.41      5.41      similar (~5)
magnetic_moment_fudge    0.655      0.921     0.896     flat is outlier (0.65 vs ~0.9)
magnetic_field_fudge     1.139      0.733     1.166     step is outlier
dof_damping              5.32e-10   1.17e-9   9.54e-10  step 2× higher
noslip_iterations        0          31        0         step uses friction solver
noslip_tolerance         1.03e-6    6.26e-4   1.17e-6   follows noslip_iters
margin                   6.92e-4    5.33e-5   4.58e-6   ↓ with terrain difficulty
```

## Key Observations

### Monotonic trends with terrain difficulty (flat → step → rough)
- **sliding_friction**: 0.41 → 0.50 → 0.63. More friction needed as terrain gets rougher.
- **solimp_midpoint**: 0.65 → 0.83 → 0.88. Impedance transition shifts toward stiffer end.
- **solimp_dmax**: 0.99 → 0.89 → 0.73. Narrower impedance range on rough terrain.
- **margin**: 6.9e-4 → 5.3e-5 → 4.6e-6. Tighter contact detection on rougher terrain.

### Step is the outlier
- **Rotational friction**: torsional ~50× and rolling ~50× higher than flat/rough. Step edges need rotational grip that heightfield bumps don't.
- **Noslip solver**: 31 iterations vs 0 for flat/rough. Binary divergence — step terrain requires the friction constraint solver; rough terrain doesn't.
- **solref_dampratio**: 1.50 vs 3.8–4.1. Step wants underdamped contacts.
- **magnetic_field_fudge**: 0.73 vs ~1.15. Step weakens the external field.

### Flat is the outlier
- **magnetic_moment_fudge**: 0.655 vs ~0.9 for step/rough. Flat wants weaker magnets.
- Compensated by higher field_fudge (1.14): net torque = 0.65 × 1.14 = 0.74 (flat) vs 0.92 × 0.73 = 0.67 (step) vs 0.90 × 1.17 = 1.05 (rough).
- **Effective magnetic torque**: rough > flat > step. Rougher terrain needs stronger drive torque to overcome surface resistance.

### Consensus parameters (all three agree within ~30%)
- **solimp_power**: 4.4–5.4 (contact impedance exponent)
- **solimp_delta_d**: 0.55–0.60 (dmax-dmin transition width, normalized)

## Cross-Terrain Generalization

| Params | On flat | On step | On rough |
|--------|---------|---------|----------|
| flat | **6.7%** mean err | ~93% (fails) | ~20% |
| step | 12.3% mean err | **~21%** | fails (overfits to edges) |
| rough | ~20% mean err | untested | **~39.5%** |

No single param set generalizes well across all three terrains. The magnetic balance (moment × field fudge) and contact model (noslip, rotational friction) diverge too much between step and non-step terrains.
