# Flat Ground Validation — 3 Param Sets Compared

**Date**: 2026-02-26
**Method**: 5 jitter trials per ref (±2°), pick trial with closest velocity to target
**Terrain**: Flat ground only (15 refs from config_new.py REFERENCE_DATA)
**Metrics**: From `optimizer_new.calculate_cost()` — yaw is deviation from initial heading [0°-180°], lateral in cm

## Param Sets

| Label | Run Dir | Optimized For | Reported Cost |
|-------|---------|---------------|---------------|
| step_argmin | `20260225T225248_step_argmin_progress` | Step terrain (16-dim) | 0.210 |
| flat_10_30_50 | `20260225T122342_flat_10_30_50` | Flat terrain (16-dim, subset f10/f30/f50) | ? |
| flat_16dim | `20260225T003517_flat_16dim_corrected_warm` | Flat terrain (16-dim, all freqs) | 0.377 |

## Results Table

```
ref                   target  | step_argmin                    | flat_10_30_50                  | flat_16dim
                      (mm/s)  |  vel   err%  lat  yaw   tmb   |  vel   err%  lat  yaw   tmb   |  vel   err%  lat  yaw   tmb
-----------------------------------------------------------------------------------------------------------------------------
scene1_f10              51.2  |  58.8  14.8  0.18   0.5 0.00  |  54.7   6.9  0.01   0.7 0.00  |  51.7   1.0  0.43 178.2 0.08
scene1_f20             126.4  |  92.6  26.8  0.09   1.8 0.00  | 103.2  18.4  0.78  13.5 0.00  | -20.9 116.5  0.03   1.7 0.00
scene1_f30             118.7  | 105.0  11.6  0.42 179.5 0.14  | 126.8   6.8  0.20   9.4 0.00  | -32.2 127.1  0.83  47.0 0.00
scene1_f50             148.3  |  82.6  44.3  0.85   2.8 0.07  | 152.2   2.7  1.62   9.3 0.00  |  -2.7 101.8  0.43  11.7 0.00
scene2_f10              83.2  |  83.9   0.9  0.35   0.3 0.00  |  83.5   0.3  0.09   4.0 0.00  | 259.5 211.9  2.51 177.9 0.04
scene2_f20             113.1  | 108.8   3.8  1.27   1.0 0.00  | 140.4  24.1  0.25   0.9 0.00  | 110.4   2.4  1.08   2.2 0.12
scene2_f30             179.6  | 172.6   3.9  0.78   8.7 0.00  | 181.4   1.0  0.54   7.1 0.00  |  15.3  91.5  1.14 175.8 0.00
scene2_f50             263.3  | 194.2  26.2  0.95   4.1 0.00  | 266.9   1.4  4.08   7.9 0.00  | -10.6 104.0  3.38 123.8 0.00
scene4_f10             112.1  |  98.5  12.1  0.01   3.1 0.00  | 113.7   1.4  0.42   1.0 0.00  |  96.3  14.1  5.39   1.3 0.12
scene4_f20             184.1  | 184.3   0.1  0.98   0.2 0.00  | 191.8   4.2  0.45   4.5 0.00  |  -8.5 104.6  1.81 177.1 0.00
scene4_f30             274.7  | 235.4  14.3  0.16   5.8 0.00  | 289.7   5.5  0.21  13.4 0.00  | -13.0 104.7  0.97 173.6 0.00
scene4_f50             327.4  | 367.7  12.3  2.68   1.7 0.00  | 309.6   5.4  6.30  26.4 0.00  |   0.1 100.0  0.20 166.9 0.00
scene_wheel_f10        143.2  | 130.8   8.7  0.07   1.6 0.00  | 157.6  10.1  0.18   2.0 0.00  | 242.5  69.3  2.97 165.0 0.07
scene_wheel_f20        305.8  | 296.7   3.0  0.55   0.3 0.00  | 332.5   8.7  0.41   1.3 0.00  |  66.3  78.3  4.00 153.5 0.00
scene_wheel_f30        449.3  | 457.5   1.8  1.14   2.8 0.00  | 463.0   3.0  0.46   4.0 0.00  |  31.1  93.1  0.17 147.9 0.00
```

Units: vel in mm/s, lat in cm, yaw in degrees (deviation from initial heading, 0=straight), tmb = tumble penalty.

## Summary Statistics

| Param Set | Mean Err | Median Err | Max Err | Worst Ref |
|-----------|----------|------------|---------|-----------|
| step_argmin | 12.3% | 11.6% | 44.3% | scene1_f50 |
| flat_10_30_50 | 6.7% | 5.4% | 24.1% | scene2_f20 |
| flat_16dim | 88.0% | 100.0% | 211.9% | scene2_f10 |

## Observations

### flat_16dim is CATASTROPHICALLY BROKEN
Despite reporting cost=0.377 during optimization, this param set produces:
- Negative velocities (robot walks backwards) for most scene1/scene4 refs
- 211.9% error on scene2_f10 (vel=259.5 vs target=83.2)
- Yaw deviation 150-178° on most refs — robot turns completely around
- Only scene1_f10 (1.0% err) and scene2_f20 (2.4% err) show reasonable matches
- **This param set is NOT usable for flat terrain**

Possible explanations:
- Non-determinism with different jitter seeds than during optimization
- The optimizer's cost function may have picked trials where things worked, masking instability
- The 3 new dims (noslip_iterations, noslip_tolerance, margin) may create brittle behavior

### step_argmin is surprisingly good on flat
Step-optimized params generalize to flat terrain with 12.3% mean error:
- Excellent on scene2 and scene_wheel (most within 5%)
- Struggles with scene1_f50 (44.3%) and scene2_f50 (26.2%)
- f50 references are hardest (not included in step training data)
- Low lateral displacement overall (<1.3cm except scene4_f50)
- Yaw mostly <5° — walks very straight
- scene1_f30 is an outlier: yaw=179.5° with tumble=0.14 — turned around but didn't tumble?
- Zero tumble on 13/15 refs

### flat_10_30_50 is the best flat performer
- 6.7% mean error, best across all refs
- Worst case is scene2_f20 at 24.1% — this freq was NOT in its training set (only f10/f30/f50)
- Zero tumble on all refs
- Yaw mostly <10° — good straight-line walking
- scene4_f50 has high lateral (6.30cm) and yaw (26.4°) — robot drifts at high speed

## Parameter Divergence: flat_10_30_50 vs step_argmin

```
param                               flat           step    ratio    diff%
-------------------------------------------------------------------------
sliding_friction                0.406742       0.498642    1.226   +22.6%
torsional_friction           0.000159883     0.00785094   49.104 +4810.4%  ***
rolling_friction             4.59135e-06    0.000201104   43.801 +4280.1%  ***
solref_timeconst              0.00204655     0.00258448    1.263   +26.3%
solref_dampratio                 3.81866        1.50062    0.393   -60.7%  ***
solimp_dmin                     0.435247       0.212921    0.489   -51.1%  ***
solimp_delta_d                  0.988954       0.860978    0.871   -12.9%
solimp_width                 4.68871e-05    0.000201772    4.303  +330.3%  ***
solimp_midpoint                 0.647338       0.830067    1.282   +28.2%
solimp_power                     5.03757        4.40897    0.875   -12.5%
magnetic_moment_fudge           0.654501       0.921361    1.408   +40.8%  ***
magnetic_field_fudge             1.13892       0.733344    0.644   -35.6%  ***
dof_damping                  5.32487e-10    1.16636e-09    2.190  +119.0%  ***
noslip_iterations               0.162362        31.3718  193.221 +19222.1%  ***
noslip_tolerance              1.0342e-06    0.000625567  604.882 +60388.2%  ***
margin                       0.000692091    5.33048e-05    0.077   -92.3%  ***
```

*** = >30% divergence

Key differences:
- **Rotational friction**: Step needs ~45× more torsional + rolling friction to grip steps
- **Magnetic balance**: Step uses stronger magnets (+41%) but weaker field (-36%) — net torque rebalance
- **Noslip solver**: Step heavily relies on friction solver (31 iters vs ~0) — flat doesn't need it
- **Contact softness**: Step wants harder penetration (solimp_dmin -51%) but softer damping (solref_dampratio -61%)
- **Joint damping**: Step needs 2× more dof_damping for stability on uneven terrain
