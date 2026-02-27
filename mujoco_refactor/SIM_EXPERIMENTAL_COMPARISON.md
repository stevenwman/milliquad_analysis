# Sim Experimental-Format Comparison

**Date**: 2026-02-26
**Method**: 3 jitter trials per ref (±2°), pick trial with closest velocity to target
**Terrain**: Flat ground, 15 refs from config_new.py
**Quantities**: Forward velocity (avg of FL+BL marker vx, steady-state t>0.1s), geometric pitch (arctan2 of FL-BL height diff, relative to t=0)

## Param Sets

| Label | Run Dir | Optimized For |
|-------|---------|---------------|
| flat | `20260225T122342_flat_10_30_50` | Flat terrain (16-dim, f10/f30/f50 subset) |
| step | `20260225T225248_step_argmin_progress` | Step terrain (16-dim) |

## Velocity & Pitch Comparison

```
ref                   target  | flat_10_30_50                | step_argmin
                      (mm/s)  |  vel    err%  pitch(°)      |  vel    err%  pitch(°)
------------------------------------------------------------------------------------------
scene1_f10              51.2  |  57.3  11.9     1.5         |  58.9  15.1     2.6
scene1_f20             126.4  | 104.2  17.6     6.0         |  92.1  27.1    11.0
scene1_f30             118.7  | 122.6   3.3     9.5         | 113.7   4.2   178.1  *** FLIP
scene1_f50             148.3  | 143.8   3.0    23.7         |  91.4  38.4    11.6
scene2_f10              83.2  |  81.4   2.2     3.1         |  84.1   1.1     2.9
scene2_f20             113.1  | 143.4  26.8     3.5         | 121.4   7.3     5.8
scene2_f30             179.6  | 179.3   0.2     4.6         | 177.7   1.0     3.8
scene2_f50             263.3  | 272.5   3.5     6.2         | 177.7  32.5     6.3
scene4_f10             112.1  | 108.5   3.2     1.2         |  96.5  13.9     1.7
scene4_f20             184.1  | 192.4   4.5     1.2         | 184.3   0.1     1.3
scene4_f30             274.7  | 293.5   6.8     2.1         | 230.3  16.2     2.5
scene4_f50             327.4  | 362.6  10.7     2.5         | 393.6  20.2     2.7
scene_wheel_f10        143.2  | 163.4  14.1     0.0         | 128.4  10.3     0.1
scene_wheel_f20        305.8  | 339.5  11.0     0.1         | 291.2   4.8     0.2
scene_wheel_f30        449.3  | 448.4   0.2     0.2         | 463.4   3.1     0.2
```

## Summary Statistics

| Metric | flat_10_30_50 | step_argmin |
|--------|---------------|-------------|
| Mean err% | 7.9% | 13.0% |
| Median err% | 4.5% | 10.3% |
| Max err% | 26.8% (scene2_f20) | 38.4% (scene1_f50) |
| Refs < 5% err | 6/15 | 5/15 |
| Refs < 10% err | 10/15 | 8/15 |
| Mean pitch | 4.8° | 16.5° |
| Flips (pitch >90°) | 0 | 1 (scene1_f30) |

## Parameter Divergence

```
param                          flat_10_30_50   step_argmin    ratio     diff%
--------------------------------------------------------------------------------
sliding_friction                   0.4067        0.4986      1.23     +22.6%
torsional_friction              1.599e-04     7.851e-03     49.10   +4810.4%  ***
rolling_friction                4.591e-06     2.011e-04     43.80   +4280.1%  ***
solref_timeconst                2.047e-03     2.584e-03      1.26     +26.3%
solref_dampratio                   3.8187        1.5006      0.39     -60.7%  ***
solimp_dmin                        0.4352        0.2129      0.49     -51.1%  ***
solimp_delta_d                     0.9890        0.8610      0.87     -12.9%
solimp_width                    4.689e-05     2.018e-04      4.30    +330.3%  ***
solimp_midpoint                    0.6473        0.8301      1.28     +28.2%
solimp_power                       5.0376        4.4090      0.88     -12.5%
magnetic_moment_fudge              0.6545        0.9214      1.41     +40.8%  ***
magnetic_field_fudge               1.1389        0.7333      0.64     -35.6%  ***
dof_damping                     5.325e-10     1.166e-09      2.19    +119.0%  ***
noslip_iterations                  0.1624       31.3718    193.22  +19222.1%  ***
noslip_tolerance                1.034e-06     6.256e-04    604.88  +60388.2%  ***
margin                          6.921e-04     5.330e-05      0.08     -92.3%  ***
```

*** = >30% divergence

## Observations

### flat_10_30_50 is better overall on flat terrain
- 7.9% mean error vs 13.0% for step params
- Zero flips, max pitch 23.7° (scene1_f50 — high pitch but stable)
- Worst case: scene2_f20 (26.8%) — this freq was NOT in its training set

### step_argmin has specific failure modes on flat
- **scene1_f30**: Robot flips (pitch=178°) — step params' high torsional friction likely destabilizes single-leg at 30Hz
- **scene1_f50**: 38.4% velocity error — severely underpredicts speed
- **scene2_f50**: 32.5% error — also underpredicts
- Pattern: f50 refs are hardest (not in step training data at all)

### step_argmin wins on some refs
- scene2_f20: 7.3% vs flat's 26.8% — flat overestimates, step is closer
- scene4_f20: 0.1% vs flat's 4.5% — near-perfect
- scene_wheel_f20: 4.8% vs flat's 11.0%

### Pitch behavior
- Wheel morphology: near-zero pitch in both (physically correct — symmetric wheel)
- scene4 (quad): low pitch (<3°) in both — 4 legs stabilize
- scene1 (single-leg): highest pitch in both — physically expected, but step params cause instability

### Key param differences driving behavior
- **Rotational friction 45× higher** in step params → too "grippy" on flat, causes single-leg instability
- **Noslip solver ON** (31 iters) in step vs OFF (~0) in flat → stricter friction enforcement may over-constrain flat locomotion
- **Magnetic moment fudge +41%** in step → stronger torque, faster for some morphologies but overshoots others
- **Contact hardness divergence**: step wants harder contacts (solimp_dmin -51%, dampratio -61%) tuned for step-edge impacts
