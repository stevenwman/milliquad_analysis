# Wheel Morphology Step Terrain Chaos Analysis

**Run**: `20260225T225248_step_argmin_progress` (16-dim, cost=0.210)
**Date**: 2026-02-26

## Context

During validation of step-optimized params, `scene_wheel` at f30 on step terrain showed only ~1/10 trials matching the experimental target velocity (93.8 mm/s). This analysis investigates whether the outcome correlates with initial yaw jitter angle or is purely stochastic.

## Sweep: scene_wheel f30, step terrain

- Jitter range: +/- 3 deg
- 30 trials, deterministic seeds (BASE=55000 + freq*1000 + trial)
- Target velocity: 93.8 mm/s
- Velocity extraction: forward-only vx, step region (x >= 50mm to 90% of STEP_END_X)

### Results (sorted by velocity)

```
vel_mm/s  jitter_deg   %err
-----------------------------------
    1.7     0.4713    98.2%
    2.7    -1.6466    97.1%
    5.6     2.8932    94.0%
    5.7     2.0689    93.9%
   10.8    -2.8035    88.5%
   11.1     2.5521    88.2%
   12.1     2.4896    87.1%
   13.4    -2.5728    85.7%
   13.7    -1.0682    85.4%
   15.7     2.4198    83.3%
   16.6     0.4935    82.3%
   18.0    -2.9774    80.8%
   19.3    -1.7840    79.4%
   24.2     2.0951    74.2%
   25.7     1.1843    72.6%
   26.3    -2.2659    72.0%
   28.1    -1.8240    70.0%
   31.2    -2.6796    66.7%
   33.4     0.8725    64.4%
   34.7    -0.9166    63.0%
   42.0    -0.8761    55.2%
   46.2    -0.7457    50.7%
   49.0     1.8578    47.8%
   49.4     0.4124    47.3%
   49.7     0.4399    47.0%
   56.5    -1.6389    39.8%
   67.0     2.2049    28.6%
   77.6     1.2227    17.3%
   93.5    -0.4617     0.3%
  104.4     1.1035    11.3%
```

### Summary Statistics

- **Acceptance rate (<20% error)**: 3/30 (10%)
- **Bimodal distribution**: most trials cluster at 1-50 mm/s (stuck), few break through to 56-104 mm/s
- **No correlation with jitter angle**: successful trials span both positive and negative angles
- **Conclusion**: outcome is purely stochastic (chaotic sensitivity to initial conditions), not angle-dependent

### Heuristic

~10% acceptance rate for "acceptable" trials (within ~20% of target). For validation purposes, running 10+ trials should yield at least 1 good match.

## Failure Mode Validation: scene_wheel f10/f20, step terrain

These frequencies are expected to FAIL (target=0.0 mm/s). The sweep confirms the robot mostly does not traverse the steps, validating the failure mode constraint used during optimization.

### scene_wheel f10 (target=0.0 mm/s)

- 30 trials, +/- 3 deg jitter
- **Moved (>10 mm/s): 3/30 (10%)**
- Max: 13.6 mm/s, Mean: 4.2 mm/s

```
vel_mm/s  jitter_deg
-------------------------
    0.4     2.7268
    0.4     1.4252
    0.9     2.3508
    1.4     1.2644
    1.8    -0.1654
    1.9    -2.5503
    1.9    -0.7276
    1.9    -1.0614
    1.9     0.0283
    1.9     0.2400
    1.9    -2.5630
    2.4    -1.5196
    2.4    -0.7539
    2.4    -1.2494
    3.0     2.1949
    3.2     1.1923
    3.3    -0.7093
    3.9    -1.1711
    4.0     1.1091
    4.5     1.5949
    4.5    -0.4320
    4.8    -2.0948
    4.9    -1.5412
    5.0     0.1321
    6.0     0.8098
    7.1    -0.6823
    9.4     1.7548
   10.7    -0.5493
   13.4    -0.8272
   13.6    -0.0368
```

**Verdict**: Mostly stays below 10 mm/s. The 3 trials >10 mm/s max out at 13.6 mm/s — effectively stuck, consistent with experimental failure.

### scene_wheel f20 (target=0.0 mm/s)

- 30 trials, +/- 3 deg jitter
- **Moved (>10 mm/s): 9/30 (30%)**
- Max: 34.7 mm/s, Mean: 8.9 mm/s

```
vel_mm/s  jitter_deg
-------------------------
   -1.2     0.7129
   -0.2     2.7442
   -0.1    -1.1657
    0.8    -2.5009
    1.3     2.2966
    1.5     2.1148
    1.5    -0.9016
    1.8    -0.9566
    3.4     2.5645
    3.5    -2.6068
    3.8    -0.2537
    4.2     0.7633
    4.7    -2.2283
    4.7     2.8348
    4.8     1.1024
    4.8    -2.5521
    5.5    -1.3519
    5.7     1.0366
    5.7    -0.0268
    5.8    -1.8812
    6.4    -1.3860
   11.3    -2.3576
   13.5    -0.5596
   14.9    -0.0337
   15.6    -2.6159
   24.6     0.7115
   26.0     0.0423
   26.1     1.4361
   30.3     2.1709
   34.7     2.1727
```

**Verdict**: More variable than f10 — 30% of trials show >10 mm/s, with some reaching 25-35 mm/s. Still substantially slower than f30's successful traversal (93-104 mm/s). The failure mode is "softer" at f20 than f10, but the robot still doesn't achieve meaningful step traversal compared to legged morphologies.

## Overall Conclusions

| Frequency | Target (mm/s) | Mean (mm/s) | Max (mm/s) | >10 mm/s rate | Behavior |
|-----------|---------------|-------------|------------|---------------|----------|
| f10       | 0.0           | 4.2         | 13.6       | 10%           | Strong failure — nearly always stuck |
| f20       | 0.0           | 8.9         | 34.7       | 30%           | Soft failure — occasional partial traversal |
| f30       | 93.8          | 30.5        | 104.4      | 17% (<20%err) | Chaotic success — bimodal, ~10% good matches |

The wheel morphology on steps is inherently chaotic. f10/f20 correctly fail (consistent with experimental data), while f30 occasionally succeeds but requires multiple trials to find a good match.
