# COT Evaluation Summary

## Parameters

- **Source**: `results/20260225T225248_step_argmin_progress`
- **Terrain**: flat
- **References**: 15 flat conditions from `config_new.REFERENCE_DATA`
- **Trials per ref**: 10
- **Top-K selection**: 3 (by smallest |v_sim - v_target|)
- **Yaw jitter**: +/-2.0 deg
- **Base seed**: 77777 (seed = 77777 + ref_idx * 100 + trial_idx)
- **Sim duration**: 3.0s, settle: 0.1s
- **COT formula**: `COT = integral(|P_ext|) dt / (m * g * d)`
  - P_ext = sum_legs(tau_ext . omega)
  - d = cumulative 2D path length

## Robot Masses

| Scene | Mass (mg) |
|-------|----------|
| scene1 | 103.00 |
| scene2 | 105.00 |
| scene4 | 109.00 |
| scene_wheel | 91.00 |

## Results

| Ref | Freq | Target (mm/s) | Sim vel (mm/s) | COT | COT std | Dist (mm) | Energy (uJ) |
|-----|------|--------------|---------------|-----|---------|-----------|-------------|
| scene1_f10 | 10 | 51.2 | 58.9 | 4.0 | 0.0 | 174.7 | 702.14 |
| scene1_f20 | 20 | 126.4 | 92.0 | 4.4 | 0.2 | 269.7 | 1207.34 |
| scene1_f30 | 30 | 118.7 | 106.0 | 4.7 | 0.1 | 315.6 | 1502.10 |
| scene1_f50 | 50 | 148.3 | 83.5 | 7.9 | 0.2 | 258.9 | 2073.39 |
| scene2_f10 | 10 | 83.2 | 83.1 | 2.2 | 0.0 | 246.0 | 560.86 |
| scene2_f20 | 20 | 113.1 | 118.6 | 3.1 | 0.2 | 347.9 | 1123.02 |
| scene2_f30 | 30 | 179.6 | 171.8 | 2.6 | 0.1 | 503.1 | 1354.44 |
| scene2_f50 | 50 | 263.3 | 190.1 | 3.6 | 0.1 | 563.2 | 2098.79 |
| scene4_f10 | 10 | 112.1 | 96.8 | 2.3 | 0.0 | 281.9 | 699.68 |
| scene4_f20 | 20 | 184.1 | 184.0 | 2.2 | 0.1 | 535.4 | 1245.71 |
| scene4_f30 | 30 | 274.7 | 233.4 | 1.7 | 0.1 | 678.9 | 1246.16 |
| scene4_f50 | 50 | 327.4 | 385.9 | 1.7 | 0.0 | 1122.5 | 2041.39 |
| scene_wheel_f10 | 10 | 143.2 | 129.4 | 1.1 | 0.0 | 378.6 | 385.59 |
| scene_wheel_f20 | 20 | 305.8 | 294.7 | 1.2 | 0.0 | 858.4 | 944.70 |
| scene_wheel_f30 | 30 | 449.3 | 466.8 | 1.1 | 0.0 | 1355.4 | 1379.92 |

## Per-Morphology Summary

| Morphology | Mean COT | Freq range |
|------------|----------|------------|
| 1-leg | 5.3 | 10-50 Hz |
| 2-leg | 2.9 | 10-50 Hz |
| 4-leg | 2.0 | 10-50 Hz |
| wheel | 1.2 | 10-30 Hz |

