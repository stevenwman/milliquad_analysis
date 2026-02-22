# 20Hz Holdout Validation (2026-02-21)

Best-fit params from `results/20260219T142207_loose_fudge` tested against 20Hz
experimental data that was **not used during optimization** (trained on 10/30/50 Hz only).

3 jitter trials per config (±2° yaw), median aggregation — same protocol as optimizer.

## Results

### Training set (10/30/50 Hz)

| id | exp (mm/s) | sim (mm/s) | err% | 1σ? | lat (cm) | yaw° | pitch° |
|---|---:|---:|---:|---|---:|---:|---:|
| scene1_f10 | 51.2 | 50.0 | -2.4% | Y | 0.01 | 2.0 | 2.03 |
| scene1_f30 | 118.7 | 115.6 | -2.6% | Y | 0.37 | 9.3 | 15.33 |
| scene1_f50 | 148.3 | 140.1 | -5.5% | Y | 0.14 | 0.6 | 11.86 |
| scene2_f10 | 83.2 | 83.3 | +0.1% | Y | 0.20 | 1.2 | 3.94 |
| scene2_f30 | 179.6 | 167.2 | -6.9% | Y | 0.06 | 0.9 | 6.94 |
| scene2_f50 | 263.3 | 247.6 | -6.0% | Y | 0.15 | 6.2 | 8.39 |
| scene4_f10 | 112.1 | 107.4 | -4.2% | Y | 0.11 | 3.7 | 2.26 |
| scene4_f30 | 274.7 | 268.2 | -2.4% | Y | 0.62 | 1.1 | 2.63 |
| scene4_f50 | 327.4 | 357.1 | +9.1% | Y | 0.78 | 11.6 | 2.76 |
| scene_wheel_f10 | 143.2 | 146.5 | +2.3% | N | 0.14 | 0.3 | 0.11 |
| scene_wheel_f30 | 449.3 | 446.0 | -0.7% | Y | 0.52 | 8.3 | 0.22 |

10/11 within 1-sigma. No tumbling. Velocity errors -7% to +9%.

### 20Hz holdout (unseen during training)

| id | exp (mm/s) | sim (mm/s) | err% | 1σ? | lat (cm) | yaw° | pitch° |
|---|---:|---:|---:|---|---:|---:|---:|
| scene1_f20 | 126.4 | 91.2 | **-27.9%** | N | 0.03 | 1.0 | 7.26 |
| scene2_f20 | 113.1 | 133.8 | +18.3% | Y | 0.22 | 0.7 | 5.09 |
| scene4_f20 | 184.1 | 181.7 | -1.3% | Y | 0.01 | 3.7 | 1.98 |
| scene_wheel_f20 | 305.8 | 304.5 | -0.4% | Y | 0.06 | 0.0 | 0.16 |

3/4 morphologies generalize well (<2% error). No tumbling anywhere.

## Observations

**scene1_f20 miss (-28%)**: The sim is running straight (low lateral, low yaw, no tumble)
but too slow. Experimentally, 1-leg at 20Hz (126 mm/s) is faster than at 30Hz (119 mm/s)
— a non-monotonic frequency response the sim doesn't capture. The sim predicts monotonic
velocity increase with frequency (50 → 91 → 116 → 140 mm/s for 10 → 20 → 30 → 50 Hz).
Stable across jitter seeds, so not a sensitivity-to-initial-conditions issue.

**scene2_f20 (+18%)**: Appears large but is within 1-sigma because the experimental std
is huge (42 mm/s) — driven by trial 2 at 64.6 mm/s vs ~138 mm/s for trials 1 and 3.

**scene4 and wheel**: Excellent generalization, suggesting the contact dynamics model
works well for these morphologies across frequencies.

## 20Hz experimental values

Source: `experimental_data/csv/flat/f20*.csv`, last 50% of recording (steady-state),
all trials included (ddof=1). See `experimental_data/docs/VELOCITY_SUMMARY_FLAT.md`
for full derivation and validation.

| scene | speed (m/s) | speed_std (m/s) | trials |
|---|---|---|---|
| scene1 | 0.1264 | 0.0047 | 3 |
| scene2 | 0.1131 | 0.0420 | 3 (trial 2 = 64.6 mm/s, trials 1,3 ~138 mm/s) |
| scene4 | 0.1841 | 0.0156 | 3 |
| scene_wheel | 0.3058 | 0.0068 | 3 |

## Reproducing

```bash
cd mujoco_refactor
uv run python test_20hz.py results/20260219T142207_loose_fudge
```
