# NoCOT + COT Numbers Reference

Auto-generated from NPZ recompute pipeline (same as nocot_065 + cot_065 scripts).

```bash
cd milliquad_opt
uv run python -m analysis.dump_numbers \
    results/20260303T192801_flat_tg \
    results/20260303T151416_step_065gate \
    results/20260303T224229_rough_tg > analysis/NOCOT_COT_NUMBERS.md
```

## Canonical Run Dirs

| Terrain | Run dir |
|---------|---------|
| flat | `results/20260303T192801_flat_tg` |
| step | `results/20260303T151416_step_065gate` |
| rough | `results/20260303T224229_rough_tg` |

## Pipeline

- **Flat**: time-gated to match experimental recording duration per condition
- **Step**: 65% spatial gate (0.05m to 0.0835m), success = full traversal to 0.1015m
- **Rough**: spatial gate to 0.155m (scene1_f10: half-gate 0.08m)
- **Trial selection**: flat/step 3 per condition, rough 5 per condition, closest to exp ref velocity
- **COT**: P = tau_ext . omega (correct under RK4), COT = energy / (m*g*distance_2d)

---

# Sim Data (selected trials, mean +/- std)

Units: velocity mm/s, pitch degrees, COT dimensionless.

## Sim Flat

| Scene | Freq | n | vx mean | vx std | pitch mean | pitch std | COT mean | COT std |
|-------|------|---|---------|--------|------------|-----------|----------|---------|
| scene1 | 10 | 3 | 57.9 | 0.1 | 1.35 | 0.07 | 0.89 | 0.00 |
| scene1 | 20 | 3 | 106.2 | 1.2 | 6.29 | 1.01 | 1.14 | 0.05 |
| scene1 | 30 | 3 | 118.9 | 2.4 | 10.14 | 1.00 | 1.68 | 0.04 |
| scene1 | 50 | 3 | 147.3 | 4.3 | 9.53 | 0.48 | 2.64 | 0.09 |
| scene2 | 10 | 3 | 89.4 | 2.2 | 3.02 | 0.28 | 0.88 | 0.02 |
| scene2 | 20 | 3 | 132.4 | 3.0 | 4.47 | 0.07 | 0.92 | 0.02 |
| scene2 | 30 | 3 | 177.2 | 4.5 | 5.57 | 0.44 | 0.96 | 0.03 |
| scene2 | 50 | 3 | 248.4 | 1.6 | 5.58 | 2.38 | 1.59 | 0.06 |
| scene4 | 10 | 3 | 102.9 | 1.3 | 1.27 | 0.06 | 0.49 | 0.00 |
| scene4 | 20 | 3 | 172.1 | 2.1 | 1.44 | 0.23 | 0.60 | 0.01 |
| scene4 | 30 | 3 | 285.4 | 3.2 | 1.76 | 0.21 | 0.65 | 0.02 |
| scene4 | 50 | 3 | 331.9 | 11.9 | 2.35 | 0.29 | 1.04 | 0.03 |
| scene_wheel | 10 | 3 | 159.4 | 0.6 | 0.03 | 0.01 | 0.17 | 0.00 |
| scene_wheel | 20 | 3 | 294.5 | 2.6 | 0.03 | 0.00 | 0.29 | 0.00 |
| scene_wheel | 30 | 3 | 389.6 | 2.5 | 0.03 | 0.00 | 0.45 | 0.00 |
| scene_wheel | 50 | 3 | 350.1 | 19.2 | 0.10 | 0.07 | 1.31 | 0.07 |

## Sim Step

| Scene | Freq | n | vx mean | vx std | pitch mean | pitch std | COT mean | COT std |
|-------|------|---|---------|--------|------------|-----------|----------|---------|
| scene1 | 10 | 3 | 18.3 | 0.4 | 3.95 | 0.15 | 2.32 | 0.06 |
| scene1 | 20 | 3 | 44.6 | 1.1 | 6.14 | 0.56 | 2.82 | 0.22 |
| scene1 | 30 | 3 | 33.2 | 6.2 | 8.38 | 1.16 | 6.08 | 1.06 |
| scene2 | 10 | 3 | 61.9 | 0.4 | 3.74 | 0.25 | 1.21 | 0.06 |
| scene2 | 20 | 3 | 82.8 | 3.6 | 5.43 | 0.78 | 1.95 | 0.06 |
| scene2 | 30 | 3 | 108.8 | 0.8 | 5.98 | 2.34 | 2.23 | 0.13 |
| scene4 | 10 | 3 | 74.4 | 9.2 | 3.00 | 0.17 | 1.25 | 0.10 |
| scene4 | 20 | 3 | 97.1 | 1.5 | 4.04 | 0.55 | 1.78 | 0.13 |
| scene4 | 30 | 3 | 75.4 | 18.5 | 3.41 | 0.15 | 2.79 | 0.63 |
| scene_wheel | 20 | 1 | 17.0 | 0.0 | 3.05 | 0.00 | 8.95 | 0.00 |
| scene_wheel | 30 | 3 | 67.4 | 22.5 | 3.62 | 0.41 | 3.80 | 1.08 |

## Sim Rough

| Scene | Freq | n | vx mean | vx std | pitch mean | pitch std | COT mean | COT std |
|-------|------|---|---------|--------|------------|-----------|----------|---------|
| scene1 | 10 | 5 | 37.8 | 3.2 | 6.38 | 0.51 | 1.47 | 0.09 |
| scene1 | 30 | 5 | 84.4 | 3.4 | 17.06 | 2.19 | 3.62 | 0.06 |
| scene1 | 50 | 5 | 58.1 | 12.6 | 15.46 | 5.98 | 11.46 | 2.36 |
| scene2 | 10 | 5 | 64.8 | 1.2 | 6.97 | 0.47 | 1.23 | 0.06 |
| scene2 | 30 | 5 | 101.9 | 5.1 | 9.04 | 1.15 | 2.60 | 0.11 |
| scene2 | 50 | 5 | 88.1 | 11.0 | 9.09 | 1.35 | 7.86 | 1.08 |
| scene4 | 10 | 5 | 76.0 | 1.8 | 6.09 | 0.49 | 0.91 | 0.05 |
| scene4 | 30 | 5 | 124.3 | 2.7 | 6.53 | 0.42 | 2.29 | 0.11 |
| scene4 | 50 | 4 | 106.3 | 41.3 | 6.54 | 0.62 | 5.74 | 2.22 |
| scene_wheel | 30 | 4 | 92.4 | 5.4 | 5.96 | 0.36 | 3.01 | 0.25 |
| scene_wheel | 50 | 5 | 109.4 | 3.3 | 6.42 | 0.69 | 5.47 | 0.20 |

---

# Experimental Data

From `experimental_data/plot_velocity_vs_freq.py` and `plot_pitch_vs_freq.py`.

## Exp Flat (velocity mm/s)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 3 | 51.2 | 2.4 |
| scene1 | 20 | 3 | 126.5 | 4.7 |
| scene1 | 30 | 4 | 118.8 | 12.7 |
| scene1 | 50 | 3 | 148.3 | 13.2 |
| scene2 | 10 | 3 | 82.4 | 0.9 |
| scene2 | 20 | 3 | 135.6 | 2.9 |
| scene2 | 30 | 3 | 179.6 | 17.9 |
| scene2 | 50 | 3 | 249.6 | 25.6 |
| scene4 | 10 | 3 | 112.1 | 6.0 |
| scene4 | 20 | 3 | 184.1 | 15.6 |
| scene4 | 30 | 4 | 274.4 | 21.0 |
| scene4 | 50 | 3 | 328.0 | 55.2 |
| scene_wheel | 10 | 3 | 143.1 | 1.4 |
| scene_wheel | 20 | 3 | 305.8 | 6.9 |
| scene_wheel | 30 | 4 | 450.3 | 18.0 |
| scene_wheel | 50 | 3 | 709.4 | 11.5 |

## Exp Flat (pitch RMS degrees)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 3 | 2.34 | 0.47 |
| scene1 | 20 | 3 | 7.90 | 1.05 |
| scene1 | 30 | 4 | 8.07 | 2.64 |
| scene1 | 50 | 3 | 14.10 | 2.58 |
| scene2 | 10 | 3 | 4.98 | 0.09 |
| scene2 | 20 | 3 | 4.04 | 0.76 |
| scene2 | 30 | 3 | 6.22 | 1.13 |
| scene2 | 50 | 3 | 4.90 | 0.55 |
| scene4 | 10 | 3 | 1.46 | 0.40 |
| scene4 | 20 | 3 | 2.22 | 0.26 |
| scene4 | 30 | 4 | 2.52 | 0.24 |
| scene4 | 50 | 3 | 4.57 | 1.95 |
| scene_wheel | 10 | 3 | 1.00 | 0.11 |
| scene_wheel | 20 | 3 | 0.75 | 0.05 |
| scene_wheel | 30 | 4 | 1.31 | 0.62 |
| scene_wheel | 50 | 3 | 0.97 | 0.09 |

## Exp Step (velocity mm/s)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 3 | 16.9 | 2.0 |
| scene1 | 20 | 3 | 46.2 | 1.8 |
| scene1 | 30 | 3 | 27.9 | 7.9 |
| scene2 | 10 | 3 | 41.5 | 21.7 |
| scene2 | 20 | 3 | 83.5 | 37.1 |
| scene2 | 30 | 3 | 109.8 | 24.5 |
| scene4 | 10 | 3 | 76.9 | 7.0 |
| scene4 | 20 | 3 | 96.3 | 15.4 |
| scene4 | 30 | 3 | 76.4 | 24.8 |
| scene_wheel | 30 | 3 | 97.2 | 4.6 |

## Exp Step (pitch RMS degrees)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 3 | 5.59 | 0.34 |
| scene1 | 20 | 3 | 8.67 | 3.15 |
| scene1 | 30 | 3 | 9.58 | 3.34 |
| scene2 | 10 | 3 | 5.76 | 0.44 |
| scene2 | 20 | 3 | 6.09 | 1.69 |
| scene2 | 30 | 3 | 5.96 | 1.56 |
| scene4 | 10 | 3 | 2.77 | 0.46 |
| scene4 | 20 | 3 | 3.80 | 1.19 |
| scene4 | 30 | 3 | 4.89 | 0.83 |
| scene_wheel | 30 | 3 | 3.58 | 0.21 |

## Exp Rough (velocity mm/s)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 4 | 42.9 | 1.0 |
| scene1 | 30 | 4 | 81.6 | 9.7 |
| scene1 | 50 | 1 | 56.5 | 0.0 |
| scene2 | 10 | 5 | 65.6 | 5.5 |
| scene2 | 30 | 4 | 128.9 | 0.3 |
| scene2 | 50 | 3 | 106.2 | 32.4 |
| scene4 | 10 | 5 | 85.7 | 6.9 |
| scene4 | 30 | 4 | 146.0 | 36.8 |
| scene4 | 50 | 1 | 101.6 | 0.0 |
| scene_wheel | 10 | 1 | 81.2 | 0.0 |
| scene_wheel | 30 | 2 | 169.3 | 12.4 |
| scene_wheel | 50 | 1 | 180.6 | 0.0 |

---

# Sim vs Exp Velocity Error

Percent error = |sim_mean - exp_mean| / exp_mean * 100.

## Flat Velocity Error (3 selected trials)

| Condition | Exp (mm/s) | Sim mean | Sim std | % Error |
|-----------|-----------|----------|---------|---------|
| L1 f10 | 51.2 | 57.9 | 0.1 | 13.1% |
| L1 f20 | 126.5 | 106.2 | 1.2 | 16.0% |
| L1 f30 | 118.8 | 118.9 | 2.4 | 0.1% |
| L1 f50 | 148.3 | 147.3 | 4.3 | 0.7% |
| L2 f10 | 82.4 | 89.4 | 2.2 | 8.5% |
| L2 f20 | 135.6 | 132.4 | 3.0 | 2.4% |
| L2 f30 | 179.6 | 177.2 | 4.5 | 1.3% |
| L2 f50 | 249.6 | 248.4 | 1.6 | 0.5% |
| L4 f10 | 112.1 | 102.9 | 1.3 | 8.2% |
| L4 f20 | 184.1 | 172.1 | 2.1 | 6.5% |
| L4 f30 | 274.4 | 285.4 | 3.2 | 4.0% |
| L4 f50 | 328.0 | 331.9 | 11.9 | 1.2% |
| WR f10 | 143.1 | 159.4 | 0.6 | 11.4% |
| WR f20 | 305.8 | 294.5 | 2.6 | 3.7% |
| WR f30 | 450.3 | 389.6 | 2.5 | 13.5% |
| WR f50 | 709.4 | 350.1 | 19.2 | 50.6% |

| Subset | Mean % Error | Median % Error | N |
|--------|-------------|----------------|---|
| All 16 conditions | 8.9% | 5.2% | 16 |
| Excl. WR f50 | 6.1% | 4.0% | 15 |

WR f50 = experimental failure (robot self-destructs at 50Hz).

## Step Velocity Error (3 selected trials)

| Condition | Exp (mm/s) | Sim mean | Sim std | % Error |
|-----------|-----------|----------|---------|---------|
| L1 f10 | 16.9 | 18.3 | 0.4 | 8.7% |
| L1 f20 | 46.2 | 44.6 | 1.1 | 3.5% |
| L1 f30 | 27.9 | 33.2 | 6.2 | 19.1% |
| L2 f10 | 41.5 | 61.9 | 0.4 | 49.0% |
| L2 f20 | 83.5 | 82.8 | 3.6 | 0.9% |
| L2 f30 | 109.8 | 108.8 | 0.8 | 0.9% |
| L4 f10 | 76.9 | 74.4 | 9.2 | 3.4% |
| L4 f20 | 96.3 | 97.1 | 1.5 | 0.8% |
| L4 f30 | 76.4 | 75.4 | 18.5 | 1.3% |
| WR f30 | 97.2 | 67.4 | 22.5 | 30.7% |

| Subset | Mean % Error | Median % Error | N |
|--------|-------------|----------------|---|
| All 10 conditions | 11.8% | 3.4% | 10 |

WR f10/f20 excluded (experimental failure). L2 f10 (49%) and WR f30 (31%) are the main outliers.
