# Plot Numbers Reference

Quick-grab reference for all values in the nocot and COT figures: raw data + plot styling.

## Run Directories

| Terrain | Run Dir |
|---------|---------|
| flat    | `results/20260228T013353_rk4_flat` |
| step    | `results/20260228T230022_step_q60_rk-warm` |
| rough   | `results/20260228T202903_rough_spatial_rk4` |

## Morphology Mapping

| Morph label | Scene name   | Color   | Plot label |
|-------------|-------------|---------|------------|
| leg         | scene1      | #1E88E5 | L1         |
| 2leg        | scene2      | #FFC107 | L2         |
| 4leg        | scene4      | #007561 | L4         |
| wheel       | scene_wheel | #D81B60 | WR         |

## Experimental Failure Modes

### Exp-only failures (sim doesn't reproduce)
- **Flat, scene_wheel @ 50 Hz** — robot self-destructs. Count: 3.

### Shared failures (both exp and sim)
- **Step, scene_wheel @ 10 Hz** — wheel can't move on steps. Count: 3.
- **Step, scene_wheel @ 20 Hz** — wheel can't move on steps. Count: 3.

### Gate thresholds (sim spatial gate)
- **Step**: robot must reach x = 0.1015 m
- **Rough**: robot must reach x = 0.155 m
- **Flat**: no gate

### Gate exempt
- scene1 @ 10 Hz on rough (traverses but too slow to reach gate)

## Dodge Widths (plot_panel)

| Mode        | Parameter             | Value   |
|-------------|-----------------------|---------|
| non-scatter | `dodge_width`         | 3.5 Hz  |
| scatter     | `scatter_dodge_width` | 15.0 Hz |
| scatter     | `intra_spread`        | 3.0 Hz  |

## Nocot Figure (`plot_megacomposite_nocot.py`)

- **Layout**: 3×4 (3 terrain rows × [vel_exp, vel_sim, pitch_exp, pitch_sim])
- **Rough row**: scatter_only mode (no shading)
- **Rough exp pitch**: blank (no data), used for legend placement
- **figsize**: 14.0 × 7.0
- **Font**: TeX Gyre Pagella, base size 8
- **DPI**: 200
- **Output**: `plots/megacomposite_nocot.png`

### GridSpec layout
- outer: 2 metric groups, `wspace=0.15`
- inner: header row (height ratio 0.07) + n data rows (1.0 each)
- `wspace=0.08`, `hspace=0.45`

### Font sizes
| Element           | Size |
|-------------------|------|
| Group headers     | 12, bold |
| Sub-headers (Experiment/Simulation) | 11 |
| Subplot letters   | 13, bold |
| Terrain row label | 12, bold |
| Legend entries     | 9 |
| Legend title       | 10 |
| x-axis note       | 11 |
| Failure count annotation | 14, bold |

### Marker sizes (post-hoc shrink)
- Scatter dots: 12 (shrunk from default 30)
- X markers: markersize 6, edgewidth 1.5

### Y-axis padding
- `y_lo = min(y_lo, -0.05 * (y_hi - y_lo))` — 5% bottom padding for both velocity and pitch pairs

## COT Figure (`plot_cot_only.py`)

- **Layout**: 3×1 (3 terrain rows × 1 COT column)
- **Rough row**: scatter_only mode
- **figsize**: 3.5 × (2.0 × n_rows) = 3.5 × 6.0
- **Font**: TeX Gyre Pagella, base size 6
- **Output**: `plots/cot_only.png`
- **tight_layout rect**: `[0, 0.06, 1, 0.95]`

### Failures (sim only, no exp COT data)
- Step, scene_wheel @ 10, 20 Hz (same shared failures)

### Font sizes
| Element           | Size |
|-------------------|------|
| Title             | 10, bold, pad=8 |
| Subplot letters   | 10, bold |
| Terrain row label | 10, bold |
| x-axis label      | 9 |
| Tick labels (x,y) | 10 |
| Legend entries     | 7 |

### Marker sizes (post-hoc shrink)
- Scatter dots: 12
- X markers: markersize 6, edgewidth 1.5

### Y-axis padding
- `bottom = -0.05 * top` — 5% below 0

## Pitch Exclusions

Currently empty (`PITCH_EXCLUDE = {}`). Inverted trials (pitch_rms > 30°) already excluded by `exclude_invalid`.

## Rough Experimental N/A Injection

`_inject_na_zeros(data, total_trials=5)` — assumes 5 trials per condition. Missing trials injected as 0.0, rendered as X markers at y=0 in scatter_only mode.

---

# Plotted Data (mean ± std of selected trials)

Units: velocity in mm/s, pitch in degrees, COT dimensionless.

## Sim Flat

| Scene | Freq | n | vx mean | vx std | pitch mean | pitch std | COT mean | COT std |
|-------|------|---|---------|--------|------------|-----------|----------|---------|
| scene1 | 10 | 3 | 53.7 | 0.2 | 3.68 | 0.09 | 0.83 | 0.01 |
| scene1 | 20 | 3 | 117.1 | 2.2 | 7.29 | 0.10 | 0.85 | 0.02 |
| scene1 | 30 | 3 | 119.7 | 2.5 | 14.39 | 0.50 | 1.46 | 0.08 |
| scene1 | 50 | 3 | 147.2 | 10.3 | 13.37 | 1.86 | 2.32 | 0.17 |
| scene2 | 10 | 3 | 80.9 | 0.2 | 3.49 | 0.39 | 0.75 | 0.02 |
| scene2 | 20 | 3 | 120.1 | 1.5 | 5.59 | 0.69 | 0.86 | 0.01 |
| scene2 | 30 | 3 | 181.0 | 1.2 | 6.36 | 0.66 | 0.88 | 0.01 |
| scene2 | 50 | 3 | 263.5 | 8.5 | 8.94 | 1.39 | 1.49 | 0.02 |
| scene4 | 10 | 3 | 103.0 | 1.0 | 1.49 | 0.12 | 0.64 | 0.01 |
| scene4 | 20 | 3 | 179.8 | 6.1 | 1.48 | 0.04 | 0.54 | 0.02 |
| scene4 | 30 | 3 | 267.5 | 6.3 | 2.77 | 0.09 | 0.77 | 0.01 |
| scene4 | 50 | 3 | 330.2 | 32.8 | 2.89 | 0.23 | 0.96 | 0.10 |
| scene_wheel | 10 | 3 | 136.1 | 1.0 | 0.09 | 0.01 | 0.21 | 0.00 |
| scene_wheel | 20 | 3 | 306.4 | 6.1 | 0.21 | 0.02 | 0.23 | 0.01 |
| scene_wheel | 30 | 3 | 447.4 | 7.0 | 0.24 | 0.03 | 0.33 | 0.02 |
| scene_wheel | 50 | 3 | 725.2 | 10.8 | 0.17 | 0.01 | 0.41 | 0.01 |

## Sim Step

| Scene | Freq | n | vx mean | vx std | pitch mean | pitch std | COT mean | COT std |
|-------|------|---|---------|--------|------------|-----------|----------|---------|
| scene1 | 10 | 3 | 17.8 | 3.8 | 5.85 | 0.09 | 3.18 | 0.76 |
| scene1 | 20 | 3 | 33.6 | 0.5 | 9.08 | 0.43 | 4.55 | 0.07 |
| scene1 | 30 | 3 | 33.3 | 2.9 | 11.93 | 0.58 | 7.89 | 0.76 |
| scene2 | 10 | 3 | 23.3 | 4.6 | 6.36 | 0.12 | 3.56 | 0.82 |
| scene2 | 20 | 3 | 60.8 | 14.0 | 9.66 | 1.03 | 2.55 | 0.55 |
| scene2 | 30 | 3 | 103.8 | 27.9 | 9.88 | 1.40 | 3.01 | 0.84 |
| scene4 | 10 | 3 | 63.0 | 13.3 | 5.86 | 0.46 | 1.06 | 0.17 |
| scene4 | 20 | 3 | 100.3 | 4.6 | 6.25 | 0.14 | 1.57 | 0.09 |
| scene4 | 30 | 3 | 116.5 | 8.3 | 6.45 | 0.60 | 2.08 | 0.06 |
| scene_wheel | 10 | 3 | 2.3 | 1.1 | 1.25 | 0.45 | 43.97 | 30.36 |
| scene_wheel | 20 | 3 | 4.2 | 1.1 | 2.05 | 1.08 | 42.34 | 14.87 |
| scene_wheel | 30 | 3 | 59.6 | 39.3 | 4.71 | 1.66 | 19.51 | 23.12 |

## Sim Rough

| Scene | Freq | n | vx mean | vx std | pitch mean | pitch std | COT mean | COT std |
|-------|------|---|---------|--------|------------|-----------|----------|---------|
| scene1 | 10 | 5 | 42.9 | 2.3 | 6.67 | 0.64 | 1.41 | 0.06 |
| scene1 | 30 | 5 | 86.9 | 4.9 | 13.32 | 0.85 | 3.80 | 0.38 |
| scene1 | 50 | 5 | 59.8 | 23.6 | 14.00 | 4.37 | 10.75 | 2.99 |
| scene2 | 10 | 5 | 69.1 | 2.8 | 7.36 | 1.04 | 1.28 | 0.05 |
| scene2 | 30 | 5 | 109.9 | 16.5 | 8.52 | 1.12 | 2.67 | 0.51 |
| scene2 | 50 | 5 | 100.5 | 9.3 | 10.40 | 2.37 | 6.74 | 0.47 |
| scene4 | 10 | 5 | 83.9 | 2.3 | 5.82 | 0.38 | 0.88 | 0.05 |
| scene4 | 30 | 5 | 127.6 | 9.0 | 7.01 | 0.88 | 2.32 | 0.27 |
| scene4 | 50 | 5 | 102.4 | 42.5 | 7.23 | 1.88 | 6.54 | 3.36 |
| scene_wheel | 10 | 5 | 28.0 | 12.6 | 4.83 | 1.10 | 2.75 | 1.30 |
| scene_wheel | 30 | 5 | 68.0 | 37.8 | 4.85 | 1.28 | 7.81 | 8.03 |
| scene_wheel | 50 | 5 | 110.9 | 5.2 | 6.38 | 1.15 | 5.34 | 0.33 |

## Exp Flat (velocity mm/s)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 3 | 51.2 | 2.0 |
| scene1 | 20 | 3 | 126.5 | 3.8 |
| scene1 | 30 | 4 | 118.8 | 11.0 |
| scene1 | 50 | 3 | 148.3 | 10.8 |
| scene2 | 10 | 3 | 82.4 | 0.7 |
| scene2 | 20 | 3 | 135.6 | 2.4 |
| scene2 | 30 | 3 | 179.6 | 14.6 |
| scene2 | 50 | 3 | 249.6 | 20.9 |
| scene4 | 10 | 3 | 112.1 | 4.9 |
| scene4 | 20 | 3 | 184.1 | 12.7 |
| scene4 | 30 | 4 | 274.4 | 18.2 |
| scene4 | 50 | 3 | 328.0 | 45.1 |
| scene_wheel | 10 | 3 | 143.1 | 1.1 |
| scene_wheel | 20 | 3 | 305.8 | 5.7 |
| scene_wheel | 30 | 4 | 450.3 | 15.6 |
| scene_wheel | 50 | 3 | 709.4 | 9.4 |

## Exp Step (velocity mm/s)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 3 | 16.9 | 1.7 |
| scene1 | 20 | 3 | 46.2 | 1.4 |
| scene1 | 30 | 3 | 27.9 | 6.4 |
| scene2 | 10 | 3 | 41.5 | 17.7 |
| scene2 | 20 | 3 | 83.5 | 30.3 |
| scene2 | 30 | 3 | 109.8 | 20.0 |
| scene4 | 10 | 3 | 76.9 | 5.7 |
| scene4 | 20 | 3 | 96.3 | 12.5 |
| scene4 | 30 | 3 | 76.4 | 20.2 |
| scene_wheel | 30 | 3 | 97.2 | 3.7 |

## Exp Rough (velocity mm/s)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 4 | 42.9 | 0.9 |
| scene1 | 30 | 4 | 81.6 | 8.4 |
| scene1 | 50 | 1 | 56.5 | 0.0 |
| scene2 | 10 | 5 | 65.6 | 4.9 |
| scene2 | 30 | 4 | 128.9 | 0.2 |
| scene2 | 50 | 3 | 106.2 | 26.5 |
| scene4 | 10 | 5 | 85.7 | 6.2 |
| scene4 | 30 | 4 | 146.0 | 31.9 |
| scene4 | 50 | 1 | 101.6 | 0.0 |
| scene_wheel | 10 | 1 | 81.2 | 0.0 |
| scene_wheel | 30 | 2 | 169.3 | 8.8 |
| scene_wheel | 50 | 1 | 180.6 | 0.0 |

## Exp Flat (pitch RMS degrees)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 3 | 2.34 | 0.38 |
| scene1 | 20 | 3 | 7.90 | 0.85 |
| scene1 | 30 | 4 | 8.07 | 2.28 |
| scene1 | 50 | 3 | 14.10 | 2.10 |
| scene2 | 10 | 3 | 4.98 | 0.07 |
| scene2 | 20 | 3 | 4.04 | 0.62 |
| scene2 | 30 | 3 | 6.22 | 0.93 |
| scene2 | 50 | 3 | 4.90 | 0.45 |
| scene4 | 10 | 3 | 1.46 | 0.33 |
| scene4 | 20 | 3 | 2.22 | 0.22 |
| scene4 | 30 | 4 | 2.52 | 0.21 |
| scene4 | 50 | 3 | 4.57 | 1.59 |
| scene_wheel | 10 | 3 | 1.00 | 0.09 |
| scene_wheel | 20 | 3 | 0.75 | 0.04 |
| scene_wheel | 30 | 4 | 1.31 | 0.54 |
| scene_wheel | 50 | 3 | 0.97 | 0.07 |

## Exp Step (pitch RMS degrees)

| Scene | Freq | n | mean | std |
|-------|------|---|------|-----|
| scene1 | 10 | 3 | 5.59 | 0.28 |
| scene1 | 20 | 3 | 8.67 | 2.57 |
| scene1 | 30 | 3 | 9.58 | 2.73 |
| scene2 | 10 | 3 | 5.76 | 0.36 |
| scene2 | 20 | 3 | 6.09 | 1.38 |
| scene2 | 30 | 3 | 5.96 | 1.28 |
| scene4 | 10 | 3 | 2.77 | 0.37 |
| scene4 | 20 | 3 | 3.80 | 0.97 |
| scene4 | 30 | 3 | 4.89 | 0.67 |
| scene_wheel | 30 | 3 | 3.58 | 0.17 |
