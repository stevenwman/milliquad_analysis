# Experimental Pitch RMS — How It's Computed

Source: `experimental_data/plot_pitch_vs_freq.py`, imported by `milliquad_opt/analysis/plot_megacomposite_nocot.py` via `extract_flat_pitch` and `extract_step_pitch_q60`.

## Flat terrain (10, 30, 50 Hz)

```
load csv, truncate to first `points` rows
theta = mass_C θ column (last θ under "mass_C" header group)
theta_ss = theta[time > steady_t]
pitch_rms = std(theta_ss)
```

Per-condition `(points, steady_t)`:

| Freq | Morph | points | steady_t |
|------|-------|--------|----------|
| 10   | leg   | 2500   | 0.3      |
| 10   | 2leg  | 1480   | 0.3      |
| 10   | 4leg  | 1199   | 0.3      |
| 10   | wheel | 910    | 0.3      |
| 30   | leg   | 1100   | 0.15     |
| 30   | 2leg  | 760    | 0.3      |
| 30   | 4leg  | 550    | 0.3      |
| 30   | wheel | 350    | 0.3      |
| 50   | leg   | 1960   | 0.35     |
| 50   | 2leg  | 1280   | 0.35     |
| 50   | 4leg  | 1060   | 0.35     |
| 50   | wheel | 620    | 0.25     |

## Flat terrain (20 Hz)

```
load csv (full length, no truncation)
theta = mass_C θ column
theta_ss = theta[len//2 :]     # last 50%
pitch_rms = std(theta_ss)
```

## Step terrain (all freqs)

```
load csv (full length)
theta = mass_C θ column
lo = int(0.45 * len)
hi = int(0.75 * len)
theta_ss = theta[lo : hi]      # 30% window centered at 60%
pitch_rms = std(theta_ss)
```

## Trial selection

Same `FLAT_CONDITIONS`, `FLAT_20HZ`, `STEP_CONDITIONS` lists as velocity. Per-condition mean and std(ddof=1) computed across selected trials.

## Note on velocity windowing

Velocity uses the **same windowing** as pitch for all frequencies:
- 10/30/50 Hz flat: `nanmean(vx[time > steady_t])` after truncating to `points` rows
- 20 Hz flat: `nanmean(vx[len//2:])` (last 50%)
- Step: `nanmean(vx[q75-150 : q75+150])` (q75 window) for `extract_step`, or `nanmean(vx[0.45*n : 0.75*n])` for `extract_step_q60`
