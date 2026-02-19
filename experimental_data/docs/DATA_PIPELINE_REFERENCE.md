# Experimental Data Reference (CSV Format + MATLAB Pipeline)

This document describes only the data and processing currently used by scripts in `experimental_data/`.

## Current Folder Layout

- `csv/flat/`: all `f*.csv` trial logs
- `csv/steps/`: all `s*.csv` trial logs
- `csv/summary/`: `Sheet 1-*.csv` summary tables
- `scripts/`: Python pipeline code
- `old_script/`: archived MATLAB `.m` scripts
- `plots/`: generated plot output target

## Scope

- Covered MATLAB workflows:
  - `old_script/Flat.m` -> `old_script/plotFlat.m`
  - `old_script/Steps.m` -> `old_script/plotSteps.m`
- Not treated as authoritative pipeline:
  - `old_script/locomotion.m` (stale file references)
  - `old_script/plotKinematics.m` (contains undefined variables)

## CSV Inventory

- Trial logs:
  - Files like `f10leg1-1.csv`, `s30w2-2.csv`, etc.
  - Shape: mostly `N x 12` numeric data, with varying `N` by trial.
- Summary sheets:
  - `Sheet 1-leg.csv`, `Sheet 1-2legged.csv`, `Sheet 1-4legged.csv`, `Sheet 1-wheel.csv`
  - Shape: `7 x 4` (small summary tables, not part of the main plotting pipeline).

## Trial CSV Layout

Trial CSVs have 2 header rows, then numeric rows.

Observed header rows:

- Row 1: `,mass_A,,,,mass_C,,,,mass_B,,mass_C`
- Row 2: `t,x,y,vx,vy,x,y,vx,vy,θ,ω,θ`

Column indices used by current MATLAB scripts (1-based):

1. `t` (time)
2. `x` (unused in active scripts)
3. `y_1` (position y source 1)
4. `vx_1`
5. `vy_1` (used in `plotSteps`)
6. `x_2` (unused in active scripts)
7. `y_2` (position y source 2)
8. `vx_2`
9. `vy_2` (used in `plotSteps`)
10. `theta_alt` (not used by active scripts)
11. `omega` (rotation/angular velocity)
12. `theta` (body angle used in active scripts)

Notes:

- `readmatrix(...)` in MATLAB automatically skips the text headers and returns numeric rows.
- Velocity and height are scaled to `mm`/`mm/s` via multiplication by `1000`.
- `vx`/`vy` signs are flipped in scripts (`-column * 1000`).

## Filename Conventions

From the driver scripts:

- Prefix:
  - `f` -> flat-ground runs
  - `s` -> step runs
- Frequency token:
  - `10`, `20`, `30`, `50` (Hz)
- Morphology token:
  - `leg` (baseline leg)
  - `2leg` (2-legged variant, appears as `102`/`302`/`502` style names)
  - `4leg` (4-legged variant, appears as `104`/`304`/`504` style names)
  - `w` (wheel)
- Trial suffix:
  - `1-1`, `2-2`, `3-3`, sometimes `4-4`

## Current MATLAB Processing Pipeline

## 1) Flat-ground pipeline (`Flat.m` + `plotFlat.m`)

Driver behavior (`Flat.m`):

- Builds groups by frequency and morphology.
- Calls `plotFlat(files, trialIdx, RGB, styles, plot_sz, titlePrefix, points, f, ti)`.
- `points` truncates each run to the first `points` rows.
- `ti` defines steady-state start time threshold (`t > ti`).

Per-trial feature extraction (`plotFlat.m`):

- `t = col1`
- `vx_1 = -col4 * 1000`
- `vx_2 = -col8 * 1000`
- `vx = 0.5 * (vx_1 + vx_2)`
- `y_raw = 0.5 * (col3 + col7)`
- `y = (y_raw - y_raw(3) + min(y_raw)) * 1000`
- `theta = col12`
- `omega = col11`

Aggregation/statistics:

- Time-series mean across trials for each signal.
- Steady-state mask: `idx_steady = t > ti`.
- Speed steady-state stats from per-trial steady means:
  - mean (`v_x_steady_mean`)
  - std (`v_x_steady_std`)
- First time mean speed reaches steady mean (`t_reach_avg`).
- For `y`, `theta`, `omega`: global steady-state mean/std using all points from all trials in steady region.

Plot output:

- One figure, 4 stacked subplots:
  1. forward speed
  2. body height
  3. body angle
  4. angular velocity
- Includes individual trial curves, mean curve, and shaded/stat reference bands.

## 2) Step pipeline (`Steps.m` + `plotSteps.m`)

Driver behavior (`Steps.m`):

- Builds groups by frequency and morphology.
- Calls `plotSteps(files, trialIdx, RGB, styles, plot_sz, titlePrefix, f, ti)`.
- In this implementation, `f` and `ti` are passed but not used in the body of `plotSteps`.

Per-trial feature extraction (`plotSteps.m`):

- `t = col1`
- `vx = 0.5 * (-col4*1000 + -col8*1000)`
- `vy = 0.5 * (-col5*1000 + -col9*1000)`
- Speed magnitude: `v = sqrt(vx^2 + vy^2)`
- Height: `y = 0.5 * (col3 + col7) * 1000`
- Body angle: `theta = col12`
- Rotation: `omega = col11`

Time alignment and averaging:

- Each trial time vector is cleaned (`finite`, sorted, deduplicated).
- A common timeline is built using median trial `dt`.
- Signals are linearly interpolated to common time (`interp1(..., 'linear', NaN)`).
- Mean curves are computed with `omitnan`.

Plot output:

- One figure, 4 stacked subplots:
  1. speed magnitude
  2. body height
  3. body angle
  4. rotation
- Shows per-trial curves and a mean curve.

## Sanity-Check Expectations for Python Port

If Python replacements are faithful, they should:

- Read the same trial CSV files and ignore the 2-row text header.
- Use the same column indices and sign/unit conversions.
- Reproduce flat vs step differences:
  - Flat: uses truncated rows (`points`) and steady-state band stats.
  - Steps: resamples/interpolates to a shared time base and averages.
- Generate the same 4-panel figures per experiment group.
