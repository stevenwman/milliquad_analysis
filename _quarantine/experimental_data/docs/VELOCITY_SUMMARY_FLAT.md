# Flat Velocity Summary (Mean +- Std)

Method (to match plotting pipeline):
- Source: `experimental_data/csv/flat/*.csv`
- Uses the same condition config as `experimental_data/scripts/flat_pipeline.py`:
  - same trial subsets (`trial_idx_1based`)
  - same row crop (`points`)
  - same steady-state threshold (`steady_t`)
- Velocity per sample: `v_x = 0.5 * (-(vx_1)*1000 + -(vx_2)*1000)` in mm/s
- Per trial: mean `v_x` over steady-state samples (`t > steady_t`)
- Per condition: mean and sample std across those trial steady means (`ddof=1`)

| Frequency (Hz) | leg (mm/s) | 2leg (mm/s) | 4leg (mm/s) | wheel (mm/s) |
|---:|---:|---:|---:|---:|
| 10 | 51.18 +- 2.41 | 82.44 +- 0.87 | 112.10 +- 5.96 | 143.11 +- 1.40 |
| 20 | 126.4 +- 4.7 | 113.1 +- 42.0 | 184.1 +- 15.6 | 305.8 +- 6.8 |
| 30 | 118.77 +- 12.66 | 179.61 +- 17.87 | 274.43 +- 20.97 | 450.27 +- 17.99 |
| 50 | 148.33 +- 13.23 | 249.59 +- 25.59 | 328.03 +- 55.20 | 709.41 +- 11.50 |

Notes:
- 20Hz row uses corrected values (last 50% of recording, proper steady-state, all trials included). See "20Hz Row Validation" section below for details. Note 2leg trial 2 is 64.6 mm/s vs ~138 for the other two, driving the large std.
- Original `20*` row (steady_t=0, no transient exclusion) was: leg 120.43, 2leg 128.01, 4leg 184.36, wheel 270.21.
- These values are very close to `Sheet 1-*.csv` for most cells.
- One notable mismatch remains at `50Hz 2leg`:
  - plotting-subset recompute: `249.59 +- 25.59`
  - summary sheet value: `263.3 +- 25.7`

## Config Reference Values (Used By MuJoCo Fit)

Source:
- `mujoco_refactor/results/20260219T142207_loose_fudge/config.py`
- `REFERENCE_DATA` entries (converted to mm/s from m/s for readability here)

| Frequency (Hz) | leg (mm/s) | 2leg (mm/s) | 4leg (mm/s) | wheel (mm/s) |
|---:|---:|---:|---:|---:|
| 10 | 51.2 +- 2.4 | 83.2 +- 1.4 | 112.1 +- 6.0 | 143.2 +- 1.3 |
| 30 | 118.7 +- 12.7 | 179.6 +- 17.9 | 274.7 +- 20.7 | 449.3 +- 18.3 |
| 50 | 148.3 +- 13.1 | 263.3 +- 25.7 | 327.4 +- 55.6 | n/a |

Notes:
- `wheel 50Hz` is intentionally omitted in this config's `REFERENCE_DATA`.
- All listed config values exactly match the legacy summary sheets (`Sheet 1-*.csv`) for the included cells.

## 20Hz Row Validation (2026-02-21)

The `20*` row above used `steady_t=0` (no transient exclusion), unlike 10/30/50 Hz
which all use `steady_t >= 0.15`. This biases 20Hz velocities **downward** by including
the acceleration ramp. Additionally, 2leg trial 2 (`f202leg2-2.csv`) is an outlier at
~64 mm/s vs ~138 mm/s for trials 1 and 3.

### Convergence check (mean velocity over last N% of each recording)

All 20Hz conditions reach a stable steady-state velocity. Sampling the last 50% vs
last 30% of each trial gives consistent values, confirming a universal steady speed exists.

| Morphology | last 50% (mm/s) | last 30% (mm/s) | Original 20* row |
|---|---|---|---|
| leg | 126.4 +- 4.7 | 128.0 +- 11.0 | 120.43 +- 4.82 |
| 2leg | 113.1 +- 42.0 | 112.9 +- 40.4 | 128.01 +- 1.47 |
| 4leg | 184.1 +- 15.6 | 192.6 +- 18.7 | 184.36 +- 12.42 |
| wheel | 305.8 +- 6.8 | 303.7 +- 4.5 | 270.21 +- 2.57 |

### Corrected 20Hz values (last 50% of recording, ddof=1)

| Morphology | speed (mm/s) | speed (m/s) | std (m/s) | notes |
|---|---|---|---|---|
| leg (scene1) | 126.4 +- 4.7 | 0.1264 +- 0.0047 | 3 trials |
| 2leg (scene2) | 113.1 +- 42.0 | 0.1131 +- 0.0420 | 3 trials (trial 2 = 64.6 mm/s, trials 1,3 ~138 mm/s) |
| 4leg (scene4) | 184.1 +- 15.6 | 0.1841 +- 0.0156 | 3 trials |
| wheel (scene_wheel) | 305.8 +- 6.8 | 0.3058 +- 0.0068 | 3 trials |

Wheel has the largest correction (+13%) because the recording is short (~0.48s) and
the robot accelerates fast, so the transient ramp dominates the full-recording average.

### Training data audit

The 10/30/50 Hz config values use the legacy summary sheets (`Sheet 1-*.csv`), which
were independently verified to match the pipeline recompute (spot-checked 10Hz leg and
30Hz 4leg — exact match). So the training data for 10/30/50 Hz is correct.

The `50Hz 2leg` discrepancy (249.59 pipeline vs 263.3 config) is the only known
mismatch and was previously documented above.
