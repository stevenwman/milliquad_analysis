# Friction Sensitivity Analysis — L2 on Rough Terrain

## Goal

Measure how each friction parameter independently affects **success rate** and **average forward velocity** for the 2-leg robot (scene2) on rough terrain, using the best rough-optimized params as baseline.

## Baseline

**Params**: `results/20260303T224229_rough_tg/optimization_bests.csv` (final row, cost=0.264)

| Parameter | Baseline value | Search space |
|-----------|---------------|--------------|
| `sliding_friction` | 0.640 | [0.01, 2.0] log |
| `torsional_friction` | 2.35e-5 | [1e-6, 10.0] log |
| `rolling_friction` | 1.48e-6 | [1e-6, 1e-3] log |

**Conditions**: scene2 × {f10, f30, f50} on rough terrain (3 frequencies).

**Experimental targets** (from REFERENCE_DATA):
- scene2_f10: 65.6 mm/s
- scene2_f30: 128.9 mm/s
- scene2_f50: 106.2 mm/s

## Sweep Design

For each friction parameter, sweep independently while holding the other two (and all 13 non-friction params) at baseline values.

### Sweep points (log-spaced, centered on baseline)

5 points per parameter, log-spaced across the search space bounds:

Concretely: `np.geomspace(lo, hi, 5)` where `lo` and `hi` are chosen to span ~2 decades below and above baseline, clamped to search space bounds.

- `sliding_friction`: `geomspace(0.01, 2.0, 5)` → [0.01, 0.06, 0.28, 1.19, 2.0]
- `torsional_friction`: `geomspace(1e-6, 1.0, 5)` → [1e-6, 3.2e-5, 1e-3, 3.2e-2, 1.0]
- `rolling_friction`: `geomspace(1e-6, 1e-3, 5)` → [1e-6, 5.6e-6, 3.2e-5, 1.8e-4, 1e-3]

### Trials per sweep point

- **N = 10** jittered trials (Y-offset ±3mm, seed=77777 base, same as optimizer)
- Total sims: 3 params × 5 values × 3 freqs × 10 trials = **450 simulations**

## Metrics (per sweep point, per frequency)

1. **Success rate**: fraction of 10 trials where robot traverses the rough section (spatial gate: `max(pos_x) >= 0.155m`). Gate-exempt: none (scene2 is not scene1_f10).
2. **Mean forward velocity** (mm/s): average `vx` over successful trials only, spatially gated (5mm to 155mm).
3. **Velocity std**: trial-to-trial variability of successful trials.

## Output

### Per-trial CSV: `friction_sweep.csv`

Columns: `param_name, param_value, ctrl_freq, trial, rng_seed, jitter_y, crash, success, max_x, vx_m_s`

Every row has enough info to reconstruct a single trial (seed + jitter_y + param overrides).

### Summary CSV: `friction_sweep_summary.csv`

Columns: `param_name, param_value, ctrl_freq, n_trials, n_success, success_rate, mean_vx_mm_s, std_vx_mm_s`

### Trajectories NPZ: `trajectories.npz`

Per non-crash trial, keyed as `{param_name}__{value:.6e}__f{freq}_t{trial}_{field}`:
- `_time`: (N,) timestamps
- `_pos`: (N, 3) body position xyz
- `_vel`: (N, 3) body velocity xyz
- `_quat`: (N, 4) chassis quaternion (w,x,y,z)

Modular: more fields (omega, joint_pos, contact) can be added to `_run_one()` by
pulling from the trajectory dict and appending to `traj_arrays`.

### Overview plots (one per friction parameter): `overview_{param_name}.png`

Grid: rows = sweep values, cols = frequencies. Each cell shows x-pos (mm) vs time
for all trials. Dashed lines = failed trials. Gate lines shown. Baseline row marked
with asterisk. Title shows `[n_success/n_total]`.

## Implementation

### Script: `friction_sensitivity.py`

```bash
uv run python friction_sensitivity.py [--n-trials 10] [--n-values 5] [--workers 16]
```

1. Load best params from rough_tg `optimization_bests.csv` (last row)
2. For each friction param: build 5-point log sweep (`geomspace`)
3. For each sweep value × freq × trial: run `simulation.run_simulation()` with modified point
4. Spatial gating: success = `max(pos_x) >= 0.155m`, velocity = displacement/time over [5mm, 155mm]
5. Save per-trial CSV + summary CSV + NPZ trajectories + overview plots

### Parallelism

`multiprocessing.Pool` (spawn). ~450 sims at ~2s each ≈ 15 minutes on 16 cores.

### Reconstructing a trial

To rerun or record video for any row in `friction_sweep.csv`:
1. Load baseline point, override `param_name` with `param_value`
2. Call `run_simulation()` with `rng_seed`, `spawn_offset=(SPAWN_X, jitter_y, SPAWN_Z_RAISE)`, `record_path=...`

### No optimizer involvement

Pure evaluation script — no CMA-ES, no cost function. Just run sims and record raw metrics.
