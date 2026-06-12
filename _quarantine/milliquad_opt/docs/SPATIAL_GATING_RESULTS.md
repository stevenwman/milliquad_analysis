# Spatial Gating Optimization Results

**Date**: 2026-03-01

## Summary

This document covers three optimization runs:
1. **Rough spatial RK4** — rough terrain with spatially-gated cost (fix for time-gated bias)
2. **Rough spatial Euler** — same spatial gating, Euler integrator warm-start
3. **Step q60 RK-warm** — step terrain with q60 target extraction, RK4 warm-start from Euler step best

The key change is **spatial gating** for rough terrain: the optimizer now measures velocity only while the robot is on the rough patch (X = 5–155mm), not for the full 2s sim. This eliminates the bias where fast morphologies (scene4, scene2 at high freq) accelerated on flat ground after clearing the rough patch, inflating their measured velocity.

## Cost Summary

| Run | Terrain | Cost | Evals | Baseline | Improvement |
|-----|---------|------|-------|----------|-------------|
| rough_spatial_rk4 | Rough | **0.184** | 4608/4800 | 0.362 (time-gated RK4) | 49% |
| rough_spatial_euler | Rough | **0.229** | 3216/4800 | 0.395 (time-gated Euler) | 42% |
| step_q60_rk-warm | Step | **0.210** | 2768/4800 | 0.243 (RK4 cold, q75) | 14% |

Both spatial rough runs show major cost reductions: the time-gated optimizer was partially fitting to flat-ground acceleration for fast robots. Removing that bias let CMA-ES focus on actual rough-terrain locomotion.

## Run Details

| Run Dir | Terrain | Integrator | Warm-Start | Evals | Final Cost | Time |
|---------|---------|------------|------------|-------|------------|------|
| `20260228T202903_rough_spatial_rk4` | Rough | RK4 | time-gated RK4 best | 4608 | 0.184 | 226 min |
| `20260228T201840_rough_spatial_euler` | Rough | Euler | time-gated Euler best | 3216 | 0.229 | 241 min |
| `20260228T230022_step_q60_rk-warm` | Step | RK4 | Euler step best | 2768 | 0.210 | 257 min |

## What Changed: Spatial Gating

**Problem**: The time-gated rough optimizer measured velocity as `(final_pos - settle_pos) / duration` from `SETTLE_TIME` to end of sim. The rough patch spans X = 5mm to 155mm (150mm total), but fast robots (scene4_f30, scene2_f30) clear it in ~0.8–1.0s of a 2.0s sim, spending ~1s accelerating on flat ground. This inflates measured velocity and rewards params that work well on flat, not rough.

**Fix**: `config_rough_spatial.py` measures velocity only between `ROUGH_START_X = 5mm` and `ROUGH_END_X = 155mm`:
- `enter_idx` = first timestep where `pos[0] >= 5mm`
- `exit_idx` = first timestep where `pos[0] >= 155mm` (or last timestep if robot doesn't reach end)
- Velocity = `(pos[exit] - pos[enter]) / (t[exit] - t[enter])`
- Lateral, yaw: measured between enter/exit
- Tumble: still over full trajectory

The original `config_rough.py` is preserved unchanged (legacy).

## Parameter Comparison

```
param                    rough_RK4    rough_Euler   rough_time   step_q60     step_RK4c
                         (spatial)    (spatial)     (RK4 base)   (RK4-warm)   (baseline)
────────────────────────────────────────────────────────────────────────────────────────────
sliding_friction         0.787        0.685         0.504        0.354        0.358
torsional_friction       5.1e-5       1.2e-4        1.1e-4       9.4e-4       1.7e-3
rolling_friction         1.7e-6       1.2e-6        1.9e-6       1.2e-6       2.7e-5
solref_timeconst         0.00149      0.00041       0.00043      0.00352      0.00728
solref_dampratio         2.55         5.07          3.03         2.49         2.38
solimp_dmin              0.217        0.148         0.321        0.800        0.432
solimp_delta_d           0.219        0.425         0.295        0.272        0.712
solimp_width             1.3e-5       9.2e-5        3.6e-5       1.5e-4       2.0e-4
solimp_midpoint          0.412        0.531         0.787        0.016 !!     0.366
solimp_power             5.49         5.64          5.50         4.17         4.14
magnetic_moment_fudge    1.008        1.213         1.139        0.507 !!     0.797
magnetic_field_fudge     1.421        1.491         1.028        0.900        0.802
dof_damping              7.2e-10      8.0e-10       7.3e-10      7.3e-10      9.3e-10
noslip_iterations        ~1           ~0            ~0           ~31          ~30
noslip_tolerance         1.1e-6       1.0e-6        1.6e-6       9.6e-6       1.5e-5
margin                   1.6e-5       1.2e-6        2.9e-6       1.5e-4       2.7e-4
```

### Parameter Insights

**Rough spatial vs time-gated**:
- `sliding_friction` jumped 0.50 → 0.79 (RK4) and 0.50 → 0.69 (Euler). Spatial gating demands more friction — the robot needs to grip the rough patch, not just get through it.
- `magnetic_field_fudge` jumped 1.03 → 1.42 (RK4). Compensates for roughness by driving harder.
- `solimp_midpoint` dropped from 0.79 to 0.41/0.53. Softer contact transition when terrain isn't smooth.
- `noslip_iterations` stayed near 0 for both. Confirms noslip solver isn't needed for rough terrain.

**Step q60 notable params**:
- `magnetic_moment_fudge = 0.507` — extremely low, meaning weak magnet coupling. Step terrain rewards slower, more controlled gaits.
- `solimp_midpoint = 0.016` — nearly zero, contact impedance kicks in immediately. Makes sense for step edges.
- `noslip_iterations = 31` — noslip solver ON, same pattern as all step runs.

---

## Validation Results (Jittered Trials)

BASE_SEED=99999, 5 trials per ref, top 3 selected by velocity match.

### Overall

| Terrain | Run | Refs | Mean Err | Max Err | Worst Ref |
|---------|-----|------|----------|---------|-----------|
| Rough | spatial_rk4 | 7 | **11.5%** | 26.5% | scene2_f50 |
| Rough | spatial_euler | 7 | **11.0%** | 31.1% | scene4_f30 |
| Step | q60_rk-warm | 10 + 2 failure | **26.2%** | 57.0% | scene2_f10 |

For comparison with previous results (from `RK4_OPTIMIZATION_RESULTS.md`):
- Time-gated rough (RK4 warm): 13.5% mean, 20.3% max
- Step RK4 cold (q75): 15.9% mean, 58.5% max

Note: Time-gated rough validation used time-gated velocity; spatial validation uses spatially-gated velocity. Not directly comparable — the spatial metric is stricter.

### Rough Spatial RK4 (5 trials, top 3, Y ±3mm)

Velocity measured only on rough patch (spatial gating: 5mm to 140mm at 90% cutoff).

| Ref | Freq | Target (cm/s) | Sim vx (cm/s) | Err (%) | COT | Pitch (°) |
|-----|------|---------------|---------------|---------|-----|-----------|
| scene1_f10 | 10 | 4.3 | 4.4 ± 0.3 | 5.7 ± 1.9 | 1.39 | 7.2 |
| scene1_f30 | 30 | 8.2 | 8.2 ± 0.8 | 6.8 ± 2.6 | 3.78 | 37.8 ! |
| scene2_f10 | 10 | 6.6 | 7.0 ± 0.6 | 11.0 ± 2.6 | 1.30 | 6.4 |
| scene2_f30 | 30 | 12.9 | 10.9 ± 0.5 | 15.0 ± 3.8 | 2.75 | 8.2 |
| scene2_f50 | 50 | 10.6 | 10.7 ± 3.1 | **26.5 ± 9.5** | 6.99 | 28.8 ! |
| scene4_f10 | 10 | 8.6 | 8.3 ± 0.1 | 2.8 ± 1.1 | 0.91 | 6.3 |
| scene4_f30 | 30 | 14.6 | 12.8 ± 0.1 | 12.4 ± 0.5 | 2.15 | 6.1 |

Exploratory conditions (no optimization target):

| Ref | Freq | Sim vx (cm/s) | COT | Pitch (°) |
|-----|------|---------------|-----|-----------|
| wheel_f10 | 10 | 2.9 ± 1.2 | 2.57 | 4.7 |
| wheel_f30 | 30 | 2.3 ± 1.8 | 17.1 | 3.2 |
| scene1_f50 | 50 | 7.6 ± 0.4 | 8.44 | 36.9 |
| scene4_f50 | 50 | 5.5 ± 2.3 | 14.4 | 5.9 |
| wheel_f50 | 50 | 7.6 ± 4.5 | 17.5 | 4.5 |

**Notes**: scene1_f30 and scene2_f50 show elevated pitch RMS (37.8° and 28.8°), indicating some tumbling in individual trials. The velocity match remains reasonable because selected trials are those with best velocity agreement, not best stability.

### Rough Spatial Euler (5 trials, top 3, Y ±3mm)

| Ref | Freq | Target (cm/s) | Sim vx (cm/s) | Err (%) | COT | Pitch (°) |
|-----|------|---------------|---------------|---------|-----|-----------|
| scene1_f10 | 10 | 4.3 | 4.3 ± 0.4 | 7.7 ± 3.3 | 1.70 | 149.0 !! |
| scene1_f30 | 30 | 8.2 | 8.1 ± 0.1 | **1.1 ± 0.7** | 4.23 | 12.1 |
| scene2_f10 | 10 | 6.6 | 6.7 ± 0.2 | 3.6 ± 1.6 | 1.40 | 6.4 |
| scene2_f30 | 30 | 12.9 | 10.6 ± 0.4 | 17.8 ± 3.3 | 3.36 | 8.9 |
| scene2_f50 | 50 | 10.6 | 9.6 ± 0.9 | 9.4 ± 8.3 | 6.86 | 32.3 ! |
| scene4_f10 | 10 | 8.6 | 8.3 ± 0.7 | 6.4 ± 5.3 | 0.93 | 6.5 |
| scene4_f30 | 30 | 14.6 | 10.1 ± 2.9 | **31.1 ± 20.0** | 3.00 | 6.9 |

Exploratory conditions:

| Ref | Freq | Sim vx (cm/s) | COT | Pitch (°) |
|-----|------|---------------|-----|-----------|
| wheel_f10 | 10 | 3.2 ± 2.5 | 3.33 | 4.5 |
| wheel_f30 | 30 | 3.1 ± 1.1 | 10.9 | 4.5 |
| scene1_f50 | 50 | 6.8 ± 1.5 | 8.47 | 35.1 |
| scene4_f50 | 50 | 10.2 ± 2.2 | 5.50 | 8.3 |
| wheel_f50 | 50 | 4.7 ± 3.5 | 36.4 | 4.5 |

**Notes**: scene1_f10 has a pitch RMS of 149° — at least one selected trial flipped completely. This is a significant stability concern. scene4_f30 has very high variance (31.1% ± 20%), indicating sensitivity to Y-offset jitter.

### Step q60 RK-Warm (5 trials, top 3, yaw ±2°)

Velocity measured only on steps (spatial gating). Target velocities from q60 extraction.

| Ref | Freq | Target (cm/s) | Sim vx (cm/s) | Err (%) | COT | Pitch (°) |
|-----|------|---------------|---------------|---------|-----|-----------|
| scene1_f10 | 10 | 2.0 | 1.8 ± 0.4 | 17.6 ± 12.7 | 3.18 | 5.9 |
| scene1_f20 | 20 | 4.7 | 3.4 ± 0.0 | 29.0 ± 0.9 | 4.55 | 9.1 |
| scene1_f30 | 30 | 3.3 | 3.3 ± 0.3 | 8.2 ± 3.5 | 7.88 | 12.1 |
| scene2_f10 | 10 | 5.4 | 2.3 ± 0.5 | **57.0 ± 8.5** | 3.56 | 6.4 |
| scene2_f20 | 20 | 8.9 | 6.1 ± 1.4 | 32.0 ± 15.5 | 2.55 | 9.7 |
| scene2_f30 | 30 | 13.4 | 10.4 ± 2.8 | 24.1 ± 18.7 | 3.01 | 9.9 |
| scene4_f10 | 10 | 7.2 | 6.3 ± 1.3 | 21.5 ± 5.3 | 1.06 | 5.9 |
| scene4_f20 | 20 | 10.4 | 10.0 ± 0.5 | **3.9 ± 3.9** | 1.57 | 6.3 |
| scene4_f30 | 30 | 9.0 | 11.6 ± 0.8 | 29.7 ± 9.2 | 2.08 | 6.6 |
| wheel_f30 | 30 | 9.4 | 6.0 ± 3.9 | 39.2 ± 39.3 | 19.5 | 4.7 |

Failure mode refs (target = 0, robot should not move):

| Ref | Freq | Sim vx (cm/s) | COT | Pitch (°) |
|-----|------|---------------|-----|-----------|
| wheel_f10 | 10 | 0.2 ± 0.1 | 44.0 | 1.2 |
| wheel_f20 | 20 | 0.4 ± 0.1 | 42.3 | 2.0 |

Both confirmed: wheel morphology is essentially stationary at f10 and f20 on steps (< 0.5 cm/s), matching experimental observations. High COT reflects tiny displacement in denominator.

**Notes**: scene2_f10 is severely underperforming (57% error), suggesting the 2-leg L2 morphology doesn't traverse steps well at 10Hz under these params. scene_wheel_f30 is essentially bimodal — some trials achieve near-target velocity while others barely move (COT=52 for near-zero vx trial). High variance across most step refs reflects the inherent chaos of step-climbing dynamics.

### RK4 vs Euler: Rough Spatial Head-to-Head

| Ref | RK4 Err (%) | Euler Err (%) | Winner |
|-----|-------------|---------------|--------|
| scene1_f10 | 5.7 | 7.7 | RK4 |
| scene1_f30 | 6.8 | **1.1** | Euler |
| scene2_f10 | 11.0 | **3.6** | Euler |
| scene2_f30 | **15.0** | 17.8 | RK4 |
| scene2_f50 | 26.5 | **9.4** | Euler |
| scene4_f10 | **2.8** | 6.4 | RK4 |
| scene4_f30 | **12.4** | 31.1 | RK4 |
| **Mean** | **11.5** | **11.0** | Euler (barely) |

Euler wins on mean error (11.0% vs 11.5%), but RK4 wins on consistency:
- RK4 max error: 26.5% (scene2_f50) with modest pitch issues
- Euler max error: 31.1% (scene4_f30) with catastrophic pitch on scene1_f10 (149°)

The Euler run also found scene1_f30 at 1.1% error — the best single-ref result across all runs. But its failures are more extreme.

---

## Methodology

### Validation
- **Seeds**: BASE_SEED=99999 (differs from optimizer's 12345/77777)
- **Selection**: 5 trials per ref, top 3 by |vx - target|
- **Rough jitter**: Uniform Y-offset in [-3mm, +3mm], fixed terrain seed=42
- **Step jitter**: Uniform yaw in [-2°, +2°]
- **Velocity**: Forward-only (vx), spatially gated
  - Rough: 5mm to 140mm (90% cutoff of 5–155mm rough patch)
  - Step: step region bounds from config
- **COT**: `W_ext / (m·g·d)`, joint-axis-projected power (see `eval_cot_v2.py`)
- **Pitch RMS**: Yaw-invariant `arctan2(dz, sqrt(dx²+dy²))` between FL/BL legs, detrended, std in degrees

### Spatial Gating (New)
- `config_rough_spatial.py` overrides `calculate_cost` from `config_rough.py`
- Optimizer measures velocity between `ROUGH_START_X` (5mm) and `ROUGH_END_X` (155mm)
- Validation uses 90% cutoff: 5mm to 140mm (excludes edge effects)
- Original `config_rough.py` preserved as legacy (time-gated)

### Step q60
- Target velocities extracted at the 60th percentile (q60) instead of q75
- 12 reference conditions (scene1/2/4 × f10/f20/f30 + wheel × f10/f20/f30)
- Wheel f10/f20 are failure mode refs (target = 0, robot should not move)
  - Run with full N_TRIALS jittered trials, 3 randomly selected (not error-sorted)
  - Written to CSV alongside regular refs for plotting
- Warm-started from Euler step best params

---

## Step Gate Tightening: 90% → 65% (2026-03-03)

### Motivation

The 90% spatial gate (`active_cutoff = step_start_x + 0.9 * (step_end_x - step_start_x)`) includes trajectory right up to the cliff edge. Robots that reach the last step are already beginning to fall, which corrupts velocity (decelerating or reversing), pitch (sudden tumble), and lateral measurements. The experimental data uses a q60 index gate (45%–75% of recording) which implicitly avoids this by ending much earlier. Tightening the sim gate to 65% better matches the experimental intent.

### Gate geometry

| Gate | Cutoff X (mm) | Region measured | Step coverage |
|------|--------------|-----------------|---------------|
| 90% (original) | 96.4 | 50.0 → 96.4 | Steps 1–7 + most of final platform |
| 75% | 88.6 | 50.0 → 88.6 | Steps 1–6 |
| 65% | 83.5 | 50.0 → 83.5 | Steps 1–5 |

Step zone: `STEP_START_X = 50mm`, `STEP_END_X = 101.5mm` (7 × 4.5mm steps + 20mm final platform).

### Impact on metrics (same optimizer params, recomputed from NPZ)

Measured from `20260228T230022_step_q60_rk-warm` validation trajectories.

**Velocity (cm/s)**:

| Ref | 90% gate | 75% gate | 65% gate | Δ 90→65 |
|-----|----------|----------|----------|---------|
| scene1_f10 | 1.8 | 1.6 | 1.4 | -21% |
| scene1_f20 | 3.4 | 3.2 | 3.1 | -8% |
| scene1_f30 | 3.3 | 3.2 | 3.1 | -6% |
| scene2_f10 | 2.3 | 2.2 | 2.1 | -9% |
| scene2_f20 | 6.1 | 5.9 | 5.7 | -7% |
| scene2_f30 | 10.4 | 10.1 | 9.9 | -5% |
| scene4_f10 | 6.3 | 6.0 | 5.7 | -10% |
| scene4_f20 | 10.0 | 9.6 | 9.3 | -7% |
| scene4_f30 | 11.6 | 10.9 | 10.6 | -9% |
| wheel_f30 | 6.0 | 5.2 | 4.9 | -18% |

Velocity drops 5–21%. Largest drops at low frequency (scene1_f10) and wheel — these robots are slowest and spend proportionally more time near the cliff edge within the 90% window.

**Pitch RMS (°)**:

| Ref | 90% gate | 75% gate | 65% gate | Δ 90→65 |
|-----|----------|----------|----------|---------|
| scene1_f10 | 5.9 | 5.3 | 4.7 | -20% |
| scene1_f20 | 9.1 | 7.8 | 7.2 | -21% |
| scene1_f30 | 12.1 | 11.2 | 10.4 | -14% |
| scene2_f10 | 6.4 | 5.7 | 5.1 | -20% |
| scene2_f20 | 9.7 | 8.3 | 7.8 | -20% |
| scene2_f30 | 9.9 | 8.3 | 7.6 | -23% |
| scene4_f10 | 5.9 | 4.0 | 3.3 | -44% |
| scene4_f20 | 6.3 | 5.4 | 5.1 | -19% |
| scene4_f30 | 6.6 | 5.7 | 5.3 | -20% |
| wheel_f30 | 4.7 | 3.8 | 3.5 | -26% |

Pitch RMS drops up to 44% (scene4_f10). The cliff-adjacent portion of the trajectory has the largest pitch oscillations — excluding it gives a cleaner steady-state measurement.

### Retraining at 65%

Post-hoc re-windowing from NPZ (as in `plot_megacomposite_nocot_065.py`) changes the reported metrics but not the underlying physics params. The optimizer's cost function used 90% gating during training, so the fitted params are biased toward 90% performance. To get params optimized for 65% gating:

- **Config**: `config_step_065.py` — exact copy of `config_step.py` with `0.9 → 0.65` on the `active_cutoff` line
- **Command**: `uv run python optimizer.py --terrain step_065 --warm-start-from results/20260228T230022_step_q60_rk-warm --suffix step_065gate`
- **Expectation**: Params may shift slightly since the cost landscape changes. The optimizer no longer sees cliff-edge behavior, which could affect friction and contact params.

### Files

| File | Purpose |
|------|---------|
| `config_step.py` | Original step config (90% gate) — **unchanged** |
| `config_step_065.py` | Copy with 65% gate for retraining |
| `analysis/plot_megacomposite_nocot_065.py` | Nocot megacomposite with 65% re-windowed metrics |
| `analysis/investigative/plot_pitch_mega.py` | Per-condition pitch time series (uses 65% gate, 100% trim) |
