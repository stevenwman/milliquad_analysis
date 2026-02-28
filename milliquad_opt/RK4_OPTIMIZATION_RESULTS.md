# RK4 Optimization Results

**Date**: 2026-02-28

## Cost Summary

| Terrain | Euler baseline | RK4 Warm | RK4 Cold | Winner |
|---------|---------------|----------|----------|--------|
| Flat | 0.128 (9 refs) | **0.188** (15 refs, 4048/4800) | 0.258 (15 refs, 1936/4800) | Warm |
| Step | 0.210 (10 refs) | 0.613 (12 refs, 3792/4800) | **0.243** (12 refs, 1296/4800) | Cold |
| Rough | 0.395 (7 refs) | **0.362** (7 refs, 1920/4800) | 12.4 (7 refs, 1200/4800) | Warm |

Note: Euler flat baseline had 9 refs (no f20, config export error). RK4 flat has 15 refs (all freqs). Not apples-to-apples.

## Run Details

| Run Dir | Terrain | Start | Evals | Cost | Status |
|---------|---------|-------|-------|------|--------|
| `20260228T013353_rk4_flat` | Flat warm | Euler flat best | 4048/4800 | 0.188 | Converged |
| `20260228T114857_rk4_flat_cold` | Flat cold | None | 1936/4800 | 0.258 | Still improving |
| `20260228T011613_rk4_step` | Step warm | Euler step best | 3792/4800 | 0.613 | Stuck |
| `20260228T093833_rk4_step_cold` | Step cold | None | 1296/4800 | 0.243 | Still improving |
| `20260228T102010_rk4_rough` | Rough warm | Euler rough best | 1920/4800 | 0.362 | Improving |
| `20260228T122805_rk4_rough_cold` | Rough cold | None | 1200/4800 | 12.4 | Failed |

## Warm vs Cold Takeaways

- **Flat**: Warm (0.188) comfortably ahead. Cold (0.258) still improving at 40% done — might close the gap but warm had a head start from good X0.
- **Step**: Cold (0.243) demolished warm (0.613) despite only 1296 evals vs 3792. Euler step warm-start trapped CMA-ES in a bad basin under RK4. Confirms the 16-dim lesson — Euler-optimal params don't transfer to RK4 for step terrain.
- **Rough**: Warm (0.362) already beating Euler baseline (0.395). Cold (12.4) hit the known failure mode — `sliding_friction` collapsed to 0.048, robots can't grip. Classic cold-start stagnation.

## Parameter Comparison (Final Best per Run)

```
param                    flat_W     flat_C     step_W     step_C     rough_W    rough_C
──────────────────────────────────────────────────────────────────────────────────────────
sliding_friction         0.368      0.314      0.565      0.358      0.504      0.048 !!
torsional_friction       1.4e-4     2.7e-5     2.6e-4     1.7e-3     1.1e-4     1.6e-2 !!
rolling_friction         4.3e-6     2.2e-6     1.8e-4     2.7e-5     1.9e-6     1.1e-4
solref_timeconst         0.00136    6.8e-5     0.00247    0.00728    0.00043    1.6e-5
solref_dampratio         6.08       8.68       1.88       2.38       3.03       6.95
solimp_dmin              0.231      0.239      0.720      0.432      0.321      0.668
solimp_delta_d           0.928      0.982      0.404      0.712      0.295      0.691
solimp_width             2.9e-5     7.9e-3     1.4e-5     2.0e-4     3.6e-5     0.121
solimp_midpoint          0.534      0.161      0.600      0.366      0.787      0.602
solimp_power             5.08       6.34       3.84       4.14       5.50       5.91
magnetic_moment_fudge    0.881      0.875      0.691      0.797      1.139 !    1.454 !!
magnetic_field_fudge     1.041      1.048      0.845      0.802      1.028      0.952
dof_damping              3.4e-10    3.8e-10    1.9e-10    9.3e-10    7.3e-10    3.2e-9
noslip_iterations        ~0         ~30        ~32        ~30        ~0         ~28
noslip_tolerance         1.0e-6     2.9e-4     1.5e-4     1.5e-5     1.6e-6     1.0e-4
margin                   1.7e-4     3.2e-4     5.0e-4     2.7e-4     2.9e-6     4.5e-3 !!
```

## Parameter-Level Patterns

### Converging across successful runs (flat_W, flat_C, step_C, rough_W)
- `sliding_friction`: 0.31–0.50 (all in locomotion range)
- `magnetic_moment_fudge`: 0.80–0.88 for flat/step, rough drifts to 1.14
- `dof_damping`: ~3–9 x 10^-10 (tight consensus)

### Noslip split persists under RK4
- Off (~0): flat warm, rough warm
- On (~30): flat cold, step warm, step cold, rough cold
- Same binary divergence as Euler. Flat warm and flat cold disagree on this.

### Rough cold is catastrophic
- `sliding_friction=0.048` — below the stagnation threshold, robots slide without gripping
- `moment_fudge=1.45`, `margin=4.5e-3` — way out of range
- Confirms cold-start rough needs warm-start, same lesson as Euler

### Step warm-start failure
- Inherited Euler's high `solimp_dmin=0.72` and low `dampratio=1.88`
- These contact params don't work under RK4's stiffer integration
- Step cold found `solimp_dmin=0.43`, `dampratio=2.38` — closer to flat consensus

### Euler vs RK4 parameter shifts
- `magnetic_moment_fudge`: Euler flat pinned at 0.655, RK4 flat shifted to 0.88. RK4 wants stronger magnets on flat.
- `solref_dampratio`: Euler flat was 3.82, RK4 flat jumped to 6.08. RK4 wants more damped contacts.
- `solimp_midpoint`: Euler flat was 0.647, RK4 flat dropped to 0.534. Softer impedance transition under RK4.

## Comparison to Euler Trifecta (from PARAM_TRIFECTA_ANALYSIS.md)

```
param                    Euler_flat  RK4_flat_W   Euler_step  RK4_step_C   Euler_rough  RK4_rough_W
────────────────────────────────────────────────────────────────────────────────────────────────────
sliding_friction         0.407       0.368        0.499       0.358        0.626        0.504
magnetic_moment_fudge    0.655       0.881        0.921       0.797        0.896        1.139
magnetic_field_fudge     1.139       1.041        0.733       0.802        1.166        1.028
solref_dampratio         3.82        6.08         1.50        2.38         4.08         3.03
solimp_dmin              0.435       0.231        0.213       0.432        0.329        0.321
solimp_midpoint          0.647       0.534        0.830       0.366        0.877        0.787
noslip_iterations        0           0            31          30           0            0
```

Key Euler-to-RK4 shifts:
- Flat `moment_fudge` jumped 0.655 -> 0.881 (RK4 needs stronger magnetic drive on flat)
- Flat `dampratio` jumped 3.82 -> 6.08 (more contact damping under RK4)
- Step `solimp_dmin` flipped 0.213 -> 0.432 (cold-start found different contact regime)
- Rough `sliding_friction` dropped 0.626 -> 0.504 (less friction needed under RK4)
- Noslip split unchanged: step ~30, flat/rough ~0

---

## Validation Results (Jittered Trials)

Validation of best parameters on unseen jitter seeds (BASE_SEED=99999).
Each ref runs N trials with terrain-appropriate jitter; the top N_SELECT by velocity match are kept.

### Overall

| Terrain | Run | Refs | Trials | Select | Mean Err | Max Err | COT Range |
|---------|-----|------|--------|--------|----------|---------|-----------|
| Flat | `rk4_flat` | 15 | 5 | 3 | 4.4% | 9.7% | 0.21–2.56 |
| Step | `rk4_step_cold` | 10 | 10 | 3 | 15.9% | 58.5% | 1.15–10.45 |
| Rough | `rk4_rough` | 7 | 5 | 3 | 13.5% | 20.3% | 1.04–5.86 |

### Flat (5 trials, top 3, yaw ±2°)

| Ref | Freq | Target (cm/s) | Sim vx (cm/s) | Err (%) | COT |
|-----|------|---------------|---------------|---------|-----|
| scene1_f10 | 10 | 5.1 | 5.4 ± 0.0 | 4.9 ± 0.4 | 0.83 |
| scene1_f20 | 20 | 12.6 | 11.7 ± 0.2 | 7.3 ± 1.8 | 0.85 |
| scene1_f30 | 30 | 11.9 | 12.0 ± 0.2 | 2.2 ± 0.6 | 1.46 |
| scene1_f50 | 50 | 14.8 | 14.7 ± 1.0 | 6.3 ± 3.0 | 2.32 |
| scene2_f10 | 10 | 8.3 | 8.1 ± 0.0 | 2.8 ± 0.3 | 0.75 |
| scene2_f20 | 20 | 11.3 | 12.0 ± 0.1 | 6.2 ± 1.3 | 0.86 |
| scene2_f30 | 30 | 18.0 | 18.1 ± 0.1 | 0.9 ± 0.5 | 0.88 |
| scene2_f50 | 50 | 26.3 | 26.3 ± 0.9 | 2.8 ± 1.7 | 1.49 |
| scene4_f10 | 10 | 11.2 | 10.3 ± 0.1 | 8.1 ± 0.9 | 0.64 |
| scene4_f20 | 20 | 18.4 | 18.0 ± 0.6 | 3.8 ± 1.4 | 0.54 |
| scene4_f30 | 30 | 27.5 | 26.7 ± 0.6 | 2.9 ± 1.9 | 0.77 |
| scene4_f50 | 50 | 32.7 | 33.0 ± 3.3 | 9.7 ± 2.6 | 0.96 |
| wheel_f10 | 10 | 14.3 | 13.6 ± 0.1 | 4.9 ± 0.7 | 0.21 |
| wheel_f20 | 20 | 30.6 | 30.6 ± 0.6 | 1.9 ± 0.7 | 0.23 |
| wheel_f30 | 30 | 44.9 | 44.7 ± 0.7 | 1.5 ± 0.6 | 0.33 |

### Step (10 trials, top 3, yaw ±2°)

Velocity measured only on steps (spatial gating: 50mm to 86mm).

| Ref | Freq | Target (cm/s) | Sim vx (cm/s) | Err (%) | COT |
|-----|------|---------------|---------------|---------|-----|
| scene1_f10 | 10 | 2.0 | 2.1 ± 0.2 | 10.3 ± 6.7 | 3.27 |
| scene1_f20 | 20 | 4.7 | 4.7 ± 0.5 | 9.6 ± 2.7 | 3.59 |
| scene1_f30 | 30 | 3.3 | 3.2 ± 0.2 | 7.9 ± 1.8 | 9.94 |
| scene2_f10 | 10 | 5.4 | 5.5 ± 0.2 | 3.3 ± 2.4 | 1.42 |
| scene2_f20 | 20 | 8.9 | 9.1 ± 0.2 | 2.5 ± 1.5 | 1.93 |
| scene2_f30 | 30 | 13.4 | 14.2 ± 0.5 | 6.1 ± 3.5 | 2.43 |
| scene4_f10 | 10 | 7.2 | 7.2 ± 0.8 | 9.8 ± 3.7 | 1.34 |
| scene4_f20 | 20 | 10.4 | 11.1 ± 0.7 | 7.2 ± 6.0 | 1.84 |
| scene4_f30 | 30 | 9.0 | 14.2 ± 2.9 | **58.5 ± 31.8** | 2.20 |
| wheel_f30 | 30 | 9.4 | 5.3 ± 1.9 | **43.4 ± 20.6** | 7.27 |

Failure mode refs (target = 0, verify robot doesn't move):

| Ref | Result | vx (cm/s) |
|-----|--------|-----------|
| wheel_f10 | PASS | 0.00 |
| wheel_f20 | PASS | 0.07 |

scene4_f30 and wheel_f30 show high variance — step terrain at 30Hz is chaotic and
sensitive to initial yaw. Even with 10 trials, the top 3 include high-error outliers.

### Rough (5 trials, top 3, Y ±3mm)

Fixed terrain seed=42.

| Ref | Freq | Target (cm/s) | Sim vx (cm/s) | Err (%) | COT |
|-----|------|---------------|---------------|---------|-----|
| scene1_f10 | 10 | 4.3 | 3.9 ± 0.2 | 8.8 ± 4.8 | 1.76 |
| scene1_f30 | 30 | 8.2 | 9.8 ± 0.9 | 19.7 ± 10.7 | 3.71 |
| scene2_f10 | 10 | 6.6 | 6.6 ± 0.5 | 6.0 ± 3.5 | 1.42 |
| scene2_f30 | 30 | 12.9 | 14.7 ± 0.8 | 14.4 ± 6.4 | 2.35 |
| scene2_f50 | 50 | 10.6 | 12.8 ± 0.7 | 20.3 ± 6.4 | 5.50 |
| scene4_f10 | 10 | 8.6 | 7.8 ± 0.9 | 11.9 ± 6.8 | 1.27 |
| scene4_f30 | 30 | 14.6 | 12.9 ± 1.6 | 13.0 ± 8.6 | 2.56 |

Exploratory conditions (no optimization target, random trial selection):

| Ref | Freq | Sim vx (cm/s) | COT |
|-----|------|---------------|-----|
| wheel_f10 | 10 | 2.7 ± 1.1 | 3.32 |
| wheel_f30 | 30 | 5.2 ± 3.8 | 12.27 |
| scene1_f50 | 50 | 5.6 ± 3.2 | 22.95 |
| scene4_f50 | 50 | 17.7 ± 9.6 | 5.93 |
| wheel_f50 | 50 | 1.8 ± 0.6 | 50.51 |

### Validation Methodology

- **Seeds**: BASE_SEED=99999 (differs from optimizer's 12345/77777) — tests generalization
- **Selection**: Run N trials with jitter, rank by |vx - target|, keep top N_SELECT
- **Flat/Step jitter**: Uniform yaw perturbation in [-2°, +2°]
- **Rough jitter**: Uniform Y-offset in [-3mm, +3mm]
- **Velocity**: Forward-only (vx). Step terrain spatially gated to step region
- **COT**: W_ext / (m·g·d), naive power formula (validated under RK4, see INTEGRATOR_ENERGY_VERIFICATION.md)
- **Failure mode**: Single trial, no jitter, verify |vx| < 0.5 cm/s
