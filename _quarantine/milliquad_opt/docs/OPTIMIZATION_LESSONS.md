# Optimization Lessons Learned

Hard-won insights from CMA-ES parameter identification across flat, step, and rough terrains.

---

## Cost Function Design

### Composition & weights

| Component | Weight | Typical share | Notes |
|-----------|--------|---------------|-------|
| Velocity error | 5 | ~92% | Dominates; quadratic relative error |
| Lateral drift | 5 | ~2% | Small because robots mostly go straight |
| Tumble penalty | 1 | <1% | Per-step normalized; threshold varies by terrain |
| Yaw penalty | 0 (rough/step) | 0% | Disabled on terrain — cliff-fall artifact corrupts yaw |
| Velocity variance | 2 | ~6% | Raw relative error (no deadzone) |
| Pitch RMS | 0 | 0% | Disabled everywhere (pitch_weight=0.0) |

### Velocity deadzone

Early configs used a deadzone: no velocity penalty if error < 1σ of experimental data. This helped avoid over-penalizing noise but made the cost landscape flat near good solutions, slowing convergence. Current approach: `VELOCITY_DEADZONE = False` — plain quadratic cost everywhere.

### Jitter trial aggregation

Each reference condition runs multiple jitter trials (different Y-offsets, collision seeds). Aggregation strategy matters:

- **Flat terrain**: MEDIAN of trials. Tolerates 1 chaotic outlier without corrupting the cost.
- **Step terrain**: BEST (argmin cost) of trials. Cliff-fall artifacts can corrupt 1-2 trials with huge yaw/tumble; taking the best avoids penalizing good physics params for bad luck.
- **Rough terrain**: MEDIAN of 3 trials. Fixed terrain seed; Y-jitter ±3mm provides variety.

---

## Search Space

### 13-dimensional base space

| Parameter | Range | Notes |
|-----------|-------|-------|
| sliding_friction | [0.01, 2.0] | Log-scale; most impactful single param |
| torsional_friction | [1e-5, 0.01] | Log-scale |
| rolling_friction | [1e-5, 0.01] | Log-scale |
| magnetic_moment_fudge | [0.75, 0.90] | Consistently pins at ~0.80 across all runs |
| dof_damping | [1e-10, 1e-8] | Log-scale; converges across morphologies |
| solref_timeconst | [1e-4, 0.02] | Diverges across morphologies (not well-constrained) |
| solref_dampratio | [1.0, 10.0] | Converges across frequencies |
| solimp_dmin | [0.8, 0.9999] | Diverges |
| solimp_delta_d | [0.001, 0.5] | Reparameterized: `dmax = dmin + delta_d * (0.9999 - dmin)` → guarantees dmax > dmin |
| solimp_width | [1e-4, 0.1] | Diverges |
| solimp_midpoint | [0.1, 0.9] | Diverges |
| solimp_power | [2.0, 7.0] | Converges across frequencies |
| field_fudge | [0.5, 1.5] | External field strength multiplier |

### `solimp_delta_d` reparameterization

MuJoCo requires `solimp[1] > solimp[0]` (dmax > dmin). Direct optimization of both can violate this constraint. Fix: optimize `delta_d ∈ [0.001, 0.5]` and compute `dmax = dmin + delta_d * (0.9999 - dmin)`. This guarantees the constraint and makes the search space rectangular.

### Parameter convergence across morphologies (per-morphology sweep, 2026-02-18)

Ran independent CMA-ES per scene (scene1/2/4/wheel) and per frequency (f10/f30/f50). Compared best params:

- **CONVERGE (same across morphologies & freqs)**: `sliding_friction` (12% CV), `magnetic_moment_fudge` (8%), `dof_damping` (9%)
- **CONVERGE freq-only**: `solref_dampratio` (2.4%), `solimp_power` (14%)
- **DIVERGE (different optimal per morphology)**: `solimp_dmin`, `solimp_width`, `solimp_midpoint`, `solref_timeconst`

Implication: divergent params are under-constrained by the cost function. They compensate for each other — many combinations give similar cost. Tightening bounds on divergent params doesn't help; they need to be wide.

### 16-dimensional extension (2026-02-23)

Added 3 MuJoCo solver params:

| Parameter | Range | MuJoCo default | Notes |
|-----------|-------|----------------|-------|
| noslip_iterations | [0, 60] (int) | 0 | Friction solver iterations; >60 causes instability |
| noslip_tolerance | [1e-6, 1e-3] | 1e-6 | Friction convergence threshold |
| margin | [0.0, 0.005] | 0.0 | Contact detection distance → `model.opt.o_margin` |

Attempted but don't exist in MuJoCo API: `o_solreffriction`, `o_solimpfriction`.

All 3 verified to affect trajectories (not dead params). 16-dim needs ~5× more CMA-ES evals than 13-dim.

---

## CMA-ES Warm-Start Strategy

### Cold-start stagnation

CMA-ES with default initialization (space midpoints) can fail catastrophically:

- `sliding_friction` log-midpoint maps to ~0.05, where robots barely move
- CMA-ES sigma collapses before finding the locomotion regime (friction ~0.3–1.0)
- Scene2 and f30 both failed this way — zero velocity throughout optimization

**Fix**: Always warm-start from known-good params. Use `--warm-start-from <dir>` to load `optimization_bests.csv` from a previous run.

### 16-dim warm-start: defaults vs midpoints (CRITICAL)

When expanding the search space with new parameters:

| Strategy | New param init | Result |
|----------|----------------|--------|
| **WRONG** | Space midpoints (noslip=30, tol≈3e-5, margin=0.0025) | cost=0.445 (17% worse than 13-dim baseline) |
| **CORRECT** | MuJoCo defaults (noslip=0, tol=1e-6, margin=0.0) | cost=0.377 (0.8% better than baseline) |

**Key insight**: New params at midpoints push the optimizer away from the 13-dim optimum. Start at defaults = the simulation already worked fine without these params, so "no change" is the correct starting point.

Correct run: sigma=0.15 (tight local refinement), N_CALLS=4800, converged at eval 1976. 11/15 reference conditions within 1σ of 13-dim baseline.

### Single-terrain overfitting

Params optimized on flat terrain fail on step terrain (92.8% mean velocity error). Params optimized on step terrain over-tune contact dynamics for steps. Multi-terrain optimization is necessary but harder (combined cost ~0.7 vs single-terrain ~0.2).

---

## Terrain-Specific Optimizer Design

### Flat (`config.py` + `optimizer.py`)

- 11 reference conditions: scene1/2/4 × f10/f30/f50 + scene_wheel × f10/f30
- 2 jitter trials per ref, MEDIAN aggregation
- Settle time before measuring velocity
- Best cost: ~0.188 (rk4_flat)

### Step (`config_step.py` + `optimizer_step.py`)

- 10 refs: scene1/2/4 × f10/f20/f30 + scene_wheel × f30
- Targets: experimental vx at q75 index ± 150 timesteps (updated to q60 windowing)
- Step geometry: 8 steps, 1mm high, 4.5mm long, 50mm flat lead, 20mm final platform
- Step XMLs generated at startup alongside originals (correct relative path resolution)
- **Spatial gating**: velocity measured only for `pos[0] >= STEP_START_X` (50mm)
- Yaw weight = 0 (cliff-fall artifact)
- BEST-trial aggregation (not median — cliff-fall corrupts individual trials)
- Best cost: ~0.210 (step_q60_rk-warm)

### Rough (`config_rough.py` + `optimizer_rough.py`)

- 7 refs: scene1/2/4 × f10/f30 + scene2 × f50 (no wheel — 40% experimental success rate too unreliable)
- Fixed terrain seed=42, Y-jitter ±3mm, MEDIAN aggregation, 3 trials/ref
- Tumble threshold = 0.17 rad (~10°), weight = 2.0 (harsher than flat)
- spawn_offset = (FLAT_LEAD + x_half, y_offset, 10mm) — start on flat lead before rough section
- Spatial gating: velocity measured for `pos[0] >= gate_start`
- Best cost: ~0.184 (rough_spatial_rk4)

### Multi-terrain (`config_multi_terrain.py`)

Combined flat + step optimization. 19 refs total (selected via correlation analysis to remove redundant f20 conditions):

- Flat: 11 refs (scene1/2/4 × f10/f30/f50 + scene_wheel f10/f30)
- Step: 8 refs (scene1/2/4 × f10/f30 + scene_wheel f20/f30)
- Hierarchical cost: terrain-level weighted sum, each terrain uses its own aggregation (flat=median, step=best)
- scene_wheel f20 step: target velocity = 0.0 m/s (experimental failure mode), weight=2.0
- Budget: N_CALLS=4800 (2× single-terrain)
- Reference IDs: terrain suffix (e.g., `scene1_f10_flat`, `scene1_f10_step`)

---

## Multiprocessing Gotchas

- `mjcf_path` must be passed in the task tuple, not via global state. `spawn` forkmethod re-imports modules and empties global dicts.
- Each worker allocates its own MuJoCo arena. 20 workers × 32MB = 640MB.
- Video recording: rank 1 only (avoid file contention).
