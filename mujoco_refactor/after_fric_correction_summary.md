# Post-Friction-Fix Sweep Summary

Date: 2026-02-18
Fix: `model.geom_friction[ground_id]` → `model.opt.o_friction[:]` in both `simulation.py` and `simulation_fast.py`
Config: CMA-ES, sigma0=0.5, cold start, 3 jitter trials, median aggregation
Evals: 600 per-morphology (×4), 1200 per-frequency (×3), 1200 combined

---

## What Changed

All previous optimization runs had **zero friction control** — the 3 friction search dimensions (`sliding_friction`, `torsional_friction`, `rolling_friction`) were silently ignored due to `mjENBL_OVERRIDE` overriding per-geom friction with global defaults `[1.0, 1.0, 0.005, 0.0001, 0.0001]`. Every prior result was effectively running at MuJoCo's default friction.

After the fix, friction sweeps from 0.001 to 5.0 produce velocity changes from 0.003 to 0.615 m/s — friction is now the most influential parameter in the system.

---

## Per-Morphology Results (one morphology, all freqs)

| Param | scene1 | scene2 | scene4 | scene_wheel | Spread | Verdict |
|---|---|---|---|---|---|---|
| **COST** | 0.0013 | 0.0164 | 0.0005 | 9.2292 | | |
| **n_eval** | 752 | 520 | 792 | 376 | | |
| sliding_friction | 3.09e-1 | 9.15e-1 | 6.15e-1 | 3.56e-3 | 34.4% | MIXED |
| torsional_friction | 2.31e-2 | 2.48e-4 | 2.83e-4 | 8.34e-3 | 28.1% | MIXED |
| rolling_friction | 1.69e-5 | 4.82e-2 | 8.73e-2 | 2.58e-2 | 53.0% | DIVERGE |
| solref_timeconst | 2.53e-3 | 4.06e-3 | 3.74e-4 | 2.30e-2 | 35.8% | DIVERGE |
| solref_dampratio | 6.77 | 2.34 | 5.99 | 2.32 | 15.5% | MIXED |
| solimp_dmin | 0.141 | 0.863 | 0.888 | 0.119 | 77.1% | DIVERGE |
| solimp_delta_d | 0.822 | 0.727 | 0.241 | 0.013 | 82.6% | DIVERGE |
| solimp_width | 3.94e-4 | 2.50e-5 | 4.15e-3 | 3.32e-4 | 31.7% | MIXED |
| solimp_midpoint | 0.738 | 0.329 | 0.114 | 0.414 | 63.7% | DIVERGE |
| solimp_power | 4.85 | 3.26 | 4.23 | 5.87 | 29.0% | MIXED |
| magnetic_moment_fudge | 0.810 | 0.800 | 0.800 | 0.823 | 5.8% | CONVERGE |
| magnetic_field_fudge | 0.898 | 0.901 | 1.199 | 1.135 | 75.4% | DIVERGE |
| dof_damping | 5.32e-10 | 6.47e-14 | 1.42e-9 | 1.35e-12 | 54.3% | DIVERGE |

## Per-Frequency Results (all morphologies, one frequency)

| Param | f10 | f30 | f50 | Spread | Verdict |
|---|---|---|---|---|---|
| **COST** | 0.0004 | 0.0005 | 0.0016 | | |
| **n_eval** | 1168 | 1176 | 1160 | | |
| sliding_friction | 6.57e-1 | 6.39e-1 | 8.56e-1 | 1.8% | CONVERGE |
| torsional_friction | 1.73e-3 | 5.24e-2 | 2.76e-2 | 21.2% | MIXED |
| rolling_friction | 1.61e-2 | 1.12e-3 | 1.28e-1 | 29.4% | MIXED |
| solref_timeconst | 2.02e-3 | 1.86e-3 | 9.63e-4 | 6.5% | CONVERGE |
| solref_dampratio | 0.40 | 2.04 | 4.85 | 36.1% | DIVERGE |
| solimp_dmin | 0.257 | 0.145 | 0.531 | 38.7% | DIVERGE |
| solimp_delta_d | 0.782 | 0.475 | 0.829 | 36.2% | DIVERGE |
| solimp_width | 1.13e-2 | 4.01e-7 | 1.07e-4 | 63.6% | DIVERGE |
| solimp_midpoint | 0.528 | 0.524 | 0.609 | 8.6% | CONVERGE |
| solimp_power | 3.35 | 2.57 | 5.80 | 35.9% | DIVERGE |
| magnetic_moment_fudge | 0.901 | 0.813 | 0.947 | 33.6% | MIXED |
| magnetic_field_fudge | 1.038 | 0.975 | 1.070 | 23.5% | MIXED |
| dof_damping | 3.56e-9 | 4.94e-10 | 7.26e-10 | 10.7% | CONVERGE |

## Cross-Reference Summary

| Param | By Morphology | By Frequency | Interpretation |
|---|---|---|---|
| magnetic_moment_fudge | **CONVERGE (5.8%)** | MIXED (33.6%) | Best candidate for global lock (~0.81); borderline freq dependence |
| sliding_friction | MIXED (34.4%) | **CONVERGE (1.8%)** | Frequency-invariant; morphology spread driven by scene_wheel outlier |
| solref_timeconst | DIVERGE (35.8%) | **CONVERGE (6.5%)** | Frequency-invariant contact timescale; morphology-dependent |
| solimp_midpoint | DIVERGE (63.7%) | **CONVERGE (8.6%)** | Morphology-specific contact shape; stable across frequencies |
| dof_damping | DIVERGE (54.3%) | **CONVERGE (10.7%)** | Frequency-invariant; morphology spread driven by near-zero outliers |
| solref_dampratio | MIXED (15.5%) | DIVERGE (36.1%) | Frequency-dependent contact response; reasonably stable by morphology |
| torsional_friction | MIXED (28.1%) | MIXED (21.2%) | Borderline both ways |
| solimp_width | MIXED (31.7%) | DIVERGE (63.6%) | Frequency-dependent; moderate morphology variation |
| solimp_power | MIXED (29.0%) | DIVERGE (35.9%) | Frequency-dependent; moderate morphology variation |
| rolling_friction | DIVERGE (53.0%) | MIXED (29.4%) | Morphology-dependent; moderate freq variation |
| solimp_dmin | DIVERGE (77.1%) | DIVERGE (38.7%) | Unstable both ways |
| solimp_delta_d | DIVERGE (82.6%) | DIVERGE (36.2%) | Unstable both ways |
| magnetic_field_fudge | DIVERGE (75.4%) | MIXED (23.5%) | Morphology-dependent; scene1/2 want ~0.9, scene4/wheel want ~1.1–1.2 |

## Combined vs Splits

| Metric | Combined | Best per-morph | Best per-freq |
|--------|----------|----------------|---------------|
| Cost | 0.166 | 0.0005 (scene4) | 0.0004 (f10) |
| Cost ratio | 1x | 330x better | 415x better |

The combined fit is ~100–400x worse than the best per-group fits. The optimizer compromises heavily:
- `rolling_friction` = 3.56 (combined) vs 0.001–0.09 (per-morph) — cranked up to balance morphologies
- `solref_timeconst` = 1.7e-5 (combined) — railing at lower bound, can't find a global optimum

---

## Key Findings

1. **`magnetic_moment_fudge` is the only true global constant.** CONVERGE by morphology (5.8%), MIXED by frequency (33.6%). All morphologies agree on ~0.80–0.82. The 19% reduction from measured moment is consistent across very different contact regimes, suggesting a real systematic offset in the moment measurement or a universal torque loss.

2. **Four params converge across frequency** (frequency-invariant physics):
   - `sliding_friction` (1.8%) — all freqs want ~0.65–0.86, far below old default of 1.0
   - `solref_timeconst` (6.5%) — ~0.001–0.002
   - `solimp_midpoint` (8.6%) — ~0.52–0.61
   - `dof_damping` (10.7%) — ~5e-10–3.6e-9

3. **Contact shape params DIVERGE across morphology** — `solimp_dmin` (77%), `solimp_delta_d` (83%), `solimp_midpoint` (64%) all need per-morphology fitting. Different leg counts create fundamentally different contact regimes.

4. **`magnetic_field_fudge` splits by morphology group** (75%): scene1/scene2 want ~0.9, scene4/scene_wheel want ~1.1–1.2. Likely absorbs inter-joint coupling errors that scale with leg count/geometry.

5. **`solref_dampratio` splits by frequency** (36%): 0.4 at 10 Hz → 4.8 at 50 Hz. Contact response timescale interacts with drive frequency — this is real physics, not noise.

6. **`scene_wheel` is unfittable** (cost 9.23 vs 0.0005–0.016 for others). The current model cannot capture wheel locomotion.

7. **Friction now matters a lot.** Sliding friction ~0.65 is far below the old default of 1.0. The robot was simulated with too much friction in all prior runs.

---

## Implications

1. **Per-morphology contact tuning is essential.** A single contact parameter set can't fit scene1 through scene_wheel. The 330x cost gap confirms this.

2. **Lock candidates for combined optimization:** `magnetic_moment_fudge` ≈ 0.81, `sliding_friction` ≈ 0.7, `solref_timeconst` ≈ 0.002, `solimp_midpoint` ≈ 0.55, `dof_damping` ≈ 1e-9. Locking these 5 reduces the search space from 13D to 8D.

3. **scene_wheel needs a different approach.** May need wheel-specific contact geometry or a different friction model.

4. **Frequency-dependent physics is missing.** `solref_dampratio`, `solimp_width`, `solimp_power`, and `solimp_dmin` all DIVERGE by frequency, consistent with the torque model lacking velocity-dependent damping.

---

## Methodology

- Spread = range of log-space positions through search bounds (for log-uniform params; linear for uniform)
- CONVERGE (<15%): shared physics — safe to lock globally
- MIXED (15–35%): borderline, may benefit from per-group tuning
- DIVERGE (>35%): strong candidate for per-group fitting or model improvement
