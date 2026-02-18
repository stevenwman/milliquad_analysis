# Per-Morphology & Per-Frequency Parameter Analysis

## Run 2: Tight Fudge Bounds (latest)

Date: 2026-02-17
Config: cold start, sigma=0.5, 1200 evals each, median jitter (3 trials)
Bounds: moment_fudge [0.8, 1.2] uniform, field_fudge [0.8, 1.2] uniform

### Per-Morphology Results (one morphology, all freqs)

| Param | scene1 | scene2 | scene4 | scene_wheel | Spread | Verdict |
|---|---|---|---|---|---|---|
| **COST** | 0.0017 | 0.0092 | 0.0030 | 0.0000 | | |
| **n_eval** | 752 | 1064 | 744 | 1176 | | |
| sliding_friction | 3.54e-2 | 1.70e-3 | 4.45e-3 | 9.92e-4 | 22.2% | MIXED |
| torsional_friction | 1.53e-1 | 2.07e-3 | 1.46e-3 | 4.47e-2 | 28.9% | MIXED |
| rolling_friction | 3.63e-4 | 2.56e-6 | 5.62e-2 | 2.93e-3 | 62.0% | DIVERGE |
| solref_timeconst | 4.47e-5 | 3.70e-3 | 4.68e-4 | 1.40e-2 | 49.9% | DIVERGE |
| solref_dampratio | 9.78 | 0.62 | 1.28 | 0.27 | 52.0% | DIVERGE |
| solimp_dmin | 0.803 | 0.828 | 0.924 | 0.066 | 85.9% | DIVERGE |
| solimp_delta_d | 0.663 | 0.351 | 0.496 | 0.277 | 39.3% | DIVERGE |
| solimp_width | 1.34e-6 | 1.57e-4 | 3.31e-4 | 4.59e-3 | 50.5% | DIVERGE |
| solimp_midpoint | 0.593 | 0.016 | 0.944 | 0.077 | 94.7% | DIVERGE |
| solimp_power | 5.41 | 5.07 | 6.22 | 3.83 | 26.5% | MIXED |
| magnetic_moment_fudge | 1.055 | 0.820 | 1.058 | 0.864 | 59.6% | DIVERGE |
| magnetic_field_fudge | 0.920 | 1.199 | 0.925 | 1.185 | 69.9% | DIVERGE |
| dof_damping | 6.61e-10 | 8.84e-10 | 4.26e-10 | 7.82e-10 | 4.0% | CONVERGE |

### Per-Frequency Results (all morphologies, one frequency)

| Param | f10 | f30 | f50 | Spread | Verdict |
|---|---|---|---|---|---|
| **COST** | 0.0001 | 0.0011 | 0.0033 | | |
| **n_eval** | 952 | 1080 | 1128 | | |
| sliding_friction | 4.26e-3 | 2.09e-4 | 1.99e-3 | 18.7% | MIXED |
| torsional_friction | 5.61e-5 | 7.15e-2 | 6.70e-4 | 44.4% | DIVERGE |
| rolling_friction | 3.38e-3 | 2.51e-6 | 1.10e-4 | 44.7% | DIVERGE |
| solref_timeconst | 1.61e-3 | 3.52e-4 | 1.37e-5 | 41.4% | DIVERGE |
| solref_dampratio | 0.49 | 6.22 | 3.37 | 36.7% | DIVERGE |
| solimp_dmin | 0.641 | 0.749 | 0.655 | 10.8% | CONVERGE |
| solimp_delta_d | 0.378 | 0.668 | 0.428 | 29.6% | MIXED |
| solimp_width | 1.23e-4 | 5.40e-4 | 1.01e-4 | 10.4% | CONVERGE |
| solimp_midpoint | 0.806 | 0.323 | 0.214 | 60.5% | DIVERGE |
| solimp_power | 6.64 | 7.55 | 4.78 | 30.7% | MIXED |
| magnetic_moment_fudge | 1.012 | 0.830 | 1.014 | 45.8% | DIVERGE |
| magnetic_field_fudge | 1.001 | 0.921 | 1.023 | 25.6% | MIXED |
| dof_damping | 2.64e-9 | 4.17e-10 | 8.49e-10 | 10.0% | CONVERGE |

### Cross-Reference Summary

| Param | By Morphology | By Frequency | Interpretation |
|---|---|---|---|
| dof_damping | **CONVERGE (4%)** | **CONVERGE (10%)** | True physics constant - lock globally |
| solimp_dmin | DIVERGE (86%) | **CONVERGE (11%)** | Morphology-dependent contact |
| solimp_width | DIVERGE (51%) | **CONVERGE (10%)** | Morphology-dependent contact |
| solimp_power | MIXED (27%) | MIXED (31%) | Borderline both ways |
| solimp_delta_d | DIVERGE (39%) | MIXED (30%) | Morphology-leaning |
| sliding_friction | MIXED (22%) | MIXED (19%) | Reasonably stable both ways |
| rolling_friction | DIVERGE (62%) | DIVERGE (45%) | Unstable both ways |
| torsional_friction | MIXED (29%) | DIVERGE (44%) | Frequency-leaning |
| solref_timeconst | DIVERGE (50%) | DIVERGE (41%) | Unstable both ways (was CONVERGE with wide fudges) |
| solref_dampratio | DIVERGE (52%) | DIVERGE (37%) | Unstable both ways |
| solimp_midpoint | DIVERGE (95%) | DIVERGE (61%) | Unstable both ways |
| magnetic_moment_fudge | DIVERGE (60%) | **DIVERGE (46%)** | Diverges BOTH ways - model deficiency |
| magnetic_field_fudge | DIVERGE (70%) | MIXED (26%) | Morphology-dependent; less freq-sensitive |

### Key Findings (Tight Fudges)

1. **Only `dof_damping` converges both ways** (4% morph, 10% freq). Safe to lock globally at ~5e-10 to 9e-10.

2. **Magnetic fudges now DIVERGE by morphology too** (60%, 70%). With wide fudge bounds [0.5, 2.0] they appeared stable across morphology (MIXED 23-33%) because they had room to absorb contact errors. Constraining to [0.8, 1.2] forces the contact/solver params to absorb those errors instead, revealing the true coupling.

3. **`moment_fudge` still diverges by frequency** (46%). Pattern: f30 wants ~0.83, while f10 and f50 want ~1.01. This confirms the torque model has a frequency-dependent error that fudge factors alone cannot fix.

4. **`solimp_dmin` and `solimp_width` converge across frequency** (11%, 10%) but diverge across morphology (86%, 51%). These are genuinely morphology-specific contact params -- different leg counts/geometries create different contact regimes.

5. **`solref_timeconst` destabilized**: was CONVERGE (11%) by frequency with wide fudges, now DIVERGE (41%). The tight fudge bounds removed slack that was hiding frequency-dependent compensation.

6. **Cost gap remains huge**: per-group costs 0.0001-0.009 vs combined cost 0.48+. The model can match any single group well, but there is no single parameter set that works across all morphologies and frequencies simultaneously.

### Implications

- The fudge factors are not just scaling the magnetic moment/field -- they are compensating for missing physics (frequency-dependent effects, morphology-dependent coupling).
- Tightening fudges to physically reasonable ranges (+-20%) exposes that the underlying torque model needs improvement, not just calibration.
- Candidate missing physics: eddy currents in substrate, velocity-dependent magnetic drag, field non-uniformity that matters differently for different leg counts, inertial coupling at high frequency.

---

## Run 1: Wide Fudge Bounds (previous)

Date: 2026-02-17
Config: cold start, sigma=0.5, 600 evals each, median jitter (3 trials)
Bounds: moment_fudge [0.5, 2.0], field_fudge [0.8, 1.2]

### Per-Morphology Results (one morphology, all freqs)

| Param | scene1 | scene2 | scene4 | scene_wheel | Spread | Verdict |
|---|---|---|---|---|---|---|
| **COST** | 0.0077 | 0.0045 | 0.0061 | 0.0001 | | |
| **n_eval** | 600 | 560 | 272 | 496 | | |
| sliding_friction | 1.21e-4 | 2.87e-4 | 2.31e-3 | 1.75e-4 | 18.3% | MIXED |
| torsional_friction | 3.55e-2 | 1.58e-2 | 1.15e-2 | 1.06e-3 | 21.8% | MIXED |
| rolling_friction | 7.91e-5 | 3.24e-1 | 6.65e-2 | 3.35e-2 | 51.6% | DIVERGE |
| solref_timeconst | 3.00e-4 | 2.33e-3 | 1.26e-3 | 2.77e-3 | 19.3% | MIXED |
| solref_dampratio | 3.04 | 3.85 | 1.36 | 0.22 | 41.7% | DIVERGE |
| solimp_dmin | 0.770 | 0.423 | 0.561 | 0.001 | 77.1% | DIVERGE |
| solimp_delta_d | 0.653 | 0.713 | 0.421 | 0.549 | 29.8% | MIXED |
| solimp_width | 1.13e-6 | 2.39e-5 | 2.09e-5 | 1.57e-3 | 44.9% | DIVERGE |
| solimp_midpoint | 0.557 | 0.921 | 0.317 | 0.694 | 61.7% | DIVERGE |
| solimp_power | 5.83 | 4.45 | 5.96 | 4.83 | 16.8% | MIXED |
| magnetic_moment_fudge | 0.587 | 0.511 | 0.500 | 0.794 | 33.4% | MIXED |
| magnetic_field_fudge | 1.194 | 1.200 | 1.164 | 1.095 | 22.6% | MIXED |
| dof_damping | 4.50e-10 | 2.27e-13 | 1.44e-11 | 4.84e-11 | 41.2% | DIVERGE |

### Per-Frequency Results (all morphologies, one frequency)

| Param | f10 | f30 | f50 | Spread | Verdict |
|---|---|---|---|---|---|
| **COST** | 0.0002 | 0.0038 | 0.0041 | | |
| **n_eval** | 576 | 600 | 392 | | |
| sliding_friction | 1.55e-4 | 1.75e-3 | 3.36e-4 | 15.0% | MIXED |
| torsional_friction | 6.05e-4 | 1.41e-4 | 3.24e-2 | 33.8% | MIXED |
| rolling_friction | 2.84e-3 | 6.87e-5 | 6.16e-3 | 27.9% | MIXED |
| solref_timeconst | 2.33e-4 | 8.07e-4 | 2.17e-4 | 11.4% | CONVERGE |
| solref_dampratio | 0.76 | 6.13 | 9.53 | 36.6% | DIVERGE |
| solimp_dmin | 0.655 | 0.619 | 0.838 | 21.9% | MIXED |
| solimp_delta_d | 0.978 | 0.794 | 0.528 | 45.9% | DIVERGE |
| solimp_width | 2.75e-5 | 2.84e-3 | 4.54e-4 | 28.8% | MIXED |
| solimp_midpoint | 0.166 | 0.274 | 0.645 | 48.8% | DIVERGE |
| solimp_power | 5.26 | 5.71 | 5.75 | 5.5% | CONVERGE |
| magnetic_moment_fudge | 0.895 | 0.568 | 1.215 | 54.8% | DIVERGE |
| magnetic_field_fudge | 0.865 | 0.992 | 0.848 | 38.7% | DIVERGE |
| dof_damping | 2.41e-9 | 1.03e-11 | 8.23e-10 | 29.6% | MIXED |

---

## Methodology

- Spread = range of log-space positions through search bounds (for log-uniform params; linear for uniform)
- CONVERGE (<15%): safe to lock globally
- MIXED (15-35%): borderline, may benefit from per-group tuning
- DIVERGE (>35%): needs per-group fitting or model improvement
