# 16-Dim Search Space Reference

All dimensions defined in `mujoco_refactor/config_new.py`, shared by both flat and step optimizers.

## Dimensions

| # | Name | Range | Scale | Description |
|---|------|-------|-------|-------------|
| 0 | sliding_friction | [0.01, 2.0] | log | Ground sliding friction |
| 1 | torsional_friction | [1e-6, 10.0] | log | Ground torsional friction |
| 2 | rolling_friction | [1e-6, 1e-3] | log | Ground rolling friction |
| 3 | solref_timeconst | [1e-5, 1.0] | log | Contact solver time constant |
| 4 | solref_dampratio | [1.0, 10.0] | log | Contact solver damping ratio |
| 5 | solimp_dmin | [0.001, 0.999] | uniform | Contact impedance min distance |
| 6 | solimp_delta_d | [0.01, 0.99] | uniform | dmax = dmin + delta_d * (0.9999 - dmin) |
| 7 | solimp_width | [1e-7, 1] | log | Contact impedance width |
| 8 | solimp_midpoint | [0.01, 0.99] | uniform | Contact impedance midpoint |
| 9 | solimp_power | [2.0, 7.0] | uniform | Contact impedance power |
| 10 | magnetic_moment_fudge | [0.5, 1.5] | uniform | Scales magnetic moment |
| 11 | magnetic_field_fudge | [0.5, 1.5] | uniform | Scales magnetic field |
| 12 | dof_damping | [1e-10, 1e-8] | log | Joint damping |
| 13 | noslip_iterations | [0, 60] | uniform (int) | Friction solver iterations (>60 unstable) |
| 14 | noslip_tolerance | [1e-6, 1e-3] | log | Friction convergence threshold |
| 15 | margin | [0.0, 0.005] | uniform | Contact detection distance |

## Parameter Convergence (from per-morphology sweeps)

**Converge across all conditions:**
- sliding_friction (~0.49, CoV=12%)
- magnetic_moment_fudge (~0.80 flat / ~0.92 step, CoV=8%)
- dof_damping (~5e-10, CoV=9%)

**Converge by frequency only:**
- solref_dampratio (CoV=2.4%)
- solimp_power (CoV=14%)

**Diverge (leave bounds wide):**
- solimp_dmin, solimp_width, solimp_midpoint, solref_timeconst

## Best Param Comparison (flat vs step)

| Param | Flat Best | Step Best | Agreement |
|-------|-----------|-----------|-----------|
| sliding_friction | 0.487 | 0.499 | Good |
| magnetic_moment_fudge | 0.653 | 0.921 | DIVERGENT |
| dof_damping | 5.0e-10 | ~5e-10 | Good |
| noslip_iterations | 0 | 31 | DIVERGENT |
| margin | 0.004 | 5e-5 | DIVERGENT |
