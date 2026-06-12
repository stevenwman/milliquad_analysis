# H3/H4: Velocity Ripple + Phase-Propulsion Plots

H3 shows the **outcome** (how smooth is forward velocity?).
H4 shows the **mechanism** (why, via angle lag and torque distribution).

## Quantities

### H3 — Outcome: "How smooth is forward velocity?"

- **Phase-folded velocity**: `vel_x` binned by `drive_angle mod 2pi` into ~36 bins. Averaged across cycles within each trial, then mean +/- std across trials. Shows velocity "shape" within one revolution.
- **Ripple coefficient**: intra-revolution `std(vel_x) / mean(vel_x)`, already computed in `velocity_ripple.py`. Summarizes smoothness as one number per (morphology, terrain, freq).
- **FFT harmonics**: amplitude at 1x, 2x, 4x drive frequency. L1 should peak at 1x, L4 at 4x.

### H4 — Mechanism: "Why is it smooth/rough?"

- **Angle lag per leg**: `(joint_pos[i] - drive_angle) mod 2pi`. The torque-generating angle. Max torque at pi/2 lag, zero at 0 or pi. Shows how each leg tracks (or lags behind) the field.
- **Phase-folded net torque**: sum of `|tau_ext|` (or forward component) across all legs, binned by `drive_angle`. Shows whether total propulsive torque is peaky or smooth.
- **Phase R-squared**: `var(bin_means) / var(all_ax)`, already computed in `phase_propulsion.py`. Summary metric: 0 = no phase structure, 1 = fully phase-locked.

## Data Available (no simulation re-run needed)

All from existing NPZ files per terrain:
- `vel_x`, `pos_x`, `time`, `pitch` -- velocity/position
- `drive_angle` -- external field phase
- `joint_pos` -- per-leg hinge angles (4 values)
- `tau_ext` -- per-leg external torques (4 values)

## Shared Utilities (already exist)

- `analysis/l2_l4/_trial_filter.py`: `is_valid_trial`, `active_mask`, `parse_key`, `find_npz`, `detect_terrain`
- `analysis/l2_l4/_plot_style.py`: `MORPH_COLORS`, `MORPH_LABELS`, `MORPH_ORDER`, `TERRAIN_TITLES`
- Existing text-only scripts with reference math: `analysis/l2_l4/velocity_ripple.py`, `analysis/l2_l4/phase_propulsion.py`

## File Structure

```
analysis/l2_l4/H3/
  __init__.py
  PLAN.md                  # this file
  plot_phase_folded.py     # 3a+4a: phase-folded vel + torque
  plot_fft_spectrum.py     # 3b: FFT harmonic spectrum
  plot_ripple_summary.py   # 3c: ripple coeff + phase R^2 bars
  plot_angle_lag.py        # 4b: angle lag distribution
  figures/
    archive/
```

## Plot Specs (4 scripts)

### Plot 3a+4a: Phase-folded velocity + torque (`plot_phase_folded.py`)

- **Layout**: 2-row figure. Top = phase-folded `vel_x` (mm/s), bottom = phase-folded net `|tau_ext|` sum (Nm). Columns = frequencies. One terrain per invocation.
- **Computation**: Bin `drive_angle mod 2pi` into 36 bins (10 deg). For each trial, average `vel_x` and `sum(|tau_ext|)` per bin across all cycles. Then mean +/- std band across trials.
- **Key visual**: Morphologies overlaid as colored lines with shaded std bands. L1 should show deep single dip in both, L4 should be flatter.
- **CLI**: same pattern as H1 -- `run_dirs` positional, `--no-save`, single terrain.

### Plot 3b: FFT harmonic spectrum (`plot_fft_spectrum.py`)

- **Layout**: Grid -- rows = terrain, cols = freq. Each panel shows FFT amplitude (mm/s) on y-axis vs normalized frequency (multiples of drive freq: 0.5x, 1x, 2x, 3x, 4x, 5x, 8x) on x-axis.
- **Computation**: For each valid trial, subtract mean from `vel_x`, take `rfft`, normalize amplitude by N. Extract amplitudes at harmonics (nearest bin to k * drive_freq). Average across trials per morphology.
- **Key visual**: L1 peaks at 1x. L4 energy shifted to 4x. WR should be flat/minimal. This is the "harmonic redistribution" evidence.
- **CLI**: multi-terrain (all run_dirs), `--no-save`.

### Plot 3c: Ripple + Phase R^2 summary (`plot_ripple_summary.py`)

- **Layout**: Bar chart grid -- rows = terrain, cols = freq. Grouped bars by morphology.
- **Computation**: Ripple coefficient (lines 107-117 of `velocity_ripple.py`: `std within revolution / mean vx`). Phase R^2 (lines 137-148 of `phase_propulsion.py`: `var(bin_means) / var(all_ax)`).
- **Dual-axis**: Left y = ripple coefficient (bars), right y = phase R^2 (diamond markers) -- or two separate rows. Keep it simple.
- **CLI**: multi-terrain, `--no-save`.

### Plot 4b: Angle lag distribution (`plot_angle_lag.py`)

- **Layout**: One subplot per morphology (1x4 grid). Each shows histogram/polar plot of `(joint_pos[leg] - drive_angle) mod 2pi` for all 4 legs as overlaid distributions.
- **Computation**: For each trial, compute angle lag per leg per timestep within active mask. Pool across trials for the histogram.
- **Key visual**: L1's single leg clusters at one lag. L4's 4 legs should span ~90 deg apart (pi/2 phase offset). Shows the "phase diversity" mechanism.
- **CLI**: single terrain + freq, diagnostic/exploratory.

## Conventions (follows H1 exactly)

- `FIGURE_DIR = Path(__file__).parent / "figures"`
- Timestamp prefix: `%Y%m%dT%H%M%S_<plotname>_<terrain>.png`
- 200 DPI, `bbox_inches="tight"`
- `plt.show()` after save
- docstring with usage example
- Imports: `from analysis.l2_l4._plot_style import ...` and `from analysis.l2_l4._trial_filter import ...`
- Refer to `analysis/l2_l4/H1/plot_contact_histogram.py` as the canonical CLI/save pattern

## Run Dirs (for CLI args)

```
results/20260228T013353_rk4_flat
results/20260228T230022_step_q60_rk-warm
results/20260228T202903_rough_spatial_rk4
```

## TODOs

1. `plot_phase_folded.py` -- phase-folded velocity (top) + net torque (bottom), per terrain
2. `plot_fft_spectrum.py` -- FFT harmonic spectrum showing energy at 1x, 2x, 4x drive freq
3. `plot_ripple_summary.py` -- ripple coefficient + phase R-squared bar chart grid
4. `plot_angle_lag.py` -- angle lag distribution per morphology
5. Update `L2_L4_ANALYSIS.md` with H3/H4 progress and findings
