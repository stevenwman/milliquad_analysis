# Exp vs Sim Comparison Plot: Shenanigans & Caveats

## The Plot

`plot_exp_vs_sim.py` produces a 2x2 figure: experimental (left) vs simulation (right) for flat (top) and step (bottom) terrains. Each panel shows forward velocity vs frequency with per-trial scatter dots and std shading bands.

## Failure Modes: Two Categories

Wheel robot has failure modes that need different treatment on exp vs sim sides:

### Exp-only failures (sim doesn't reproduce)
- **Flat WR f50**: Robot physically self-destructs at 50Hz in experiments. Sim happily runs at 720 mm/s because it doesn't model mechanical failure.
- Treatment: exp panel gets shading tapered to 0 + X marker. Sim panel shows actual data (no X).

### Shared failures (both exp and sim)
- **Step WR f10, f20**: Wheel can't move on steps at low frequency. This is a physical limitation the sim reproduces (near-zero velocity).
- Treatment: both panels get shading tapered to 0 + X markers.

## Shading Taper Convention

When a morphology fails at a frequency, we don't just drop the data point — we inject `mean=0, std=0` at that frequency so the `fill_between` shading smoothly tapers from the nearest real data point down to zero. This matches the style in `experimental_data/plots/velocity_vs_freq_flat_clean.png`.

Key detail: injected zeros must be **sorted by frequency** with the real data, otherwise `fill_between` draws crossed/garbled bands. The reference `plot_velocity_vs_freq.py` prepends zeros (which works because failures are always at freq extremes), but `plot_exp_vs_sim.py` appends then sorts.

## Scatter Points at Failures

Scatter dots at failure frequencies are **stripped** (not shown). Only the X marker at y=0 remains. This prevents misleading non-zero dots from experimental noise (e.g., wheel f50 flat had some gentle-actuation trials with nonzero velocity that aren't representative).

## Wheel f50 Flat Sim Data

Not in the original flat validation CSV (config_flat.py only has WR at f10/f20/f30). We ran 5 extra trials with the validate_params protocol (yaw jitter ±2°, BASE_SEED=99999, ref_idx=15) and appended to the CSV. Results: ~719 mm/s mean, COT ~0.41. Video in `results/20260228T013353_rk4_flat/wheel_f50_flat.mp4`.

## Shared Y-Axis

Each row shares y-axis limits between exp and sim panels for fair visual comparison. This means flat sim's WR f50 at ~720 mm/s stretches the y-axis for the entire flat row.

## No Mean Lines

Plots show scatter dots + std shading only. No mean lines connecting the dots — matches the experimental plot style. The shading band implicitly shows where the mean is (center of the band).

## Pitch Comparison (`plot_exp_vs_sim_pitch.py`)

### f50 excluded from flat (same as velocity)
Sim robots tumble at 50Hz (pitch RMS 65-83°), blowing up the y-axis. Experimental flat pitch at f50 is 8-17°. Both exp and sim f50 stripped from the flat pitch comparison.

### Step pitch requires spatial gating
`compute_pitch_rms` originally used only time-based gating (`settle_time`). On step terrain this included the cliff-fall at the end of the staircase, where the robot tumbles off and accumulates huge unwrapped rotation (40-500° RMS). Fix: added `step_start_x`/`step_end_x` spatial gating to `compute_pitch_rms` (same window as `compute_cot` and `extract_velocity`: step_start_x to 90% of step_end_x). After fix: 5-23° RMS, comparable to experimental 3-12°.
