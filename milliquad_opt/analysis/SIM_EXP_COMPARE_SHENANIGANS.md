# Exp vs Sim Comparison Plot: Shenanigans & Caveats

## The Plot

`plot_exp_vs_sim_composite.py` produces a 2×4 figure: rows = flat, step; col-pairs = [velocity exp|sim, pitch exp|sim]. Each panel shows per-trial scatter dots and std shading bands.

Legacy standalone scripts (`plot_exp_vs_sim.py`, `plot_exp_vs_sim_pitch.py`) still exist but are superseded by the composite.

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

Now in `config_flat.py` REFERENCE_DATA as a failure mode (`speed=0, weight=0`). Runs natively through `validate_params` — no manual CSV patching needed. Results: ~719 mm/s mean, COT ~0.41, pitch_rms ~0.18° (stable, no tumbling).

## Shared Y-Axis

Each row shares y-axis limits between exp and sim panels for fair visual comparison. This means flat sim's WR f50 at ~720 mm/s stretches the y-axis for the entire flat row.

## No Mean Lines

Plots show scatter dots + std shading only. No mean lines connecting the dots — matches the experimental plot style. The shading band implicitly shows where the mean is (center of the band).

## Pitch Comparison (`plot_exp_vs_sim_pitch.py`)

### Yaw-invariant pitch computation
The original `compute_pitch_rms` measured pitch in the world XZ plane (`arctan2(dz, dx)` between FL and BL legs). When a robot yawed 180°, the FL-BL x-distance flipped sign, registering as ~180° "pitch" even though the robot never tilted. This caused false "inverted" detections at f50 (scene1: 65-83°, scene4: 83° — actually just yaw rotations, confirmed by video).

Fix: `compute_pitch_series` now uses `arctan2(dz, sqrt(dx²+dy²))` — horizontal distance is always positive, so yaw rotations don't affect pitch. Result bounded to [-90, +90]°. After fix, all f50 trials show reasonable pitch (scene1: 12°, scene4: 3°). `PITCH_EXCLUDE` is empty — no frequency exclusions needed.

### Step pitch requires spatial gating
`compute_pitch_rms` originally used only time-based gating (`settle_time`). On step terrain this included the cliff-fall at the end of the staircase, where the robot tumbles off and accumulates huge unwrapped rotation (40-500° RMS). Fix: added `step_start_x`/`step_end_x` spatial gating to `compute_pitch_rms` (same window as `compute_cot` and `extract_velocity`: step_start_x to 90% of step_end_x). After fix: 5-23° RMS, comparable to experimental 3-12°.

## Experimental Step Data Windowing (q60 vs q75)

Experimental step terrain data has two windowing methods. Both only apply to **experimental** data — simulation uses spatial gating (`step_start_x` to 90% of `step_end_x`) which is terrain-absolute and more robust.

### Original windowing
- **Velocity**: q75 ± 150 fixed samples — center at 75% of recording, ±150 samples window. Matches `config_step.py` targets.
- **Pitch**: 50%–90% of recording — skip first half (transient) and last 10% (cliff-fall).

### q60 windowing (student's method)
- 30% window centered at 60% of recording: indices 45%–75%.
- Trajectory-relative — adapts to different trial lengths/speeds (unlike q75±150 which is absolute sample count).
- Functions: `extract_step_q60()` in `plot_velocity_vs_freq.py`, `extract_step_pitch_q60()` in `plot_pitch_vs_freq.py`.
- Old functions preserved alongside new ones (`extract_step()` / `extract_step_pitch()` unchanged).

### q60 vs q75 velocity impact on optimization targets

Current `config_step.py` REFERENCE_DATA was built from q75 (matches to <0.5%). Switching to q60 shifts most targets:

| Scene | Freq | q75 (mm/s) | q60 (mm/s) | Shift |
|-------|------|-----------|-----------|-------|
| scene1 | 10 | 19.9 | 16.9 | -15% |
| scene1 | 20 | 47.3 | 46.2 | -2% |
| scene1 | 30 | 33.1 | 27.9 | -16% |
| scene2 | 10 | 54.2 | 41.5 | -23% |
| scene2 | 20 | 89.5 | 83.5 | -7% |
| scene2 | 30 | 134.1 | 109.8 | -18% |
| scene4 | 10 | 71.6 | 76.9 | +7% |
| scene4 | 20 | 104.2 | 96.3 | -8% |
| scene4 | 30 | 90.0 | 76.4 | -15% |
| wheel | 30 | 94.0 | 97.2 | +3% |

- Most conditions drop 10–23% (q60 window is earlier → robot still accelerating on steps).
- Two outliers go UP: scene4 f10 (+7%) and wheel f30 (+3%) — different trajectory velocity profiles.
- q60 stds are often higher (less stable window), especially scene2.
- Pitch RMS barely changes between windowing methods (±5–15%, no systematic bias).
- **q60 adopted**: `config_step_q60.py` has updated targets. The `step_q60_rk-warm` run was optimized against q60 targets. `plot_exp_vs_sim_composite.py` uses `extract_step_q60()` / `extract_step_pitch_q60()` for experimental step panels. Original `config_step.py` still has q75 targets (legacy).

**CRITICAL**: When comparing exp vs sim for step terrain, the experimental extraction windowing MUST match the config targets the sim was trained on. If the sim used q60 targets, the exp panel must use `extract_step_q60`, not `extract_step`. Mismatch introduces 10-23% systematic bias.

---

## Trial Validity Filtering

### Gate-clearing criterion (rough/step)

Trials on rough and step terrain are filtered by whether the robot reached the end of the terrain section (`max_x >= gate_end`). This replaces the old `min_window_vx` velocity threshold, which had false negatives (slow but completing robots like scene1_f10) and false positives (fast starts that stall mid-course).

- **Rough**: `gate_end = 0.155m` (ROUGH_END_X from `config_rough_spatial.py`)
- **Step**: `gate_end = 0.1015m` (STEP_END_X from `config_step.py`)
- **Flat**: no gate — all trials pass the position check

**Gate exemption**: `GATE_EXEMPT = {("scene1", 10.0)}` — scene1_f10 (1-leg @ 10Hz) successfully traverses rough terrain but moves too slowly to reach 155mm. All 5 trials top out at 101–122mm. Visually confirmed as valid locomotion, not stuck. Exempt from gate check on all terrains.

The `max_x` column is computed natively by `validate_params.py` (`max(pos_x)` per trial) and written to all CSVs (flat included, though flat has no gate).

Implementation: `_is_valid_trial(r, gate_end)` in `plot_validation.py`. `GATE_END` dict maps terrain → threshold. `GATE_EXEMPT` set of `(scene, freq)` tuples bypass gate check.

### Inverted pitch check (all terrains)

Trials with `pitch_rms > 30°` are excluded regardless of terrain. This catches robots that flip over.

### "All failed" X markers

When ALL selected trials for a (scene, freq) combo are invalid (didn't clear gate OR inverted), a scene-colored X is placed at y=0 on plots. This distinguishes "no data because all trials failed" from "data point not shown because it's off-screen".

### Flat terrain: no filtering needed in practice

Flat trials all pass both checks (no gate, no inversions). WR f50 pitch_rms is ~0.18° (stable).

---

## Script Inventory

| Script | Purpose | Status |
|--------|---------|--------|
| `plot_validation.py` | Sim-only: velocity/COT/pitch per terrain + 3×3 composite | **Primary** |
| `plot_exp_vs_sim_composite.py` | Exp vs sim: 2×4 (flat+step × vel+pitch × exp+sim) | **Primary** |
| `plot_exp_vs_sim.py` | Exp vs sim velocity only (old 2×2) | Legacy |
| `plot_exp_vs_sim_pitch.py` | Exp vs sim pitch only (old 2×2) | Legacy |

All primary scripts import from `plot_validation.py` for shared utilities (`plot_panel`, `build_plot_data`, `build_all_failed_freqs`, `GATE_END`, etc.).
