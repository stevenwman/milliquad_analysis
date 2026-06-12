# Exp vs Sim Comparison Plot: Shenanigans & Caveats

## The Plot

`plot_exp_vs_sim_composite.py` produces a 2×4 figure: rows = flat, step; col-pairs = [velocity exp|sim, pitch exp|sim]. Each panel shows per-trial scatter dots and std shading bands.

Legacy standalone scripts (`plot_exp_vs_sim.py`, `plot_exp_vs_sim_pitch.py`) still exist but are superseded by the composite.

## Failure Modes: Two Categories

Wheel robot has failure modes that need different treatment on exp vs sim sides:

### Exp-only failures (sim doesn't reproduce)
- **Flat WR f50**: Robot physically self-destructs at 50Hz in experiments. Sim doesn't model mechanical failure. Exp: 709.4 ± 11.5 mm/s (n=3). Sim: 725.2 ± 13.2 mm/s (n=3 selected). ~2% sim-over-exp.
- Treatment: exp panel gets shading tapered to 0 + X marker. Sim panel shows actual data (no X).

### Shared failures (both exp and sim)
- **Step WR f10, f20**: Wheel can't move on steps at low frequency. This is a physical limitation the sim reproduces (near-zero velocity).
- Treatment: both panels get shading tapered to 0 + X markers.

## Failure Frequency Handling

When a morphology fails at a frequency, `_strip_failure_freqs()` removes the frequency entirely from the data (trials, mean, std). No shading connects to or from the failure point. An X marker at y=0 indicates the failure independently.

Previously, `_inject_failure_zeros()` inserted `mean=0, std=0` to taper the shading band to zero. This was misleading — it implied a gradual decline when the reality is binary (works or doesn't). Replaced with clean stripping.

Scatter dots at failure frequencies are also stripped. Only the X marker at y=0 remains. This prevents misleading non-zero dots from experimental noise.

Failure X markers appear on **all panel types** (velocity, pitch, COT) — not just velocity. Both exp and sim panels get their respective failure dicts.

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

### Per-trial gate failure X markers (all modes)

When `exclude_invalid=True`, `build_plot_data` inserts `0.0` for any gate-failed or inverted trial. In `plot_panel`, both scatter_only (rough) AND non-scatter (flat/step) modes split rendering: `val > 0` → scatter dot, `val <= 0` → X marker. This applies to all metrics (velocity, pitch, COT).

Previously, non-scatter mode rendered 0.0 as regular dots — this caused misleading near-zero dots in step sim panels. Example: scene_wheel f30 step has 3 selected trials but only 2 pass the gate (max_x >= 101.5mm). The third (max_x = 79mm) now correctly renders as an X marker instead of a dot at y=0.

### "All failed" X markers

When ALL selected trials for a (scene, freq) combo are invalid (didn't clear gate OR inverted), a scene-colored X is placed at y=0 on plots. This distinguishes "no data because all trials failed" from "data point not shown because it's off-screen".

`build_all_failed_freqs` returns `{scene: {freq: count}}` — the count is the number of trials that failed. Used for count annotations (see below).

### X marker count annotations

When multiple failures collapse into a single X marker, a count number is annotated above the X (fontsize=14, bold, scene-colored). Single failures show a plain X with no number.

Three sources of X markers, each with count logic:
1. **Per-trial gate failures (scatter_only)**: failures grouped per-freq, one X + count replaces N stacked X's. Only valid trials participate in the intra-spread layout.
2. **Per-trial gate failures (non-scatter)**: same grouping — one X + count per freq instead of overlapping X's at the same dodged x-position.
3. **All-failed combos** (`all_failed` dict): `build_all_failed_freqs` provides the count directly. Example: step sim WR f10/f20 each have 3 selected trials that all fail the gate → X with "3".

Hardcoded categorical failures (`failures` dict from `_strip_failure_freqs`) remain plain single X's — no count, because the trial data is stripped before `plot_panel` sees it.

### Flat terrain: no filtering needed in practice

Flat trials all pass both checks (no gate, no inversions). WR f50 pitch_rms is ~0.18° (stable).

---

## Scatter-Only Mode (Rough Terrain)

Rough terrain has high trial-to-trial variance and low sample counts (some conditions only 1-2 successful trials out of 5). Standard `fill_between` shading is misleading with <3 data points — a band implies a distribution where there's barely a sample. Instead, rough panels use `scatter_only=True`:

- **No shading** (`fill_between` suppressed)
- **Individual trial dots** spread horizontally within each morphology's dodge slot
- **Sorted left-to-right** by value (lowest on left, highest on right) so visual spread indicates variance
- **Wider dodge**: `scatter_dodge_width = 15.0` Hz between morphologies (vs `dodge_width = 3.5` Hz for flat/step). Frequency ticks are 20 Hz apart.
- **Intra-morphology spread**: `intra_spread = 3.0` Hz — 5 trials span ±1.5 Hz within each morphology slot
- **Clearance constraint**: morphology gap = `sdw / (n-1)` must be > `intra_spread`, otherwise adjacent morphology groups bleed into each other. Current: gap = 15/3 = 5.0 Hz, clearance = 5.0 - 3.0 = 2.0 Hz.
- **Xlim padding**: auto-computed as `sdw/2 + intra_spread/2 + 1` for scatter_only (vs fixed ±3 Hz for flat/step)
- **Figure width**: megacomposite uses 8 inches/column (vs default 6) to give scatter dots more physical space

Activated by `terrain.startswith("rough")` in both `plot_validation.py` (standalone + 3×3 composite) and `plot_megacomposite.py` (rough row, all 5 columns).

## Unified Bracket Ticks + Grey Gap Bands

All rows (flat, step, rough) use the same visual language: bracket tick marks bounding each frequency zone, with grey `axvspan` bands in the gaps between zones (white inside brackets). This replaced earlier inconsistencies where rough used alternating bands and flat/step used a different pattern.

Implementation (single unified block in `plot_panel`):
- **Bracket half-spread**: scatter_only = `sdw/2 + intra_spread/2` (= 9.0 Hz), non-scatter = `dw/2 + 0.75` (= 2.5 Hz)
- Two tick marks per frequency at ±half_spread (no labels), centered frequency labels via minor ticks with `length=0`
- Grey bands (`#f0f0f0`) fill edges and inter-bracket gaps; bracket interiors stay white
- Y-only grid (`ax.grid(axis="y")`) — no vertical grid lines on any row

Previously went through several iterations: alternating bands → vertical separator lines → grey gap bands (matching flat/step). The current unified approach is simplest and most consistent.

## Experimental Rough n/a Trials

`extract_rough()` from `experimental_data/plot_velocity_vs_freq.py` silently drops `n/a` trials from the CSV. This hides failure information — e.g., wheel f10 has 4/5 n/a (20% success rate) but the plot only shows 1 dot.

Fix: `_inject_na_zeros()` in `plot_megacomposite.py` adds zeros for each missing trial (`5 - n_success` per condition). In scatter_only mode, `plot_panel` separates `val > 0` (success dots) from `val <= 0` (X markers at y=0). All trials (successes + failures) share the same intra_spread spacing so they don't overlap.

This is velocity-only — rough experimental pitch data doesn't exist yet.

## Dodge on X Markers

All X markers (failure modes, all-invalid combos) now receive morphology-based horizontal offset (`dx`), matching the scatter dot positions. Previously X markers sat at the exact frequency tick, misaligned from their morphology's dots. Uses the same `dodge_width` or `scatter_dodge_width` depending on mode.

---

## Pitch Mega Figure: Sim vs Exp Gating Mismatch

`analysis/investigative/plot_pitch_mega.py` produces per-condition pitch time series and XY paths for step terrain. Sim and exp use **different windowing methods** for pitch RMS — this is intentional, not a bug.

### Sim pitch RMS: spatial gating
- Window: `STEP_START_X` (50mm) to `CUTOFF_X` (65% of step region = 83.5mm)
- Previously 90% (96.4mm), tightened to 65% to avoid cliff-edge artifacts and better match exp's q60 intent
- Calculation: `std(pitch[enter_idx : gate_idx+1])`
- Trials that never reach `CUTOFF_X` are excluded (e.g., wheel f30 trial 0 stuck at 79mm)
- Display: trajectory plotted to 100% (`TRIM_X = STEP_END_X`), shading shows 65% gate region

### Exp pitch RMS: q60 index gating
- Window: indices `[0.45*n : 0.75*n]` of the full CSV recording (30% centered at 60%)
- Calculation: `std(theta[lo : hi])`
- Consistent with `extract_step_pitch_q60()` in `experimental_data/plot_pitch_vs_freq.py`
- No spatial information available in exp data — cannot apply spatial gating

### Why they differ
Experimental CSVs don't have forward position data in a sim-compatible coordinate frame, so spatial gating is impossible. The q60 window was chosen by the student as the best trajectory-relative approximation: skip the initial transient (first 45%), capture steady-state locomotion on the steps (45%–75%), and avoid cliff-fall artifacts (last 25%). The sim's 65% spatial gate achieves the same intent more precisely using known step geometry — both exclude roughly the last third of the trajectory where cliff-fall artifacts dominate.

### Gate tightening (90% → 65%)
The original optimizer and validation pipeline used 90% spatial gate, which included trajectory near the cliff edge. This inflated both pitch RMS (up to 44% higher) and velocity (up to 21% higher) compared to 65%. The 65% gate better matches exp's q60 intent and gives cleaner steady-state measurements. See `SPATIAL_GATING_RESULTS.md` for detailed comparison tables.

**Optimizer retraining**: `config_step_065.py` retrains with 65% gate. The 90% optimizer (`config_step.py`) is preserved unchanged. Post-hoc re-windowing from NPZ (`plot_megacomposite_nocot_065.py`) can approximate 65% metrics without retraining, but the fitted params are biased toward 90% performance.

### Verification against nocot
Sim pitch RMS from the mega figure matches the `pitch_rms` column in the validation CSV (same `compute_pitch_rms` function, same spatial gate, same trial exclusion via gate-clearing). The `_065` variant recomputes from NPZ at 65%. Exp step pitch is **not yet in the nocot megacomposite** (panels are blank for step exp pitch). The mega figure is the first visualization of exp step pitch.

### XY path data
- **Sim**: `pos_x` (forward) vs `pos_z` (lateral) from MuJoCo qpos. Thin line = full traj to 100% (TRIM_X), thick = spatially gated portion (65%). Gate start/end shown as vertical lines.
- **Exp**: Average of mass_A and mass_C marker positions (CSV cols 1-2 and 5-6). `x = (avg_x[0] - avg_x) * 1000` (flipped to match sim forward direction). Thin = full recording, thick = q60 window.

---

## Script Inventory

| Script | Purpose | Status |
|--------|---------|--------|
| `plot_validation.py` | Sim-only: velocity/COT/pitch per terrain + 3×3 composite | **Primary** |
| `plot_megacomposite.py` | 3×5 megacomposite: 3 terrains × (vel exp\|sim, pitch exp\|sim, COT) | **Primary** |
| `plot_megacomposite_nocot.py` | 3×4 no-COT: 3 terrains × (vel exp\|sim, pitch exp\|sim) | **Primary** |
| `plot_cot_only.py` | 3×1 COT column: 3 terrains × sim COT | **Primary** |
| `plot_exp_vs_sim_composite.py` | Exp vs sim: 2×4 (flat+step × vel+pitch × exp+sim) | **Primary** |
| `plot_exp_vs_sim.py` | Exp vs sim velocity only (old 2×2) | Legacy |
| `plot_exp_vs_sim_pitch.py` | Exp vs sim pitch only (old 2×2) | Legacy |

All primary scripts import from `plot_validation.py` for shared utilities (`plot_panel`, `build_plot_data`, `build_all_failed_freqs`, `GATE_END`, etc.).
