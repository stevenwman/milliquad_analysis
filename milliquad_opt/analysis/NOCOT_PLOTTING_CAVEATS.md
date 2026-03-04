# Megacomposite NoCOT 065 — Plotting Caveats

Script: `analysis/20260303_plot_megacomposite_nocot_065.py`

Produces a 3x4 figure: 3 terrain rows (flat, step, rough) x 4 columns (exp velocity, sim velocity, exp pitch, sim pitch).

---

## Architecture: TerrainPlotSpec

All terrain-specific rules are declared in `TERRAIN_SPECS`, a dict of frozen `TerrainPlotSpec` dataclass instances. The main loop is generic — no `if terrain == "step"` branches anywhere.

Each spec defines: gating rules, NPZ recompute function, experimental data extractors, failure dicts, trial selection count, and display mode (scatter vs shaded).

---

## Data Pipeline (per terrain)

Order of operations in the main loop:

1. **Load CSV** — `load_validation_csv(csv_path)` reads the validation_trials.csv. New CSV format (from refactored `validate_params.py`) omits `vx`, `pitch_rms`, `cot`, `max_x` columns. `load_validation_csv` uses `.get()` with None fallbacks for all optional columns.

2. **NPZ recompute** — If `spec.recompute` is not None, replaces CSV rows entirely with rows built from NPZ trajectories. All three terrains now use recompute because the new CSV format lacks the needed columns. Each recompute function applies terrain-appropriate gating to compute `vx` and `pitch_rms`.

3. **Trial selection** — If `spec.n_select` is not None, subsamples trials per (scene, freq) down to `n_select`. Uses `_build_ref_velocities` to extract experimental mean velocities, then picks trials closest to exp reference speed.

4. **Build plot data** — `build_plot_data` / `build_all_failed_freqs` from `plot_validation.py` aggregate rows into the format `plot_panel` expects.

5. **Plot** — `plot_panel` renders scatter dots, shading bands, X markers.

---

## NPZ Recompute: Per-Terrain Details

All three terrains rebuild rows from NPZ rather than using CSV values. The CSV serves only as a trial inventory (which ref_ids and trials exist); all metrics are recomputed.

### Flat: Time-Gating (`_recompute_flat_tg`)

Sim velocity and pitch are measured over a window matching the mean experimental recording length for each condition.

- **Window**: `[SETTLE_TIME, SETTLE_TIME + trial_duration]` where `SETTLE_TIME = 0.1s` (magnetic field onset) and `trial_duration` is per-condition from `_FLAT_TRIAL_DURATION` dict.
- **`_FLAT_TRIAL_DURATION`**: Hardcoded lookup `{(scene, freq): seconds}`. Values computed from experimental CSVs in `experimental_data/csv/flat/`. Example: scene1_f10 = 2.625s, scene4_f50 = 0.547s.
- **WR f50**: Time-gated at 0.316s (mean of 3 experimental recordings: 0.312, 0.328, 0.309s). Treated as exp-only failure in the plot (X marker on exp panel, real data on sim panel) because `extract_flat()` replaces WR f50 with 0.0 as a failure mode — the robot self-destructs at 50Hz experimentally. Experimental CSVs do exist (`f50w1-1.csv` etc.).
- **Velocity**: `dx / dt` between settle_idx and end_idx.
- **Pitch RMS**: `std(pitch[settle:end] - pitch[settle])` — zero-referenced to start of window.
- **Gate**: None (flat terrain, all trials valid).

### Step: 65% Spatial Gate (`_recompute_step_065`)

Two-stage gating: full traversal for success, 65% gate for measurement.

- **Success criterion**: `max(pos_x) >= STEP_END_X` (0.1015m). Trials that don't reach the end of the step section are failures (`vx=0.0, pitch_rms=0.0`).
- **Measurement window**: From `STEP_START_X` (0.05m, where steps begin) to `CUTOFF_065` (0.05 + 0.65 * (0.1015 - 0.05) = 0.0835m). This is 65% of the step section.
- **Why 65% not 90%**: The 90% gate used by the original optimizer (`config_step.py`) includes trajectory near the cliff edge where robots fall off the last step. This corrupts velocity (up to 21% inflated) and pitch RMS (up to 44% inflated). The 65% gate avoids cliff-edge artifacts and approximates the experimental q60 index window in intent.
- **Velocity**: `dx / dt` between enter_idx and gate_idx.
- **Pitch RMS**: `std(pitch[enter:gate] - pitch[enter])`.
- **Gate for `build_plot_data`**: `gate_end = STEP_END_X` (0.1015m) — trials that didn't fully traverse are marked invalid.

### Rough: Spatial Gate with Half-Gate Exception (`_recompute_rough`)

- **Default gate**: `_ROUGH_END_X = 0.155m`. Trials that don't reach this are failures.
- **Half-gate exception**: `scene1_f10` uses `_ROUGH_HALF_GATE = 0.08m`. At full gate (0.155m), only 2/10 trials pass. At half gate, 9/10 pass. This 1-leg 10Hz condition is too slow to traverse the full rough section but is still valid locomotion.
- **Measurement window**: From `_ROUGH_START_X` (0.005m) to the gate threshold (0.155m or 0.08m).
- **Velocity**: `dx / dt` between enter and exit indices.
- **Pitch RMS**: `std(pitch[enter:exit] - pitch[enter])`.
- **Gate exempt**: `{("scene1", 10.0)}` in the spec's `gate_exempt` — tells `build_all_failed_freqs` not to mark scene1_f10 trials as gate-failed even if `max_x < 0.155`. This is needed because the recompute uses 0.08m but `build_all_failed_freqs` checks against the spec's `gate_end` (0.155m).

---

## Trial Selection (`_select_trials`)

All three terrains subsample trials before plotting via `_select_trials`.

| Terrain | Sim trials/condition | Selected | Method |
|---------|---------------------|----------|--------|
| Flat    | 10                  | 3        | closest to exp ref velocity |
| Step    | 10                  | 3        | closest to exp ref velocity |
| Rough   | 10                  | 5        | closest to exp ref velocity |

### Selection priority (per condition):

1. **Exp failure conditions** (e.g. step WR f10/f20): Pick sim failures (vx=0.0) first, fill remaining slots randomly (seed=42).
2. **Ref velocity available**: Sort all trials by `|vx * 1000 - ref_vx_mm_s|`, pick the `n_select` closest. Reference velocities are extracted from the same experimental data extractors used for exp panels (`_build_ref_velocities`).
3. **Fallback**: Random sample with seed=42.

### Why closest-to-reference (not random):

- Rough terrain has high trial-to-trial variance (10 jitter trials). Random selection can pick outliers that misrepresent the typical behavior. Closest-to-reference selects trials that best match what the experiment measured.
- Step terrain also benefits — the 3 selected trials are the ones whose velocities most closely match experimental means, giving a fairer exp-vs-sim comparison.
- Side effect: originally designed for rough only, but applying to step too made panel (f) look better. Kept intentionally.

---

## Experimental Data

### Extractors

| Terrain | Velocity | Pitch |
|---------|----------|-------|
| Flat    | `extract_flat` | `extract_flat_pitch` |
| Step    | `extract_step_q60` | `extract_step_pitch_q60` |
| Rough   | `extract_rough` | None (no exp pitch data) |

All imported from `experimental_data/plot_velocity_vs_freq.py` and `plot_pitch_vs_freq.py`.

**CRITICAL**: Step experimental data uses q60 windowing (30% window centered at 60% of recording length = indices 45%-75%). This MUST match the config targets the sim was trained on. The sim's 65% spatial gate approximates the same intent (skip transient, avoid cliff-fall).

### Morph name remapping

Experimental extractors use morphology names (`"leg"`, `"2leg"`, `"4leg"`, `"wheel"`). Sim uses scene names (`"scene1"`, `"scene2"`, `"scene4"`, `"scene_wheel"`). `_remap_exp_data()` translates via `_MORPH_TO_SCENE`.

---

## Failure Handling

### Exp-only failures (hardcoded in spec)

| Terrain | Condition | Count | Reason |
|---------|-----------|-------|--------|
| Flat    | WR f50    | 3     | Robot self-destructs at 50Hz in experiments. Sim doesn't model mechanical failure. |
| Step    | WR f10    | 3     | Wheel can't move on steps at low frequency. |
| Step    | WR f20    | 3     | Same physical limitation. |

Treatment: `_strip_failure_freqs` removes these from exp data entirely (no shading taper). X markers with count annotations placed by `plot_panel` using `exp_failure_counts`.

### Sim failures (dynamic)

Sim failures are never hardcoded. `build_all_failed_freqs` dynamically identifies conditions where ALL selected trials fail the gate check. These are rendered as X markers with count annotations.

### X markers on all panel types

Gate-failed trials get `vx=0.0` and `pitch_rms=0.0` from recompute. `build_plot_data` with `exclude_invalid=True` keeps these zeros. `plot_panel` splits rendering: `val > 0` -> dot, `val <= 0` -> X marker. This applies to velocity AND pitch panels — a gate failure shows X on both.

---

## Rough-Specific: n/a Injection

Experimental rough data (`extract_rough`) silently drops n/a trials from CSV. `_inject_na_zeros` adds zeros for missing trials (`na_total_trials - n_success` per condition), rendered as X markers at y=0 in scatter_only mode. This shows that e.g. WR f10 had only 1/5 successful trials (1 dot + 4 X's), not just 1 trial.

No rough experimental pitch data exists — the exp pitch panel is replaced by the legend (axis turned off, used as `legend_ax`).

---

## Display Modes

### Flat & Step: Shaded bands (scatter_only=False)

- `fill_between` shading showing mean +/- std
- Mean line connecting across frequencies
- Dodge width: 3.5 Hz between morphologies
- Per-trial scatter dots overlaid on shading

### Rough: Scatter only (scatter_only=True)

- No `fill_between` shading — misleading with 1-2 successful trials per condition
- Individual trial dots spread horizontally within each morphology's dodge slot
- `scatter_dodge_width = 8.0` Hz between morphologies
- `intra_spread = 0.0` Hz (dots stacked vertically, not spread)
- `scatter_mean_line = True` — thin line connecting mean values across frequencies

---

## Y-Axis Sharing

Each terrain row shares y-limits between exp and sim panels (within velocity pair and within pitch pair). `_share_ylim` computes the union of both panels' auto-limits, then adds bottom padding for X-marker count annotations that sit below y=0.

Padding is computed in data coordinates: 24 points (6pt offset + 14pt font + 4pt breathing room) converted via the axis's physical height.

---

## Figure Layout

- **GridSpec**: Two outer columns (velocity, pitch), each subdivided into `n_rows + 1` rows (1 header + n terrain rows) x 2 columns (exp, sim).
- **Figure size**: 14.0 x 7.0 inches.
- **Metric headers**: "Velocity (mm/s)" and "Pitch RMS (deg)" as text above the columns.
- **Row labels**: "Flat", "Step", "Rough" on the left y-axis of the exp velocity panel.
- **Subplot letters**: (a) through (k), sequential across all visible panels, top-left of each.
- **Column headers**: "Experiment" / "Simulation" on top row only.

---

## Legend Placement

When rough terrain is present, its exp pitch panel doesn't exist (no data). That axis is turned off and reused as `legend_ax` to hold the morphology legend, x-axis label, and failure X marker explanation.

If no such axis is available, falls back to `fig.legend` at the bottom.

---

## Output

Filename: `plots/{timestamp}_megacomposite_nocot_065.png` where timestamp is `YYYYMMDDTHHMMSS`. Can be overridden with `--output`.

DPI: 200. Tight bounding box.

---

## Constants Reference

| Constant | Value | Source |
|----------|-------|--------|
| `_SETTLE_TIME` | 0.1 s | `config.SETTLE_TIME` — magnetic field onset |
| `_STEP_START_X` | 0.05 m | Where steps begin in step terrain XML |
| `_STEP_END_X` | 0.1015 m | End of step section (after final platform) |
| `_CUTOFF_065` | 0.0835 m | 65% of step section for measurement |
| `_ROUGH_START_X` | 0.005 m | Start of rough terrain measurement |
| `_ROUGH_END_X` | 0.155 m | Full rough terrain traversal gate |
| `_ROUGH_HALF_GATE` | 0.08 m | Half-distance gate for scene1_f10 |
| `_SELECT_SEED` | 42 | RNG seed for trial selection fallback |

---

## Canonical Run Dirs (as of 2026-03-04)

| Terrain | Run dir | Trials/condition | Params |
|---------|---------|-----------------|--------|
| Flat    | `results/20260303T192801_flat_tg` | 10 | flat_tg (with 20Hz) |
| Step    | `results/20260303T151416_step_065gate` | 10 | step_065 gate params |
| Rough   | `results/20260303T224229_rough_tg` | 10 | rough spatial RK4 |

Tested and rejected:
- `flat_tg_no20` (`results/20260304T111457_flat_tg_no20`): worse fit across frequency range without 20Hz training data.
- `step_065_b32` (`results/20260304T010054_step_065_b32`): visually worse step panels.
