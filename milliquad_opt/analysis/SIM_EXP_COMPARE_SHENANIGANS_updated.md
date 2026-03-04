# Exp vs Sim Comparison Plot: Shenanigans & Caveats (Updated 2026-03-04)

Supersedes the original `SIM_EXP_COMPARE_SHENANIGANS.md`. Reflects the `TerrainPlotSpec` refactor and NPZ-as-source-of-truth pipeline in `20260303_plot_megacomposite_nocot_065.py`.

---

## The Plot

`20260303_plot_megacomposite_nocot_065.py` produces a 3×4 figure: rows = flat, step, rough; col-pairs = [velocity exp|sim, pitch exp|sim]. All terrain-specific rules live in a `TerrainPlotSpec` dataclass. The main loop is generic — no `if terrain ==` branches.

---

## Data Pipeline

### Sim data: NPZ is the source of truth (flat, step)

For flat and step terrains, the plotting script **builds rows directly from NPZ trajectory arrays**, bypassing CSV trial selection. This means:
- All N trials per condition appear in the plot (typically 5), not just the top-N selected by `validate_params.py`
- Velocity and pitch are recomputed from raw trajectories with terrain-specific windowing
- No cherry-picking: if 2/5 trials fail, 2 X markers + 3 dots appear

The CSV is only used for initial terrain detection and as a fallback if no NPZ exists.

**Rough terrain** still uses CSV rows (10 trials, 5 selected) because rough has more trials and the CSV selection is meaningful.

### NPZ key structure

Keys follow `{scene}_{freq}_t{trial}_{suffix}` pattern, e.g. `scene_wheel_f30_t2_pos_x`. A regex parser extracts `(scene, freq, trial_idx)` from pos_x keys. Standard suffixes: `pos_x`, `pos_y`, `pos_z`, `time`, `pitch`, `yaw`, `omega`, plus 18 extended fields.

### Row construction

Each NPZ trial becomes a row dict with fields matching what `build_plot_data` expects: `scene`, `freq`, `vx`, `pitch_rms`, `max_x`, `crash=False`, `selected=True`, `cot=None` (COT cannot be recomputed from NPZ — no omega stored).

---

## Terrain-Specific Rules

### Flat terrain

| Rule | Value |
|------|-------|
| Gate | None (all trials valid) |
| Gate exempt | N/A |
| Measurement window | Time-gated: `[SETTLE_TIME, SETTLE_TIME + trial_duration]` |
| Trial duration source | Hardcoded `_FLAT_TRIAL_DURATION` dict, keyed by `(scene, freq)` |
| Settle time | 0.2s |
| Exp failures | WR f50 (3 of 5 self-destruct) |
| Sim failures | None expected (all flat trials succeed) |
| Display mode | Shading + scatter dots |
| WR f50 | Skipped entirely (no experimental recording → no trial_duration entry) |

**Why time-gate flat?** Experimental recordings vary from 0.38s (WR f30) to 2.6s (L1 f10). The sim runs for 3s. Without time-gating, sim measures a longer steady-state window than experiment. Effect is most pronounced on WR f20/f30 where the robot is still accelerating at settle time — the short experimental window captures transient, not steady state.

### Step terrain

| Rule | Value |
|------|-------|
| Success gate | `max_x >= STEP_END_X (101.5mm)` — full step traversal |
| Measurement gate | 65% spatial: `STEP_START_X` to `STEP_START_X + 0.65 * (STEP_END_X - STEP_START_X)` = 83.5mm |
| Gate exempt | None |
| Velocity | `dx/dt` from step entry (50mm) to 65% gate |
| Pitch RMS | `std(pitch - pitch[0])` from step entry to 65% gate |
| Failed trial output | `vx=0.0, pitch_rms=0.0` → renders as X marker |
| Exp failures | WR f10, f20 (3/3 fail in experiment) |
| Sim failures | Dynamic — detected from NPZ `max_x` per trial |
| Display mode | Shading + scatter dots |

**Success vs measurement distinction**: A trial that reaches 85mm (past the 65% gate at 83.5mm) but doesn't clear 101.5mm is still a FAILURE. Measurement gating only applies to successful trials. This prevents cliff-adjacent trajectory corruption from inflating velocity/pitch.

**Why 65% not 90%?** The original 90% gate included trajectory near the cliff edge where robots decelerate and oscillate before falling off. Impact: velocity up to 21% higher, pitch RMS up to 44% higher at 90% vs 65%. The 65% gate matches the intent of the experimental q60 window (skip transient + avoid end artifacts).

**Trial selection (n_select=3)**: Step selects 3 of 5 NPZ trials per condition to match experiment's 3 trials/condition. Selection prioritizes matching the experimental outcome:
- For exp failure conditions (WR f10, f20): pick all sim failures first (up to 3), backfill remainder randomly from passes. Example: WR f10 has 2/5 sim fails → selected = 2 fails + 1 random pass.
- For normal conditions: random sample of 3.
- This avoids the old problem where `validate_params.py` cherry-picked the 3 best-matching (= most successful) trials, hiding sim failures that exist.

### Rough terrain

| Rule | Value |
|------|-------|
| Gate | `max_x >= 155mm` (full rough section traversal) |
| Gate exempt | `{("scene1", 10.0)}` — L1 f10 successfully locomotes but too slow to reach 155mm |
| Measurement | Full CSV values (no NPZ recompute) |
| Exp failures | None hardcoded — `_inject_na_zeros` handles missing trials |
| Sim failures | Dynamic from `build_all_failed_freqs` |
| Display mode | `scatter_only=True` (no shading, individual dots) |
| n/a injection | 5 total trials expected per condition; missing = zero-valued X markers |

**Why scatter_only?** High trial-to-trial variance + low success rates (some conditions 1-2/5) make `fill_between` shading misleading. Individual dots + X markers show the actual distribution.

---

## Failure Handling

### Categories

1. **Exp-only failures** (declared in `TerrainPlotSpec.exp_failures`):
   - Flat WR f50: mechanical self-destruction at 50Hz
   - Step WR f10, f20: wheel can't move on steps at low frequency
   - Treatment: exp panel gets X marker + count annotation. Sim panel shows actual data.

2. **Sim failures** (always dynamic, never hardcoded):
   - Detected per-trial from NPZ/CSV `max_x` via `build_all_failed_freqs`
   - Treatment: X marker at y=0 on all panel types (velocity, pitch, COT)
   - Count annotation when multiple failures collapse to one X

### Failure rendering

- `_strip_failure_freqs()` removes failure frequencies from shading data entirely (no taper to zero)
- X markers at y=0 are independent of shading — they always appear
- X markers receive morphology-based dodge offset (same as scatter dots)
- Count annotation (bold, scene-colored) when count > 1

### Previous issues fixed
- **CSV trial selection bias**: `validate_params.py` selects top-N by velocity match, which cherry-picks successful trials. Example: step WR f10 has 3/5 passing, selection picks those 3 → looks like 100% success. NPZ-based rows + failure-prioritized selection now shows 2 fails + 1 pass for WR f10 (matching exp failure pattern).
- **Hardcoded sim failures**: Previously step WR f10/f20 were hardcoded as shared failures. Now sim failures are purely dynamic — if the optimizer finds params where WR f10 succeeds, no X appears.

---

## Experimental Data Sources

| Terrain | Velocity | Pitch |
|---------|----------|-------|
| Flat | `extract_flat()` | `extract_flat_pitch()` |
| Step | `extract_step_q60()` | `extract_step_pitch_q60()` |
| Rough | `extract_rough()` | None (no exp rough pitch data) |

All from `experimental_data/plot_velocity_vs_freq.py` and `plot_pitch_vs_freq.py`. Morphology names remapped: `leg→scene1, 2leg→scene2, 4leg→scene4, wheel→scene_wheel`.

### Experimental windowing

- **Flat**: full recording (variable length per condition, 0.38–2.6s)
- **Step q60**: 30% window centered at 60% of recording (indices 45%–75%)
- **Rough**: full recording, n/a trials silently dropped by extractor → re-injected as zeros

---

## Display Rules

### Shared across all terrains
- Bracket ticks + grey gap bands delineate frequency zones
- Y-axis shared within each exp/sim pair per terrain row
- X markers dodged by morphology offset
- Panel letters (a)–(k) in top-left
- No mean lines (shading center implies mean)

### Scatter-only mode (rough)
- `scatter_dodge_width=8.0` Hz between morphologies
- `intra_spread=0.0` Hz (dots stacked at morphology center)
- `scatter_mean_line=True` (horizontal line at mean)
- X markers at y=0 for failed/n/a trials

### Non-scatter mode (flat, step)
- `dodge_width=3.5` Hz between morphologies
- `fill_between` shading (mean ± std)
- Dots at individual trial values

---

## Gate Exemptions

`gate_exempt` is a `frozenset[tuple[str, float]]` per terrain, passed to `_is_valid_trial`, `build_plot_data`, and `build_all_failed_freqs`.

- **Flat**: empty (no gate)
- **Step**: empty (all conditions must clear full step)
- **Rough**: `{("scene1", 10.0)}` — L1 f10 is too slow to reach 155mm but locomotes successfully

The `gate_exempt` parameter was added to `plot_validation.py` functions as a backward-compatible keyword arg (defaults to module-level `GATE_EXEMPT` if not provided).

---

## Script Inventory

| Script | Purpose | Status |
|--------|---------|--------|
| `20260303_plot_megacomposite_nocot_065.py` | 3×4 megacomposite with TerrainPlotSpec, NPZ-based rows, 65% step gate, flat time-gating | **Primary** |
| `plot_validation.py` | Sim-only validation plots + shared utilities (`plot_panel`, `build_plot_data`, etc.) | **Primary** |
| `plot_cot_only.py` | 3×1 COT column | **Primary** |
| `plot_megacomposite_nocot.py` | Original 3×4 (pre-TerrainPlotSpec, 90% step gate) | Legacy |
| `plot_megacomposite.py` | 3×5 with COT | Legacy |
| `plot_exp_vs_sim_composite.py` | 2×4 flat+step | Legacy |

---

## Known Limitations

1. **COT not recomputable from NPZ**: NPZ doesn't store angular velocity (`omega` suffix exists but is body angular velocity, not the `cvel` used for COT). COT panels require CSV values from `validate_params.py`.

2. **Flat time-gating makes WR velocity drop**: Short experimental windows (0.38–0.97s) capture wheel still accelerating past settle time. This is correct for comparison but looks like sim underperforms vs old full-window plots.

3. **Rough pitch**: No experimental rough terrain pitch data exists. The rough pitch panel shows legend/metadata instead.

4. **Step optimizer bias**: Current step params were optimized with 90% spatial gate. The 65% post-hoc re-windowing changes measured velocity/pitch but the params themselves are biased toward 90% performance. A `config_step_065.py` retrain exists.
