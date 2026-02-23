# Multi-Terrain Validation Script — Plan

## Goal

Create two new files in `mujoco_refactor/`:
1. **`terrain_config.py`** — terrain preset definitions (separate from optimizer config)
2. **`terrain_test.py`** — validation script that runs best-fit params on various terrains

This is a **validation** tool (not optimization) — we want to see how the fitted
model behaves on unseen terrain types.

---

## terrain_config.py

Separate file so terrain definitions don't pollute `config.py` (optimizer-only).
Contains named presets as plain dicts.

```python
"""Terrain presets for terrain_test.py. Separate from optimizer config."""

# External magnetic field amplitude (bookkeeping — actual value comes from fitted params)
B_FIELD_MT = 2.0  # mT, same across all experiments

# Jitter defaults (can be overridden via CLI)
DEFAULT_JITTER_TRIALS = 1
DEFAULT_JITTER_DEG = 2.0
JITTER_BASE_SEED = 99999

TERRAIN_PRESETS: dict[str, dict] = {
    # --- Flat (baseline) ---
    "flat": {
        "type": "flat",
        "ctrl_freq": 30.0,           # Hz actuation frequency
    },

    # --- Step presets ---
    "step_default": {
        "type": "step",
        "ctrl_freq": 30.0,           # Hz
        "step_height": 0.0015,       # 1.5 mm
        "step_length": 0.0045,       # 4.5 mm
        "step_count": 5,
        "final_step_length": 0.02,   # 20 mm extended platform after last rise
        "step_width": 0.1,           # 100 mm (wide enough robot can't fall off)
        "flat_lead": 0.0075,         # 7.5 mm flat ground before first step
    },
    "step_tall": {
        "type": "step",
        "ctrl_freq": 30.0,
        "step_height": 0.003,        # 3 mm — double default
        "step_length": 0.0045,
        "step_count": 5,
        "final_step_length": 0.02,
        "step_width": 0.1,
        "flat_lead": 0.0075,
    },
    "step_many": {
        "type": "step",
        "ctrl_freq": 30.0,
        "step_height": 0.001,        # 1 mm — shorter but more steps
        "step_length": 0.0045,
        "step_count": 10,
        "final_step_length": 0.02,
        "step_width": 0.1,
        "flat_lead": 0.0075,
    },

    # --- Rough presets ---
    "rough_mild": {
        "type": "rough",
        "ctrl_freq": 30.0,
        "height_mean": 0.002,        # 2 mm mean tile height
        "height_std": 0.0005,        # 0.5 mm std
        "tile_size": 0.005,          # 5 mm square tiles
        "grid_nx": 20,
        "grid_ny": 10,
        "z_safe": 0.001,             # 1 mm min material thickness
        "seed": 42,
        "flat_lead": 0.05,           # 50 mm flat ground before terrain
    },
    "rough_harsh": {
        "type": "rough",
        "ctrl_freq": 30.0,
        "height_mean": 0.002,
        "height_std": 0.001,         # 1 mm std — double the mild
        "tile_size": 0.005,
        "grid_nx": 20,
        "grid_ny": 10,
        "z_safe": 0.001,
        "seed": 42,
        "flat_lead": 0.05,
    },
}
```

Each preset includes `ctrl_freq` so you know exactly what actuation frequency was
used. `B_FIELD_MT` is recorded at the top for bookkeeping (always 2 mT in current
experiments; the actual value used in sim comes from `magnetic_field_fudge * B_FIELD`
in the fitted params).

Users can add new presets by copying an existing one and tweaking values.
CLI: `--preset step_default` (or `--preset rough_mild`, etc.)
CLI overrides for individual params are still supported and override the preset values.

---

## Terrain Types

### 1. Flat (baseline)
- No MJCF modification. Uses original `MJCF_PATHS[scene]` as-is.
- Purpose: baseline comparison for velocity on terrain vs velocity on flat.

### 2. Steps (staircase)
- Append box geoms to the MJCF worldbody via ElementTree.
- Each step is a box with configurable height, length, width.
- Steps stack linearly in Z (staircase pattern along +X).

**Preset keys:** `step_height`, `step_length`, `step_count`, `final_step_length`,
`step_width`, `flat_lead`

**Geometry (from legacy `visualize_rollout_step.py`):**
```
for i in range(step_count):
    is_final = (i == step_count - 1)
    length = final_step_length if is_final else step_length
    pos_x = flat_lead + i * step_length + length / 2.0  (final adjusted)
    pos_z = (i + 1) * step_height - step_height / 2.0
    size = (length/2, step_width/2, step_height/2)   # MuJoCo half-extents
```

### 3. Rough (random heightfield)
- Uses `utils/terrain_mesh.py::generate_terrain_hfield()` to create a reproducible
  random heightfield PNG, then injects `<hfield>` asset + geom into the MJCF.
- Terrain is seeded, so the same seed always produces the same terrain.

**Preset keys:** `height_mean`, `height_std`, `tile_size`, `grid_nx`, `grid_ny`,
`z_safe`, `seed`, `flat_lead`

**Heightfield injection (adapted from legacy `visualize_rollout_rough.py`):**
1. Call `generate_terrain_hfield(nX, nY, sL, height_mean, height_std, z_safe, seed, output_path)`
2. Get back `(heights, (x_half, y_half, z_top, z_bottom))` size tuple
3. Add `<hfield>` asset element with absolute PNG path and size
4. Add `<geom type="hfield">` to worldbody at `pos=(flat_lead + x_half, 0, 0)`
5. Write edited XML to a temp directory

---

## Friction / Contact Params

**No special handling needed.** `simulation_fast.py` uses `mjENBL_OVERRIDE` which
applies the fitted `o_solref`, `o_solimp`, and `o_friction` **globally** to all
geom-geom contacts (including terrain geoms). It also sets `geom_condim[:] = 6`
on all geoms. So any new terrain geoms automatically inherit the fitted contact params.

---

## Param Loading

Same pattern as `replay_best.py`:
1. Accept `run_dir` positional arg (results directory)
2. Read `optimization_bests.csv` → get final best ID
3. Look up full-precision params in `multi_optimization_results.csv`
4. Convert via `sim_params_from_point()`

---

## Reference Configs

Two modes for selecting which (scene, freq) combos to run:

**Mode 1: Preset frequency (default)**
Use the `ctrl_freq` from the preset. Runs all 4 morphologies at that single frequency.
- `--scenes scene1 scene4` can filter morphologies

**Mode 2: Sweep frequencies**
`--freqs 10 20 30 50` overrides the preset's `ctrl_freq` and runs all listed
frequencies for each morphology. This is useful for comparing terrain impact
across the full frequency range.
- `--scenes` still filters morphologies

In both modes, each (scene, freq) pair is a "config".

---

## Jitter / Statistical Aggregation

- `--jitter-trials N` (int, overrides preset default of 1)
- `--jitter-deg D` (float, overrides preset default of 2.0°)
- Seeds are deterministic: `base_seed + config_idx * N + trial_idx`
  (base_seed from terrain_config.py)
- When N > 1, report **median** velocity (same as optimizer convention),
  plus mean and std across trials.
- When N = 1, no jitter applied (yaw_jitter_deg = 0).

---

## Output

### Console Table
```
preset: step_default (h=1.5mm, l=4.5mm, n=5, lead=7.5mm)
jitter: ±2.0° yaw, 3 trials, median aggregation

  config              vel(mm/s)  std    flat_vel  ratio  tumble  lat(cm)  yaw°
  ─────────────────────────────────────────────────────────────────────────────
  scene1_f10            48.2    1.3      50.0     0.96    N      0.03    1.2
  scene1_f30           102.1    5.7     115.6     0.88    N      0.45    8.1
  ...
```

Columns:
- `vel`: median forward velocity on terrain (mm/s)
- `std`: std across jitter trials (if N > 1)
- `flat_vel`: velocity from flat baseline (same params, same jitter protocol)
- `ratio`: terrain_vel / flat_vel — how much terrain slows the robot
- `tumble`: Y/N if tumble_penalty > threshold in any trial
- `lat`, `yaw`: lateral displacement and yaw deviation

### Recording (optional)
- `--record` flag: save mp4 videos to `{run_dir}/terrain_videos/`
- One video per config, named `{ref_id}_{preset}.mp4`
- Only the median-velocity trial gets recorded (run twice: first headless for stats,
  then with `record_path` for the best trial if requested)

### CSV (optional)
- `--csv` flag: write results to `{run_dir}/terrain_results_{preset}.csv`

---

## CLI Interface

```bash
cd mujoco_refactor

# Flat baseline (all 15 configs, no jitter)
uv run python terrain_test.py results/20260222T... --preset flat

# Steps with defaults (all configs, 3 jitter trials)
uv run python terrain_test.py results/20260222T... --preset step_default --jitter-trials 3

# Rough terrain, filtered scenes
uv run python terrain_test.py results/20260222T... --preset rough_mild \
    --scenes scene4 scene_wheel --jitter-trials 3

# Override a single preset param (e.g. taller steps)
uv run python terrain_test.py results/20260222T... --preset step_default \
    --step-height 0.003 --jitter-trials 3

# Record videos
uv run python terrain_test.py results/20260222T... --preset step_default --record

# List available presets
uv run python terrain_test.py --list-presets
```

---

## MJCF Editing Strategy

- Use `tempfile.TemporaryDirectory()` for all edited XMLs and heightmap PNGs
- Context manager cleans up automatically on exit
- MuJoCo resolves `<hfield file=...>` relative to XML path, so PNG goes in same temp dir

---

## Implementation Notes

### simulation_fast.py interface
```python
simulation_fast.run_simulation(
    sim_params,                    # dict from sim_params_from_point()
    mjcf_path=edited_xml_path,     # str, absolute path to modified MJCF
    sim_duration=SIM_DURATION,     # from config (3.0s default)
    wall_timeout=SIMULATION_TIMEOUT,
    visualize=False,               # headless for stats
    record_path=None,              # or path for video
    init_yaw_jitter_deg=jitter_deg,
    rng_seed=seed,
)
# Returns: list[dict] trajectory, or None on failure
```

`sim_params` needs `drive_freq` set per-config:
```python
sp = dict(sim_params)
sp['drive_freq'] = ref['ctrl_freq']
```

### Imports from config.py (optimizer config — read-only)
```python
from config import (
    MJCF_PATHS, SIM_DURATION, SIMULATION_TIMEOUT, SETTLE_TIME,
    reference_rows, sim_params_from_point, space,
)
```

### Imports from terrain_config.py
```python
from terrain_config import (
    TERRAIN_PRESETS, DEFAULT_JITTER_TRIALS, DEFAULT_JITTER_DEG, JITTER_BASE_SEED,
)
```

### Imports from utils
```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from terrain_mesh import generate_terrain_hfield
```

### Cost extraction (for tumble/lateral/yaw)
```python
from optimizer import calculate_cost
cost_data = calculate_cost(trajectory, target_velocity=ref['speed'], verbose=False)
# cost_data keys: avg_forward_velocity, tumble_penalty, lateral_displacement,
#                 yaw_deviation_deg, pitch_rms_deg, total_cost
```

### Flat baseline run
When preset type != "flat", we also need flat velocities for the ratio column.
Run flat silently first, cache results, then run terrain, print combined table.

---

## File Structure

```
mujoco_refactor/
├── terrain_config.py        # <-- new: terrain preset definitions
├── terrain_test.py          # <-- new: validation script
├── config.py                # optimizer config (unchanged, read-only)
├── simulation_fast.py       # sim engine (unchanged)
├── optimizer.py             # calculate_cost (unchanged)
└── ...
```

No changes to existing files.

---

## Edge Cases

- **Wheel morphology + steps**: The wheel's MJCF is in `wheel_milli_quad/`, not
  `multi_milli_quad/`. The step/rough terrain injection just appends to worldbody,
  so it works for any MJCF structure.
- **Sim failure**: If `run_simulation` returns None, mark that trial as FAIL and
  continue. Report FAIL in the table.
- **SIM_DURATION**: Use config default (3.0s). Terrain sims may need longer if the
  robot is slow on rough terrain — add `--duration` override.
- **Preset not found**: Print available presets and exit with clear error.
- **CLI override + preset**: CLI args like `--step-height 0.003` shallow-merge into
  the preset dict, so you can tweak one param without redefining everything.

---

## Implementation Notes (post-implementation)

### MJCF relative path resolution
MuJoCo resolves `<include>`, mesh files, and textures relative to the XML file's
directory. We cannot copy the XML to a temp dir because the robot XMLs use relative
`<include file="robot_4.xml">` and mesh references (`assets/magnet.stl`).

**Solution**: Write edited XMLs **alongside the original** in the same source directory
with a `_terrain_tmp.xml` suffix. Track these in `_temp_xml_files` list and clean up
via `cleanup_temp_xmls()` in a `try/finally` block.

### MuJoCo hfield z_bottom
`generate_terrain_hfield()` in `utils/terrain_mesh.py` returns `z_bottom=0.0`.
MuJoCo requires all hfield size params to be strictly positive. Fixed in
`_inject_rough()` by clamping `z_bottom` to `0.001` when zero.

### Step terrain tumble/yaw artifact
Step presets produce tumble=Y and yaw~180° for most configs. This is real physics:
the robot climbs the staircase then falls off a cliff at the end (total elevation =
`step_count * step_height`, e.g. 7.5mm for step_default). The 20mm `final_step_length`
is not long enough to prevent the robot from reaching the edge during 3s sim. Options:
- Increase `final_step_length` (e.g. 0.1m) to keep the robot on the platform
- Increase `--duration` to see post-fall recovery
- Accept it as a realistic terrain challenge metric

### Rough terrain mild preset
`rough_mild` (0.5mm height std) barely affects any morphology (ratio 0.94–1.04).
Use `rough_harsh` (1.0mm std) for meaningful differentiation.
