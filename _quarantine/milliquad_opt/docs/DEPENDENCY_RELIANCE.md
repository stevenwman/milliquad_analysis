# External Dependencies in `milliquad_opt/`

Audit date: 2026-02-28

Goal: make `milliquad_opt/` fully self-contained — no runtime imports or file
reads reaching into `/experimental_data` or `/mujoco_refactor` (or project-root
`/utils`).

## Files with real external dependencies

### 1. `analysis/plot_exp_vs_sim.py` — 🔴 Hard import

- **Line 26**: `_EXP_DIR = ...parent.parent.parent / "experimental_data"`
- **Line 30**: `from plot_velocity_vs_freq import extract_flat, extract_step`
- Adds `experimental_data/` to `sys.path` and imports two functions from
  `experimental_data/plot_velocity_vs_freq.py`.

### 2. `analysis/plot_exp_vs_sim_pitch.py` — 🔴 Hard import

- **Line 26**: `_EXP_DIR = ...parent.parent.parent / "experimental_data"`
- **Line 30**: `from plot_pitch_vs_freq import extract_flat_pitch, extract_step_pitch`
- Same pattern as above, imports from `experimental_data/plot_pitch_vs_freq.py`.

### 3. `analysis/validate_params.py` — 🟡 File read (graceful fallback)

- **Lines 73-76**: reads `experimental_data/csv/random_terrain_raw.csv` via pathlib.
- Only used for rough-terrain exploratory conditions. Returns `[]` if file is
  missing, so it won't crash — but still reaches outside.

### 4. `generate_terrain_xmls.py` — 🔴 Hard import

- **Line 23**: `PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent`
- **Line 26**: `from utils.terrain_mesh import generate_heightmap`
- Imports `generate_heightmap` from project-root `utils/terrain_mesh.py`.

## Files with comment-only references (no dependency)

These mention `experimental_data` or `mujoco_refactor` in comments/docstrings
only. No runtime coupling — safe to ignore.

- `analysis/plot_validation.py` — docstring mentions plot filenames for style
  reference; line 31 comment notes color origin.
- `generate_terrain_xmls.py` — lines 41, 53 note where constants were copied
  from.

## Fix options

1. **Copy needed functions/data into `milliquad_opt/`** — inline
   `generate_heightmap`, copy `extract_flat`/`extract_step` and pitch
   equivalents, copy the CSV.
2. **Remove the external-dependent features** if they aren't needed in the
   standalone version.
3. **Hybrid** — keep `validate_params.py` as-is (graceful fallback) and only
   fix the three hard-import files.

## All 41 `.py` files checked

37 files are fully self-contained. The 4 listed above are the only ones with
external dependencies.
