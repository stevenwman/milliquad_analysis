# Plan: RK4 Re-optimization — `milliquad_opt/` Restructure

## Context

Euler integrator injects phantom energy through soft contacts, making the naive COT formula (`P = τ·ω`) produce negative energy for some conditions (5/15 on rough, 4/15 on flat). RK4 fixes this completely but Euler-optimized params don't transfer (16.9% mean error under RK4 vs 11.6% under Euler). We need to re-optimize all 3 terrains with RK4.

**Goal**: Create a clean `milliquad_opt/` directory with RK4 baked into all XMLs, organized robot files, and the same 3-terrain optimization pipeline ready to run.

## Directory Structure

```
milliquad_opt/
├── robots/
│   ├── quad/                         # shared by scene1, scene2, scene4
│   │   ├── assets/                   # copied from mujoco_refactor/multi_milli_quad/assets/
│   │   ├── robot_1.xml, robot_2.xml, robot_4.xml   # unchanged
│   │   ├── scene_1_flat.xml          # scene_1 + integrator="RK4" + energy flag
│   │   ├── scene_2_flat.xml, scene_4_flat.xml
│   │   ├── scene_1_step.xml          # flat + 8 step box geoms baked in
│   │   ├── scene_2_step.xml, scene_4_step.xml
│   │   ├── scene_1_rough.xml         # flat + hfield + <size memory="128M"/>
│   │   ├── scene_2_rough.xml, scene_4_rough.xml
│   │   └── rough_heightmap.png       # pre-generated (seed=42)
│   └── wheel/
│       ├── assets/                   # copied from mujoco_refactor/wheel_milli_quad/assets/
│       ├── robot_wheel.xml           # unchanged
│       ├── scene_wheel_flat.xml
│       ├── scene_wheel_step.xml
│       ├── scene_wheel_rough.xml
│       └── rough_heightmap.png
│
├── config.py              # shared: 16-dim space, sim constants, param conversion
├── config_flat.py         # flat: REFERENCE_DATA, cost fn, MJCF_PATHS, CMAES_X0
├── config_step.py         # step: REFERENCE_DATA, cost fn, step geometry, CMAES_X0
├── config_rough.py        # rough: REFERENCE_DATA, cost fn, terrain params, CMAES_X0
├── simulation.py          # unified sim engine (new + rough merged)
├── optimizer.py           # CMA-ES loop (--terrain flat|step|rough)
├── show_bests.py          # result display
├── generate_terrain_xmls.py  # one-time: creates step/rough XMLs + heightmap PNGs
└── results/
```

**Robot grouping rationale**: scene1/2/4 share the same mesh assets — grouping under `quad/` avoids 3× asset duplication. MuJoCo's `meshdir="assets"` relative path still works.

## Files to Create (13 files)

### 1. Flat scene XMLs (4 files)

Copy each `scene_X.xml` from mujoco_refactor, inject:
```xml
<option integrator="RK4">
    <flag energy="enable"/>
</option>
```
This is the ONLY change from the original scene files.

### 2. `generate_terrain_xmls.py`

One-time script. Reads flat scene XMLs, generates terrain variants.

**Step terrain** — reuses logic from `mujoco_refactor/optimizer_step.py::_inject_steps()`:
- 8 box geoms: 1mm high, 4.5mm long, 50mm flat lead, 20mm final platform, 100mm wide
- Same STEP_PRESET constants as `mujoco_refactor/config_step.py`

**Rough terrain** — reuses logic from `mujoco_refactor/optimizer_rough.py::_inject_rough_terrain()`:
- Heightmap via `utils/terrain_mesh.py::generate_heightmap(nX=10, nY=6, std=1mm, seed=42)`
- Tiles 3× along X, upsamples 8×, saves as uint16 PNG
- Adds `<hfield>`, `<geom type="hfield">`, `<size memory="128M"/>`

All generated XMLs already have `integrator="RK4"` from the flat base.

### 3. `config.py` — Shared base

**Source**: `mujoco_refactor/config_new.py`

Contains (all unchanged values):
- `PACKAGE_DIR`
- Sim constants: `SIM_TIMESTEP=1/2000`, `SETTLE_TIME=0.1`, `INITIAL_Z_HEIGHT`, `INITIAL_QUATERNION`, `INITIAL_LEG_ANGLES`, `LEG_BODY_OFFSET`, `STUCK_*`
- Physics: `MAGNETIC_MOMENT`, `MAGNETIC_FIELD_MAGNITUDE`, `MU0_OVER_4PI`, `R_EPS`
- Video: `VIDEO_FRAMERATE`, `VIDEO_WIDTH/HEIGHT`, `CAMERA_DISTANCE_*`
- 16-dim `space` list (all bounds identical)
- Functions: `point_to_params()`, `sim_params_from_point()`, `_make_ref_id()`
- Helper generators: `reference_rows(ref_data)`, `reference_ids(ref_data)`, `csv_fieldnames(ref_data, mjcf_paths)`

**NOT** in config.py (terrain-specific): REFERENCE_DATA, MJCF_PATHS, CMAES_X0, cost weights, calculate_cost, N_CALLS, SIM_DURATION

### 4. `config_flat.py`

**Sources**: flat parts of `config_new.py` + `calculate_cost` from `optimizer_new.py`

- `MJCF_PATHS` → `robots/quad/scene_X_flat.xml`, `robots/wheel/scene_wheel_flat.xml`
- `REFERENCE_DATA` — 15 rows (identical to config_new.py)
- Cost weights: VEL=5, TMB=1, LAT=5, YAW=1, VAR=2, TUMBLE_THRESHOLD=0.0
- `calculate_cost()` — time-gated (settle_time), no spatial gating
- Jitter: yaw ±2°, 3 trials, aggregation=**median**
- `CMAES_X0` = best from `20260225T122342_flat_10_30_50`
- `CMAES_SIGMA0=0.15`, `N_CALLS=4800`, `SIM_DURATION=3.0`

### 5. `config_step.py`

**Sources**: `mujoco_refactor/config_step.py` + `calculate_cost` from `optimizer_step.py`

- `MJCF_PATHS` → `robots/quad/scene_X_step.xml`, `robots/wheel/scene_wheel_step.xml`
- `REFERENCE_DATA` — 11 rows (scene1/2/4 × f10/f20/f30 + wheel f10/f20/f30)
- Cost weights: VEL=5, TMB=1, LAT=1, YAW=0, PROGRESS=2
- `calculate_cost()` — spatial-gated (STEP_START_X), progress penalty
- `STEP_START_X=0.05`, `STEP_END_X=0.1015`
- Jitter: yaw ±2°, 3 trials, aggregation=**best** (argmin)
- `CMAES_X0` = best from `20260225T225248_step_argmin_progress`
- `CMAES_SIGMA0=0.5`, `N_CALLS=4800`, `SIM_DURATION=5.0`

### 6. `config_rough.py`

**Sources**: `mujoco_refactor/config_rough.py` + `calculate_cost` from `optimizer_rough.py`

- `MJCF_PATHS` → `robots/quad/scene_X_rough.xml` (no wheel — 40% success)
- `REFERENCE_DATA` — 7 rows (scene1/2/4 × f10/f30 + scene2 f50)
- Cost weights: VEL=5, TMB=2, LAT=5, YAW=1, TUMBLE_THRESHOLD=0.17
- `calculate_cost()` — time-gated
- Spawn: `SPAWN_OFFSET` computed from terrain geometry, `Y_JITTER=0.003`
- Jitter: Y-offset ±3mm, 3 trials, aggregation=**median**
- `CMAES_X0` = best from `zzz_rough_v2`
- `CMAES_SIGMA0=0.3`, `N_CALLS=4800`, `SIM_DURATION=2.0`

### 7. `simulation.py` — Unified engine

**Sources**: `simulation_fast_new.py` + `spawn_offset` from `simulation_fast_rough.py`

- Superset: supports both `init_yaw_jitter_deg` AND `spawn_offset`
- Imports from `config` (not `config_new`)
- All physics functions unchanged: `_apply_magnetic_forces`, `_compute_external_torques`, `_compute_interjoint_torques`, `_record_state`, `_check_instability`, video recording
- The ONLY additions vs simulation_fast_new.py:
  - `spawn_offset` parameter in `run_simulation()` and `_initialize_pose()`
  - `_initialize_pose` applies spawn_offset AFTER standard init

### 8. `optimizer.py` — Unified CMA-ES

**Source**: `mujoco_refactor/optimizer_new.py` with terrain dispatch

- New CLI arg: `--terrain flat|step|rough` (required)
- Dynamic import: `config_mod = importlib.import_module(f"config_{args.terrain}")`
- Reads from config_mod: `REFERENCE_DATA`, `MJCF_PATHS`, `calculate_cost`, cost weights, jitter config, aggregation method, `CMAES_X0`, `CMAES_SIGMA0`, `N_CALLS`, `SIM_DURATION`
- All existing CLI args preserved: `--suffix`, `--scenes`, `--freqs`, `--n-calls`, `--warm-start-from`, `--resume-from`
- Worker dispatches jitter type based on config:
  - flat/step: `run_simulation(init_yaw_jitter_deg=..., rng_seed=...)`
  - rough: `run_simulation(spawn_offset=...)`
- Aggregation dispatches based on config:
  - flat/rough: median
  - step: argmin (best trial)
- CMA-ES loop, CSV output, video recording — all identical

### 9. `show_bests.py`

**Source**: `mujoco_refactor/show_bests.py`, updated imports.

## Implementation Order

1. Create directory structure + copy robot files + assets
2. Create flat scene XMLs (inject RK4 option into copies)
3. Write + run `generate_terrain_xmls.py` (creates step/rough XMLs + heightmaps)
4. Write `config.py` (shared base)
5. Write `config_flat.py`, `config_step.py`, `config_rough.py`
6. Write `simulation.py` (merge new + rough)
7. Write `optimizer.py` (terrain dispatch)
8. Write `show_bests.py`
9. Smoke test: `--terrain flat --n-calls 1`

## Usage

```bash
cd milliquad_opt

# Flat (warm-started from Euler flat best)
uv run python optimizer.py --terrain flat --suffix rk4_flat

# Step (warm-started from Euler step best)
uv run python optimizer.py --terrain step --suffix rk4_step

# Rough (warm-started from Euler rough best)
uv run python optimizer.py --terrain rough --suffix rk4_rough
```

## Verification

1. `generate_terrain_xmls.py` succeeds — all XMLs load in MuJoCo
2. Quick Python check: load a generated XML, verify `model.opt.integrator == 4` (RK4)
3. `--n-calls 1` smoke test per terrain — CSV output, no crashes
4. Single-point eval comparison: same params, same terrain, euler vs rk4 — verify trajectories differ (confirming RK4 is active)

## What does NOT change

- 16-dim search space (identical bounds)
- Magnetic torque physics (external + inter-joint dipole coupling)
- Contact model (condim=6, override flags, o_solref/solimp/friction)
- Cost function logic per terrain (moved to config, not modified)
- CMA-ES mechanics (batch ask/tell, CSV format)
- All reference data (target velocities, weights, speed_std)
- Jitter strategies per terrain
- Aggregation methods (median for flat/rough, argmin for step)
- Step/rough terrain geometry (8 steps 1mm, seed=42 heightmap)
