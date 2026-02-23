# Plan: Step Terrain Optimizer

## Context

We have a flat-terrain optimizer (`optimizer.py` + `config.py`) that fits 13 contact/friction/magnetic params to match experimental flat-surface velocities across 4 morphologies and 3-4 frequencies (15 references). The best flat-terrain params (`results/20260219T142207_loose_fudge`) generalize poorly to step terrain: **62% mean absolute velocity error** against experimental step data (from `STEP_TERRAIN_VALIDATION.md`).

The goal is to build a **separate** step-terrain optimizer that fits the same 13 params to experimental step-terrain velocities. This is decoupled from the flat optimizer so we can explore whether step-specific contact params exist, how much they diverge from flat-best, and whether overfitting occurs. The flat optimizer is not modified.

## Experimental Step Terrain Data

Source: `experimental_data/csv/steps/` processed by `mujoco_refactor/analysis/analyze_step_terrain.py`.
Velocity extraction: **forward-only `vx`, q75-300 window** (75% index +/- 150 timesteps, clamped to recording bounds), average of two tracking points (mass_A, mass_C), negated (camera convention), m/s units.

Why q75-300: The 75% index captures later, more steady-state locomotion compared to mid-300 (50% index). For scene1_f30, mid-300 had one trial with negative velocity at midpoint (std nearly equal to mean), while q75-300 reduces std from 24.8 to 6.6 mm/s. For short recordings, the window is clamped to bounds — end-of-recording sanity check confirmed no cliff-fall or anomalous behavior (vx stays positive, height increases consistent with step climbing). See `STEP_TERRAIN_VALIDATION.md` for the full method comparison.

10 conditions with experimental data (3 trials each, averaged):

| config | exp_vel (m/s) | exp_std (m/s) | notes |
|---|---:|---:|---|
| scene1_f10 | 0.0199 | 0.0018 | 1-leg, 10Hz |
| scene1_f20 | 0.0473 | 0.0106 | 1-leg, 20Hz |
| scene1_f30 | 0.0331 | 0.0066 | 1-leg, 30Hz |
| scene2_f10 | 0.0542 | 0.0105 | 2-leg, 10Hz |
| scene2_f20 | 0.0894 | 0.0275 | 2-leg, 20Hz |
| scene2_f30 | 0.1335 | 0.0129 | 2-leg, 30Hz |
| scene4_f10 | 0.0716 | 0.0074 | 4-leg, 10Hz |
| scene4_f20 | 0.1038 | 0.0120 | 4-leg, 20Hz |
| scene4_f30 | 0.0898 | 0.0202 | 4-leg, 30Hz |
| wheel_f30 | 0.0938 | 0.0097 | wheel, 30Hz only |

No wheel data at 10Hz or 20Hz. These are the **only** targets.

## Step Terrain Geometry

From `terrain_config.py` preset `step_default`:
- 8 steps, each 1mm high, 4.5mm long
- 100mm wide (effectively infinite for ~6mm robot)
- 20mm extended platform after last step (prevents cliff-fall artifact)
- **50mm flat lead** before first step (uniform for all morphologies)

One step preset for all scenes. 4 step XMLs total (scene1, scene2, scene4, scene_wheel), all with the same geometry.

**Step field x-range**: `flat_lead` (0.05m) to `flat_lead + 7*step_length + final_step_length` = 0.05 + 0.0315 + 0.02 = **0.1015m**. This is used by the step-aware cost function to measure on-step velocity.

## Cost Function Changes (vs flat optimizer)

The flat optimizer's `calculate_cost()` uses global average velocity: `(final_pos[0] - start_pos[0]) / active_duration`. This includes the 50mm flat lead where the robot accelerates freely, inflating the velocity relative to experimental on-step measurements.

**Step-aware cost function**: Measure forward velocity **only in the step region** (x > flat_lead):
1. Find first trajectory state where `pos[0] >= flat_lead` and `time >= SETTLE_TIME` (robot enters step field)
2. Use the last trajectory state as the endpoint
3. Compute `avg_step_velocity = (pos_exit[0] - pos_enter[0]) / (t_exit - t_enter)`
4. If the robot never reaches the step field, return `COST_FAILURE`

**Kept**: tumble penalty, lateral displacement, yaw spin-out, velocity variance
**Removed**: pitch RMS (weight=0, remove entirely from the code)

## Architecture: Files to Create

### 1. `mujoco_refactor/config_step.py` — Step-specific configuration
Thin overlay on `config.py`:
- `REFERENCE_DATA` with mid-300 vx targets (10 rows above)
- Same 13-param search space (imported from `config.py`)
- `VELOCITY_DEADZONE = False`
- `CMAES_X0` warm-started from no-deadzone best params
- `SIM_DURATION = 5.0` (longer than flat's 3.0s)
- `N_CALLS = 2400`
- Single `STEP_PRESET` dict with step geometry (50mm flat_lead for all)
- `reference_rows()`, `reference_ids()`, `csv_fieldnames()` defined here

Import `space`, `sim_params_from_point`, `point_to_params` from `config.py`.

### 2. `mujoco_refactor/optimizer_step.py` — Step terrain optimizer
Copy of `optimizer.py` with:
- Imports from `config_step` instead of `config`
- **Step-aware `calculate_cost()`**: velocity measured only in the step region (x >= flat_lead)
- No pitch RMS computation
- Pre-builds step XMLs at startup using `_inject_steps()` (copied from `terrain_test.py`)
- XML filenames encode step specs: `scene_1_step_8x1mm_4p5L_50lead.xml`
- Passes `mjcf_path` through the task tuple to workers
- Copies both `config_step.py` and `config.py` into results dir

### 3. `mujoco_refactor/show_bests_step.py` — Results viewer
Copy of `show_bests.py` with import changed from `config` to `config_step`.

### 4. No other files modified
- `simulation_fast.py` — reused as-is (takes any `mjcf_path`)
- `show_bests.py` — unchanged
- `terrain_test.py` — unchanged (`_inject_steps` logic copied, not imported)
- `config.py` — unchanged
- `optimizer.py` — unchanged

## Detailed Implementation

### Step-aware calculate_cost()

```python
def calculate_cost(
    trajectory: list[dict],
    target_velocity: float,
    speed_std: float = 0.0,
    step_start_x: float = 0.0,
    verbose: bool = True,
) -> dict[str, float]:
    if not trajectory:
        return {"total_cost": COST_FAILURE, "avg_forward_velocity": 0, ...}

    # Find when robot enters step field (x >= step_start_x AND t >= SETTLE_TIME)
    enter_state = None
    for state in trajectory:
        if state["time"] >= SETTLE_TIME and state["pos"][0] >= step_start_x:
            enter_state = state
            break

    if enter_state is None:
        # Robot never reached the steps
        return {"total_cost": COST_FAILURE, "avg_forward_velocity": 0, ...}

    final_state = trajectory[-1]
    active_duration = final_state["time"] - enter_state["time"]
    if active_duration > 1e-6:
        forward_displacement = final_state["pos"][0] - enter_state["pos"][0]
        avg_forward_velocity = forward_displacement / active_duration
    else:
        avg_forward_velocity = 0.0

    # Velocity error (relative squared, no deadzone)
    vel_deviation = avg_forward_velocity - target_velocity
    velocity_error = (vel_deviation / target_velocity) ** 2

    # Tumble, lateral, yaw — same as flat optimizer but measured from enter_state
    # ... (identical logic, just using enter_state instead of start_state)

    total_cost = (
        VELOCITY_COST_WEIGHT * velocity_error
        + TUMBLE_COST_WEIGHT * tumble_penalty
        + LATERAL_COST_WEIGHT * lateral_error
        + YAW_COST_WEIGHT * yaw_penalty
    )
    # No pitch RMS term
```

The `step_start_x` parameter comes from `STEP_PRESET["flat_lead"]` and is passed in from the worker.

### Worker function changes

```python
def _evaluate_one_scene(args):
    point_index, point, ref_row, trial_index, show_progress, global_point_index, mjcf_path = args
    # ... same as flat optimizer but:
    # 1. Use mjcf_path directly (passed in tuple, not from global dict)
    # 2. Pass step_start_x to calculate_cost
    cost_data = calculate_cost(
        trajectory, target_velocity,
        speed_std=speed_std,
        step_start_x=STEP_START_X,  # from config_step.STEP_PRESET["flat_lead"]
        verbose=False,
    )
```

### XML generation and naming

```python
# In __main__, before Pool creation:
h_mm = STEP_PRESET["step_height"] * 1000
l_mm = STEP_PRESET["step_length"] * 1000
n = STEP_PRESET["step_count"]
lead_mm = STEP_PRESET["flat_lead"] * 1000
step_tag = f"step_{n}x{h_mm:.0f}mm_{l_mm:.1f}L_{lead_mm:.0f}lead"
# => "step_8x1mm_4.5L_50lead"

for scene, base_xml in BASE_MJCF_PATHS.items():
    src_dir = pathlib.Path(base_xml).parent
    stem = pathlib.Path(base_xml).stem
    out_xml = str(src_dir / f"{stem}_{step_tag}.xml")
    _inject_steps(base_xml, STEP_PRESET, out_xml)
    MJCF_STEP_PATHS[scene] = out_xml
```

XMLs written alongside originals (same dir = correct relative path resolution).
Descriptive names prevent collision; no cleanup needed.

### Multiprocessing: pass mjcf_path in task tuple

Workers re-import module and see empty `MJCF_STEP_PATHS`. Fix: pass path directly.

```python
tasks = [
    (i, point, ref_row, trial_idx, False, n_done + i,
     MJCF_STEP_PATHS[ref_row["scene"]])
    for i, point in enumerate(points)
    for ref_row in _REF_ROWS
    for trial_idx in range(n_trials)
]
```

### Config snapshot

Copy both configs into results dir:
```python
shutil.copy2(pathlib.Path(__file__).parent / "config_step.py", run_dir_results / "config_step.py")
shutil.copy2(pathlib.Path(__file__).parent / "config.py", run_dir_results / "config.py")
```

## What Could Go Wrong (Caveats)

### 1. Sim duration
With 50mm flat lead + ~44mm of steps, robot must travel ~94mm. At ~16mm/s (scene1_f10, slowest mid-300 target), that takes ~6s. With settle time and acceleration, **5.0s may be tight**. If scene1_f10 returns COST_FAILURE (never reaches steps), increase to 7.0s.

### 2. Robot never reaching step field
With step_start_x = 0.05m, slow morphologies (scene1_f10) may not reach x=0.05 within 5s if contact params are bad. The cost function returns COST_FAILURE in this case, which is correct behavior — CMA-ES will explore away from these params.

### 3. Wheel behavior
The flat-fit sim wheel gets stuck on 1mm steps (13.8mm/s). The optimizer should find params that unstick it, but early iterations may have many failures for wheel_f30.

### 4. MJCF relative paths
Generated XMLs MUST be in the same directory as the original scene XMLs (MuJoCo resolves relative paths from XML location).

### 5. scene1_f30 noise
Mid-300 target for scene1_f30 is 0.0241 m/s with std 0.0248 (one trial went backwards at -2.5 mm/s). This condition may be inherently chaotic. Consider giving it lower weight or dropping it if it destabilizes optimization.

## Files Summary

| File | Action | Description |
|---|---|---|
| `mujoco_refactor/config_step.py` | **CREATE** | Step config: 10 refs (mid-300 vx), step geometry, warm-start X0 |
| `mujoco_refactor/optimizer_step.py` | **CREATE** | Step optimizer: step-aware cost, CMA-ES loop, builds step XMLs |
| `mujoco_refactor/show_bests_step.py` | **CREATE** | Copy of show_bests.py importing from config_step |
| `mujoco_refactor/simulation_fast.py` | unchanged | Reused as-is |
| `mujoco_refactor/config.py` | unchanged | Flat config stays untouched |
| `mujoco_refactor/optimizer.py` | unchanged | Flat optimizer stays untouched |
| `mujoco_refactor/terrain_test.py` | unchanged | `_inject_steps` logic copied (not imported) |

## Usage

```bash
cd mujoco_refactor

# Full run (all 10 refs, warm-started from no-deadzone best)
uv run python optimizer_step.py --suffix step_v1

# Quick test (1 morphology, small budget)
uv run python optimizer_step.py --scenes scene4 --n-calls 16 --suffix step_smoke

# View results
uv run python show_bests_step.py results/YYYYMMDDTHHMMSS_step_v1/optimization_bests.csv
```

## Verification

1. **Smoke test**: `--scenes scene4 --n-calls 16 --suffix step_smoke` — 2 batches, check for non-1e6 costs
2. **XML check**: Verify `multi_milli_quad/scene_4_step_8x1mm_4.5L_50lead.xml` exists during the run
3. **Cost sanity**: First batch costs in 0.1–10 range (not 1e6 failures)
4. **Velocity check**: `show_bests_step.py` shows sim velocities in 10–200 mm/s range
5. **Step-aware velocity**: Verify that reported velocities exclude flat-lead acceleration (should be lower than global-average equivalent)
6. **Full run**: 2400 evals, expect 3-5 hours
7. **Post-run**: Compare step-optimized params with flat-optimized (especially sliding_friction, moment_fudge, dof_damping)
