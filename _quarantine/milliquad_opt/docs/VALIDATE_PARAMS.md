# validate_params.py — Design & Refactor Notes

## Purpose

Run jittered simulation trials for each reference condition using optimized
params, save raw trajectories (NPZ) and metadata (CSV). All analysis
(velocity, COT, pitch, trial selection) is deferred to downstream plotting code.

## Usage

```bash
cd milliquad_opt
uv run python -m analysis.validate_params results/<run_dir> --terrain <terrain> --csv [--record] [--n-trials N]
```

- `--terrain`: required if auto-detect fails (e.g. `flat_tg`, `step_065`, `rough_tg`)
- `--csv`: write NPZ + CSV + trajectory overview plot
- `--record`: record video of all non-crash trials
- `--n-trials`: number of jittered trials per ref (default: 5)

## Output Files

All saved in `results/<run_dir>/`:

```
<timestamp>_validation_trajectories.npz   # raw trajectory arrays
<timestamp>_validation_trials.csv          # trial metadata
<timestamp>_videos/                        # mp4s (if --record)
trajectory_overview.png                    # x-pos vs time per ref
```

## Refactor (2026-03-04)

### What was removed

- **Trial selection**: previously selected top-N trials by velocity error.
  Now `selected=True` for all trials (crash or not). Selection is a downstream
  plotting concern.
- **Velocity computation**: `extract_velocity()` no longer called. The velocity
  shown in trajectory overview is computed from NPZ arrays inline (displacement/time).
- **COT / pitch / min-window-velocity**: removed from validate. Downstream
  plotting code computes these from NPZ with terrain-appropriate windowing.
- **`--n-select` arg**: removed (no selection).
- **Console summary stats**: removed (mean error, COT range). Just prints
  trial count and crash count.

### Why

Different terrain configs use different measurement windows:

| Terrain    | Gating method                    | Window per condition    |
|------------|----------------------------------|------------------------|
| flat       | time: SETTLE_TIME to end         | full trajectory (~2.9s) |
| flat_tg    | time: SETTLE_TIME to SETTLE_TIME + trial_duration | per-condition (0.38–2.6s) |
| step/step_065 | spatial: step_start_x to gate_fraction * step_end_x | position-based |
| rough      | spatial: FLAT_LEAD to FLAT_LEAD + 2*X_HALF | position-based |

validate_params was computing velocity with a single `extract_velocity()` that
didn't know about per-condition time gating (flat_tg). Rather than making
validate_params terrain-aware, we moved all analysis downstream where the
plotting code already handles terrain-specific windowing.

### CSV schema (new)

```
ref_id, scene, ctrl_freq, target_speed, trial, rng_seed,
jitter_type, jitter_value, crash, selected
```

Removed columns: `vx`, `velocity_error_pct`, `cot`, `min_window_vx`,
`pitch_rms`, `max_x`. These are now computed from NPZ by plotting code.

`selected` is always `True` — kept for backward compatibility with downstream
code that filters on `selected == "True"`.

### Trajectory overview plot

`plot_trajectories.py` changes:
- Legend shows per-trial velocity: `t0: 143 mm/s` (computed from NPZ)
- Removed selected/unselected rendering split (all trials rendered equally)
- Removed `min_window_vx` annotation box
- Gate lines drawn per terrain:
  - **step/rough**: horizontal lines at step_start_x / step_end_x (spatial gate)
  - **flat_tg**: vertical line at SETTLE_TIME + trial_duration (time gate, per subplot)
  - **flat** (no trial_duration): no gate lines

## Jitter & Seeds

- Base seed: 99999 (different from optimizer's 12345/77777 to test generalization)
- Flat/step: yaw jitter (matching `INIT_YAW_JITTER_DEG` from config)
- Rough: Y-position jitter (matching `Y_JITTER` from config)
- Seed per trial: `BASE_SEED + ref_idx * N_TRIALS + trial_idx`

## NPZ Contents

Per trial `{ref_id}_t{trial_idx}`:
- `_time`, `_pos_x/y/z`, `_vel_x/y/z`: kinematics
- `_pitch`: pitch angle series (degrees)
- `_joint_pos`: (T, 4) joint positions
- `_drive_angle`: magnetic drive angle (if present)
- `_tau_ext`, `_omega`: external torques + angular velocity (for COT)
- `_leg_xquat`, `_joint_vel`, `_leg_xpos`: leg state (for energy)
- `_leg_in_contact`, `_leg_normal_force`, `_leg_tangent_force`, `_leg_contact_pos`: contact
- `_body_in_contact`, `_body_normal_force`, `_body_tangent_force`: chassis contact
- `_total_ncon`: total contact count

## Time Gating (flat_tg)

`config_flat_tg.py` adds `trial_duration` to each REFERENCE_DATA entry — the
mean experimental recording length. The optimizer cost function truncates
sim velocity measurement to `[SETTLE_TIME, SETTLE_TIME + trial_duration]`.

Trial durations (from `experimental_data/csv/flat/`):

| Condition      | Duration |
|----------------|----------|
| scene1_f10     | 2.625s   |
| scene1_f20     | 1.093s   |
| scene1_f30     | 1.197s   |
| scene1_f50     | 1.023s   |
| scene2_f10     | 1.567s   |
| scene2_f20     | 1.021s   |
| scene2_f30     | 0.827s   |
| scene2_f50     | 0.663s   |
| scene4_f10     | 1.245s   |
| scene4_f20     | 0.712s   |
| scene4_f30     | 0.589s   |
| scene4_f50     | 0.547s   |
| wheel_f10      | 0.965s   |
| wheel_f20      | 0.478s   |
| wheel_f30      | 0.384s   |

Wheel f50 excluded (weight=0, robot self-destructs experimentally).
