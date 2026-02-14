# mujoco_refactor — Developer Context

Read this before modifying any file. It covers the non-obvious decisions and gotchas.

## Two Simulation Engines

| File                 | Role                                             | Status                                                |
| -------------------- | ------------------------------------------------ | ----------------------------------------------------- |
| `simulation.py`      | **Reference implementation** (scipy-based)       | **DO NOT MODIFY** — kept for correctness verification |
| `simulation_fast.py` | **Production implementation** (vectorized numpy) | Active development target                             |

They share ~80% identical scaffolding (stuck detection, video recording, `run_simulation` structure). Only the magnetic force computation differs — `simulation_fast.py` uses batched numpy instead of per-leg Python loops.

### Numerical Equivalence (not identity)

The fast version produces **ULP-level floating-point differences** (~1e-16 per step) due to different operation ordering (inline quaternion rotation vs scipy matrix construction). Over 10,000 steps of chaotic contact dynamics, these **accumulate and cause trajectory divergence**. This is expected. `verify_sim.py` quantifies the divergence. The physics model is mathematically equivalent, not bit-identical.

## Data Flow: How Parameters Reach the Simulation

```
config.py: space[] (13 dimensions)
    → config.py: sim_params_from_point(point) → sim_params dict
        → caller injects sim_params["drive_freq"] (NOT in sim_params_from_point!)
            → simulation_fast.run_simulation(sim_params, ...)
```

**Key gotcha**: `drive_freq` is **not** part of the 13D search space and is **not** set by `sim_params_from_point()`. Every caller (`optimizer.py`, `replay.py`, `visualize_rollout.py`, `compare_cot.py`) must inject it manually from the reference row data.

## Instability Handling Pipeline

1. MuJoCo C-level warnings suppressed: `set_mju_user_warning(lambda msg: None)`
2. Same conditions detected in Python: `_check_instability()` checks `qacc`, `solver_niter`, `solver_fwdinv`, `warning.number`
3. Raises `ValueError("Simulation unstable: ...")` → caught in `run_simulation` → returns `None`
4. Caller interprets `None` as `COST_FAILURE = 1e6`

## Physics: Magnetic Torques

Two torque sources (intentionally separated for COT analysis):

1. **External drive** (`_compute_external_torques`): rotating field aligns magnets to goal direction. `τ = kp_mag * (north × goal)`. **This is the energy input.**
2. **Inter-joint coupling** (`_compute_interjoint_torques`): dipole-dipole between legs. `τ = m × B` with `B` from μ₀/4π model. **Conservative/internal, excluded from COT.**

### omega / cvel Convention

MuJoCo stores `data.cvel` as `[ω, v]` (angular first, then linear). The code reads `cvel[:, :3]` for angular velocity — this is correct.

omega is captured **before** `mj_step()` so it's at the same instant as the applied torques, for correct power computation `P = τ · ω`.

## Reference Data (config.py)

Target velocities come from physical experiments on the real robot:

| Scene          | 10 Hz      | 30 Hz      | 50 Hz      |
| -------------- | ---------- | ---------- | ---------- |
| scene1 (1 leg) | 0.0512 m/s | 0.1187 m/s | 0.1483 m/s |
| scene2 (2 leg) | 0.0832 m/s | 0.1796 m/s | 0.2633 m/s |
| scene4 (4 leg) | 0.1121 m/s | 0.2747 m/s | 0.3274 m/s |
| scene_wheel    | 0.1432 m/s | 0.4493 m/s | —          |

## COT Formula

`COT = E_ext / (m * g * d)` where `E_ext = ∫|P_ext| dt` — absolute value counts both driving and braking phases. Inter-joint coupling is conservative and excluded.

## Known Rough Edges

- `simulation.py` and `simulation_fast.py` share duplicated scaffolding — bug fixes to shared logic (stuck detection, video recording, etc.) must be applied to both
- `_get_all_magnet_states()` in `simulation_fast.py` returns a **view** of `data.xpos`, not a copy — safe currently (consumed before `mj_step`) but fragile
- `_quat_rotate_vec()` (single-vector version) in `simulation_fast.py` is dead code — only the batch version is used
- `show_bests.py` uses an unclosed file handle
