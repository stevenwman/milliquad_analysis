# mujoco_refactor Context

## Architecture (3-file split)
- `config.py` — constants, 13-dim search space, `sim_params_from_point()` (single source of truth)
- `simulation.py` — core MuJoCo sim engine, `run_simulation()` entry point
- `optimizer.py` — batch Bayesian optimization using scikit-optimize

Self-contained: no imports from old `mujoco/` folder.

## simulation.py Physics

Two magnetic torque sources (intentionally separated for COT analysis):
1. `_compute_external_torques`: rotating drive field aligns magnets to goal direction — **this is the energy cost**
2. `_compute_interjoint_torques`: dipole-dipole coupling between legs (tau = m x B) — **conservative/internal, excluded from COT**

### omega capture timing
Angular velocity grabbed from `data.cvel[body_idx, :3]` *before* `mj_step` so it's at the same instant as torques. Intentional for power computation: P = tau . omega.

### Key parameters
- `progress=True`: enables tqdm in headless mode (not for optimizer workers)
- `ignore_stuck_detection=True`: skip early termination for stuck robots
- `benchmark=True`: print step-level timing breakdown

## Consumers
- `visualize_rollout.py` — reads CSV, runs sim, computes COT via `compute_locomotion_metrics()`
- `compare_cot.py` — compares top-N results across scenes

## Constants (config.py)
- `SETTLE_TIME = 0.1s` — wait before driving starts
- `SIM_TIMESTEP = 1/2000` — 2 kHz physics
- `MAGNETIC_MOMENT = 1.13e-3`, `MAGNETIC_FIELD_MAGNITUDE = 2e-3`
- Target velocities: scene4=21cm/s, scene2=14cm/s

## COT Formula
`COT = E_ext / (m * g * d)` where `E_ext = integral(|P_ext| dt)` — absolute value counts both driving and braking.
