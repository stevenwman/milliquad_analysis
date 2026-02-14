# MuJoCo Milliquad Simulation & Optimization

MuJoCo-based simulation of LEGO-scale magnetically actuated milliquad robots, with Bayesian optimization of contact/friction parameters to match experimentally observed locomotion velocities.

## Architecture

```
config.py                 ← Single source of truth: constants, search space, reference data
    ↓
simulation.py             ← Reference simulation engine (scipy-based rotations)
simulation_fast.py        ← Optimized simulation (~5x faster, equivalent physics)
    ↓
optimizer.py              ← Batch Bayesian optimization (scikit-optimize)
    ↓
multi_optimization_results.csv    ← All evaluated points
optimization_bests.csv            ← Running best log (reset per run)
```

## Files

| File                   | Purpose                                                                        |
| ---------------------- | ------------------------------------------------------------------------------ |
| `config.py`            | Shared constants, 13D search space, reference data, parameter conversion       |
| `simulation.py`        | Original simulation engine using scipy `Rotation` objects                      |
| `simulation_fast.py`   | Vectorized simulation — inline quaternion math, batched magnet states          |
| `optimizer.py`         | Batch Bayesian optimization with parallel multi-scene evaluation               |
| `replay.py`            | Replay a specific CSV result in the MuJoCo interactive viewer                  |
| `visualize_rollout.py` | Visualize/record rollouts with COT (Cost of Transport) analysis                |
| `compare_cot.py`       | Compare COT across top-N results for all scenes                                |
| `verify_sim.py`        | Verify `simulation_fast.py` produces identical trajectories to `simulation.py` |
| `show_bests.py`        | Pretty-print `optimization_bests.csv` as readable tables                       |

## Physics

The robot is a planar-walking milliquad with 4 magnetized legs. Locomotion is driven by:

1. **External rotating field** — each leg magnet experiences τ = kp × (north × goal), where the goal direction rotates at `drive_freq` Hz around the y-axis.
2. **Inter-joint dipole coupling** — pairwise magnetic dipole-dipole torques τ = m × B between all 4 legs (μ₀/4π model).

The MuJoCo model uses `xfrc_applied` to inject these torques at each timestep.

### Instability Handling

MuJoCo's C-level "Nan/Inf in QACC" warnings are suppressed via `set_mju_user_warning(lambda msg: None)`. The same conditions are detected programmatically in `_check_instability()` (checking `data.qacc`, `solver_niter`, `solver_fwdinv`, and `data.warning.number`) and result in a `COST_FAILURE` penalty.

## Optimization

The optimizer tunes 13 contact/friction parameters to minimize velocity error across multiple robot configurations and drive frequencies:

- **Scenes**: 1-leg (`scene1`), 2-leg (`scene2`), 4-leg (`scene4`), wheel (`scene_wheel`)
- **Frequencies**: 10, 30, 50 Hz per scene
- **Cost function**: weighted sum of velocity error², tumble penalty, lateral drift², pitch RMS, and cross-reference velocity variance

### Search Space (13D)

| Parameter               | Range        | Scale  |
| ----------------------- | ------------ | ------ |
| `sliding_friction`      | 1e-5 – 0.8   | log    |
| `torsional_friction`    | 1e-5 – 0.1   | log    |
| `rolling_friction`      | 1e-5 – 0.1   | log    |
| `solref_timeconst`      | 0.001 – 0.1  | linear |
| `solref_dampratio`      | 0.1 – 2.0    | linear |
| `solimp_dmin`           | 0.8 – 0.99   | linear |
| `solimp_dmax`           | 0.95 – 0.999 | linear |
| `solimp_width`          | 1e-4 – 1e-2  | log    |
| `solimp_midpoint`       | 0.1 – 0.9    | linear |
| `solimp_power`          | 1.0 – 6.0    | linear |
| `magnetic_moment_fudge` | 0.5 – 1.5    | linear |
| `magnetic_field_fudge`  | 0.5 – 1.5    | linear |
| `dof_damping`           | 7e-11 – 7e-9 | log    |

## Quick Start

```bash
cd mujoco_refactor

# Run optimization (200 evals, batches of 8)
uv run python optimizer.py

# View running best results
uv run python show_bests.py

# Replay best result in viewer
uv run python replay.py

# Replay specific ID for one scene
uv run python replay.py f37ba9db scene4

# Visualize with COT analysis
uv run python visualize_rollout.py --rank 1 --scene scene4

# Record video
uv run python visualize_rollout.py --record rollout.mp4

# Compare COT across top-3
uv run python compare_cot.py --top 3

# Verify fast sim matches original
uv run python verify_sim.py
```

## Optimizer Tuning

All surrogate-model hyperparameters live in `config.py`:

| Constant                 | Default      | Purpose                                                     |
| ------------------------ | ------------ | ----------------------------------------------------------- |
| `BASE_ESTIMATOR`         | `"gp"`       | `"gp"` (Gaussian process) or `"rf"` (random forest)         |
| `ACQ_FUNC`               | `"gp_hedge"` | Acquisition function: `"EI"`, `"LCB"`, `"PI"`, `"gp_hedge"` |
| `ACQ_FUNC_KWARGS`        | `{}`         | e.g. `{"kappa": 1.5}` for LCB, `{"xi": 0.01}` for EI        |
| `N_INITIAL_POINTS`       | `20`         | Random exploration before surrogate kicks in                |
| `OPTIMIZER_NOISE`        | `"gaussian"` | GP noise: `"gaussian"` (auto), float (fixed σ²), or `None`  |
| `OPTIMIZER_RANDOM_STATE` | `42`         | Seed for reproducibility                                    |

### Recipes

**Good coverage first** (recommended for first run):

```python
BASE_ESTIMATOR = "gp"
ACQ_FUNC = "gp_hedge"
N_INITIAL_POINTS = 30
OPTIMIZER_NOISE = "gaussian"
```

**Refine an existing good result**:

```python
BASE_ESTIMATOR = "gp"
ACQ_FUNC = "LCB"
ACQ_FUNC_KWARGS = {"kappa": 0.5}
N_INITIAL_POINTS = 10
OPTIMIZER_NOISE = "gaussian"
```

**Escape local minima**:

```python
BASE_ESTIMATOR = "gp"
ACQ_FUNC = "LCB"
ACQ_FUNC_KWARGS = {"kappa": 3.0}
N_INITIAL_POINTS = 40
OPTIMIZER_NOISE = "gaussian"
```

**Fast throughput** (GP gets slow at n>300):

```python
BASE_ESTIMATOR = "rf"
ACQ_FUNC = "EI"
OPTIMIZER_NOISE = None
```

## Module Selector

Both `optimizer.py`, `replay.py`, `visualize_rollout.py`, and `compare_cot.py` contain a `SIM_MODULE` variable (default: `"simulation_fast"`). Change to `"simulation"` to use the original scipy-based engine for debugging or verification.
