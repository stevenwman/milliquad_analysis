# COT & Energy Computation Notes

## Power Formula

### Wrong (v1, eval_cot.py)
```
P = sum_j dot(tau_ext[j], data.cvel[leg_body_j, :3])
```
`cvel` is the **total** body angular velocity in world frame = base rotation + joint rotation. The `tau · omega_base` cross-terms are spurious and can flip the sign of the energy integral.

### Correct (v2, eval_cot_v2.py)
```
P = sum_j dot(tau_ext[j], axis_world[j]) * joint_vel[j]
```
- `tau_ext[j]`: world-frame torque on leg body j (from `kp_mag * cross(north, goal)`)
- `axis_world[j]`: joint axis in world frame (rotate body-local `[0,0,1]` by `leg_xquat`)
- `joint_vel[j]`: scalar joint angular velocity from `data.qvel[6+j]` (rad/s)

For a 1-DOF hinge, only the torque component along the joint axis does work. The rest is absorbed by the joint constraint.

### Verification (verify_power.py)
Compared against MuJoCo's rotational Jacobian (`mj_jacBody`):
- `jacr[:, dof_j]` = joint axis in world frame (MuJoCo's ground truth)
- `J^T @ tau_ext` = generalized joint torque (standard robotics formula)
- **Result**: NEW matches Jacobian to machine precision at every timestep. OLD does not.
- Signed energy: OLD = -498 uJ (wrong sign), NEW = +804 uJ, JAC = +804 uJ

## Joint Axes
All 4 hinge joints have `axis="0 0 1"` in body-local frame (from MJCF). The world-frame axis is the 3rd column of each leg body's rotation matrix, computed from stored `leg_xquat`.

## COT Formula
```
COT = E_ext / (m * g * d)
E_ext = integral(P_ext * dt)    # signed — drive field does net positive work
d = cumulative 2D path length   # sum(||pos[i+1][:2] - pos[i][:2]||)
m = sum(model.body_mass)        # total robot mass from MJCF
```

## Signed vs Absolute Energy
- **Signed** (`E_signed = integral(P dt)`): net energy into joints. Should be positive (drive field does work). This is the physically meaningful quantity.
- **Absolute** (`E_abs = integral(|P| dt)`): counts both driving and braking phases. ~30-40% higher than signed. Inflated by oscillation.
- v2 reports both; COT uses signed.

## Typical COT Values (flat params, flat terrain)
| Morphology | COT range | Pattern |
|------------|-----------|---------|
| 1-leg      | 6.6 – 11.7 | Highest; drops from f10 to f20 then flat |
| 2-leg      | 2.2 – 3.0  | ~3x more efficient than 1-leg |
| 4-leg      | 1.0 – 1.8  | Most efficient legged |
| wheel      | 0.5 – 1.1  | Best overall |

## Files
- `eval_cot.py`: v1 (wrong power, uses cvel). Kept for reference.
- `eval_cot_v2.py`: v2 (correct power, uses joint projection). Outputs `cot_results_v2.csv`.
- `verify_power.py`: Jacobian-based verification script.
- `plot_cot.py`: Reads both v1 (`cot` column) and v2 (`cot_signed` column) CSVs.
- `visualize_rollout.py`: Has same v1 bug in `compute_locomotion_metrics()` — not fixed (separate legacy file).

## Data
- Trajectory stores `tau_ext` (4x3), `leg_xquat` (4x4), `joint_vel` (4,) — all needed for correct power. No simulation code changes required.
- `omega` in trajectory (`data.cvel[:3]`) is kept for backward compatibility but should NOT be used for power.
