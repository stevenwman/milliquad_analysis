# Agent Handoff: Custom Velocity-Dependent Damping

## What We're Doing
Replacing MuJoCo's built-in `dof_damping` (purely viscous: `τ = -c * ω`) with a velocity-dependent damping model where the **coefficient itself** scales with velocity:

```
c(ω) = a * |ω| + b
τ = -(a * |ω| + b) * ω
```

- `a` = `damp_quad`: quadratic damping coeff (dominates at high freq/speed)
- `b` = `damp_linear`: linear viscous coeff (same as old `dof_damping`)

Applied to the 4 leg hinge DOFs via `data.qfrc_applied[-4:]` using joint velocities `data.qvel[-4:]`.

**Why:** Optimization shows structural frequency-dependent velocity bias (+5% at 10Hz, -22% at 50Hz). A velocity-dependent damping coefficient can capture this — pure viscous damping cannot.

## Current State of Code

### Changes already made (both files modified, not yet working):

**`config.py`** — DONE:
- Removed `dof_damping` from search `space`
- Added `damp_quad` [1e-12, 1e-6] and `damp_linear` [1e-11, 1e-7] (log-uniform)
- Updated `sim_params_from_point()` to emit `damp_quad` and `damp_linear` instead of `dof_damping`

**`simulation_fast.py`** — DONE but has a bug:
- Added `_apply_custom_damping(data, damp_quad)` function (line ~268)
- `model.dof_damping[-4:]` set to `params['damp_linear']` (implicit solver handles linear part)
- `_do_simulation_step()` signature updated with `damp_quad` param
- Both call sites updated (viewer ~567, headless ~590)
- `default_params` updated in `__main__` block
- Added `data.warning.number[:] = 0` reset before `mj_step` (line ~407)

**`simulation_fast_backup.py`** — backup of the working pre-change version.

### The Bug

`run_simulation()` fails with `mjWARN_BADQACC` around step 210. The warning fires in the **base body** (contact acceleration spike during initial drop, qacc_max=1550), NOT in the leg DOFs. The custom damping forces are negligible at that point (~1e-39).

**Key findings from debugging:**
1. Backup code (`simulation_fast_backup.py`) with identical model params works perfectly (6000 steps, 0.264 m/s)
2. Manual step-by-step test with the new code works fine (no warnings for 300+ steps)
3. `run_simulation()` with the new code fails — `_check_instability` catches a cumulative `mjWARN_BADQACC`
4. Writing exact zeros to `qfrc_applied[-4:]` does NOT cause the issue
5. Writing tiny non-zero values (1e-39) to `qfrc_applied[-4:]` DOES trigger it
6. MuJoCo's `data.warning.number` is **cumulative** — once set, never auto-resets

**Attempted fix:** Added `data.warning.number[:] = 0` before `mj_step` to reset per-step. This has NOT been tested yet (conversation ran out of context right before the test).

**Root cause hypothesis:** Writing any non-zero value to `qfrc_applied` (even subnormal floats like 1e-39) changes how MuJoCo's constraint solver handles the contact, triggering `mjWARN_BADQACC` on a transient contact spike that doesn't happen with pure `dof_damping`. The warning counter then persists and `_check_instability` aborts.

### What to Do Next

1. **Test the warning reset fix** — run the simulation headless:
```bash
cd mujoco_refactor && uv run python -c "
from simulation_fast import run_simulation
from config import MAGNETIC_MOMENT, MAGNETIC_FIELD_MAGNITUDE, SETTLE_TIME
m_fudge = 0.4; b_fudge = 1.3606385
m_mag = MAGNETIC_MOMENT * m_fudge
kp_mag = m_mag * MAGNETIC_FIELD_MAGNITUDE * b_fudge
params = {
    'ground_friction': [0.01, 0.0178, 0.00156],
    'damp_quad': 1e-9, 'damp_linear': 9.24e-10,
    'solref': [0.001, 1.607],
    'solimp': [0.9, 0.969, 0.001, 0.283, 2.0],
    'kp_mag': kp_mag, 'drive_freq': 30.0,
    'mag_params': {'m_mag': m_mag},
}
traj = run_simulation(params, sim_duration=3.0, visualize=False)
if traj:
    start = next(s for s in traj if s['time'] >= SETTLE_TIME)
    final = traj[-1]
    dt = final['time'] - start['time']
    vel = (final['pos'][0] - start['pos'][0]) / dt if dt > 1e-6 else 0
    print(f'OK - {len(traj)} steps, velocity={vel:.4f} m/s (target ~0.2747)')
else:
    print('FAILED')
"
```

2. If warning reset doesn't work, **alternative approach**: skip `qfrc_applied` when damping force magnitude is below a threshold (e.g., 1e-20) to avoid subnormal float issues.

3. Once simulation runs, verify the optimizer works end-to-end with a short run (~50 evals).

## File Map
- `config.py` — search space, constants, `sim_params_from_point()` (MODIFIED)
- `simulation_fast.py` — simulation engine with custom damping (MODIFIED, has bug)
- `simulation_fast_backup.py` — pre-change backup (working reference)
- `optimizer.py` — Bayesian optimization loop (unchanged for this feature, but was modified earlier for relative velocity error + jitter seed fix)
- `simulation.py` — reference implementation, DO NOT MODIFY

## Important Constraints
- User prefers no fallback logic / magic defaults — params explicit and required
- GUI-based testing blocked (WSL2) — always test headless
- `simulation.py` is the reference implementation — do not touch it
- The backup sim can always be restored: `cp simulation_fast_backup.py simulation_fast.py`
