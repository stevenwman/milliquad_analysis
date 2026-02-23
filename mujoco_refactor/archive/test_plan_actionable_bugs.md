# Test Plan: Actionable Bug Verification

Date: 2026-02-17
Targets: Bug #2 (friction), Bug #3 (torque magnitudes), Bug #9 (frequency-blind torque)

---

## Bug #2: Floor-Only Friction

### Background

Friction is set only on the `floor` geom. Robot geoms (leg, leg_box, magnet, body) have no friction attribute and use MuJoCo defaults `[1.0, 0.005, 0.0001]`.

MuJoCo's `mjENBL_OVERRIDE` (enabled in simulation.py:477) overrides `solref` and `solimp` globally but does **not** override friction — friction is always resolved from the geom pair.

Across all optimization runs, converged `sliding_friction` values range from 1e-4 to 0.074 — all far below 1.0. If MuJoCo uses element-wise max, these values are completely inert.

### Test A: Verify MuJoCo Friction Combination Rule

**Hypothesis:** MuJoCo resolves contact friction as `max(geom1_friction, geom2_friction)` per element.

**Method:** Minimal MuJoCo script — create a model with two geoms with known friction, generate a contact, read back `data.contact[0].friction`.

```python
"""Test how MuJoCo combines friction from two contacting geoms."""
import mujoco
import numpy as np

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"
          friction="0.01 0.002 0.0003"/>
    <body pos="0 0 0.1">
      <freejoint/>
      <geom name="box" type="box" size="0.05 0.05 0.05" mass="1"
            friction="0.5 0.008 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

print(f"Floor friction:   {model.geom_friction[0]}")
print(f"Box friction:     {model.geom_friction[1]}")
print(f"Num contacts:     {data.ncon}")
if data.ncon > 0:
    print(f"Contact friction: {data.contact[0].friction}")
    # Expected if max:  [0.5, 0.008, 0.001]
    # Expected if mean: [0.255, 0.005, 0.00065]
    # Expected if geom: [sqrt(0.005), sqrt(0.000016), sqrt(0.0000003)]
```

**Expected outcome:** Contact friction = element-wise max of the two geoms.

**Implications:**
- If max: sliding friction below 1.0 is dead (confirmed by optimizer convergence data).
- If geometric mean: sliding friction has sqrt-scale influence (partial effect).
- If average: sliding friction has linear but halved influence.

### Test B: Sensitivity of Simulation to Sliding Friction

**Hypothesis:** Changing sliding friction from 0.001 to 0.999 produces no measurable change in robot velocity (because the robot default 1.0 dominates).

**Method:** Run `simulation_fast.run_simulation()` with best-known params, varying only sliding friction across [0.001, 0.01, 0.1, 0.5, 0.999, 2.0, 5.0, 10.0]. Use scene_4 at 30 Hz, 3 seconds, same seed.

**Measurements:**
- Average forward velocity
- Lateral displacement
- Tumble penalty

**Expected outcome:**
- Values 0.001–0.999: identical behavior (all masked by robot default 1.0)
- Values >1.0: potentially different behavior (now exceeding robot default)
- If no change even above 1.0: friction has ~zero effect on this system (contact-dominated by geometry)

### Test C: Effect of Setting Robot Geom Friction

**Hypothesis:** Setting friction on robot leg geoms gives the optimizer actual control over contact behavior.

**Method:** Modify simulation to also set friction on all robot leg/box geoms (not just floor). Run the same sweep as Test B but with both floor AND robot friction set identically.

```python
# After setting floor friction, also set robot geom friction:
for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name and name != "floor":
        model.geom_friction[geom_id] = params['ground_friction']
```

**Expected outcome:** Simulation velocity now responds to friction across the full search range, because both sides of the contact pair are controlled.

### Test D: Quantify Dead-Parameter Impact on Optimization

**Hypothesis:** The optimizer wastes evaluations exploring sliding friction values that don't affect the cost function.

**Method:** Analyze existing optimization CSVs:
1. Compute Spearman correlation between `sliding_friction` and `cost` across all evaluated points.
2. Compare to correlation for other parameters (e.g., `rolling_friction`, `solref_dampratio`).

**Expected outcome:** Near-zero correlation for `sliding_friction` (confirming it's a dead parameter), meaningful correlation for others.

### Case Analysis for Bug #2

| Scenario | sliding_friction | Robot default | Effective contact | Optimizer effect |
|----------|-----------------|---------------|-------------------|-----------------|
| Current (value < 1.0) | 0.01 | 1.0 | 1.0 (robot wins) | **None** |
| Current (value > 1.0) | 5.0 | 1.0 | 5.0 (floor wins) | Active |
| Fix: set both geoms | 0.01 | 0.01 | 0.01 | **Full control** |
| Fix: narrow range [1, 10] | 2.0 | 1.0 | 2.0 (floor wins) | Active but limited |

**Recommendation priority:**
1. Run Test A first (5 min) — confirms the combination rule definitively
2. Run Test B (10 min) — confirms the dead-parameter hypothesis empirically
3. If confirmed, either narrow the search range OR set both geom frictions (Test C)

---

## Bug #3: Inter-Joint vs External Torque Magnitudes

### Background

The predecessor claims inter-joint torque is ~2x the external drive torque. This has implications for how `moment_fudge` vs `field_fudge` affect the system and whether the inter-joint model is correct.

### Test E: Log Torque Magnitudes During Simulation

**Hypothesis:** |τ_int| / |τ_ext| ≈ 1.5–2.0 at fudge=1.

**Method:** Run a single simulation with best-known params, extract `tau_ext` and `tau_int` from trajectory (already recorded in `step_cache`), compute per-leg torque magnitudes.

```python
from simulation_fast import run_simulation
from config import sim_params_from_point, space, SETTLE_TIME
import numpy as np

# Use midpoint or best-known params
params = { ... }  # best params from optimization
sim_params = sim_params_from_point([params[d.name] for d in space])
sim_params["drive_freq"] = 30.0

traj = run_simulation(sim_params, sim_duration=3.0, visualize=False)

# Extract torque magnitudes after settle
tau_ext_mags = []
tau_int_mags = []
for state in traj:
    if state["time"] < SETTLE_TIME:
        continue
    if "tau_ext" in state and "tau_int" in state:
        tau_ext_mags.append(np.linalg.norm(state["tau_ext"], axis=1))  # (4,)
        tau_int_mags.append(np.linalg.norm(state["tau_int"], axis=1))  # (4,)

tau_ext_mags = np.array(tau_ext_mags)  # (T, 4)
tau_int_mags = np.array(tau_int_mags)  # (T, 4)

print("Per-leg mean |tau_ext|:", tau_ext_mags.mean(axis=0))
print("Per-leg mean |tau_int|:", tau_int_mags.mean(axis=0))
print("Ratio |tau_int|/|tau_ext|:", tau_int_mags.mean() / tau_ext_mags.mean())
```

**Measurements:**
- Per-leg mean, max, std of |τ_ext| and |τ_int|
- Time-series of the ratio (does it vary within a gait cycle?)
- Sensitivity to fudge factors (run at fudge=1.0 and at best-fit fudge values)

**Expected outcomes:**
- Ratio ≈ 1.5–2.0 at fudge=1.0
- Ratio changes with `moment_fudge` (since it scales m_mag, which affects inter-joint quadratically but external linearly)
- If ratio >> 2: inter-joint coupling is too strong, may indicate incorrect leg spacing in XML vs physical robot

### Case Analysis for Bug #3

| moment_fudge | m_mag scaling | External torque | Inter-joint torque | Ratio |
|-------------|---------------|-----------------|-------------------|-------|
| 1.0 | 1x | kp = m·B·field_fudge | ∝ m² / r³ | ~1.5–2x |
| 0.5 | 0.5x | 0.5x (via kp) | 0.25x (quadratic in m) | ~0.75–1x |
| 1.5 | 1.5x | 1.5x (via kp) | 2.25x (quadratic) | ~2.25–3x |

The quadratic scaling of inter-joint torque with `moment_fudge` means that this parameter has **asymmetric leverage** on the two torque sources. This is important for interpreting optimization results:
- Low `moment_fudge` → inter-joint coupling drops faster than external drive → legs more decoupled
- High `moment_fudge` → inter-joint dominates even more → legs more tightly coupled

---

## Bug #9: Frequency-Blind Torque Model

### Background

The torque law `τ = kp * cross(north, goal)` is the same at all frequencies. The only frequency-dependent quantity is the drive angle rate. Real magnetic systems have frequency-dependent losses (eddy currents, phase lag). The optimizer's `moment_fudge` diverges ~46% across frequency (per predecessor analysis), suggesting the model can't fit all frequencies simultaneously.

### Test F: Quantify Frequency-Dependent Velocity Bias

**Hypothesis:** With a single parameter set, the model systematically over- or under-predicts velocity at specific frequencies in a pattern consistent with missing frequency-dependent damping.

**Method:** Take the best combined-fit parameters and run simulations at each frequency independently. Compare sim velocity to target velocity.

```python
# For each (scene, freq) in REFERENCE_DATA:
#   Run simulation with best combined params
#   Record: sim_velocity, target_velocity, relative error

# Expected pattern if damping is missing:
#   Low freq (10 Hz): sim velocity too high (not enough damping)
#   High freq (50 Hz): sim velocity too low (needs more damping at high speed)
# OR the reverse, depending on the dominant physics
```

**Measurements:**
- Relative velocity error per (scene, freq) combination
- Is the error systematic with frequency? (monotonic trend across 10/30/50 Hz)
- Per-morphology: does each scene show the same frequency trend?

**Expected outcome:** Systematic frequency-dependent bias that can't be eliminated by any single-point parameter set. This would confirm the need for frequency-dependent physics.

### Test G: Torque Phase Lag Analysis

**Hypothesis:** At higher frequencies, the magnet north vector lags behind the drive goal by a larger angle, reducing effective torque transfer.

**Method:** From the trajectory data (which records both `tau_ext` and `drive_angle`), compute the instantaneous angle between each magnet's north direction and the goal direction as a function of time. Compare across frequencies.

```python
# For each timestep after settle:
#   goal = [sin(angle), 0, cos(angle)]
#   north_i = magnet north direction (from quat)
#   phase_lag_i = arccos(dot(north_i, goal))
# Average over gait cycles, compare 10 Hz vs 30 Hz vs 50 Hz
```

**Measurements:**
- Mean phase lag per leg per frequency
- Phase lag distribution (is it steady-state or oscillating?)
- Correlation between phase lag and instantaneous torque magnitude

**Expected outcome:** Phase lag increases with frequency. At 50 Hz, magnets trail the field more, reducing the effective cross product and therefore the net propulsive torque.

### Test H: Velocity-Dependent Damping Feasibility (if pursuing fix)

**Hypothesis:** The `mjWARN_BADQACC` bug from the previous damping attempt was caused by writing subnormal floats to `qfrc_applied`, not by the damping physics.

**Method (from HANDOFF.md):**
1. Restore the damping code from HANDOFF.md
2. Apply the untested fix: `data.warning.number[:] = 0` before each `mj_step`
3. Additionally, clamp damping force to zero when |force| < 1e-20 (avoid subnormals)
4. Run headless with the test command from HANDOFF.md

**Fallback:** If warning reset doesn't work, apply damping via `data.qfrc_applied[-4:]` only when |force| > threshold, writing exact 0.0 otherwise.

### Case Analysis for Bug #9

| Frequency | Steps/rev | Expected phase lag | Damping need | Current model behavior |
|-----------|-----------|-------------------|--------------|----------------------|
| 10 Hz | 200 | Small (~2°) | Low | Tends to over-predict velocity |
| 30 Hz | 67 | Medium (~5°) | Medium | Best-fit point (most data) |
| 50 Hz | 40 | Large (~9°+) | High | Tends to under-predict velocity |

**What frequency-dependent damping would fix:**

```
τ_damping = -(a·|ω| + b) · ω
```

- At 10 Hz (low ω): damping ≈ b·ω (small, mostly linear viscous)
- At 50 Hz (high ω): damping ≈ a·ω² (quadratic term dominates)

This naturally produces more damping at high frequency, which would reduce the 50 Hz velocity prediction (currently too high in isolated fits) and produce a better cross-frequency fit.

**Risk assessment:**
- Low risk: subnormal float fix is straightforward
- Medium risk: adding 2 search dimensions (damp_quad, damp_linear) increases optimizer burden
- Mitigation: can fix the existing `dof_damping` as `damp_linear`, add only `damp_quad` as new dimension (12→13D, or keep at 13D if we remove the dead `sliding_friction`)

---

## Execution Order

| Step | Test | Time est. | Blocks |
|------|------|-----------|--------|
| 1 | **A** — MuJoCo friction combination rule | 5 min | Nothing |
| 2 | **B** — Sliding friction sensitivity sweep | 10 min | Confirms A |
| 3 | **E** — Log torque magnitudes | 10 min | Nothing |
| 4 | **F** — Frequency-dependent velocity bias | 15 min | Nothing |
| 5 | **G** — Phase lag analysis | 15 min | Uses F data |
| 6 | **D** — Dead-parameter correlation analysis | 10 min | Uses existing CSVs |
| 7 | **C** — Robot geom friction test | 20 min | Requires code change |
| 8 | **H** — Damping feasibility | 30 min | Requires code change |

Tests 1–6 are read-only / analysis-only (no code changes needed).
Tests 7–8 require code modifications and should be done on a branch.
