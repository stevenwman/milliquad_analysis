# Potential Bugs Review — Independent Verification

Date: 2026-02-17
Source: `potential_bugs.md` (predecessor audit), verified against current codebase

---

## #1. "Variance penalty uses unfiltered reference data" — FALSE POSITIVE

**Claim:** Line 411 of `optimizer.py` calls `reference_rows()` (all 11 rows) instead of `_REF_ROWS` (filtered), causing incorrect variance penalty when `--scenes`/`--freqs` filters are active.

**Verification:** The variance computation at `optimizer.py:411-419`:
```python
ref_rows_local = reference_rows()          # all 11 rows
targets_by_id = {row["id"]: row["speed"] for row in ref_rows_local}
rel_errors = [
    (ref_avg_velocities[rid] - targets_by_id[rid]) / targets_by_id[rid]
    for rid in ref_avg_velocities      # iterates FILTERED set only
    if targets_by_id.get(rid, 0) > 0
]
```

The outer loop iterates over `ref_avg_velocities`, which only contains IDs for scenes that were actually simulated (the filtered set). Extra entries in `targets_by_id` are never accessed. Using `_REF_ROWS` instead produces identical results — just a wasteful allocation, not a bug.

**Verdict:** No fix needed.

---

## #2. "Only floor friction is tuned" — CONFIRMED BUG, all 3 friction dims are dead

**Claim:** Friction is set only on the floor geom; robot leg/magnet geoms use MuJoCo defaults `[1.0, 0.005, 0.0001]`.

**Verification:** The original claim was correct but understated the severity. Empirical testing reveals the root cause is **not** the friction combination rule — it's `mjENBL_OVERRIDE`.

`simulation.py:477` enables `mjENBL_OVERRIDE`:
```python
model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
```

This flag causes MuJoCo to use global override parameters (`model.opt.o_friction`, `model.opt.o_solref`, `model.opt.o_solimp`) for **all** contacts, completely ignoring per-geom friction values. The code sets `solref` and `solimp` via the correct override slots (`model.opt.o_solref`, `model.opt.o_solimp`), but writes friction to the per-geom slot (`model.geom_friction[ground_id]`), which is ignored.

**Empirical confirmation:** Sweeping floor friction from 0.0 to 10,000 across all 3 dimensions produces **bit-identical** simulation results (velocity, lateral displacement, everything). The parameter has literally zero effect.

**Impact:** All 3 friction search dimensions (`sliding_friction`, `torsional_friction`, `rolling_friction`) have been completely inert across every optimization run. The optimizer has been wasting evaluations exploring a 3D subspace that cannot influence the cost function.

**Fix:** Set `model.opt.o_friction` instead of `model.geom_friction[ground_id]`. Note that `o_friction` has 5 elements `[slide1, slide2, torsional, rolling1, rolling2]` vs the 3-element per-geom friction `[sliding, torsional, rolling]`.

**Verdict:** Confirmed bug. All 3 friction parameters are dead. HIGH priority.

---

## #3. "Inter-joint dipole torques dominate external drive by ~2x" — APPROXIMATELY CORRECT

**Claim:** Inter-joint torque ~4.7e-6 N·m vs external ~2.3e-6 N·m.

**Verification (fudges=1):**
- External: `kp_mag = 1.13e-3 × 2e-3 = 2.26e-6 N·m`
- Inter-joint: closest leg pair (FR-FL) at r ≈ 0.006m (from XML positions ±0.00298m in y). Dipole field B ≈ `1e-7 × 1.13e-3 / (0.006)³ × 2 ≈ 1.05e-3 T`. Torque per source ≈ `1.13e-3 × 1.05e-3 ≈ 1.2e-6 N·m`. With 3 source magnets at varying distances, total ≈ 3–4e-6 N·m.

Ratio is ~1.5–2x, consistent with claim. The conclusion that `moment_fudge` has more leverage than `field_fudge` follows correctly (moment scales both torque types, field scales only external).

**Verdict:** Approximately correct. Worth logging torque magnitudes during simulation to get exact numbers.

---

## #4. "Half-step time lag in drive angle computation" — CORRECT, not a bug

**Claim:** Drive angle computed from `data.time` before `mj_step()` creates a frequency-dependent phase lag.

**Verification:** Standard explicit integration — forces at time t applied during step t → t+dt. The magnitude table is accurate:

| Drive freq | Steps/rev | Angle error/step | % of cycle |
|------------|-----------|-------------------|------------|
| 10 Hz      | 200       | 1.8°              | 0.5%       |
| 30 Hz      | 67        | 5.4°              | 1.5%       |
| 50 Hz      | 40        | 9.0°              | 2.5%       |

**Verdict:** Correct observation, not a bug. Standard for explicit integration schemes.

---

## #5. "Asymmetric contact geometry in scene_2" — WRONG

**Claim:** In `robot_2.xml`, FR leg has 2 `<geom class="leg">` entries, FL has 3.

**Verification:** Counted active (non-commented) `<geom class="leg">` entries in `robot_2.xml`:

| Leg | Active leg geoms | Lines |
|-----|-----------------|-------|
| FR  | 2 | 32, 34 |
| FL  | 2 | 46, 47 |
| BR  | 2 | 56, 57 |
| BL  | 2 | 64, 69 |

All legs have exactly 2 active leg geoms. The predecessor miscounted — several geoms are commented out with `<!-- -->`. The naming convention across robot files corresponds to leg geom count:
- `robot_1.xml`: 1 leg geom per leg
- `robot_2.xml`: 2 per leg
- `robot_4.xml`: 4 per leg

All files are internally symmetric.

**Verdict:** Claim is incorrect. No asymmetry exists.

---

## #6. "Contact solver timescale interacts with drive frequency" — VALID observation

**Claim:** `solref_timeconst` comparable to drive cycle period causes frequency-dependent contact behavior.

**Verification:** Correct physics reasoning. At 50 Hz (20ms cycle), a `solref_timeconst` of ~10ms is half the drive period. At 10 Hz (100ms cycle), same timeconst is 10% of the period. A single `solref_timeconst` can't be optimal across all frequencies.

**Verdict:** Valid physics observation, not a code bug.

---

## #7. "Dipole field singularity at R_EPS = 1e-6" — CORRECT, zero risk

**Claim:** At R_EPS distance (1μm), dipole field reaches ~3e8 T (unphysical).

**Verification:** `config.py:47` confirms `R_EPS = 1e-6`. Math: `B ≈ 1e-7 × 1.13e-3 / (1e-6)³ ≈ 1.13e8 T`. Correct.

However, the inter-magnet distances are **geometrically constant** throughout the simulation. Each magnet sits on its hinge rotation axis (magnet pos = `(0, 0, 0.00074)` in body-local frame, hinge axis = body-local z). Rotating the hinge leaves the magnet position unchanged. All 4 hinge anchors are fixed relative to the chassis rigid body, so inter-magnet distances are:
- Lateral neighbors (FR↔FL, BR↔BL): ~6.0 mm
- Longitudinal neighbors (FR↔BR, FL↔BL): ~6.8 mm
- Diagonal (FR↔BL, FL↔BR): ~9.1 mm

R_EPS = 1e-6 m is 1000x below the minimum possible separation. The guard can never be triggered by any physically realizable state.

**Verdict:** Zero risk. No action needed.

---

## #8. "No explicit condim in XML" — CORRECT, not a problem

**Claim:** No contact dimensionality specified in any XML file.

**Verification:** Grep confirms no `condim` in any XML. MuJoCo defaults apply (condim=3 for most contacts: normal + 2 tangent friction). Appropriate for this application.

**Verdict:** Correct observation. No action needed.

---

## #9. "Torque model is frequency-blind" — CORRECT, referenced fix is stale

**Claim:** Torque magnitude `kp_mag * cross(north, goal)` has no frequency dependence. References in-progress velocity-dependent damping (`damp_quad`, `damp_linear` in `simulation_fast.py`).

**Verification:** The torque model is confirmed frequency-blind — only the drive angle rate depends on frequency.

However, `damp_quad`/`damp_linear` **do not exist** in the current `simulation_fast.py` (grep returns nothing). `HANDOFF.md` documents a custom damping feature that was implemented but hit a `mjWARN_BADQACC` bug. The current `simulation_fast.py` appears to have been reverted to the pre-change backup. The referenced fix is not present in the codebase.

**Verdict:** Physics observation is correct. Code reference to in-progress fix is stale — the feature was reverted.

---

## Summary Table

| # | Claim | Verdict | Priority |
|---|-------|---------|----------|
| 1 | Variance penalty bug | **False positive** | None |
| 2 | Floor-only friction | **Confirmed bug** — ALL 3 friction dims dead (`mjENBL_OVERRIDE` ignores per-geom) | **CRITICAL** |
| 3 | Inter-joint > external (~2x) | **~Correct** | Medium (verify empirically) |
| 4 | Half-step time lag | **Correct** — standard, not a bug | None |
| 5 | Asymmetric scene_2 geoms | **Wrong** — all legs symmetric | None |
| 6 | solref × frequency | **Valid** physics observation | Low |
| 7 | R_EPS singularity | **Correct** but zero risk (constant inter-magnet distances) | None |
| 8 | No condim | **Correct** — defaults fine | None |
| 9 | Frequency-blind torque | **Correct**, damping code reverted | HIGH (if pursuing freq fit) |

## Recommended Actions

1. **#2 — Friction (CRITICAL)**: `mjENBL_OVERRIDE` causes all per-geom friction to be ignored. All 3 friction search dimensions are completely dead. Fix: write to `model.opt.o_friction` (5-element array: `[slide1, slide2, torsional, rolling1, rolling2]`) instead of `model.geom_friction[ground_id]`.
2. **#9 — Frequency-dependent damping**: Decide whether to re-attempt the velocity-dependent damping feature. The `HANDOFF.md` documents the approach and the `mjWARN_BADQACC` bug. The fix (resetting `data.warning.number` per step) was proposed but never tested.
3. **#3 — Torque magnitudes**: Log `tau_ext` and `tau_int` during a representative simulation to get empirical confirmation of the ~2x ratio.
