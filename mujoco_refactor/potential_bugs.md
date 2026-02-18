# Potential Bugs & Missing Physics — Deep Dive Audit

Date: 2026-02-17
Sources: 4 parallel code audits (magnetic torque, simulation timing, optimizer pipeline, MuJoCo model setup)

---

## CONFIRMED BUG

### 1. Variance penalty uses unfiltered reference data

**Location:** `optimizer.py`, `_aggregate_scene_results()`, line 411

**Code:**
```python
ref_rows_local = reference_rows()   # BUG: returns ALL 11 rows
targets_by_id = {row["id"]: row["speed"] for row in ref_rows_local}
```

**Problem:** When running with `--scenes` or `--freqs` filters, the variance penalty compares velocity errors against ALL 11 reference targets, not the filtered subset. The cost function uses filtered `_REF_ROWS` for velocity/tumble/lateral, but the variance term uses unfiltered data.

**Impact:** Every per-morphology and per-frequency run we've done has an inconsistent variance penalty. The optimizer received conflicting signals.

**Fix:** Change `reference_rows()` to `_REF_ROWS` on line 411.

---

## HIGH SEVERITY

### 2. Only floor friction is tuned — robot geom friction stays at MuJoCo defaults

**Location:** `simulation.py:466`

```python
ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
model.geom_friction[ground_id] = params['ground_friction']
```

**Problem:** Friction is set ONLY on the floor geom. Robot leg/magnet geoms have no friction defined in any XML file and use MuJoCo's built-in defaults `[1.0, 0.005, 0.0001]`. MuJoCo resolves contact friction from both geoms in a contact pair — the optimizer controls only one side.

**Impact:** The optimizer's friction parameters have reduced leverage over actual contact behavior. Robot-side friction is fixed at defaults regardless of what the optimizer tries.

**Investigation needed:** Check how MuJoCo combines friction from both geoms in a pair (max? average? min?). Consider also setting friction on robot leg geoms.

### 3. Inter-joint dipole torques dominate external drive by ~2x

**Estimated magnitudes:**
- External drive torque: ~2.3e-6 N-m (`kp_mag * cross(north, goal)`)
- Inter-joint coupling torque: ~4.7e-6 N-m (dipole-dipole at ~3mm leg spacing)

**Implication:** The inter-joint coupling is not a small correction — it's the dominant torque source. This means:
- `moment_fudge` (scales both external AND inter-joint via `m_mag`) has ~3x the total torque leverage of `field_fudge` (scales only external via `kp_mag`)
- The two fudges are NOT redundant — they have different sensitivity profiles
- If inter-joint coupling magnitude is wrong (e.g., leg spacing in XML doesn't match reality), fudge factors absorb that error

**Investigation needed:** Verify actual leg-to-leg distances in XML match physical robot. Log torque magnitudes during simulation to confirm the 2x ratio.

---

## MEDIUM SEVERITY

### 4. Half-step time lag in drive angle computation

**Location:** `simulation.py`, `_compute_drive_angle()`

The drive angle is computed from `data.time` BEFORE `mj_step()` advances the state. Forces computed at angle θ(t) are applied during the step from t to t+dt.

**Frequency dependence:**
| Drive freq | Steps/revolution | Angle error per step | % of cycle |
|------------|-----------------|---------------------|------------|
| 10 Hz | 200 | 1.8° | 0.5% |
| 30 Hz | 67 | 5.4° | 1.5% |
| 50 Hz | 40 | 9.0° | 2.5% |

**Impact:** Creates a small but systematic frequency-dependent phase lag. Not a bug (standard for explicit integration), but contributes to frequency-dependent parameter divergence.

### 5. Asymmetric contact geometry in scene_2 (2-leg)

**Location:** `multi_milli_quad/robot_2.xml`

FR leg has 2 `<geom class="leg">` entries, FL has 3. Different legs have different numbers of contact-capable meshes.

**Impact:** Left/right contact asymmetry that uniform damping/friction can't capture. May explain some of scene_2's higher optimization cost relative to scene_1 and scene_4.

### 6. Contact solver timescale interacts with drive frequency

`solref_timeconst` (optimizer range: [1e-5, 1.0]) controls contact response time. At 50 Hz drive (20ms cycle), if `solref_timeconst` is comparable to the cycle period, contact dynamics and field rotation interact differently than at 10 Hz (100ms cycle).

**Impact:** Even with identical parameters, the effective contact behavior is frequency-dependent. This is physics, not a bug, but it means a single `solref_timeconst` can't be optimal across all frequencies.

---

## LOW SEVERITY

### 7. Dipole field singularity at R_EPS = 1e-6

**Location:** `simulation.py`, `_dipole_field()`

```python
r = max(np.linalg.norm(r_vec), R_EPS)  # R_EPS = 1e-6 m
```

At R_EPS distance, dipole field reaches ~3e8 T (unphysical). If legs ever come within 1 micrometer (numerical collision), inter-joint torque explodes.

**Fix:** Use physically-motivated floor based on magnet size (~1mm), not numerical artifact.

### 8. No explicit condim in XML

No contact dimensionality is specified in any XML file. MuJoCo defaults to condim=3 or 4 depending on context. Probably fine, but not explicitly verified.

---

## MISSING PHYSICS (root cause of frequency divergence)

### 9. Torque model is frequency-blind

The only place frequency enters the simulation is the drive angle rate:
```python
angle = (sim_time - settle_time) * drive_freq * 2 * np.pi
```

The torque magnitude `kp_mag * cross(north, goal)` is identical at all frequencies. Missing:

- **Velocity-dependent damping:** Real magnetic systems have eddy current losses proportional to dB/dt (i.e., frequency). Higher frequency = more electromagnetic braking.
- **Phase lag feedback:** The torque law is purely proportional (no derivative term). At higher frequencies, legs lag behind the field more, but the torque doesn't compensate.
- **Resonance effects:** Legs have natural frequency from inertia + contact stiffness. Driving near/far from resonance changes energy transfer efficiency.

**Evidence:** `moment_fudge` diverges 46% across frequency (f30 wants 0.83, f10/f50 want 1.01). This pattern is NOT fixable by calibration — it requires modeling frequency-dependent physics.

**Prior work:** `HANDOFF.md` documents an in-progress velocity-dependent damping feature (`damp_quad`, `damp_linear` in `simulation_fast.py`) that addresses this but has a `mjWARN_BADQACC` bug.

---

## ACTION ITEMS

| # | Action | Priority | Effort |
|---|--------|----------|--------|
| 1 | Fix variance penalty bug (line 411: `reference_rows()` → `_REF_ROWS`) | **NOW** | 1 line |
| 2 | Verify inter-joint torque magnitude (log during sim) | HIGH | ~30 min |
| 3 | Test setting friction on robot leg geoms too | HIGH | ~1 hour |
| 4 | Fix velocity-dependent damping (`simulation_fast.py` BADQACC bug) | HIGH | ~2 hours |
| 5 | Increase R_EPS to ~1mm | LOW | 1 line |
| 6 | Verify scene_2 leg geometry matches physical robot | LOW | Manual check |
