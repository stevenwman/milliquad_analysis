# Handoff: Vectorize Magnetic Force Computation in `simulation_fast.py`

## Context

This is a MuJoCo simulation of a miniature quadruped robot (~6mm) with 4 magnetically-actuated legs. The simulation runs at 2kHz (10,000 steps per 5s run) and the per-step magnetic force computation in Python is the dominant wall-clock bottleneck. The optimization loop runs 48 simulations per batch (8 param sets x 2 scenes x 3 jitter trials) across 20 workers.

## What Has Already Been Done (Options 1 & 2)

`simulation_fast.py` is a copy of `simulation.py` with two optimizations already applied:

### Option 1: Eliminate redundant magnet state computation
- **Original**: `_get_magnet_state()` was called **12 times** per step:
  - 4x in `_compute_external_torques()`
  - 4x in `_compute_interjoint_torques()`
  - 4x in `_apply_magnetic_forces()` (just to get `body_idx`, which is trivially `i + LEG_BODY_OFFSET`)
- **Fixed**: Added `_get_all_magnet_states(data)` which computes all 4 legs once, returning `(pos, north)` as `(4,3)` arrays. Called once in `_apply_magnetic_forces()`, results passed to both torque functions.

### Option 2: Replace scipy with inline quaternion rotation
- **Original**: Each `_get_magnet_state()` call constructed a `scipy.spatial.transform.Rotation` object via `R.from_quat(quat, scalar_first=True).as_matrix()` then did matrix-vector multiply. This is ~45 flops + Python object overhead per call.
- **Fixed**: Added `_quat_rotate_vec(q_wxyz, v)` using the direct formula (Rodrigues-like, ~15 flops, pure numpy):
  ```python
  def _quat_rotate_vec(q_wxyz, v):
      w = q_wxyz[0]
      q_xyz = q_wxyz[1:4]
      t = 2.0 * np.cross(q_xyz, v)
      return v + w * t + np.cross(q_xyz, t)
  ```
- Also replaced `R.from_euler('y', angle).as_matrix() @ [0,0,1]` in `_compute_external_torques()` with direct trig: `goal = [sin(angle), 0, cos(angle)]`.

### What was NOT changed
- `_initialize_pose()` — called once, uses scipy for quaternion composition. Not hot path.
- `_update_viewer_overlays()` — viewer-only, runs at display framerate. Not hot path.
- `_dipole_field()` — still called individually 12 times per step (4 legs x 3 neighbors). This is your target.
- The inner loop structure in `_compute_interjoint_torques()` — still Python `for i/for j`. This is your target.
- The loop in `_compute_external_torques()` — still Python `for i in range(4)`. This is your target.
- The loop in `_get_all_magnet_states()` — still Python `for i in range(4)`. This is your target.

## Your Task: Option 3 — Full Vectorization

Replace all Python `for` loops in the hot path with batched numpy operations on `(4,3)` arrays. There are 4 functions to vectorize:

### 3a. Vectorize `_get_all_magnet_states(data)`

Current (loop over 4 legs):
```python
def _get_all_magnet_states(data):
    pos = np.empty((4, 3))
    north = np.empty((4, 3))
    for i in range(4):
        body_idx = i + LEG_BODY_OFFSET
        pos[i] = data.xpos[body_idx]
        quat = data.xquat[body_idx]
        body_dir = _BODY_DIR_POS_X if i in (0, 2) else _BODY_DIR_NEG_X
        n = _quat_rotate_vec(quat, body_dir)
        norm = np.linalg.norm(n)
        if norm > 0:
            n /= norm
        north[i] = n
    return pos, north
```

Target: Batch all 4 quaternion rotations into a single vectorized call. Key details:
- `LEG_BODY_OFFSET = 2`, so body indices are `[2, 3, 4, 5]`
- `data.xpos` is shape `(n_bodies, 3)`, `data.xquat` is shape `(n_bodies, 4)` in MuJoCo's `(w,x,y,z)` format
- Body directions differ by leg: legs 0,2 use `[+1,0,0]`, legs 1,3 use `[-1,0,0]`. This can be expressed as: `body_dirs = np.array([[1,0,0],[-1,0,0],[1,0,0],[-1,0,0]], dtype=float)` (pre-allocate as module-level constant)
- Write a batched `_quat_rotate_vec_batch(quats, vecs)` where `quats` is `(4,4)` and `vecs` is `(4,3)`:
  ```python
  def _quat_rotate_vec_batch(q_wxyz, v):
      # q_wxyz: (N, 4), v: (N, 3)
      w = q_wxyz[:, 0:1]       # (N, 1)
      q_xyz = q_wxyz[:, 1:4]   # (N, 3)
      t = 2.0 * np.cross(q_xyz, v)   # (N, 3)
      return v + w * t + np.cross(q_xyz, t)  # (N, 3)
  ```
- Normalization: `np.linalg.norm(north, axis=1, keepdims=True)` then divide, with a max against epsilon to avoid division by zero.

### 3b. Vectorize `_compute_external_torques()`

Current:
```python
def _compute_external_torques(data, angle, kp_mag, settle_time, north):
    tau_ext = np.zeros((4, 3))
    if data.time <= settle_time:
        return tau_ext
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    goal = np.array([sin_a, 0.0, cos_a])
    for i in range(4):
        tau_ext[i] = kp_mag * np.cross(north[i], goal)
    return tau_ext
```

Target: `goal` is the same for all 4 legs, so:
```python
tau_ext = kp_mag * np.cross(north, goal)  # (4,3) cross with broadcast (3,)
```
One line replaces the loop.

### 3c. Vectorize `_compute_interjoint_torques()` — the big one

Current (nested Python loop + per-pair function call):
```python
def _compute_interjoint_torques(m_mag, pos, north):
    tau_int = np.zeros((4, 3))
    if m_mag == 0.0:
        return tau_int
    m = m_mag * north
    for i in range(4):
        Bi = np.zeros(3)
        for j in range(4):
            if j == i:
                continue
            Bi += _dipole_field(m[j], pos[i] - pos[j])
        tau_int[i] = np.cross(m[i], Bi)
    return tau_int
```

Where `_dipole_field` is:
```python
def _dipole_field(mj, r_vec):
    r = max(np.linalg.norm(r_vec), R_EPS)
    rhat = r_vec / r
    return MU0_OVER_4PI * (1.0 / r**3) * (3.0 * np.dot(mj, rhat) * rhat - mj)
```

This calls `_dipole_field` 12 times per step. Target: fully vectorized with numpy broadcasting:

```python
def _compute_interjoint_torques(m_mag, pos, north):
    if m_mag == 0.0:
        return np.zeros((4, 3))

    m = m_mag * north  # (4, 3) dipole moments

    # Displacement vectors: r_vecs[i,j] = pos[i] - pos[j], shape (4, 4, 3)
    r_vecs = pos[:, None, :] - pos[None, :, :]

    # Distances with epsilon floor, shape (4, 4)
    r_norms = np.linalg.norm(r_vecs, axis=-1)
    np.maximum(r_norms, R_EPS, out=r_norms)

    # Unit displacement vectors, shape (4, 4, 3)
    r_hat = r_vecs / r_norms[..., None]

    # 1/r^3, shape (4, 4)
    inv_r3 = 1.0 / (r_norms ** 3)

    # dot(m[j], r_hat[i,j]) for each (i,j) pair, shape (4, 4)
    # m[j] is indexed by second axis, r_hat[i,j] by both
    m_dot_rhat = np.einsum('jk,ijk->ij', m, r_hat)

    # B_ij = MU0_OVER_4PI / r^3 * (3 * (m_j . r_hat) * r_hat - m_j)
    # shape (4, 4, 3)
    B_ij = MU0_OVER_4PI * inv_r3[..., None] * (
        3.0 * m_dot_rhat[..., None] * r_hat - m[None, :, :]
    )

    # Zero out self-interaction (i == j) on the diagonal
    diag_idx = np.arange(4)
    B_ij[diag_idx, diag_idx, :] = 0.0

    # Total field at each leg: sum over source legs j
    B_total = B_ij.sum(axis=1)  # (4, 3)

    # Torque: tau = m x B
    return np.cross(m, B_total)  # (4, 3)
```

**Important**: The `np.einsum('jk,ijk->ij', m, r_hat)` computes `dot(m[j], r_hat[i,j])` for all pairs. Make sure the axes are correct: `m` is indexed by its first axis as `j` (source dipole), and `r_hat` has `i` on axis 0 (target) and `j` on axis 1 (source).

### 3d. Vectorize remaining loops in `_apply_magnetic_forces()`

Current:
```python
for i in range(4):
    body_idx = i + LEG_BODY_OFFSET
    data.xfrc_applied[body_idx, 3:6] += tau_ext[i] + tau_int[i]

omega = np.zeros((4, 3))
for i in range(4):
    omega[i] = data.cvel[i + LEG_BODY_OFFSET, :3]
```

Target (use array slicing):
```python
body_indices = slice(LEG_BODY_OFFSET, LEG_BODY_OFFSET + 4)
data.xfrc_applied[body_indices, 3:6] += tau_ext + tau_int

omega = data.cvel[body_indices, :3].copy()
```

## Verification Protocol

A verification script already exists at `mujoco_refactor/verify_sim.py`. It:
1. Runs `simulation.py` (original) and `simulation_fast.py` (your version) with identical params and seeds
2. Compares the full floating-base trajectory (pos, vel, quat) at every timestep
3. Reports max element-wise differences and speedup
4. Runs 5 trials: 1 with default params + 4 with randomized params across both scenes

Run it with:
```bash
cd mujoco_refactor && uv run python verify_sim.py
```

**Expected behavior regarding numerical differences**: The inline quaternion rotation formula uses different floating-point operation ordering than scipy's matrix construction + matmul. This produces ULP-level differences (~1e-16 per step). Over 10,000 steps of chaotic contact dynamics, these tiny differences CAN accumulate and cause trajectory divergence. This is expected and acceptable — what matters is that the physics is mathematically equivalent, not bit-identical. The verify script reports the divergence so the user can assess.

If you want to achieve exact bit-match, you would need to keep scipy's `R.from_quat().as_matrix() @ v` in `_get_all_magnet_states` and only eliminate the redundant calls (option 1 only). But the user has accepted that ULP-level divergence is fine — the goal is speed.

## File Map

| File | Role |
|------|------|
| `simulation.py` | Original implementation. DO NOT MODIFY. Reference for correctness. |
| `simulation_fast.py` | Your working copy. Currently has options 1+2. Apply option 3 here. |
| `verify_sim.py` | Compares `simulation.py` vs `simulation_fast.py`. Run after changes. |
| `config.py` | Constants and search space. Read-only for this task. |
| `optimizer.py` | Optimization loop. Currently imports `simulation`, not `simulation_fast`. Will be switched after verification passes. |

## Physics Constants Used in Hot Path

From `config.py`:
- `MU0_OVER_4PI = 1e-7` (N/A^2)
- `R_EPS = 1e-6` (meters, floor for dipole distance to avoid 1/r^3 blow-up)
- `LEG_BODY_OFFSET = 2` (body index offset: leg `i` → body `i + 2`)

## Hot Path Call Stack (per simulation step)

```
_do_simulation_step()
  └─ _apply_magnetic_forces()        ← orchestrator, called at 2kHz
       ├─ _compute_drive_angle()     ← trivial (one modulo), ignore
       ├─ _get_all_magnet_states()   ← 4 quat rotations (YOUR TARGET)
       ├─ _compute_external_torques()← 4 cross products (YOUR TARGET)
       ├─ _compute_interjoint_torques()← 12 dipole fields + 4 cross products (YOUR TARGET, biggest win)
       └─ write to data.xfrc_applied + read cvel (YOUR TARGET, minor)
  └─ mujoco.mj_step()               ← C code, not touchable
  └─ _check_instability()            ← cheap checks, ignore
  └─ _record_state()                 ← dict append, ignore
```

## Style Notes

- The user prefers no unnecessary abstractions, no fallback logic, no magic defaults.
- Keep functions explicit and readable.
- Pre-allocate module-level constants (like the body direction array) rather than creating them per call.
- The `_dipole_field` function can be removed entirely if you inline the vectorized version. Check that it's not used elsewhere first (it's also used in the viewer overlay — actually no, check: it's only used in `_compute_interjoint_torques`).
- `scipy.spatial.transform.Rotation` import (`R`) must stay because `_initialize_pose()` and `_update_viewer_overlays()` still use it. Those are NOT hot path.

## Summary of What to Do

1. Open `simulation_fast.py`
2. Apply vectorizations 3a through 3d as described above
3. Run `uv run python verify_sim.py` to compare against `simulation.py`
4. Report the max trajectory differences and the speedup factor
5. If the user is satisfied, they will swap `simulation_fast.py` → `simulation.py` in `optimizer.py`'s import
