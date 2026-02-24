# Plan: Add 5 New MuJoCo Solver Parameters to Optimization System

## Context

The current optimizer searches over 13 physical parameters (friction, contact solver, magnetic scaling, joint damping). To improve sim-to-real matching, we want to expand the search space with 5 additional MuJoCo contact solver parameters that are currently left at defaults:

**CRITICAL: This creates a PARALLEL system with new `*_new.py` files. ZERO modifications to existing files. Current optimization runs can continue unaffected.**

1. **noslip_iterations** — number of friction solver iterations (int)
2. **noslip_tolerance** — friction convergence tolerance (float)
3. **margin** — contact detection distance threshold (float)
4. **solreffriction** — friction-specific contact solver reference (2-element array)
5. **solimpfriction** — friction-specific contact impedance (5-element array)

This expands the search space from 13 → 18 dimensions. The existing codebase is well-designed for this: search space is defined dynamically, CSV columns auto-generate from dimension names, and parameter flow is centralized through `sim_params_from_point()`.

**Key constraint**: User wants to verify that each new parameter actually affects trajectories (no "dead" parameters) before running expensive optimization.

## Critical Files

- **Current system (13 params)** — **UNTOUCHED, NO MODIFICATIONS**:
  - `/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/mujoco_refactor/simulation_fast.py` — **left as-is**
  - `/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/mujoco_refactor/config.py` — **left as-is**
  - `/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/mujoco_refactor/optimizer.py` — **left as-is**
  - `/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/mujoco_refactor/config_step.py` — **left as-is**
  - `/home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/mujoco_refactor/optimizer_step.py` — **left as-is**

- **New system (18 params)** — **BRAND NEW FILES ONLY**:
  - `simulation_fast_new.py` — **NEW FILE** (copy of simulation_fast.py + 5 new params)
  - `config_new.py` — **NEW FILE** (copy of config.py + 18-dim space)
  - `optimizer_new.py` — **NEW FILE** (copy of optimizer.py + updated imports)
  - `verify_new_params.py` — **NEW FILE** (validation script)

## Implementation Plan

### Step 1: Create `simulation_fast_new.py`

**Source**: Copy from `simulation_fast.py` (687 lines)

**Changes to `run_simulation()` function** (lines 508-525):

```python
# EXISTING (lines 509-525):
model.dof_damping[-4:] = params['dof_damping']
model.opt.o_solref = params['solref']
model.opt.o_solimp = params['solimp']
gf = params['ground_friction']
model.opt.o_friction[:] = [gf[0], gf[0], gf[1], gf[2], gf[2]]

# ADD AFTER LINE 515 (before timestep):
# New solver parameters
if 'noslip_iterations' in params:
    model.opt.noslip_iterations = int(params['noslip_iterations'])
if 'noslip_tolerance' in params:
    model.opt.noslip_tolerance = float(params['noslip_tolerance'])
if 'margin' in params:
    model.opt.margin = float(params['margin'])
if 'solreffriction' in params:
    model.opt.o_solreffriction = params['solreffriction']  # 2-element array
if 'solimpfriction' in params:
    model.opt.o_solimpfriction = params['solimpfriction']  # 5-element array
```

**Docstring update** (line 492):
```python
params: Simulation parameters dict. Expected keys:
    ground_friction, dof_damping, solref, solimp, kp_mag, mag_params,
    noslip_iterations (optional), noslip_tolerance (optional),
    margin (optional), solreffriction (optional), solimpfriction (optional)
```

**Backward compatibility**: All 5 new params are optional (use `if 'key' in params`), so existing code continues to work.

### Step 2: Create `config_new.py`

**Source**: Copy from `config.py` (first ~400 lines containing constants + search space + conversion functions)

**Add 5 new dimensions to `space` list** (after line 204):

```python
space: list[Real] = [
    # ... existing 13 dimensions ...
    Real(1e-10, 1e-8, "log-uniform", name="dof_damping"),

    # NEW: Friction solver parameters
    Real(0, 100, "uniform", name="noslip_iterations"),           # int, but Real for CMA-ES
    Real(1e-6, 1e-3, "log-uniform", name="noslip_tolerance"),
    Real(0.0, 0.005, "uniform", name="margin"),

    # NEW: Friction-specific contact solver (similar to solref/solimp)
    Real(1e-5, 1.0, "log-uniform", name="solreffriction_timeconst"),
    Real(1.0, 10.0, "log-uniform", name="solreffriction_dampratio"),
    Real(0.001, 0.999, "uniform", name="solimpfriction_dmin"),
    Real(0.01, 0.99, "uniform", name="solimpfriction_delta_d"),
    Real(1e-7, 1, "log-uniform", name="solimpfriction_width"),
    Real(0.01, 0.99, "uniform", name="solimpfriction_midpoint"),
    Real(2.0, 7.0, "uniform", name="solimpfriction_power"),
]
```

**Note**: `noslip_iterations` should be int, but we use `Real` because CMA-ES operates on continuous space. Will cast to int in simulation.

**Update `sim_params_from_point()`** (lines 285-309):

```python
def sim_params_from_point(point: list[float]) -> dict[str, Any]:
    """Build the sim_params dict consumed by simulation_fast_new.run_simulation()."""
    params = point_to_params(point)
    m_mag = MAGNETIC_MOMENT * params["magnetic_moment_fudge"]
    kp_mag = m_mag * MAGNETIC_FIELD_MAGNITUDE * params["magnetic_field_fudge"]

    # Existing params
    base_params = {
        "ground_friction": [params["sliding_friction"], params["torsional_friction"], params["rolling_friction"]],
        "solref": [params["solref_timeconst"], params["solref_dampratio"]],
        "solimp": [
            params["solimp_dmin"],
            params["solimp_dmin"] + params["solimp_delta_d"] * (0.9999 - params["solimp_dmin"]),
            params["solimp_width"],
            params["solimp_midpoint"],
            params["solimp_power"],
        ],
        "dof_damping": params["dof_damping"],
        "kp_mag": kp_mag,
        "mag_params": {"m_mag": m_mag},
    }

    # New params
    base_params.update({
        "noslip_iterations": int(round(params["noslip_iterations"])),
        "noslip_tolerance": params["noslip_tolerance"],
        "margin": params["margin"],
        "solreffriction": [
            params["solreffriction_timeconst"],
            params["solreffriction_dampratio"],
        ],
        "solimpfriction": [
            params["solimpfriction_dmin"],
            params["solimpfriction_dmin"] + params["solimpfriction_delta_d"] * (0.9999 - params["solimpfriction_dmin"]),
            params["solimpfriction_width"],
            params["solimpfriction_midpoint"],
            params["solimpfriction_power"],
        ],
    })

    return base_params
```

**Update `CMAES_X0` warm-start dict** (line 133):
```python
CMAES_X0: dict[str, float] | None = None  # Cold-start until we have good initial values
```

**Reason**: We don't have optimized values for the 5 new params yet. Set to `None` for cold-start, or populate with sensible defaults:
```python
CMAES_X0 = {
    # ... existing 13 params ...
    "noslip_iterations": 0,          # MuJoCo default
    "noslip_tolerance": 1e-6,        # MuJoCo default
    "margin": 0.0,                   # MuJoCo default
    "solreffriction_timeconst": 0.002,   # consensus from flat solref
    "solreffriction_dampratio": 1.0,     # standard critical damping
    "solimpfriction_dmin": 0.9,          # typical solimp values
    "solimpfriction_delta_d": 0.09,
    "solimpfriction_width": 0.001,
    "solimpfriction_midpoint": 0.5,
    "solimpfriction_power": 2.0,
}
```

**Change imports at top of file**:
```python
# OLD:
# from simulation_fast import run_simulation

# NEW:
from simulation_fast_new import run_simulation
```

### Step 3: Create `verify_new_params.py`

**Purpose**: Test each of the 5 new parameters (11 dimensions total) individually to verify they affect trajectory.

**Strategy**:
1. Load baseline params from a known-good run (e.g., `results/20260222T181114_with_20hz_no-deadzone`)
2. For each new param dimension, run 3 simulations:
   - Low value (e.g., 10th percentile of range)
   - Mid value (50th percentile)
   - High value (90th percentile)
3. Compare final positions — if all three are identical, param is "dead"
4. Report which params have effect and which are dead

**Script structure**:
```python
#!/usr/bin/env python3
"""Verify that new solver parameters actually affect simulation trajectories."""

import csv
import sys
from pathlib import Path
import numpy as np

from config_new import sim_params_from_point, space
from simulation_fast_new import run_simulation

# Baseline params from flat optimizer
BASELINE_RUN = "results/20260222T181114_with_20hz_no-deadzone"
baseline_csv = Path(BASELINE_RUN) / "optimization_bests.csv"

if not baseline_csv.exists():
    print(f"Error: {baseline_csv} not found")
    sys.exit(1)

rows = list(csv.DictReader(open(baseline_csv)))
best = rows[-1]

# Extract baseline point (13 params)
old_param_names = [
    'sliding_friction', 'torsional_friction', 'rolling_friction',
    'solref_timeconst', 'solref_dampratio', 'solimp_dmin', 'solimp_delta_d',
    'solimp_width', 'solimp_midpoint', 'solimp_power',
    'magnetic_moment_fudge', 'magnetic_field_fudge', 'dof_damping'
]
point_baseline = [float(best[p]) for p in old_param_names]

# Add default values for new params (middle of range)
new_param_defaults = {
    "noslip_iterations": 50,
    "noslip_tolerance": 1e-4,
    "margin": 0.0025,
    "solreffriction_timeconst": 0.002,
    "solreffriction_dampratio": 5.0,
    "solimpfriction_dmin": 0.5,
    "solimpfriction_delta_d": 0.5,
    "solimpfriction_width": 0.001,
    "solimpfriction_midpoint": 0.5,
    "solimpfriction_power": 4.5,
}

# Build full 18-param point
point_full = point_baseline + [new_param_defaults[dim.name] for dim in space[13:]]

# Test scene
MJCF_PATH = "multi_milli_quad/scene_4.xml"
CTRL_FREQ = 30
SIM_DURATION = 3.0

def run_test(params_dict, label):
    """Run simulation and return final position."""
    params_dict['drive_freq'] = CTRL_FREQ
    try:
        trajectory = run_simulation(
            params=params_dict,
            mjcf_path=MJCF_PATH,
            sim_duration=SIM_DURATION,
            rng_seed=42,
        )
        if trajectory:
            return trajectory[-1]['pos'].copy()
        else:
            return None
    except Exception as e:
        print(f"  {label} FAILED: {e}")
        return None

print("=" * 80)
print("PARAMETER EFFECT VERIFICATION (New Solver Params)")
print("=" * 80)
print(f"Baseline: {BASELINE_RUN}")
print(f"Test scene: {MJCF_PATH}, freq={CTRL_FREQ}Hz, duration={SIM_DURATION}s")
print()

# Test each new param dimension
new_param_indices = list(range(13, 18))  # Indices 13-17 for the 5 new params (11 dims)

results = []

for idx in new_param_indices:
    dim = space[idx]
    param_name = dim.name
    lo, hi = dim.low, dim.high

    # 3 test values: 10th, 50th, 90th percentile
    if dim.prior == "log-uniform":
        vals = np.logspace(np.log10(lo), np.log10(hi), 10)[[1, 5, 9]]
    else:
        vals = np.linspace(lo, hi, 10)[[1, 5, 9]]

    positions = []
    for val in vals:
        test_point = point_full.copy()
        test_point[idx] = val
        params = sim_params_from_point(test_point)
        pos = run_test(params, f"{param_name}={val:.4g}")
        positions.append(pos)

    # Check if all 3 positions are identical (dead parameter)
    if all(p is not None for p in positions):
        diffs = [np.linalg.norm(positions[i] - positions[0]) for i in range(1, 3)]
        max_diff = max(diffs)
        is_dead = max_diff < 1e-6

        status = "DEAD" if is_dead else "ACTIVE"
        print(f"{param_name:<30} {status:>8}  max_diff={max_diff:.2e}")

        results.append({
            "param": param_name,
            "status": status,
            "max_diff": max_diff,
        })
    else:
        print(f"{param_name:<30} {'ERROR':>8}  (simulation failed)")

print()
print("=" * 80)
active = [r for r in results if r["status"] == "ACTIVE"]
dead = [r for r in results if r["status"] == "DEAD"]

print(f"SUMMARY: {len(active)} active, {len(dead)} dead out of {len(results)} tested")
if dead:
    print(f"Dead params: {', '.join([r['param'] for r in dead])}")
else:
    print("All params have measurable effect on trajectories!")
```

**Expected outcome**: All 11 new param dimensions should be ACTIVE. If any are DEAD, investigate why (e.g., MuJoCo ignores the parameter, or we're not applying it correctly).

### Step 4: Check `optimizer.py` compatibility

**Read `optimizer.py`** to identify if any changes are needed.

**Expected**: No changes needed because:
- It imports `space`, `sim_params_from_point`, `reference_rows` from `config`
- It loops over `space` dynamically (no hardcoded dimension count)
- CSV columns auto-generate from `[dim.name for dim in space]`

**If changes needed**: Create `optimizer_new.py` with modifications. Otherwise, just update imports:
```python
# OLD:
from config import ...

# NEW:
from config_new import ...
```

**If optimizer.py is fully compatible**: Just create a symlink or wrapper:
```bash
cd mujoco_refactor
cp optimizer.py optimizer_new.py
# Edit line 29: from config import ... → from config_new import ...
# Edit line 30: from simulation_fast import ... → from simulation_fast_new import ...
```

### Step 5: Update MEMORY.md

Add section documenting the new 18-param system:

```markdown
## Extended Solver Parameters (18-dim, 2026-02-23)
- **New system**: `config_new.py` + `simulation_fast_new.py` + `optimizer_new.py`
- Added 5 MuJoCo solver params (11 dimensions total):
  - `noslip_iterations` (int 0-100): friction solver iterations
  - `noslip_tolerance` (float 1e-6 to 1e-3): friction convergence threshold
  - `margin` (float 0.0 to 0.005): contact detection distance
  - `solreffriction` (2-dim): friction-specific contact damping
  - `solimpfriction` (5-dim): friction-specific contact impedance
- Verified all new params affect trajectories (not dead)
- Cold-start recommended until initial values established
- Curse of dimensionality: 18-dim needs ~10× more evals than 13-dim (estimate 6000+ calls)
```

## Verification Steps

After implementation:

1. **Test simulation_fast_new.py standalone**:
   ```bash
   cd mujoco_refactor
   python simulation_fast_new.py  # Should run default params + visualize
   ```

2. **Run parameter verification**:
   ```bash
   python verify_new_params.py
   ```
   Expected: All 11 new param dimensions show ACTIVE status (max_diff > 1e-6)

3. **Smoke test optimizer_new.py**:
   ```bash
   python optimizer_new.py --scenes scene4 --freqs 30 --n-calls 16 --suffix smoke_test_18dim
   ```
   Expected: 2 CSV files created with 18 param columns, optimization runs without errors

4. **Check CSV output**:
   ```bash
   cd results/<latest_run>
   head -n 2 multi_optimization_results.csv | cut -d',' -f1-25  # Check first 25 columns
   ```
   Expected: All 18 param names present + solimp_dmax + solimpfriction_dmax

5. **Compare 13-dim vs 18-dim on same conditions**:
   - Run 100 evals with old 13-dim system
   - Run 100 evals with new 18-dim system
   - Compare convergence speed and final cost
   - Expected: 18-dim converges slower (more parameters) but potentially reaches lower cost

## Files to Create (NO MODIFICATIONS TO EXISTING FILES)

**LEGACY PROTECTION**: All existing files (`simulation_fast.py`, `config.py`, `optimizer.py`, `config_step.py`, `optimizer_step.py`) remain **completely unchanged**. Current optimization runs (e.g., `results/20260223T144835_corrected_flip_20hz_cold/`) continue unaffected.

1. **`mujoco_refactor/simulation_fast_new.py`** — NEW FILE (copy + ~20 lines added)
2. **`mujoco_refactor/config_new.py`** — NEW FILE (copy + ~80 lines added/changed)
3. **`mujoco_refactor/optimizer_new.py`** — NEW FILE (copy + 2 import lines changed)
4. **`mujoco_refactor/verify_new_params.py`** — NEW FILE (~150 lines)
5. **Update `~/.claude/projects/.../memory/MEMORY.md`** — APPEND new section (+10 lines)

## Risks and Mitigations

1. **Risk**: New params are "dead" (don't affect simulation)
   - **Mitigation**: Run `verify_new_params.py` before expensive optimization
   - **Fallback**: Remove dead params from search space

2. **Risk**: 18-dim search space is too large (curse of dimensionality)
   - **Mitigation**: Start with fixed-value sweeps (e.g., noslip_iterations ∈ {0, 50, 100})
   - **Fallback**: Add only 2-3 most impactful params first

3. **Risk**: MuJoCo ignores some parameters due to flags or ordering
   - **Mitigation**: Check `model.opt.*` values after application with print debugging
   - **Fallback**: Consult MuJoCo docs for parameter dependencies

4. **Risk**: Optimization convergence is much slower (10× more evals needed)
   - **Mitigation**: Use warm-start from 13-dim best params + default new params
   - **Fallback**: Run per-morphology sweeps first, then combined run

## Implementation Order

1. Create `simulation_fast_new.py` (15 min)
2. Create `config_new.py` (30 min)
3. Create `verify_new_params.py` (20 min)
4. Run verification script (5 min)
5. Check `optimizer.py` compatibility (10 min)
6. Create `optimizer_new.py` if needed (5 min) or update imports (2 min)
7. Run smoke test (10 min)
8. Update MEMORY.md (5 min)

**Total estimated time**: 1.5-2 hours
