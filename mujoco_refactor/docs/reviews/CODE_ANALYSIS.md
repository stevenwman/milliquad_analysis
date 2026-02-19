# Code Analysis: tune_multi_params_optimized.py & sim_optimizer_couple.py

**Analysis Date**: 2026-01-31
**Files Analyzed**:
- `mujoco/tune_multi_params_optimized.py`
- `mujoco/sim_optimizer_couple.py`

**Focus Areas**: Optimal simulation performance, code readability, maintainability

---

## Critical Issues

### 1. Dead Code: Unused Worker Function
**Location**: [tune_multi_params_optimized.py:16-34](mujoco/tune_multi_params_optimized.py#L16-L34)

**Issue**: The `run_simulation_worker()` function is defined but **never used**. The actual worker is `_evaluate_one_scene()` at line 194.

**Impact**: Confusing code, wasted lines, potential for accidentally calling wrong function.

**Fix**: Delete lines 13-34 entirely.

---

### 2. Dangerous Constant Synchronization
**Location**: [tune_multi_params_optimized.py:69](mujoco/tune_multi_params_optimized.py#L69)

**Issue**: `COST_SETTLE_TIME = 0.1` must manually match `sim_optimizer_couple.SETTLE_TIME`. The comment admits this fragility: "Must match sim_optimizer_couple.SETTLE_TIME"

**Impact**: High risk of bugs if one is changed without updating the other. Silent failures in cost calculation.

**Fix**: Import and use directly instead of duplicating:
```python
from sim_optimizer_couple import SETTLE_TIME as COST_SETTLE_TIME
```

---

### 3. Duplicated Parameter Conversion Logic
**Location**: [tune_multi_params_optimized.py:174-191](mujoco/tune_multi_params_optimized.py#L174-L191)

**Issue**: The pattern of converting `point → params → sim_params` appears in multiple functions. This violates DRY principle and creates maintenance burden.

**Impact**: Changes to parameter structure require updates in multiple places. High chance of inconsistency.

**Fix**: Create a single source of truth for this transformation. Consider a dedicated module or class.

---

### 4. Global Mutable State
**Location**: [tune_multi_params_optimized.py:77-78](mujoco/tune_multi_params_optimized.py#L77-L78)

**Issue**: `all_results = []` and `pool = None` as globals makes the code untestable and not reusable.

**Impact**:
- Cannot run multiple optimizations in same process
- Difficult to unit test
- Risk of state pollution between runs

**Fix**: Move these into the main function or adopt a class-based approach:
```python
class OptimizationRunner:
    def __init__(self):
        self.all_results = []
        self.pool = None
```

---

## Moderate Issues

### 5. Magic Numbers in solimp Construction
**Location**: [Lines 187, 408-409](mujoco/tune_multi_params_optimized.py#L187)

**Issue**: The values `0.5, 1.0` for solimp midpoint and power are hardcoded in multiple places.

**Impact**: Unclear what these numbers mean, difficult to change consistently.

**Fix**: Define as named constants at the top:
```python
SOLIMP_MIDPOINT = 0.5  # Default midpoint for solimp contact model
SOLIMP_POWER = 1.0     # Default power for solimp contact model
```

---

### 6. Unclear Separation of Concerns
**Location**: Throughout `tune_multi_params_optimized.py`

**Issue**: The optimization file handles too many responsibilities:
- Space definition
- Cost calculation
- Simulation orchestration
- CSV writing
- Video recording

**Impact**: File is 431 lines, hard to navigate, difficult to test individual components.

**Fix**: Consider splitting into:
```
optimization_config.py    # Search space, constants, target velocities
cost_functions.py         # Cost calculation logic
tune_multi_params_optimized.py  # Orchestration only
```

---

### 7. No Validation of Required mag_params
**Location**: [sim_optimizer_couple.py:162-168](mujoco/sim_optimizer_couple.py#L162-L168)

**Issue**: `mag_params` validation happens deep in the simulation code, but calling code doesn't guarantee structure until runtime in worker processes.

**Impact**: Errors only surface after expensive optimization setup. Poor error messages.

**Fix**: Validate params structure early in `_sim_params_from_point()`:
```python
def _sim_params_from_point(point):
    """Build sim_params dict from one point (list). Validates structure."""
    params = _point_to_params(point)

    # Validate required fields early
    required = ['magnetic_moment_fudge', 'magnetic_field_fudge']
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"Missing required params: {missing}")

    # ... rest of conversion
```

---

## Minor Issues

### 8. Missing Type Hints
**Location**: Throughout both files

**Issue**: No function has type annotations.

**Impact**: Harder to catch bugs, unclear interfaces, poor IDE support.

**Fix**: Add type hints progressively, starting with public functions:
```python
def calculate_cost(
    trajectory: list[dict],
    target_velocity: float,
    verbose: bool = True
) -> dict:
    """Calculate cost with detailed metrics."""
```

---

### 9. Inconsistent String Formatting
**Location**: Throughout both files

**Issue**: Mix of f-strings and other formatting approaches.

**Impact**: Harder to read, inconsistent style.

**Fix**: Standardize on f-strings throughout (already mostly done, just finish).

---

### 10. Long `_run_batch_optimization()` Function
**Location**: [tune_multi_params_optimized.py:244-338](mujoco/tune_multi_params_optimized.py#L244-L338)

**Issue**: This function does everything: setup, batch loop, progress printing, CSV writing, pool management. 95 lines long.

**Impact**: Hard to understand flow, difficult to test individual pieces.

**Fix**: Extract smaller functions:
```python
def _initialize_optimizer(seed_point=None):
    """Create optimizer with optional seed."""

def _print_batch_summary(batch_num, results, n_done, n_total):
    """Print progress for one batch."""

def _run_batch_optimization():
    """Main optimization loop - orchestrates pieces."""
```

---

## Recommended Action Plan

### Phase 1: High ROI, Low Risk (2-3 hours)
**Immediate wins with minimal refactoring:**

1. ✅ Delete dead `run_simulation_worker()` code (lines 13-34)
2. ✅ Import `SETTLE_TIME` instead of duplicating `COST_SETTLE_TIME`
3. ✅ Extract solimp magic numbers to named constants
4. ✅ Add docstring clarifications where logic is complex

**Expected Impact**: Reduce lines by ~20, eliminate one critical bug source.

---

### Phase 2: Improves Maintainability (4-6 hours)
**Better structure without major rewrites:**

4. Move globals (`all_results`, `pool`) into main function scope
5. Add validation for params dict structure early in conversion
6. Add type hints to key functions (`calculate_cost`, `_sim_params_from_point`, `run_simulation`)
7. Create shared constants module for values used across files

**Expected Impact**: Easier testing, better error messages, clearer interfaces.

---

### Phase 3: Structural Improvements (1-2 days)
**Only if planning significant future development:**

8. Split `tune_multi_params_optimized.py` into:
   - `optimization_config.py` - constants, space definition
   - `cost_functions.py` - cost calculation utilities
   - `tune_multi_params_optimized.py` - orchestration
9. Refactor `_run_batch_optimization()` into smaller functions
10. Unify parameter conversion into single source of truth (possibly a dataclass)
11. Add comprehensive unit tests for cost calculation and parameter conversion

**Expected Impact**: Highly maintainable codebase, easy to add new features, testable components.

---

## Priority Ranking

**If you can only do one thing**: Fix #2 (constant synchronization) - it's the most likely to cause silent bugs.

**If you have an afternoon**: Do Phase 1 - quick wins that make the code noticeably better.

**If you're planning to extend this code**: Do Phases 1 & 2 first, then evaluate if Phase 3 is worth it based on your roadmap.

---

## Notes

- The core simulation logic in `sim_optimizer_couple.py` is well-structured with good separation of concerns
- The optimization script works but has grown organically and could benefit from refactoring
- No critical performance issues identified - parallelization strategy is sound
- CSV writing and result tracking is functional but tightly coupled

---

## Questions to Consider

1. Will you be adding more scenes/robots to optimize? (Affects whether splitting files is worth it)
2. Do you need to support multiple optimization algorithms? (Affects abstraction level)
3. Is the current CSV output format final? (Affects how tightly to couple it)
4. Will you need to replay/visualize arbitrary parameter sets? (Affects validation needs)
