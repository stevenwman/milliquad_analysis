# File Reorganization Guide

**Date**: 2026-02-25
**Purpose**: Promote 16-dim system as primary, move multi-terrain experiments to subdirectory

---

## Step 1: Manual File Moves

Execute these commands:

```bash
cd mujoco_refactor

# Create experiments directory
mkdir -p experiments

# Move multi-terrain files
mv config_multi_terrain.py experiments/
mv optimizer_multi_terrain.py experiments/
mv show_bests_multi_terrain.py experiments/
mv MULTI_TERRAIN_OPTIMIZER_SUMMARY.md experiments/
mv test_terrain_cost_balance.py experiments/
mv verify_reference_velocities.py experiments/
mv test_batch_size.py experiments/

# Rename config_step files (promote 16-dim as primary)
mv config_step.py config_step_13dim.py
mv config_step_new.py config_step.py
```

---

## Step 2: Fix Imports

After moving files, run:

```bash
cd mujoco_refactor
./fix_imports_after_move.sh
```

This script will update all import statements in:
- `test_batch_size.py` → imports from `experiments.config_multi_terrain`
- `verify_reference_velocities.py` → imports from `experiments.config_multi_terrain`
- `test_terrain_cost_balance.py` → imports from `experiments.config_multi_terrain` and `experiments.optimizer_multi_terrain`
- `optimizer_step.py` → imports from `config_step` (now 16-dim)
- `test_flat_params_on_steps.py` → imports from `config_step_13dim` (old 13-dim)
- `archive/old files/bodyflip_analysis/compare_bodyflip_steps.py` → imports from `config_step_13dim`

**Auto-fixed by rename** (no edits needed):
- `show_bests_step.py` → already imports `config_step`, which is now the 16-dim version
- `test_16dim_params_on_steps.py` → already imports `config_step`, which is now the 16-dim version

---

## Step 3: Validate Repeatability

Run short optimization trials to verify configurations are repeatable:

```bash
cd mujoco_refactor
./validate_optimization_repeatability.sh
```

This will:
1. Re-run each of the 3 most recent configs for **64 evaluations** (~10-15 min total)
2. Compare `multi_optimization_results.csv` entries (parameter points and costs)
3. Check for repeatability (max diff < 1e-6)
4. Report any X0 confounding issues

**Expected outcome**: All 3 runs should be **perfectly repeatable** (same random seed → same parameter proposals → same costs).

**If NOT repeatable**, check:
- X0 (warm-start dict) differs between original and validation run
- Random seed (`OPTIMIZER_RANDOM_STATE`) changed in `config.py` or `config_step.py`
- Simulation jitter seeds changed (`INIT_JITTER_SEED`)
- Config file modified between runs

---

## Import Dependency Map

### Multi-Terrain Files (moved to `experiments/`)

**config_multi_terrain.py**:
- Imported by: `optimizer_multi_terrain.py`, `show_bests_multi_terrain.py`, `test_terrain_cost_balance.py`, `test_batch_size.py`, `verify_reference_velocities.py`
- Imports from: `config_new` (stays in main directory)

**optimizer_multi_terrain.py**:
- Imported by: `test_terrain_cost_balance.py`
- Imports from: `experiments.config_multi_terrain` (same dir after move)

**show_bests_multi_terrain.py**:
- Standalone script (no external imports)
- Imports from: `experiments.config_multi_terrain`

---

### Config_Step Files (renamed)

**config_step_13dim.py** (formerly `config_step.py`):
- Imported by: `test_flat_params_on_steps.py`, `archive/old files/bodyflip_analysis/compare_bodyflip_steps.py`
- Imports from: `config` (13-dim base)

**config_step.py** (formerly `config_step_new.py`, **now primary 16-dim**):
- Imported by: `optimizer_step.py`, `show_bests_step.py`, `test_16dim_params_on_steps.py`
- Imports from: `config_new` (16-dim base)

---

## File Inventory After Reorganization

### Main Directory (`mujoco_refactor/`)

**Primary 16-dim system**:
- `config_new.py` — 16-dim flat terrain config (primary)
- `config_step.py` — 16-dim step terrain config (primary, renamed from `config_step_new.py`)
- `optimizer_new.py` — 16-dim flat optimizer
- `optimizer_step.py` — 16-dim step optimizer
- `simulation_fast_new.py` — 16-dim MuJoCo engine
- `show_bests.py` — 16-dim flat results viewer
- `show_bests_step.py` — 16-dim step results viewer

**Legacy 13-dim system** (for reference):
- `config.py` — 13-dim flat terrain config
- `config_step_13dim.py` — 13-dim step terrain config (renamed from `config_step.py`)
- `optimizer.py` — 13-dim flat optimizer
- `simulation_fast.py` — 13-dim MuJoCo engine

**Test scripts**:
- `test_16dim_params_on_steps.py` — Uses `config_new` + `config_step` (16-dim)
- `test_flat_params_on_steps.py` — Uses `config` + `config_step_13dim` (13-dim)
- `terrain_test.py` — Terrain validation (flat/step/rough)
- `terrain_config.py` — Terrain geometry presets

---

### Experiments Directory (`mujoco_refactor/experiments/`)

**Multi-terrain optimizer** (experimental, 13-dim):
- `config_multi_terrain.py` — Combined flat+step config (19 refs)
- `optimizer_multi_terrain.py` — Multi-terrain optimizer (~1100 lines)
- `show_bests_multi_terrain.py` — Multi-terrain results viewer
- `MULTI_TERRAIN_OPTIMIZER_SUMMARY.md` — Design documentation
- `test_terrain_cost_balance.py` — Cost balance testing
- `verify_reference_velocities.py` — Reference data validation
- `test_batch_size.py` — Batch size profiling

---

## Verification Checklist

After reorganization, verify:

- [ ] `experiments/` directory contains 7 multi-terrain files
- [ ] `config_step_13dim.py` and `config_step.py` both exist in main directory
- [ ] No files import from `config_step_new` (renamed to `config_step`)
- [ ] No files import from `config_multi_terrain` without `experiments.` prefix
- [ ] Validation script runs successfully (all 3 runs repeatable)
- [ ] Import fix script completes without errors

**Quick verification commands**:

```bash
cd mujoco_refactor

# Check experiments/ directory
ls -1 experiments/

# Check config_step files
ls -1 config_step*.py

# Check for old imports (should return nothing)
grep -r 'from config_step_new import' *.py 2>/dev/null
grep -r 'from config_multi_terrain import' *.py 2>/dev/null | grep -v experiments

# Run import fix script
./fix_imports_after_move.sh

# Run validation (optional, ~10-15 min)
./validate_optimization_repeatability.sh
```

---

## Rollback Instructions

If issues arise, rollback with:

```bash
cd mujoco_refactor

# Restore multi-terrain files
mv experiments/config_multi_terrain.py .
mv experiments/optimizer_multi_terrain.py .
mv experiments/show_bests_multi_terrain.py .
mv experiments/MULTI_TERRAIN_OPTIMIZER_SUMMARY.md .
mv experiments/test_terrain_cost_balance.py .
mv experiments/verify_reference_velocities.py .
mv experiments/test_batch_size.py .

# Restore config_step files
mv config_step.py config_step_new.py
mv config_step_13dim.py config_step.py

# Revert import changes
git checkout -- *.py
```

---

## Notes

**X0 Confounding**:
- All 3 recent runs use **warm-start X0** from previous optimizations
- Validation runs will use the **same X0** (copied config.py to results dir)
- If X0 differs, optimizer will explore from different starting point → different trajectory

**Random Seed**:
- CMA-ES uses `OPTIMIZER_RANDOM_STATE = 69420` (fixed across all runs)
- Jitter uses `INIT_JITTER_SEED = 12345` (fixed)
- Should guarantee repeatability if config unchanged

**16-Dim Promotion**:
- 16-dim system is now **primary** (`config_new.py`, `config_step.py`)
- 13-dim system kept for reference and legacy tests
- All recent optimization runs already using 16-dim (noslip_iterations, noslip_tolerance, margin)

**Multi-Terrain Status**:
- Experimental approach, moved to `experiments/`
- Found to be challenging due to flat/step parameter divergence (12/16 params >20% different)
- Current approach: separate single-terrain optimizers (16-dim flat + 16-dim step)
