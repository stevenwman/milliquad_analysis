# Multi-Terrain Optimizer: Final Configuration Summary

## Executive Summary

The multi-terrain optimizer addresses the critical overfitting problem discovered in single-terrain optimization, where parameters optimized on flat terrain fail catastrophically on step terrain (92.8% mean error). This optimizer simultaneously optimizes across **both flat and step terrain** to find parameters that generalize.

**Key Design Principles:**
- **Hierarchical cost structure**: Within-terrain aggregation → across-terrain weighted sum
- **Terrain-specific cost functions**: Different penalties for flat vs step (e.g., yaw enabled on flat, disabled on step)
- **Consistency enforcement**: Velocity variance penalty encourages uniform performance across morphologies/frequencies
- **Reduced reference set**: 19 references (down from 23) based on correlation analysis, saving 19% compute
- **Failure mode constraint**: Scene_wheel f20 explicitly constrained to velocity=0 (weight=2.0)

**Expected Outcome**: Parameters that work reasonably well on both terrains, trading off peak flat performance for step terrain capability.

---

## Technical Configuration

### Reference Set (19 total)
```
FLAT TERRAIN (11 refs):
  - scene1 × [f10, f30, f50]
  - scene2 × [f10, f30, f50]
  - scene4 × [f10, f30, f50]
  - scene_wheel × [f20*, f30]

STEP TERRAIN (8 refs):
  - scene1 × [f10, f30]
  - scene2 × [f10, f30]
  - scene4 × [f10, f30]
  - scene_wheel × [f20*, f30]

* = Failure mode constraint (target=0.0, weight=2.0)
```

**Computational load**: 19 refs × 2 jitter trials = **38 simulations per evaluation**

### Cost Function Structure

**Within-Terrain Aggregation:**
- **Flat terrain**: MEDIAN of 2 jitter trials
- **Step terrain**: BEST (argmin) of 2 jitter trials (robust to cliff-fall artifacts)
- **Velocity variance penalty**: Added to each terrain's aggregate cost
  - Penalizes inconsistent performance across references within terrain
  - `variance_cost = 2.0 × var(relative_errors)`
  - Skips failure mode constraints (target=0) when computing variance

**Across-Terrain Aggregation:**
```python
total_cost = FLAT_TERRAIN_WEIGHT × flat_cost + STEP_TERRAIN_WEIGHT × step_cost
           = 1.0 × flat_cost + 1.0 × step_cost
```

**Component Weights (Terrain-Specific):**
```python
FLAT:  velocity=5, lateral=5, tumble=1, yaw=1
STEP:  velocity=5, lateral=1, tumble=1, yaw=0
```

### CMA-ES Settings

```python
N_CALLS = 4800              # 2× single-terrain budget
BATCH_SIZE = 8              # Parallel evaluations
CMAES_SIGMA0 = 0.5          # Larger exploration radius
INIT_JITTER_TRIALS = 2
SIM_DURATION = 3.0          # Seconds (both terrains)
```

**Warm-start**: From best flat-terrain params (cost=0.380 on flat, ~12.3 on step)
- Shows ~16.7× cost imbalance initially
- Optimizer will explore to balance both terrains

**Search space**: 13 dimensions (same as flat/step optimizers)

---

## Design Rationale

### 1. Why terrain-specific cost weights?
Flat and step terrain have different physics:
- **Flat**: Lateral drift and yaw deviation are critical failure modes
- **Step**: Lateral drift matters less (constrained by step walls), yaw corrupted by cliff-fall artifact

Using the same weights would mis-prioritize on each terrain.

### 2. Why velocity variance penalty?
Without it, the optimizer can "cheat" by:
- Optimizing perfectly for one morphology at the expense of others
- Causing high variation in performance across frequencies

Variance penalty encourages **consistent** performance across all conditions.

### 3. Why unequal terrain-level weights (1.0, 1.0) despite 16.7× imbalance?
Two philosophies:
- **Balanced contributions**: Adjust weights so flat and step contribute equally (would need STEP_WEIGHT=0.06)
- **Equal priorities**: Keep weights equal, let optimizer find natural balance

We chose **equal priorities** because:
- Step terrain is the harder problem (16.7× higher cost)
- Optimizer will naturally focus effort where cost is highest
- We want step capability, not just flat optimization with step as afterthought

If optimizer stagnates on step at expense of flat, we can rebalance.

### 4. Why f20 dropout for most morphologies?
Correlation analysis showed f20 highly correlated with f10 and f30 (redundant information). **Exception**: scene_wheel f20 is a failure mode (robot doesn't move), which is NOT redundant with f10/f30 success — it's a distinct constraint.

Kept f20 only for scene_wheel, dropped elsewhere → 19% compute savings.

### 5. Why sigma0=0.5 instead of 0.3?
Warm-start params work well on flat but fail on step. This is a **large exploration problem**:
- Need to escape the "flat-optimized basin"
- Need to find compromises in parameter space

Larger sigma0 (0.5) encourages more exploration initially.

### 6. Why 4800 evaluations?
Multi-terrain optimization is harder than single-terrain:
- Larger effective search space (balancing two objectives)
- More simulations per evaluation (19 refs vs 11 or 10)
- Need more exploration to find compromise solutions

2× single-terrain budget (2400 → 4800) is a conservative estimate.

---

## Expected Behavior During Optimization

**Phase 1 (evals 0-500): Initial exploration**
- CMA-ES explores broadly around warm-start params
- Cost should drop rapidly from ~12.3 toward ~5-8 range
- Flat cost may increase slightly, step cost should decrease significantly

**Phase 2 (evals 500-2000): Refinement**
- Optimizer balances flat vs step performance
- Cost improvements slow down
- Expect cost in range 2-5 (rough estimate)

**Phase 3 (evals 2000-4800): Fine-tuning**
- Diminishing returns
- Cost may plateau around 1-3 (best case scenario)
- Parameter variance (sigma) shrinks as CMA-ES converges

**Best-case outcome**: Cost ~1-2 (both terrains working reasonably well)
**Worst-case outcome**: Cost ~5-10 (stuck in poor compromise, may need rebalancing)

**CSVs written**:
- `multi_optimization_results.csv`: All evaluations
- `optimization_bests.csv`: Progressive best parameters

**Video recording**: Only for rank-1 (best) evaluations

---

## Computational Estimates

```
Time per evaluation:
  - 38 sims × 3.0s = 114 sim-seconds
  - Batch size 8 → 114/8 = 14.3 seconds per batch (assuming perfect parallelization)
  - Add overhead (physics engine, I/O, cost calculation): ~17-22s per batch

Total evaluations: 4800
Total batches: 4800 / 8 = 600 batches
Estimated runtime: 600 × 17s = 10,200s ≈ 2.8 hours (best case)
                   600 × 22s = 13,200s ≈ 3.7 hours (realistic)

Conservative estimate accounting for non-parallel overhead: 4-5 hours
```

**Note**: Reduced from 3 jitter trials to 2 (33% speedup). With batch_size=8 and multiprocessing, substantially faster than original estimate.

---

## Launch Command

```bash
cd mujoco_refactor
python optimizer_multi_terrain.py --suffix multi_v1
```

**Output directory**: `results/YYYYMMDDTHHMMSS_multi_v1/`

**Files generated**:
- `multi_optimization_results.csv`: Full evaluation history
- `optimization_bests.csv`: Progressive best parameters
- `config.py`: Snapshot of config_multi_terrain.py
- `cmaes_state.pkl`: CMA-ES state for potential resumption

---

## Post-Run Analysis

**Check convergence**:
```bash
python show_bests_multi_terrain.py results/<run_dir>
```

**Validate on terrains**:
```bash
python test_terrain_cost_balance.py  # Using best params from run
```

**Compare to single-terrain optimizers**:
- Flat-only: cost=0.380 on flat, ~93% error on steps
- Step-only: (pending run)
- Multi-terrain: (goal) cost ~1-3 on both, balanced performance

---

## Known Risks and Mitigations

**Risk 1**: Optimizer stagnates on step terrain, sacrifices flat performance
- **Mitigation**: Adjust STEP_TERRAIN_WEIGHT down (e.g., 0.5) and re-run
- **Detection**: Monitor `cost_flat` and `cost_step` columns in CSV

**Risk 2**: 4800 evaluations insufficient to converge
- **Mitigation**: Extend run with `--warm-start-from <multi_v1_dir>` for another 2400 evals
- **Detection**: Check if cost is still improving at eval 4800

**Risk 3**: Failure mode constraint (f20) dominates cost function
- **Mitigation**: Reduce weight from 2.0 → 1.0 if it's causing problems
- **Detection**: Check `cost_scene_wheel_f20_flat` and `cost_scene_wheel_f20_step` columns

**Risk 4**: Variance penalty causes optimizer to "give up" on hard references
- **Mitigation**: Reduce VELOCITY_VARIANCE_WEIGHT from 2.0 → 1.0 or 0.5
- **Detection**: Check if velocity variance stays high while component costs drop

---

## Success Criteria

**Minimum viable**:
- ✓ Flat terrain velocity error < 30% on average
- ✓ Step terrain velocity error < 50% on average
- ✓ No catastrophic failures (tumble/yaw) on either terrain

**Target performance**:
- ✓ Flat terrain velocity error < 15%
- ✓ Step terrain velocity error < 30%
- ✓ Velocity variance (per terrain) < 0.1

**Stretch goal**:
- ✓ Both terrains < 10% velocity error
- ✓ Generalizes to rough terrain without re-optimization

---

## File Inventory

**Core optimizer files**:
- `config_multi_terrain.py`: Configuration and reference data
- `optimizer_multi_terrain.py`: Multi-terrain optimization loop
- `show_bests_multi_terrain.py`: Results visualization
- `test_terrain_cost_balance.py`: Cost magnitude testing

**Step terrain XMLs** (auto-generated at startup):
- `multi_milli_quad/scene_1_step_8x1mm_4.5L_50lead.xml`
- `multi_milli_quad/scene_2_step_8x1mm_4.5L_50lead.xml`
- `multi_milli_quad/scene_4_step_8x1mm_4.5L_50lead.xml`
- `wheel_milli_quad/scene_wheel_step_8x1mm_4.5L_50lead.xml`

**Analysis scripts**:
- `analyze_ref_correlations.py`: Reference redundancy analysis

---

## Implementation History

**2026-02-23**: Multi-terrain optimizer created
- Discovered single-terrain overfitting (13-dim: 92.8% error, 16-dim: 114.2% error on steps)
- Designed hierarchical cost structure with terrain-specific weights
- Added velocity variance penalty for consistency
- Configured warm-start from best flat params (cost=0.380)
- Empirically measured 16.7× cost imbalance (flat: 0.74, step: 12.31)
- Set equal terrain-level weights (1.0, 1.0) for step-focused optimization
- Configured CMA-ES with sigma0=0.5 for larger exploration
- Budget: 4800 evaluations (~6-8 hours estimated)

---

## Ready to Launch

All files verified:
- ✓ `config_multi_terrain.py` imports successfully
- ✓ `optimizer_multi_terrain.py` imports successfully
- ✓ `test_terrain_cost_balance.py` runs without errors
- ✓ Step XMLs generated and present
- ✓ Cost functions handle divide-by-zero edge case
- ✓ Terrain-specific weights configured correctly

**Status**: Ready for launch. Awaiting go-ahead to start optimization run.
