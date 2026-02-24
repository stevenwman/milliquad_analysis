# AGENT TRANSFER: Multi-Terrain Optimizer Context

**Date**: 2026-02-24
**Status**: Multi-terrain optimizer running (started 13:18, ~3.5-4 hours remaining)
**Background Task ID**: b2328d7

---

## Current Situation

The multi-terrain optimizer is **RUNNING IN BACKGROUND** with output at:
```
/tmp/claude-1000/-home-sman-Work-CMU-Research-LEGO-milliquad-mujoco/tasks/b2328d7.output
```

**Monitor progress**:
```bash
tail -f /tmp/claude-1000/-home-sman-Work-CMU-Research-LEGO-milliquad-mujoco/tasks/b2328d7.output
```

**Results directory**:
```
mujoco_refactor/results/20260224T131848_multi_terrain_multi_v1/
```

---

## Problem: Single-Terrain Optimization Overfits

- 13-dim flat-optimized: 92.8% error on steps
- 16-dim flat-optimized: 114.2% error on steps
- **Solution**: Multi-terrain optimizer (flat + step simultaneously)

---

## Configuration Summary

### References: 19 total (38 sims/eval)
- Flat: scene1/2/4 × f10/f30/f50 + scene_wheel f10/f30 (11 refs)
- Step: scene1/2/4 × f10/f30 + scene_wheel f20/f30 (8 refs, f20 = failure mode)
- Jitter: 2 trials per reference

### Cost Structure
- Hierarchical: within-terrain → across-terrain
- Flat: MEDIAN of jitter trials
- Step: BEST (argmin) of jitter trials
- Variance penalty: 2.0 × var(relative_errors) per terrain

### Terrain Weights
- Flat terrain: vel=5, lateral=5, tumble=1, yaw=1
- Step terrain: vel=5, lateral=1, tumble=1, yaw=0 (disabled due to cliff-fall)
- Terrain-level: flat_weight=1.0, step_weight=1.0

### CMA-ES
- N_CALLS=4800, BATCH_SIZE=8, SIGMA0=0.5
- Warm-start from flat params (cost=0.380→~12.3 on step)
- Large sigma for exploration out of flat-optimized basin

---

## Key Files

### Must Read First
1. **MULTI_TERRAIN_OPTIMIZER_SUMMARY.md** - Complete design doc
2. **config_multi_terrain.py** - Configuration (19 refs, weights)
3. **optimizer_multi_terrain.py** - Main optimizer (~1100 lines)

### Supporting
- config.py - Base 13-dim search space
- simulation_fast.py - MuJoCo engine
- show_bests_multi_terrain.py - Results viewer
- analyze_ref_correlations.py - Why 19 refs not 23

---

## Monitoring & Analysis

### Check Progress
```bash
tail -f /tmp/claude-1000/-home-sman-Work-CMU-Research-LEGO-milliquad-mujoco/tasks/b2328d7.output
```

### View Best Results
```bash
cd mujoco_refactor
python show_bests_multi_terrain.py results/20260224T131848_multi_terrain_multi_v1
```

### Quick CSV Check
```bash
cd mujoco_refactor/results/20260224T131848_multi_terrain_multi_v1
tail -n 10 optimization_bests.csv | column -t -s','
```

---

## Expected Behavior

### Initial (Observed)
- Cost: 14.43 (best) to 555k (failed sims)
- Best: flat=1.50, step=12.93
- 16.7× imbalance as expected

### Phase 1 (0-500 evals): Exploration
- Cost drops 12→5-8 range
- Step cost improves significantly
- Flat may increase slightly

### Phase 2 (500-2000): Refinement
- Balancing act between terrains
- Cost 2-5 range

### Phase 3 (2000-4800): Fine-tuning
- Plateau around 1-3 (best case)
- Diminishing returns

### Success Criteria
- Min: Flat <30%, Step <50% error
- Target: Flat <15%, Step <30%
- Stretch: Both <10%

---

## When Finished

1. **Analyze**: `python show_bests_multi_terrain.py results/...`
2. **Compare** to single-terrain (flat=0.380, step=92.8% error)
3. **Validate** on rough terrain, different step heights
4. **Next steps** (if successful): Generalization testing
5. **Next steps** (if failed): Adjust weights, extend run, or increase sigma

---

## Critical Design Decisions

1. **19 refs not 23**: f20 correlated except scene_wheel (failure mode)
2. **Terrain-specific weights**: Flat needs yaw+lateral, step doesn't (cliff-fall artifact)
3. **Equal terrain weights**: Let optimizer focus on harder problem (step)
4. **MEDIAN vs BEST**: Flat tolerates outliers, step avoids cliff-fall
5. **sigma0=0.5**: Large exploration to escape flat-optimized basin

---

## Known Issues

1. **Scene naming**: REFERENCE_DATA uses `scene1`, MJCF uses `scene_1.xml`
2. **Divide-by-zero**: scene_wheel f20 target=0.0, check before normalizing
3. **Cliff-fall**: Robot falls off last step → yaw corruption → disabled for steps
4. **Step XMLs**: Must be alongside originals for relative paths

---

## System Info

- CPU: Intel i9-13900H (10 physical, 20 logical cores)
- RAM: 15GB
- Optimal batch_size: 8-10 (currently 8)

---

## If Problems Occur

### Stalled/Failed
1. Check simulation failures (cost > 1000)
2. Check imbalance (one terrain dominating)
3. Extend: `--warm-start-from <dir> --n-calls 2400`
4. Rebalance: Edit STEP_TERRAIN_WEIGHT in config
5. Reduce variance: VELOCITY_VARIANCE_WEIGHT 2.0→1.0

### Need Help
- Read: MULTI_TERRAIN_OPTIMIZER_SUMMARY.md (comprehensive)
- Check: memory/MEMORY.md (architecture, lessons learned)
- Code: optimizer_multi_terrain.py (implementation)

---

## TL;DR for New Agent

**Status**: Multi-terrain optimizer running in background (task b2328d7)
**Goal**: Find params that work on BOTH flat and step terrain (not just flat)
**Why**: Single-terrain optimizers overfit catastrophically (93-114% error)
**Monitor**: `tail -f /tmp/.../b2328d7.output`
**Results**: `mujoco_refactor/results/20260224T131848_multi_terrain_multi_v1/`
**When done**: Run `show_bests_multi_terrain.py` and compare to flat-only (0.380 cost, 93% step error)

Good luck! 🚀
