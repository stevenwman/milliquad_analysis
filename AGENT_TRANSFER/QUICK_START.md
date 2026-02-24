# QUICK START: What to Do Right Now

## Background Task is RUNNING

Task ID: `b2328d7`
Started: 13:18 (2026-02-24)
Estimated completion: ~16:48-17:18 (3.5-4 hours total)

**DO NOT STOP THIS TASK** unless there's an error.

---

## Immediate Actions

### 1. Check if optimizer is still running
```bash
tail -n 50 /tmp/claude-1000/-home-sman-Work-CMU-Research-LEGO-milliquad-mujoco/tasks/b2328d7.output
```

**Look for**:
- Batch progress printouts (e.g., "Batch 150/600")
- Cost decreasing over time
- NO Python tracebacks or errors

### 2. Check current progress
```bash
cd /home/sman/Work/CMU/Research/LEGO-milliquad-mujoco/mujoco_refactor
python show_bests_multi_terrain.py results/20260224T131848_multi_terrain_multi_v1
```

**Expected**:
- Cost should be decreasing from initial ~14
- Both flat and step costs should be present
- No catastrophic failures (cost > 1000)

### 3. Monitor in real-time (optional)
```bash
watch -n 10 'tail -n 30 /tmp/claude-1000/-home-sman-Work-CMU-Research-LEGO-milliquad-mujoco/tasks/b2328d7.output'
```

---

## What's Running

**Multi-Terrain Optimizer**:
- Optimizing MuJoCo params for BOTH flat and step terrain
- 4800 evaluations total, batch_size=8
- 19 references × 2 jitter = 38 sims per evaluation
- Warm-started from flat-optimized params (which fail on steps)

**Goal**: Find params that generalize to both terrains instead of overfitting to flat.

---

## When Optimizer Finishes

### Immediate Analysis
```bash
cd mujoco_refactor
python show_bests_multi_terrain.py results/20260224T131848_multi_terrain_multi_v1
```

### Check Final Cost
Look at last line of `optimization_bests.csv`:
```bash
tail -n 1 results/20260224T131848_multi_terrain_multi_v1/optimization_bests.csv
```

**Interpret**:
- `cost`: Total combined cost
- `cost_flat`: Flat terrain cost
- `cost_step`: Step terrain cost
- `velocity_flat`: Avg flat velocity
- `velocity_step`: Avg step velocity

### Success Metrics
- **Minimum viable**: Total cost < 10, both terrains functional
- **Good**: Total cost < 5, flat <15% error, step <30% error
- **Excellent**: Total cost < 2, both <10% error

### Compare to Baseline
- Flat-only optimizer: cost=0.380 on flat, 92.8% error on steps (FAILED)
- Multi-terrain should: ~1-5 cost, both terrains <30% error (SUCCESS)

---

## If Something Goes Wrong

### Optimizer Crashed
```bash
# Check for errors in output
grep -i "error\|traceback\|exception" /tmp/claude-1000/.../tasks/b2328d7.output | tail -20
```

### Costs Exploding
- Check if simulations are failing (cost > 1000)
- May need to adjust weights or sigma

### Cost Not Improving
- Check if optimizer is exploring (sigma shrinking?)
- May need to extend run or increase sigma

### Need to Stop/Restart
```bash
# Find PID
ps aux | grep optimizer_multi_terrain.py

# Kill if needed (last resort)
kill <PID>

# Restart with warm-start
cd mujoco_refactor
uv run python optimizer_multi_terrain.py --suffix multi_v1_restart --warm-start-from results/20260224T131848_multi_terrain_multi_v1
```

---

## Next Steps (After Completion)

### 1. Validate Results
- Run terrain_test.py on best params (TODO: create multi-terrain version)
- Test on rough terrain, different step heights
- Generate videos

### 2. Compare to Single-Terrain
- Flat-only: 0.380 cost on flat, 93% error on steps
- Step-only: (not yet run)
- Multi-terrain: (check results)

### 3. If Successful
- Document in MEMORY.md
- Consider extending to rough terrain
- Test generalization to new morphologies

### 4. If Unsuccessful
- Adjust terrain weights (config_multi_terrain.py)
- Extend run (--warm-start-from, --n-calls 2400)
- Increase exploration (CMAES_SIGMA0 = 0.7)
- Reduce variance penalty (VELOCITY_VARIANCE_WEIGHT = 1.0)

---

## Key Files for New Agent

**Must read**:
1. README.md (this directory) - Full context
2. MULTI_TERRAIN_OPTIMIZER_SUMMARY.md - Design doc
3. config_multi_terrain.py - Configuration

**Reference**:
- optimizer_multi_terrain.py - Implementation
- memory/MEMORY.md - Project lessons learned
- results/20260224T131848_multi_terrain_multi_v1/ - Output

---

## Commands Cheat Sheet

```bash
# Check if running
tail /tmp/claude-1000/-home-sman-Work-CMU-Research-LEGO-milliquad-mujoco/tasks/b2328d7.output

# View progress
cd mujoco_refactor
python show_bests_multi_terrain.py results/20260224T131848_multi_terrain_multi_v1

# Check CSV
tail results/20260224T131848_multi_terrain_multi_v1/optimization_bests.csv

# Kill if needed
ps aux | grep optimizer_multi_terrain.py
kill <PID>

# Restart with warm-start
uv run python optimizer_multi_terrain.py --warm-start-from results/20260224T131848_multi_terrain_multi_v1 --n-calls 2400 --suffix extended
```

---

## Contact Points

- **Task output**: /tmp/claude-1000/.../tasks/b2328d7.output
- **Results**: mujoco_refactor/results/20260224T131848_multi_terrain_multi_v1/
- **Config**: mujoco_refactor/config_multi_terrain.py
- **Code**: mujoco_refactor/optimizer_multi_terrain.py
- **Docs**: AGENT_TRANSFER/MULTI_TERRAIN_OPTIMIZER_SUMMARY.md
