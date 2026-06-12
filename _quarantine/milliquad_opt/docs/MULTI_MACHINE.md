# Multi-Machine Setup Guide

## First-Time Setup on a New Machine

```bash
git checkout mjc_clean
git pull origin mjc_clean
cd milliquad_opt
uv sync                              # install deps
uv run python generate_terrain_xmls.py   # REQUIRED: regenerates rough XMLs with correct absolute paths
```

**Why `generate_terrain_xmls.py`?** Rough terrain XMLs embed absolute paths to heightmap PNGs (MuJoCo requirement). These paths are machine-specific and won't work if copied from another machine via git. Step and flat XMLs use relative paths and work everywhere.

## Running Optimizations

```bash
# Warm-start (uses CMAES_X0 from config)
uv run python optimizer.py --terrain flat --suffix rk4_flat
uv run python optimizer.py --terrain step --suffix rk4_step
uv run python optimizer.py --terrain rough --suffix rk4_rough

# Cold-start (X0=None, wider sigma)
uv run python optimizer.py --terrain flat_cold --suffix rk4_flat_cold
uv run python optimizer.py --terrain step_cold --suffix rk4_step_cold
uv run python optimizer.py --terrain rough_cold --suffix rk4_rough_cold
```

## Cold-Start Configs

`config_{terrain}_cold.py` are thin overlays that inherit everything from the base config but set:
- `CMAES_X0 = None` (no warm-start)
- `CMAES_SIGMA0 = 0.5` (wider exploration)

## Git Workflow Across Machines

### While runs are active
- **Safe**: `git fetch`, `git log`, `git diff` (read-only)
- **Safe**: editing non-running code, committing, pushing
- **UNSAFE**: `git checkout` / `git pull` that overwrites files the running process has open (especially CSVs in `results/`). On Linux, the process keeps writing to the old inode, which gets unlinked — writes are lost when the process exits.

### Merging results from multiple machines
1. Wait for runs to finish on each machine
2. Commit + push results from each machine to its own branch (or to `mjc_clean` sequentially)
3. Results dirs don't overlap (unique timestamps), so merges are conflict-free

### Merging code changes
- `optimizer.py` is the most likely conflict point
- After merging, verify with `git diff branch_a branch_b -- milliquad_opt/optimizer.py`

## Terrain Type Matching

The `--terrain` arg maps to `config_{terrain}.py` via `importlib`. The optimizer uses `startswith("rough")` / `startswith("step")` to dispatch terrain-specific logic (spawn offsets, jitter type, CSV columns). So `rough_cold`, `step_cold` etc. all route correctly.
