# Reproducibility Notes

## Final 3 Optimization Runs

All runs use `optimizer.py` with 16-dim search space, RK4 integrator, and terrain-specific configs.

### 1. Flat — `20260228T013353_rk4_flat`

- **Config**: `config_flat.py` (snapshotted in results dir)
- **Cost**: 0.188
- **Warm-start**: X0 hardcoded in config from `mujoco_refactor/results/20260225T122342_flat_10_30_50` (Euler, cost=0.1276)
- **CLI** (reconstructed): `uv run python optimizer.py --suffix rk4_flat`
- **No `--warm-start-from`** — X0 was manually copied into `CMAES_X0` dict

### 2. Step — `20260228T230022_step_q60_rk-warm`

- **Config**: `config_step_q60.py` → imports `config_step.py` (snapshotted in results dir)
- **Cost**: 0.210
- **Warm-start**: X0 hardcoded in config from `20260228T093833_rk4_step_cold` (RK4, cost=0.243)
- **CLI** (reconstructed): `uv run python optimizer.py --suffix step_q60_rk-warm`
- **No `--warm-start-from`** — X0 was manually copied into `CMAES_X0` dict

### 3. Rough — `20260228T202903_rough_spatial_rk4`

- **Config**: `config_rough_spatial.py` → imports `config_rough.py` (spatial snapshotted, rough NOT snapshotted)
- **Cost**: 0.184
- **Warm-start**: `--warm-start-from` at runtime from `results/archive/20260228T102010_rk4_rough` (cost=0.362)
  - That run's X0 came from `config_rough.py` CMAES_X0 (Euler `zzz_rough_v2`, cost=0.395)
- **CLI** (reconstructed): `uv run python optimizer.py --suffix rough_spatial_warm_rk4 --warm-start-from results/20260228T102010_rk4_rough`
- **Note**: `config_rough.py` was NOT snapshotted in results dir. The inherited CMAES_X0 was overridden by `--warm-start-from`, so config_rough's X0 was irrelevant at runtime. The actual X0 is the best params from `results/archive/20260228T102010_rk4_rough/optimization_bests.csv`.

## Warm-Start Provenance Chain

```
Flat:   Euler flat (mujoco_refactor/) → [copied X0] → rk4_flat (final)
Step:   rk4_step_cold → [copied X0] → step_q60_rk-warm (final)
Rough:  Euler rough (zzz_rough_v2) → [config X0] → rk4_rough → [--warm-start-from] → rough_spatial_rk4 (final)
```

## Key Archived Runs (in results/archive/)

| Dir | Role | Cost |
|-----|------|------|
| `20260228T102010_rk4_rough` | Rough warm-start source for final run | 0.362 |
| `20260228T122805_rk4_rough_cold` | Rough cold-start (failed, cost=12.39) | 12.395 |
