# File Triage & Restructuring Task

## Goal

Classify every `.py`, `.md`, and `.sh` file in `mujoco_refactor/` into one of:
1. **Core** — needed every time you run an optimization (stay in root)
2. **Post-run tools** — used after every optimization run to inspect/replay results (stay in root)
3. **Analysis** — reusable but occasional (move to `analysis/`)
4. **Archive** — one-off debugging scripts for bugs that are now fixed, or investigation notes that have been captured elsewhere (move to `archive/`)

Then propose the moves and execute after user approval.

## Context

The core workflow is:
```
config.py  →  optimizer.py  →  show_bests.py / replay_best.py
```

Everything else is either a diagnostic tool, a one-off debugging script, or documentation.

## Classification Rules

- If a script was written to test a specific bug fix (condim=6, solimp overlap, solref dead zone), and that fix is now permanently in the codebase, it's **archive**.
- If a script is useful every time a new optimization run finishes (viewing results, replaying sims, plotting diagnostics), it's **post-run tools**.
- If a script was part of a one-time experiment (per-morphology sweep, COT comparison) but could be rerun if the experiment is repeated, it's **analysis**.
- If a `.md` documents a resolved investigation whose findings are already captured in MEMORY.md or config.py comments, it's **archive**.

## Already-completed triage (from subagent analysis)

### Core (stay in root)
| File | Why |
|---|---|
| `config.py` | Single source of truth for constants, search space, references |
| `simulation.py` | Reference simulation engine |
| `simulation_fast.py` | Optimized simulation (~5x faster) |
| `optimizer.py` | CMA-ES optimization loop |

### Post-run tools (stay in root)
| File | Why |
|---|---|
| `show_bests.py` | Pretty-print optimization_bests.csv — used after every run |
| `replay_best.py` | Replay exact best-cost sims with deterministic seeds |
| `replay.py` | Replay a specific CSV result in interactive viewer |
| `visualize_rollout.py` | Visualize/record rollouts with COT analysis |
| `plot_torques.py` | Deep 7-subplot diagnostic (torque, velocity, angles, RPY) |
| `test_20hz.py` | Holdout validation script — reusable for any frequency holdout |
| `replay_cmaes_state.py` | Reconstruct CMA-ES state for resuming interrupted runs |
| `verify_sim.py` | Regression test: confirms simulation_fast matches simulation.py |

### Analysis (move to `analysis/`)
| File | Why |
|---|---|
| `compare_morphology_params.py` | Per-morphology sweep comparison (2026-02-18 experiment). Reusable if sweep is repeated. |
| `compare_cot.py` | COT comparison across top-N results. Depends on `visualize_rollout.py`. |
| `check_param_bounds.py` | Shows where top solutions cluster across all results dirs. Uses old `solimp_dmax` column name. |
| `run_per_morphology.sh` | Launcher for per-morphology/per-frequency sweep experiments |
| `test_param_sensitivity.py` | Tests every search dimension affects sim output. Useful when search space changes. |

### Archive (move to `archive/`)
| File | Why |
|---|---|
| `check_solimp_overlap.py` | Tested solimp_dmin >= dmax bug — now fixed by delta_d reparameterization |
| `test_condim_fix.py` | Tested condim=6 torsional/rolling friction sweeps — condim=6 is now permanent |
| `test_friction_detail.py` | Verified friction values reach MuJoCo contacts at condim=6 — done |
| `test_solref_tc.py` | Found solref_timeconst dead-zone boundary — findings already in MEMORY.md |
| `config_20260219_loose_fudge_backup.py` | Config backup from before adding 20Hz data |

### Documentation
| File | Action | Why |
|---|---|---|
| `README.md` | Keep, but **needs update** — search space table is outdated (old ranges, missing delta_d) |
| `20HZ_VALIDATION.md` | Keep — active reference for 20Hz holdout results |
| `optimization_journey.md` | Archive — historical narrative, findings captured in MEMORY.md |
| `test_plan_actionable_bugs.md` | Archive — past bug investigation plan, bugs resolved |
| `try_fix_scene2.md` | Archive — scene2 CMA-ES stagnation analysis, fix (warm-start) already implemented |

## Proposed directory structure after cleanup

```
mujoco_refactor/
├── config.py
├── simulation.py
├── simulation_fast.py
├── optimizer.py
├── show_bests.py
├── replay.py
├── replay_best.py
├── replay_cmaes_state.py
├── visualize_rollout.py
├── plot_torques.py
├── test_20hz.py
├── verify_sim.py
├── README.md
├── 20HZ_VALIDATION.md
├── analysis/
│   ├── compare_morphology_params.py
│   ├── compare_cot.py
│   ├── check_param_bounds.py
│   ├── test_param_sensitivity.py
│   └── run_per_morphology.sh
├── archive/
│   ├── check_solimp_overlap.py
│   ├── test_condim_fix.py
│   ├── test_friction_detail.py
│   ├── test_solref_tc.py
│   ├── config_20260219_loose_fudge_backup.py
│   ├── optimization_journey.md
│   ├── test_plan_actionable_bugs.md
│   └── try_fix_scene2.md
├── multi_milli_quad/          (MJCF models - unchanged)
├── wheel_milli_quad/          (MJCF models - unchanged)
└── results/                   (optimization outputs - unchanged)
```

## Execution steps

1. `mkdir -p analysis archive`
2. `git mv` each file to its new location
3. Update any cross-file imports if needed (analysis scripts that import from root — they'll need `sys.path` or relative imports)
4. Verify nothing broke: `uv run python -c "from config import reference_rows; print(len(reference_rows()))"`
5. Update README.md search space table to match current `space` in config.py
