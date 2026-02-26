# AGENT_TRANSFER2

Handoff package for new agents. Created 2026-02-26.

## Files

| File | What it covers |
|------|---------------|
| `CONTEXT_BRIEFING.md` | Full project overview: architecture, configs, CLI patterns, lessons learned |
| `SEARCH_SPACE.md` | All 16 search dimensions, convergence analysis, flat vs step param comparison |
| `VALIDATION_RESULTS.md` | Current best results, wheel chaos analysis summary, cross-terrain gaps |

## Quick Orientation

1. Read `CONTEXT_BRIEFING.md` first for the big picture
2. All active code lives in `mujoco_refactor/`
3. Two optimizers: `optimizer_new.py` (flat) and `optimizer_step.py` (step)
4. Both use 16-dim search space from `config_new.py`
5. Best flat cost: 0.377, best step cost: 0.210

## What's NOT in this package

- The actual config/optimizer Python files (they're in `mujoco_refactor/`)
- Raw CSV results (they're in `mujoco_refactor/results/`)
- The old `AGENT_TRANSFER/` directory (stale, from Feb 24)
