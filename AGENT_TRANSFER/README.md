# Agent Transfer Package

This folder contains everything a fresh Claude Code agent needs to become productive on this project immediately, even on a different device/account with no prior context.

## Quick Start

1. Copy this entire repo to the new machine
2. Place `MEMORY.md` at `~/.claude/projects/<project-hash>/memory/MEMORY.md` (Claude Code auto-loads this)
3. `CLAUDE.md` should already be in the repo root (it's checked in)
4. Read `CONTEXT_BRIEFING.md` at the start of your first conversation

## Files

| File | Purpose |
|---|---|
| `README.md` | This file — how to bootstrap a new agent |
| `MEMORY.md` | Auto-memory snapshot — architecture, lessons learned, parameter insights |
| `CONTEXT_BRIEFING.md` | Self-contained briefing: project state, active work, key decisions, file map |
| `CLAUDE.md` | Project instructions (also in repo root, included here for completeness) |

## What the New Agent Should Do First

1. Read `CONTEXT_BRIEFING.md` (covers everything)
2. If working on the **step terrain optimizer**: read `mujoco_refactor/config_step.py`, `mujoco_refactor/optimizer_step.py`, `mujoco_refactor/STEP_OPTIMIZER_PLAN.md`
3. If working on the **flat optimizer**: read `mujoco_refactor/config.py`, `mujoco_refactor/optimizer.py`
4. Run `uv sync` to install dependencies

## Key Commands

```bash
# Install deps
uv sync

# Run flat optimizer
cd mujoco_refactor && uv run python optimizer.py --suffix flat_run

# Run step optimizer
cd mujoco_refactor && uv run python optimizer_step.py --n-calls 1200 --suffix step_v1

# View results
cd mujoco_refactor && uv run python show_bests.py        # flat
cd mujoco_refactor && uv run python show_bests_step.py   # step

# Smoke test (quick validation)
cd mujoco_refactor && uv run python optimizer_step.py --scenes scene4 --n-calls 16 --suffix step_smoke
```
