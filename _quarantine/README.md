# Quarantine

Non-paper files moved here on branch `paper-release`. **Not deleted** — restore anytime with `git mv`.

## Layout

| Path | Contents |
|------|----------|
| `milliquad_opt/docs/` | Internal `.md` plans (EXECUTION_PLAN, FRICTION_SENSITIVITY, analysis notes, etc.) |
| `milliquad_opt/configs/` | Unused terrain config variants (`config_flat.py`, `config_step.py`, …) |
| `milliquad_opt/scripts/` | Friction sweep, camera recording, terrain gen, energy analysis, `.bak` files |
| `milliquad_opt/analysis/` | Legacy plots, investigative/, l2_l4/, sim-vs-exp composites |
| `milliquad_opt/results/` | `archive/`, `prev_params/`, `mujoco_defaults/`, friction sensitivity runs |
| `milliquad_opt/plots/extra/` | Non-canonical figure PNGs (drafts, overlays, old megacomposite) |
| `milliquad_opt/opt_archive/` | Old cold-start config snapshots |
| `milliquad_opt/videos_local/` | Untracked MP4 recordings (gitignored) |
| `experimental_data/` | Spreadsheets, generated plots, scripts, calculation notebooks |
| `root/` | `CLAUDE.md`, `REPO_MINDMAP.html` |

## Why quarantined

These files supported development and side experiments but are **not required** to reproduce the paper figures (Fig. 3, Fig. 7, Table I) or the three canonical optimization runs.

## Restore example

```bash
git mv _quarantine/milliquad_opt/docs/EXECUTION_PLAN.md milliquad_opt/
```
