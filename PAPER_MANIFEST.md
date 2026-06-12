# Paper Release Manifest

Branch: `paper-release` (from `final`)

Paper: `Milliquad_Paper.pdf` — IROS submission comparing L1/L2/L4/WR morphologies across flat, step, and rough terrain with MuJoCo system ID.

## Paper ↔ Code mapping

| Paper artifact | Source |
|----------------|--------|
| **Fig. 3** — exp/sim velocity + pitch (3×4) | `milliquad_opt/analysis/20260303_plot_megacomposite_nocot_065.py` |
| **Fig. 7** — sim COT | `milliquad_opt/analysis/plot_cot_065.py` |
| **Table I** — optimized params | `milliquad_opt/analysis/param_table_expanded_tg_with_default.tex` |
| **Fig. 4, 5** — step/rough snapshots | Manual video frames (not in repo) |
| **Fig. 6** — L2 trajectory vs default params | Likely manual / one-off; `plot_trajectories.py` available for sim trajectories |

## Canonical optimization runs (used in figures)

| Terrain | Results dir | Config |
|---------|-------------|--------|
| Flat | `results/20260303T192801_flat_tg` | `config_flat_tg.py` |
| Step | `results/20260303T151416_step_065gate` | `config_step_065.py` |
| Rough | `results/20260303T224229_rough_tg` | `config_rough_tg.py` |

## Kept files (release tree)

```
milliquad_opt/
├── config.py                    # 16-dim search space + sim constants
├── config_flat_tg.py            # flat terrain refs + cost
├── config_step_065.py           # step terrain (65% gate params)
├── config_rough_tg.py           # rough terrain refs + cost
├── simulation.py                # MuJoCo rollout engine
├── optimizer.py                 # CMA-ES (reproduce optimization)
├── robots/                      # MJCF models (quad + wheel)
├── analysis/
│   ├── _common.py
│   ├── validate_params.py       # jittered validation → NPZ/CSV
│   ├── plot_validation.py       # shared plot utilities
│   ├── 20260303_plot_megacomposite_nocot_065.py   # Fig. 3
│   ├── plot_cot_065.py          # Fig. 7
│   ├── plot_trajectories.py     # trajectory plots
│   ├── dump_numbers.py          # extract numbers for paper/LaTeX
│   ├── param_table_expanded_tg.tex
│   └── param_table_expanded_tg_with_default.tex
├── results/
│   ├── 20260303T192801_flat_tg/
│   ├── 20260303T151416_step_065gate/
│   └── 20260303T224229_rough_tg/
└── plots/
    ├── 20260303_megacomposite_nocot_065.png   # Fig. 3 output
    └── 20260304T174414_cot_065.png            # Fig. 7 output

experimental_data/
├── csv/                         # raw trial CSVs
├── plot_velocity_vs_freq.py     # exp velocity extractors
└── plot_pitch_vs_freq.py        # exp pitch extractors
```

## Reproduce figures

```bash
uv sync
cd milliquad_opt

# Regenerate validation (optional; NPZ/CSV already in results/)
uv run python -m analysis.validate_params results/20260303T192801_flat_tg --terrain flat_tg --csv
uv run python -m analysis.validate_params results/20260303T151416_step_065gate --terrain step_065 --csv
uv run python -m analysis.validate_params results/20260303T224229_rough_tg --terrain rough_tg --csv

# Fig. 3
uv run python -m analysis.20260303_plot_megacomposite_nocot_065 \
  results/20260303T192801_flat_tg \
  results/20260303T151416_step_065gate \
  results/20260303T224229_rough_tg

# Fig. 7
uv run python -m analysis.plot_cot_065 \
  results/20260303T192801_flat_tg \
  results/20260303T151416_step_065gate \
  results/20260303T224229_rough_tg
```

## Quarantined (see `_quarantine/README.md`)

~1.8 GB moved to `_quarantine/`: internal docs, legacy configs, friction sensitivity, archive results, investigative analysis, l2_l4, videos, dev scripts.

Nothing deleted — recover with `git mv` from `_quarantine/` back.
