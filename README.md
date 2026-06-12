# LEGO Milliquad MuJoCo

MuJoCo simulation + CMA-ES system identification for magnetically-actuated ~100 mg Milliquad robots (L1 / L2 / L4 / wheel morphologies).

Paper: [`Milliquad_Paper.pdf`](Milliquad_Paper.pdf)  
Detailed file map: [`PAPER_MANIFEST.md`](PAPER_MANIFEST.md)

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Linux with EGL for headless MuJoCo (set automatically by scripts)

```bash
uv sync
```

## Repository layout

```
milliquad_opt/          # simulation, optimization, analysis
├── config*.py          # shared 16-param space + per-terrain cost functions
├── simulation.py       # MuJoCo rollout (magnetic torques, RK4 scenes)
├── optimizer.py        # CMA-ES parameter search
├── robots/             # MJCF models (quad + wheel, flat/step/rough)
├── analysis/           # validation + paper figures
└── results/            # timestamped optimization runs (3 canonical dirs included)

experimental_data/      # raw trial CSVs + exp plotting extractors (needed for figures)
_quarantine/            # dev junk moved aside (not needed for paper repro)
```

## Canonical results (paper figures)

These runs are already in the repo — you can plot figures without re-optimizing:

| Terrain | Directory | Config |
|---------|-----------|--------|
| Flat | `milliquad_opt/results/20260303T192801_flat_tg` | `config_flat_tg.py` |
| Step | `milliquad_opt/results/20260303T151416_step_065gate` | `config_step_065.py` |
| Rough | `milliquad_opt/results/20260303T224229_rough_tg` | `config_rough_tg.py` |

## Reproduce paper figures

All commands run from `milliquad_opt/`:

```bash
cd milliquad_opt

# Fig. 3 — velocity + pitch (3 terrains × exp/sim)
uv run python -m analysis.20260303_plot_megacomposite_nocot_065 \
  results/20260303T192801_flat_tg \
  results/20260303T151416_step_065gate \
  results/20260303T224229_rough_tg

# Fig. 7 — cost of transport
uv run python -m analysis.plot_cot_065 \
  results/20260303T192801_flat_tg \
  results/20260303T151416_step_065gate \
  results/20260303T224229_rough_tg
```

Outputs land in `milliquad_opt/plots/`.

## Full pipeline (optimize → validate → plot)

```bash
cd milliquad_opt

# 1. Optimize (hours per terrain at full budget; see smoke test below for a quick check)
uv run python optimizer.py --terrain flat_tg --suffix flat_tg
uv run python optimizer.py --terrain step_065 --suffix step_065gate
uv run python optimizer.py --terrain rough_tg --suffix rough_tg

# 2. Validate best params (jittered trials → NPZ + CSV)
uv run python -m analysis.validate_params results/<run_dir> --terrain flat_tg --csv
uv run python -m analysis.validate_params results/<run_dir> --terrain step_065 --csv
uv run python -m analysis.validate_params results/<run_dir> --terrain rough_tg --csv

# 3. Plot (use your new run dirs, or the canonical dirs above)
uv run python -m analysis.20260303_plot_megacomposite_nocot_065 results/<flat> results/<step> results/<rough>
```

`--terrain` must match a `config_{terrain}.py` file in `milliquad_opt/`.

## Smoke test (verify CMA-ES runs)

One CMA generation (16 evals), single condition (`scene1` @ 10 Hz) — finishes in a few minutes per terrain:

```bash
cd milliquad_opt

uv run python optimizer.py --terrain flat_tg  --n-calls 16 --scenes scene1 --freqs 10 --suffix smoke_flat
uv run python optimizer.py --terrain step_065 --n-calls 16 --scenes scene1 --freqs 10 --suffix smoke_step
uv run python optimizer.py --terrain rough_tg --n-calls 16 --scenes scene1 --freqs 10 --suffix smoke_rough
```

Expect a new `results/<timestamp>_smoke_*/` folder with `multi_optimization_results.csv` and `optimization_bests.csv`. Full runs use `--n-calls 4800` (default in each config).
