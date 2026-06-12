# LEGO Milliquad MuJoCo

MuJoCo simulation and system identification for magnetically-actuated ~100 mg Milliquad robots.

Paper: see `Milliquad_Paper.pdf`. Release layout documented in `PAPER_MANIFEST.md`.

## Setup

```bash
uv sync
```

## Quick start (reproduce paper figures)

```bash
cd milliquad_opt
uv run python -m analysis.20260303_plot_megacomposite_nocot_065 \
  results/20260303T192801_flat_tg \
  results/20260303T151416_step_065gate \
  results/20260303T224229_rough_tg
```

See `PAPER_MANIFEST.md` for full pipeline (optimize → validate → plot).
