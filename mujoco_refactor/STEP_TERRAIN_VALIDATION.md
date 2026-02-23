# Step Terrain: Sim vs Experiment (2026-02-22)

Best-fit params from `results/20260219T142207_loose_fudge` (trained on flat terrain only).
Step preset: 8 steps, 1mm high, 4.5mm long, 20mm final platform, 7.5mm flat lead.
Wheel runs used 50mm flat lead for momentum buildup.

## Velocity Comparison

| config | exp (mm/s) | exp std | sim (mm/s) | error% |
|---|---:|---:|---:|---:|
| scene1_f10 | 19.2 | 4.3 | 43.2 | +125% |
| scene1_f20 | 53.4 | 9.5 | 73.4 | +37% |
| scene1_f30 | 31.1 | 5.6 | 69.7 | +124% |
| scene2_f10 | 52.4 | 11.4 | 68.1 | +30% |
| scene2_f20 | 91.8 | 33.3 | 101.4 | +11% |
| scene2_f30 | 138.1 | 16.3 | 159.3 | +15% |
| scene4_f10 | 74.3 | 9.5 | 100.3 | +35% |
| scene4_f20 | 105.0 | 17.1 | 151.6 | +44% |
| scene4_f30 | 94.6 | 21.3 | 202.9 | +114% |
| wheel_f10 | N/A | N/A | 15.9 | N/A |
| wheel_f20 | N/A | N/A | 17.8 | N/A |
| wheel_f30 | 96.0 | 10.3 | 13.8 | -86% |

Mean absolute error: 62%. Median: 41%.

## Terrain Traversal Ratio (step_vel / flat_vel)

| config | sim ratio | exp ratio | gap |
|---|---:|---:|---:|
| scene1_f10 | 0.86 | 0.38 | +0.49 |
| scene1_f20 | 0.79 | 0.42 | +0.37 |
| scene1_f30 | 0.60 | 0.26 | +0.34 |
| scene2_f10 | 0.82 | 0.64 | +0.19 |
| scene2_f20 | 0.77 | 0.81 | -0.05 |
| scene2_f30 | 0.98 | 0.77 | +0.21 |
| scene4_f10 | 0.93 | 0.66 | +0.27 |
| scene4_f20 | 0.81 | 0.57 | +0.24 |
| scene4_f30 | 0.76 | 0.35 | +0.41 |
| wheel_f30 | 0.03 | 0.21 | -0.18 |

## Observations

**Legged morphologies: sim traverses steps too easily.**
Sim ratios 0.60-0.98 vs experimental 0.26-0.81. The sim doesn't slow down enough
on steps — the contact model lets the legs step over obstacles more easily than
reality. Worst for scene1 (1-leg) which has the least ability to recover from
step impacts. scene2 is closest to experiment (within 0.05-0.21 ratio gap).

**Wheel: sim is inverted.**
Sim wheel gets completely stuck on 1mm steps (ratio 0.03) while the real wheel
manages 96 mm/s (ratio 0.21). The sim wheel's geometry may be physically unable
to roll over the sharp step edge, whereas the real wheel likely has enough
compliance or imprecision to get over it.

**Not surprising given training data.**
These params were fitted on flat terrain only. The fact that contact dynamics
don't fully generalize to step terrain is expected — the optimizer found friction
and contact params that reproduce flat-surface locomotion, but step climbing
depends on additional physics (impact dynamics, edge geometry interaction) that
flat-surface fitting can't constrain.

## Experimental Data Source

Step CSVs: `experimental_data/csv/steps/`
- Naming: `s{freq}{morph}{trial}-{trial}.csv` (e.g. `s30leg1-1.csv`)
- 3 trials per condition, last 50% of recording for steady-state
- No wheel data at 10Hz or 20Hz (only 30Hz was tested)

## Reproducing

```bash
cd mujoco_refactor

# Legged morphologies (uses default 7.5mm flat_lead)
uv run python terrain_test.py results/20260219T142207_loose_fudge \
    --preset step_default --scenes scene1 scene2 scene4 --freqs 10 20 30 --record

# Wheel (50mm flat_lead for momentum)
uv run python terrain_test.py results/20260219T142207_loose_fudge \
    --preset step_default --scenes scene_wheel --freqs 10 20 30 --flat-lead 0.05 --record

# Comparison script
uv run python analyze_step_terrain.py
```
