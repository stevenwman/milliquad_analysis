# Step Terrain: Sim vs Experiment (2026-02-22, revised 2026-02-23)

Best-fit params from `results/20260219T142207_loose_fudge` (trained on flat terrain only).
Step preset: 8 steps, 1mm high, 4.5mm long, 20mm final platform, 7.5mm flat lead.
Wheel runs used 50mm flat lead for momentum buildup.

## Experimental Velocity Extraction Methods

The experimental step CSVs (`experimental_data/csv/steps/`) contain instantaneous
velocity from two tracking points (mass_A columns 3-4, mass_C columns 7-8).
Forward velocity: `vx = 0.5 * (-vx_A + -vx_C)` (negated for camera convention, m/s to mm/s).

Four extraction methods were compared:

| method | description |
|---|---|
| **vx_last50%** | Forward velocity `vx`, averaged over last 50% of timesteps |
| **\|v\|_last50%** | Total velocity `sqrt(vx^2 + vy^2)`, averaged over last 50% of timesteps |
| **vx_mid300** | Forward velocity, averaged over middle 300 timesteps (50% index +/- 150) |
| **vx_q75-300** | Forward velocity, averaged over 300 timesteps centered at 75% index (clamped to bounds) |

Recordings are short (0.5-3.0s, 400-2900 timesteps) and capture mainly the step
traversal itself — there is no long flat-ground lead-in captured on camera. The
robot starts at x ~ 61-68mm and traverses in the -x direction.

### Method Comparison (all values mm/s, mean across 3 trials)

| config | vx_last50% | \|v\|_last50% | vx_mid300 | vx_q75-300 | student Table 1-1 |
|---|---:|---:|---:|---:|---:|
| scene1_f10 | 19.2 | 52.3 | 15.6 | 19.9 | 52.6 |
| scene1_f20 | 53.4 | 108.0 | 42.1 | 47.3 | 100.0 |
| scene1_f30 | 31.1 | 90.0 | 24.1 | 33.1 | 89.9 |
| scene2_f10 | 52.4 | 83.7 | 31.9 | 54.2 | 77.3 |
| scene2_f20 | 91.8 | 114.1 | 91.4 | 89.4 | 111.4 |
| scene2_f30 | 138.1 | 153.8 | 135.1 | 133.5 | 157.9 |
| scene4_f10 | 74.3 | 82.2 | 80.0 | 71.6 | 87.7 |
| scene4_f20 | 105.0 | 116.5 | 99.6 | 103.8 | 109.7 |
| scene4_f30 | 94.6 | 109.7 | 78.3 | 89.8 | 99.1 |
| wheel_f30 | 96.0 | 101.3 | 89.6 | 93.8 | 98.2 |

**Key finding**: The student's Table 1-1 values match **total velocity `|v|`**, not
forward-only `vx`. No single column perfectly reproduces the student's numbers,
suggesting a slightly different windowing or averaging method was used.

The forward-only vs total velocity gap is largest for scene1 (1-leg), where the
robot drifts laterally on steps: `vx=19.2` vs `|v|=52.3` (2.7x) at 10Hz.
For scene4 and wheel the gap is small (5-18%) because those morphologies travel
more straight.

### Mid-300 vs q75-300

The mid-300 window (50% index) can land in the initial acceleration phase for
longer recordings, and scene1_f30 had one trial with negative velocity at midpoint
(std 0.0248 vs mean 0.0241). The q75-300 window (75% index) captures later,
more steady-state locomotion.

For short recordings (scene2_f30: 428 samples), q75+150 exceeds the recording
length. The window is clamped: `start = max(0, q75-150)`, `end = min(N, q75+150)`.
End-of-recording sanity check confirmed no cliff-fall or anomalous behavior:
vx remains positive (11-18 cm/s) and height increases (+0.7-1.7mm, consistent
with climbing steps).

**Decision**: Use **vx_q75-300** for optimizer targets. The q75 window gives more
stable estimates for slow/long recordings (scene1_f30 std drops from 24.8 to 6.6 mm/s)
while remaining safe to clamp for short recordings.

### Which method to use for sim comparison?

The simulation cost function measures **forward displacement only**: `pos[0]`
(x-axis displacement divided by active time). For an apples-to-apples comparison,
the experimental target should also be **forward-only `vx`**.

The step-aware cost function measures velocity only after the robot enters the
step field (`pos[0] >= flat_lead`), avoiding flat-lead acceleration inflation.

## Velocity Comparison (using vx_last50%, forward-only)

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
- Columns: `t, x_A, y_A, vx_A, vy_A, x_C, y_C, vx_C, vy_C, omega_B, theta_B, theta_C`
- 3 trials per condition
- Recording duration: 0.5s (fast) to 3.0s (slow), 400-2900 timesteps
- Robot starts at x ~ 61-68mm, traverses in -x direction (camera convention)
- No wheel data at 10Hz or 20Hz (only 30Hz was tested)
- Velocity extraction script: `mujoco_refactor/analysis/analyze_step_terrain.py`

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
uv run python analysis/analyze_step_terrain.py
```
