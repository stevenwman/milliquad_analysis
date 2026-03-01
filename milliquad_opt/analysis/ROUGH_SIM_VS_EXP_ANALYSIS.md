# Rough Terrain: Simulation vs Experiment Failure Analysis

Rough terrain (1mm std heightfield, seed=42) shows a large gap between experimental and simulated failure rates. This document breaks down the data condition-by-condition.

**Data sources**:
- Experimental: `experimental_data/csv/random_terrain_raw.csv` (5 trials per condition)
- Simulation: `results/20260228T202903_rough_spatial_rk4/validation_trials.csv` (10 trials, top 5 selected)

---

## Combined Table

| Morphology | Freq (Hz) | Exp success | Exp n/a | Exp mean (mm/s) | Sim valid | Sim invalid | Sim mean (mm/s) | Agreement |
|------------|-----------|-------------|---------|-----------------|-----------|-------------|-----------------|-----------|
| L1 (scene1) | 10 | 4/5 | 1/5 | 42.9 | 5/5 | 0/5 | 42.9 | Excellent (velocity match, sim slightly more robust) |
| L1 (scene1) | 30 | 4/5 | 1/5 | 81.6 | 5/5 | 0/5 | 86.9 | Good (velocity +6%, sim more robust) |
| L1 (scene1) | 50 | 1/5 | 4/5 | 56.5 | 2/5 | 3/5 | 88.1 | Partial (both struggle, but sim velocity 56% higher when it works) |
| L2 (scene2) | 10 | 5/5 | 0/5 | 65.6 | 5/5 | 0/5 | 69.1 | Excellent (+5%) |
| L2 (scene2) | 30 | 4/5 | 1/5 | 128.9 | 5/5 | 0/5 | 109.9 | Moderate (velocity -15%, sim more robust) |
| L2 (scene2) | 50 | 3/5 | 2/5 | 106.2 | 5/5 | 0/5 | 100.5 | Good (velocity -5%, sim much more robust) |
| L4 (scene4) | 10 | 5/5 | 0/5 | 85.7 | 4/5 | 1/5 | 84.0 | Excellent (-2%, sim slightly less robust) |
| L4 (scene4) | 30 | 4/5 | 1/5 | 146.0 | 5/5 | 0/5 | 127.6 | Moderate (velocity -13%, sim more robust) |
| L4 (scene4) | 50 | 1/5 | 4/5 | 101.6 | 4/5 | 1/5 | 115.2 | Poor (sim much more robust: 80% vs 20% success) |
| WR (wheel)  | 10 | 1/5 | 4/5 | 81.2 | 0/5 | 5/5 | - | Both fail, but differently (exp: 1 lucky trial; sim: total gate failure) |
| WR (wheel)  | 30 | 2/5 | 3/5 | 169.3 | 1/5 | 4/5 | 94.5 | Both struggle. Sim velocity 44% lower when it works |
| WR (wheel)  | 50 | 1/5 | 4/5 | 180.6 | 3/5 | 2/5 | 111.5 | Inverted: sim more robust (60% vs 20%) but 38% slower |

---

## Aggregate Failure Rates

|  | Total trials | Failed | Success rate |
|--|-------------|--------|--------------|
| Experiment | 60 | 25 | **58%** |
| Simulation | 60 | 16 | **73%** |

Sim has 15 percentage points higher success rate overall.

---

## Failure Rate by Morphology

| Morphology | Exp success rate | Sim success rate | Gap |
|------------|-----------------|-----------------|-----|
| L1 (1-leg) | 9/15 = 60% | 12/15 = 80% | Sim +20pp |
| L2 (2-leg) | 12/15 = 80% | 15/15 = 100% | Sim +20pp |
| L4 (4-leg) | 10/15 = 67% | 13/15 = 87% | Sim +20pp |
| WR (wheel) | 4/15 = 27% | 4/15 = 27% | Match |

Sim is consistently ~20pp more robust than experiment for legged morphologies. Wheel is equally bad in both.

## Failure Rate by Frequency

| Frequency | Exp success rate | Sim success rate | Gap |
|-----------|-----------------|-----------------|-----|
| 10 Hz | 15/20 = 75% | 14/20 = 70% | Sim -5pp |
| 30 Hz | 14/20 = 70% | 16/20 = 80% | Sim +10pp |
| 50 Hz | 6/20 = 30% | 14/20 = 70% | Sim +40pp |

50 Hz is the critical gap: sim succeeds 70% of the time where experiment only succeeds 30%. This aligns with the hypothesis that high-frequency experimental failures are mechanical (leg detachment, coupling loss, resonance-induced wobble) rather than physics-based.

---

## Why Sim Fails Less

### Failure modes the sim DOES capture
- **Gate failure (max_x < 155mm)**: Robot gets stuck or moves too slowly to traverse the full rough section. All 16 sim failures are gate failures. Wheel is most affected — the single contact patch can't maintain traction on rough terrain.
- **scene4 f10 near-miss**: One trial reached 150.9mm (gate=155mm). Barely failed — the 4-leg robot at 10Hz is slow but persistent.
- **scene_wheel f30 near-misses**: Two trials at 150.9mm and 153.8mm. Wheel barely fails the gate on rough terrain.

### Failure modes the sim DOES NOT capture
- **Mechanical failure / leg detachment**: At high frequencies (f50), real LEGO legs can detach or lose magnetic coupling. Sim legs are rigidly attached.
- **Manufacturing variance**: Real robots have asymmetric legs, imperfect magnet placement. Sim has perfect symmetry.
- **Surface interaction detail**: Real rough surfaces can wedge or catch specific leg geometries. Heightfield approximation smooths these features.
- **Magnetic coupling loss**: At high frequency + rough terrain, the magnet can skip beats or lose synchronization with the driving field. Sim magnetic model is idealized.
- **Accumulated damage**: Real robots degrade across trials (especially at 50Hz). Sim starts fresh each trial.

### The wheel paradox
Wheel shows equal failure rates (27%) in both exp and sim, but for different reasons:
- **Experiment**: Mechanical coupling failure (magnet can't maintain rotation on bumps)
- **Simulation**: Gate failure (single contact point loses traction, robot stalls mid-course)
- The velocity when successful also diverges: exp wheel achieves 81–181 mm/s, sim only 94–115 mm/s. Experimental successes may be "lucky" high-coupling events that the sim can't reproduce because it doesn't model stochastic coupling dynamics.

---

## Implications for Plotting

- Rough exp velocity uses `scatter_only` mode with X markers at y=0 for each n/a trial
- `_inject_na_zeros()` infers failure count as `5 - n_success` per condition
- Sim failures shown via `all_failed` X markers (only when ALL 5 selected trials fail — currently only wheel f10)
- The visual gap between exp X clusters and sim dots is real and expected: the sim is systematically more robust than the physical robot on rough terrain
