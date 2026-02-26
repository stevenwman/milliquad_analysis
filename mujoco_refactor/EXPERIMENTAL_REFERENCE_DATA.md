# Experimental Reference Data

**Total: 27 experimental conditions**
- Flat terrain: 15 conditions (4 morphologies × 10/20/30/50 Hz, wheel only up to 30 Hz)
- Step terrain: 12 conditions (4 morphologies × 10/20/30 Hz)

---

## Flat Terrain (15 conditions)

| Morphology | Scene        | Freq (Hz) | Speed (m/s) | ID            |
|------------|--------------|-----------|-------------|---------------|
| 2-legged   | scene1       | 10        | 0.0512      | scene1_f10    |
| 2-legged   | scene1       | 20        | 0.1264      | scene1_f20    |
| 2-legged   | scene1       | 30        | 0.1187      | scene1_f30    |
| 2-legged   | scene1       | 50        | 0.1483      | scene1_f50    |
| 4-legged   | scene2       | 10        | 0.0832      | scene2_f10    |
| 4-legged   | scene2       | 20        | 0.1131      | scene2_f20    |
| 4-legged   | scene2       | 30        | 0.1796      | scene2_f30    |
| 4-legged   | scene2       | 50        | 0.2633      | scene2_f50    |
| leg-only   | scene4       | 10        | 0.1121      | scene4_f10    |
| leg-only   | scene4       | 20        | 0.1841      | scene4_f20    |
| leg-only   | scene4       | 30        | 0.2747      | scene4_f30    |
| leg-only   | scene4       | 50        | 0.3274      | scene4_f50    |
| wheel      | scene_wheel  | 10        | 0.1432      | scene_wheel_f10 |
| wheel      | scene_wheel  | 20        | 0.3058      | scene_wheel_f20 |
| wheel      | scene_wheel  | 30        | 0.4493      | scene_wheel_f30 |

---

## Step Terrain (12 conditions)

| Morphology | Scene        | Freq (Hz) | Speed (m/s) | ID            | Notes           |
|------------|--------------|-----------|-------------|---------------|-----------------|
| 2-legged   | scene1       | 10        | 0.0199      | scene1_f10    |                 |
| 2-legged   | scene1       | 20        | 0.0473      | scene1_f20    |                 |
| 2-legged   | scene1       | 30        | 0.0331      | scene1_f30    |                 |
| 4-legged   | scene2       | 10        | 0.0542      | scene2_f10    |                 |
| 4-legged   | scene2       | 20        | 0.0894      | scene2_f20    |                 |
| 4-legged   | scene2       | 30        | 0.1335      | scene2_f30    |                 |
| leg-only   | scene4       | 10        | 0.0716      | scene4_f10    |                 |
| leg-only   | scene4       | 20        | 0.1038      | scene4_f20    |                 |
| leg-only   | scene4       | 30        | 0.0898      | scene4_f30    |                 |
| wheel      | scene_wheel  | 10        | 0.0000      | scene_wheel_f10 | **FAILURE**    |
| wheel      | scene_wheel  | 20        | 0.0000      | scene_wheel_f20 | **FAILURE**    |
| wheel      | scene_wheel  | 30        | 0.0938      | scene_wheel_f30 |                 |

---

## Statistics by Morphology

### Flat Terrain

| Morphology | Frequencies      | Speed Range (m/s) | Mean Speed (m/s) |
|------------|------------------|-------------------|------------------|
| 2-legged   | 10, 20, 30, 50   | 0.051 - 0.148     | 0.111            |
| 4-legged   | 10, 20, 30, 50   | 0.083 - 0.263     | 0.160            |
| leg-only   | 10, 20, 30, 50   | 0.112 - 0.327     | 0.225            |
| wheel      | 10, 20, 30       | 0.143 - 0.449     | 0.299            |

### Step Terrain

| Morphology | Frequencies      | Speed Range (m/s) | Mean Speed (m/s) |
|------------|------------------|-------------------|------------------|
| 2-legged   | 10, 20, 30       | 0.020 - 0.047     | 0.033            |
| 4-legged   | 10, 20, 30       | 0.054 - 0.134     | 0.092            |
| leg-only   | 10, 20, 30       | 0.072 - 0.104     | 0.088            |
| wheel      | 10, 20, 30       | 0.000 - 0.094     | 0.031            |

---

## Flat vs Step Comparison

| Morphology | Freq | Flat (m/s) | Step (m/s) | Step/Flat Ratio | Difference (m/s) |
|------------|------|------------|------------|-----------------|------------------|
| 2-legged   | 10   | 0.0512     | 0.0199     | 38.9%           | +0.0313          |
| 2-legged   | 20   | 0.1264     | 0.0473     | 37.4%           | +0.0791          |
| 2-legged   | 30   | 0.1187     | 0.0331     | 27.9%           | +0.0856          |
| 4-legged   | 10   | 0.0832     | 0.0542     | 65.1%           | +0.0290          |
| 4-legged   | 20   | 0.1131     | 0.0894     | 79.1%           | +0.0237          |
| 4-legged   | 30   | 0.1796     | 0.1335     | 74.3%           | +0.0461          |
| leg-only   | 10   | 0.1121     | 0.0716     | 63.9%           | +0.0405          |
| leg-only   | 20   | 0.1841     | 0.1038     | 56.4%           | +0.0803          |
| leg-only   | 30   | 0.2747     | 0.0898     | 32.7%           | +0.1849          |
| wheel      | 10   | 0.1432     | 0.0000     | 0.0%            | +0.1432          |
| wheel      | 20   | 0.3058     | 0.0000     | 0.0%            | +0.3058          |
| wheel      | 30   | 0.4493     | 0.0938     | 20.9%           | +0.3555          |

---

## Key Findings

### Terrain Impact
- **Step terrain reduces speed to 21-79% of flat** (morphology/frequency dependent)
- **Worst ratios**: wheel @30Hz (21%), leg-only @30Hz (33%), 2-legged @30Hz (28%)
- **Best ratio**: 4-legged @20Hz (79%)

### Failure Modes
- **Wheel completely fails on steps @10/20 Hz** (0.0 m/s)
- Only succeeds @30 Hz (0.094 m/s, but still only 21% of flat performance)

### Speed Rankings (Flat Terrain)
1. **Wheel**: 0.299 m/s avg (fastest, but limited to ≤30 Hz testing)
2. **Leg-only**: 0.225 m/s avg
3. **4-legged**: 0.160 m/s avg
4. **2-legged**: 0.111 m/s avg (slowest)

### Frequency Trends
- **Generally speed increases with frequency on flat terrain**
  - Exception: 2-legged @30Hz dips slightly below @20Hz
- **On steps, trends are less consistent**
  - 2-legged peaks @20Hz then drops
  - leg-only peaks @20Hz then drops significantly @30Hz

### Morphology-Specific Observations
- **4-legged most robust on steps** (maintains 65-79% of flat speed)
- **Wheel most specialized** (fastest on flat, but catastrophic failures on steps)
- **2-legged struggles most on steps** (28-39% of flat speed)
- **Leg-only has dramatic drop @30Hz on steps** (33% vs 64% @10Hz)

---

## Usage for Validation

Compare these experimental values against simulation rollouts using fitted parameters:
1. Load best params from optimization runs
2. Run simulations for all 27 conditions
3. Use jitter trials (±3 range) and average results
4. Calculate error metrics: velocity error, lateral drift, tumbling, etc.
5. Compare sim vs experimental speeds to quantify fit quality

**Source files:**
- Flat terrain refs: `config_new.py` → `REFERENCE_DATA`
- Step terrain refs: `config_step.py` → `REFERENCE_DATA`
