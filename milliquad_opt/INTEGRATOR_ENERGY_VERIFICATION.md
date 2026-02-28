# INTEGRATOR ENERGY VERIFICATION — RK4 OPTIMIZED PARAMS

**Date**: 2026-02-28
**Params**: Per-terrain best from RK4 optimization (flat warm, step cold, rough warm)
**Integrator**: RK4 (baked into all XMLs, `<flag energy="enable"/>`)
**Sim**: dt=0.5ms (2kHz), no jitter, single trial per ref

All energy values in uJ.

## Methodology

Same as `mujoco_refactor/INTEGRATOR_ENERGY_ANALYSIS.md`. Key definitions:

```
P_ext   = Σ_j  τ_ext[j] · ω_leg[j]       (naive: field torque dot full body angular velocity)
P_int   = Σ_j  τ_int[j] · ω_leg[j]       (inter-joint coupling power)
P_joint = Σ_j  (τ_ext[j] · â_j) · q̇_j    (joint-projected: torque on joint axis × joint speed)

W_ext   = ∫ P_ext dt        (total work by external field)
W_int   = ∫ P_int dt        (total work by inter-joint coupling — should be ≈0, conservative)
W_xfrc  = W_ext + W_int     (total work by all applied torques)
W_joint = ∫ P_joint dt      (total work through joint-projected formula)
dE      = ΔE_mujoco         (change in MuJoCo's KE+PE over steady state)
Dissip  = W_xfrc - dE       (must be ≥ 0: friction/damping can only remove energy)
```

**Physical requirement**: Dissipation must be positive. Negative dissipation means the
integrator is injecting phantom energy through the contact solver.

**COT formula**: Under RK4, `COT = W_ext / (m·g·d)` using the naive formula is correct
because W_ext is always positive and the energy budget closes. No need for the
joint-projected workaround that was required under Euler.

## Flat Terrain

**Params**: `results/20260228T013353_rk4_flat` (cost=0.188)
**Refs**: 15 (scene1/2/4 × f10/f20/f30/f50 + scene_wheel × f10/f20/f30)
**Duration**: 3.0s, settle=0.1s

| Ref | vx (mm/s) | target | err% | W_ext | W_int | W_xfrc | W_joint | dE | Dissip |
|-----|-----------|--------|------|-------|-------|--------|---------|-----|--------|
| scene1_f10 | 51.7 | 51.2 | 0.9% | +128.5 | -11.0 | +117.5 | +1686.1 | +0.5 | +117.0 |
| scene1_f20 | 101.4 | 126.4 | 19.8% | +291.7 | -5.1 | +286.6 | +1674.6 | +0.3 | +286.3 |
| scene1_f30 | 131.2 | 118.7 | 10.5% | +478.1 | -3.6 | +474.5 | +1944.6 | +2.0 | +472.6 |
| scene1_f50 | 155.2 | 148.3 | 4.7% | +1012.7 | -2.3 | +1010.5 | +3192.5 | +4.3 | +1006.2 |
| scene2_f10 | 80.4 | 83.2 | 3.4% | +179.7 | -12.5 | +167.2 | +624.1 | +4.2 | +163.0 |
| scene2_f20 | 127.5 | 113.1 | 12.7% | +318.4 | -5.5 | +312.9 | +888.5 | +4.1 | +308.8 |
| scene2_f30 | 192.8 | 179.6 | 7.4% | +487.8 | -1.7 | +486.1 | +978.5 | +3.1 | +483.1 |
| scene2_f50 | 285.0 | 263.3 | 8.2% | +1186.7 | -1.9 | +1184.8 | +2400.9 | +7.9 | +1176.9 |
| scene4_f10 | 100.9 | 112.1 | 10.0% | +207.1 | +0.0 | +207.1 | +406.7 | +0.7 | +206.4 |
| scene4_f20 | 187.1 | 184.1 | 1.6% | +305.5 | -0.1 | +305.5 | +656.5 | +1.7 | +303.7 |
| scene4_f30 | 298.2 | 274.7 | 8.6% | +649.0 | +0.6 | +649.6 | +1353.9 | +3.4 | +646.2 |
| scene4_f50 | 363.4 | 327.4 | 11.0% | +964.5 | +1.8 | +966.2 | +1666.3 | +11.6 | +954.6 |
| scene_wheel_f10 | 136.8 | 143.2 | 4.5% | +74.9 | -0.9 | +74.1 | +177.8 | +1.0 | +73.1 |
| scene_wheel_f20 | 314.1 | 305.8 | 2.7% | +194.4 | +0.5 | +194.9 | +923.9 | +5.2 | +189.7 |
| scene_wheel_f30 | 493.1 | 449.3 | 9.8% | +390.4 | +6.6 | +397.0 | +2139.1 | +16.3 | +380.7 |

**Negative dissipation: 0/15**

## Step Terrain

**Params**: `results/20260228T093833_rk4_step_cold` (cost=0.243)
**Refs**: 12 (scene1/2/4 × f10/f20/f30 + scene_wheel × f10/f20/f30)
**Duration**: 5.0s, settle=0.1s
**Note**: scene_wheel f10/f20 are failure-mode refs (target=0, robot should not move)

| Ref | vx (mm/s) | target | err% | W_ext | W_int | W_xfrc | W_joint | dE | Dissip |
|-----|-----------|--------|------|-------|-------|--------|---------|-----|--------|
| scene1_f10 | 44.9 | 19.9 | 125.8% | +372.2 | -34.5 | +337.7 | +431.7 | -1.2 | +338.9 |
| scene1_f20 | 95.3 | 47.3 | 101.6% | +858.0 | -9.6 | +848.4 | +1123.6 | -1.0 | +849.4 |
| scene1_f30 | 94.7 | 33.1 | 186.0% | +1470.1 | -10.8 | +1459.2 | +1926.9 | +2.3 | +1456.9 |
| scene2_f10 | 85.6 | 54.2 | 57.9% | +380.3 | -10.5 | +369.8 | +449.3 | +2.5 | +367.3 |
| scene2_f20 | 118.5 | 89.4 | 32.5% | +825.6 | -11.3 | +814.4 | +999.3 | +1.4 | +812.9 |
| scene2_f30 | 166.4 | 133.5 | 24.7% | +1582.9 | -5.9 | +1577.0 | +1860.0 | +3.2 | +1573.7 |
| scene4_f10 | 100.9 | 71.6 | 41.0% | +449.1 | -0.2 | +449.0 | +632.7 | -0.1 | +449.1 |
| scene4_f20 | 239.6 | 103.8 | 130.9% | +784.2 | -5.9 | +778.3 | +969.9 | +3.9 | +774.4 |
| scene4_f30 | 238.5 | 89.8 | 165.6% | +1387.5 | -1.8 | +1385.7 | +1689.4 | +3.2 | +1382.5 |
| scene_wheel_f10 | 10.0 | 0.0 | — | +277.5 | -1.0 | +276.5 | +445.6 | +0.5 | +276.0 |
| scene_wheel_f20 | 10.9 | 0.0 | — | +838.7 | +3.6 | +842.4 | +1287.4 | +1.6 | +840.8 |
| scene_wheel_f30 | 12.6 | 93.8 | 86.6% | +1493.0 | +9.8 | +1502.8 | +2142.9 | +3.7 | +1499.1 |

**Negative dissipation: 0/12**

Note: Step velocity errors are high because this energy audit uses time-gated velocity
(settle_time=0.1s to end), not the spatial-gated measurement (STEP_START_X to 90% of
STEP_END_X) that the optimizer uses. The energy budget is the relevant metric here.

## Rough Terrain

**Params**: `results/20260228T102010_rk4_rough` (cost=0.362)
**Refs**: 7 (scene1/2/4 × f10/f30 + scene2 f50)
**Duration**: 2.0s, settle=0.1s

| Ref | vx (mm/s) | target | err% | W_ext | W_int | W_xfrc | W_joint | dE | Dissip |
|-----|-----------|--------|------|-------|-------|--------|---------|-----|--------|
| scene1_f10 | 31.1 | 42.9 | 27.5% | +121.5 | -6.8 | +114.7 | +387.7 | -0.2 | +114.9 |
| scene1_f30 | 83.4 | 81.6 | 2.1% | +678.8 | -0.7 | +678.1 | +1807.5 | +7.6 | +670.5 |
| scene2_f10 | 57.6 | 65.6 | 12.1% | +184.9 | -9.9 | +175.0 | +289.6 | +0.6 | +174.4 |
| scene2_f30 | 157.5 | 128.9 | 22.2% | +670.7 | -6.4 | +664.3 | +1321.1 | +3.5 | +660.8 |
| scene2_f50 | 170.4 | 106.2 | 60.4% | +1363.9 | +4.4 | +1368.3 | +2418.0 | +9.1 | +1359.3 |
| scene4_f10 | 55.3 | 85.7 | 35.5% | +206.9 | -2.1 | +204.8 | +441.3 | +0.0 | +204.8 |
| scene4_f30 | 208.2 | 146.0 | 42.6% | +700.5 | -1.6 | +698.9 | +1309.5 | +5.1 | +693.7 |

**Negative dissipation: 0/7**

## Summary

| Terrain | Refs | Neg. Dissipation | W_ext range (uJ) | W_int range (uJ) | W_joint/W_ext |
|---------|------|------------------|-------------------|-------------------|---------------|
| Flat | 15 | **0/15** | +74.9 to +1186.7 | -12.5 to +6.6 | 1.7–13.1× |
| Step | 12 | **0/12** | +277.5 to +1582.9 | -34.5 to +9.8 | 1.2–1.6× |
| Rough | 7 | **0/7** | +121.5 to +1363.9 | -9.9 to +4.4 | 1.5–3.2× |
| **Total** | **34** | **0/34** | | | |

### Comparison to Euler (from INTEGRATOR_ENERGY_ANALYSIS.md)

| | Euler (old params) | RK4 (new params) |
|---|---|---|
| Flat negative dissipation | 4/15 (27%) | **0/15** |
| Rough negative dissipation | 5/15 (33%) | **0/7** |
| W_ext sign | Mixed (+/-), 9 negative | **All positive** |
| W_int drift | Up to -45 uJ | ≤ -35 uJ (mostly < 12 uJ) |
| W_joint/W_ext ratio | 3–10× (undefined when W_ext<0) | 1.2–13× |

### Key Findings

1. **Zero negative dissipation**: 0/34 refs across all 3 terrains. The energy budget
   closes correctly for every reference condition. Friction and damping only remove
   energy — no phantom injection from the integrator.

2. **W_ext always positive**: The external field consistently injects energy into the
   system, as physics demands. Under Euler, 9/30 refs had negative W_ext (unphysical).

3. **W_int ≈ 0**: Inter-joint coupling is conservative (range: -35 to +10 uJ).
   The largest value (-34.5 uJ on scene1_f10 step) is a single outlier; most are < 12 uJ.

4. **dE ≈ 0**: Change in total mechanical energy is near zero for all refs (range: -1.2 to
   +16.3 uJ), confirming steady-state locomotion.

5. **W_joint/W_ext gap persists but is harmless**: The joint-projected formula still
   overcounts by 1.2–13× because `data.cvel` includes base body rotation. But since
   W_ext is always positive under RK4, the naive formula gives the correct sign and the
   joint-projected workaround is unnecessary.

6. **COT validation**: `COT = W_ext / (m·g·d)` using `P = Σ τ_ext · ω` (naive formula)
   is the physically correct formulation under RK4. This is what `compute_cot()` in
   `analysis/_common.py` implements.

---

*Generated by `energy_analysis.py` on 2026-02-28.*
*Params optimized with RK4 integrator baked into all scene XMLs.*
