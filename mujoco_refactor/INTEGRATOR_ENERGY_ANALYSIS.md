# INTEGRATOR ENERGY ANALYSIS — EULER VS RK4

**Date**: 2026-02-27
**Params**: flat_10_30_50 best (results/20260225T122342_flat_10_30_50)
**Terrain**: flat ground, all 15 references
**Sim**: 2s duration, settle=0.1s, dt=0.5ms (2kHz)

All energy values in uJ.

## Methodology

### System definition

The "system" is the robot: 1 main body + 4 leg bodies, connected by 4 hinge joints.
MuJoCo tracks the system's total mechanical energy via `data.energy`:

```
E_total = KE (translational + rotational, all bodies) + PE (gravitational)
```

### Forces acting on the system

Two torques are applied to the leg bodies via `data.xfrc_applied`:

1. **τ_ext** — external magnetic field torque (the "actuator"). `τ = kp * (m × B_goal)`
2. **τ_int** — inter-joint dipole-dipole coupling (internal to the robot, conservative)

MuJoCo internally handles gravity (in PE), contact forces (normal + friction), and
joint damping (`dof_damping`). These are not applied via `xfrc_applied`.

### Power computation

At each simulation timestep, we compute instantaneous power:

```
P_ext  = Σ_j  τ_ext[j] · ω_leg[j]     (field power into each leg body)
P_int  = Σ_j  τ_int[j] · ω_leg[j]     (coupling power)
P_joint = Σ_j (τ_ext[j] · â_j) · q̇_j  (joint-projected: torque on joint axis × joint speed)
```

where:
- `τ_ext[j]`, `τ_int[j]` are (3,) world-frame torques from `step_cache` (recorded BEFORE `mj_step`)
- `ω_leg[j]` = `data.cvel[leg_body, :3]` = full angular velocity of leg body in world frame
  (also recorded BEFORE `mj_step`, so τ and ω are at the same time instant)
- `â_j` = joint axis `[0,0,1]` rotated into world frame by leg body quaternion
- `q̇_j` = `data.qvel[6+j]` = scalar hinge joint velocity

### Energy integration

Integrate power over the steady-state window (t > 0.1s settle time):

```
W_ext   = ∫ P_ext dt        (total work by external field)
W_int   = ∫ P_int dt        (total work by inter-joint coupling)
W_xfrc  = W_ext + W_int     (total work by all applied torques)
W_joint = ∫ P_joint dt      (total work through joint-projected formula)
```

### Energy balance

From first principles:

```
dE_total = W_xfrc - W_dissipation
```

where `W_dissipation` = energy removed by contact friction + joint damping (always ≥ 0).

We measure `dE_total` from MuJoCo's `data.energy` and `W_xfrc` from our power integration,
then solve for dissipation:

```
Dissipation = W_xfrc - dE_total
```

**Physical requirement**: Dissipation must be positive (friction/damping can only remove energy).
If dissipation comes out **negative**, it means the integrator's numerical error is injecting
phantom energy into the system — the energy budget doesn't close.

### Column definitions

| Column | Meaning |
|--------|---------|
| vx (mm/s) | Simulated forward velocity (mm/s), averaged over steady state |
| target | Experimental target velocity (mm/s) |
| err% | \|vx - target\| / target × 100 |
| W_ext | Work done by external field τ_ext (uJ). Positive = field injects energy |
| W_int | Work done by inter-joint coupling τ_int (uJ). Should be ≈0 (conservative) |
| W_xfrc | W_ext + W_int = total work by all applied torques (uJ) |
| W_joint | Work computed by joint-projected formula (uJ). Always positive by construction |
| dE | Change in MuJoCo's total energy KE+PE over steady state (uJ). ≈0 if steady state |
| Dissip | W_xfrc - dE (uJ). **Must be positive** for physical consistency |

## Euler (semi-implicit, current default)

| Ref | vx (mm/s) | target | err% | W_ext | W_int | W_xfrc | W_joint | dE | Dissip |
|-----|-----------|--------|------|-------|-------|--------|---------|-----|--------|
| scene1_f10 | 57.7 | 51.2 | 12.7% | -511.4 | -22.4 | -533.8 | +2099.7 | -0.8 | -533.0 |
| scene1_f20 | 95.3 | 126.4 | 24.6% | -238.1 | -24.7 | -262.9 | +2064.7 | +1.2 | -264.0 |
| scene1_f30 | 134.7 | 118.7 | 13.5% | -40.2 | -9.7 | -49.9 | +2290.3 | +3.5 | -53.4 |
| scene1_f50 | 154.9 | 148.3 | 4.5% | +420.0 | -10.0 | +410.1 | +3291.5 | +4.5 | +405.6 |
| scene2_f10 | 89.5 | 83.2 | 7.6% | +29.3 | -4.1 | +25.2 | +620.3 | +1.4 | +23.8 |
| scene2_f20 | 146.8 | 113.1 | 29.8% | +157.9 | -3.0 | +154.8 | +981.0 | +1.7 | +153.1 |
| scene2_f30 | 177.4 | 179.6 | 1.2% | +285.6 | +8.4 | +294.0 | +1274.6 | +3.5 | +290.5 |
| scene2_f50 | 269.7 | 263.3 | 2.4% | +809.7 | +7.2 | +816.8 | +2408.2 | +7.7 | +809.1 |
| scene4_f10 | 114.7 | 112.1 | 2.3% | +106.0 | -3.0 | +103.0 | +338.0 | +0.7 | +102.3 |
| scene4_f20 | 197.7 | 184.1 | 7.4% | +217.5 | -10.1 | +207.4 | +665.3 | +1.8 | +205.6 |
| scene4_f30 | 306.4 | 274.7 | 11.5% | +378.3 | -3.2 | +375.1 | +1076.2 | +8.6 | +366.4 |
| scene4_f50 | 418.5 | 327.4 | 27.8% | +855.2 | +3.4 | +858.6 | +1921.1 | +11.1 | +847.5 |
| scene_wheel_f10 | 166.9 | 143.2 | 16.6% | +28.4 | +0.5 | +28.9 | +176.4 | +1.4 | +27.5 |
| scene_wheel_f20 | 343.0 | 305.8 | 12.2% | +100.2 | +0.1 | +100.3 | +418.5 | +6.1 | +94.2 |
| scene_wheel_f30 | 447.7 | 449.3 | 0.4% | -2.8 | -7.3 | -10.2 | +1199.8 | +13.5 | -23.7 |

## RK4

| Ref | vx (mm/s) | target | err% | W_ext | W_int | W_xfrc | W_joint | dE | Dissip |
|-----|-----------|--------|------|-------|-------|--------|---------|-----|--------|
| scene1_f10 | 61.0 | 51.2 | 19.2% | +174.1 | -6.9 | +167.3 | +363.2 | -1.3 | +168.6 |
| scene1_f20 | 117.2 | 126.4 | 7.3% | +392.7 | -6.9 | +385.8 | +897.5 | +3.6 | +382.2 |
| scene1_f30 | 97.0 | 118.7 | 18.3% | +617.6 | -6.5 | +611.1 | +1375.2 | +1.7 | +609.4 |
| scene1_f50 | 153.7 | 148.3 | 3.6% | +1374.8 | -2.6 | +1372.2 | +2385.1 | +0.6 | +1371.6 |
| scene2_f10 | 97.0 | 83.2 | 16.6% | +248.4 | -9.9 | +238.5 | +432.7 | -0.3 | +238.8 |
| scene2_f20 | 133.6 | 113.1 | 18.1% | +402.8 | -7.5 | +395.3 | +693.6 | +3.0 | +392.3 |
| scene2_f30 | 170.0 | 179.6 | 5.4% | +565.6 | -3.2 | +562.4 | +857.7 | +3.4 | +559.0 |
| scene2_f50 | 345.2 | 263.3 | 31.1% | +1425.5 | -1.0 | +1424.5 | +1934.7 | +6.9 | +1417.6 |
| scene4_f10 | 107.8 | 112.1 | 3.9% | +170.9 | +0.9 | +171.8 | +270.1 | +0.3 | +171.5 |
| scene4_f20 | 209.5 | 184.1 | 13.8% | +347.2 | -0.6 | +346.7 | +513.4 | +4.5 | +342.1 |
| scene4_f30 | 281.5 | 274.7 | 2.5% | +616.8 | -0.1 | +616.7 | +909.9 | +6.2 | +610.5 |
| scene4_f50 | 542.9 | 327.4 | 65.8% | +1094.0 | -1.5 | +1092.4 | +1282.3 | +24.1 | +1068.3 |
| scene_wheel_f10 | 176.0 | 143.2 | 22.9% | +67.4 | -1.1 | +66.3 | +85.3 | +1.5 | +64.8 |
| scene_wheel_f20 | 334.9 | 305.8 | 9.5% | +194.8 | -1.5 | +193.3 | +240.2 | +5.3 | +188.0 |
| scene_wheel_f30 | 521.5 | 449.3 | 16.1% | +382.7 | -0.1 | +382.6 | +464.6 | +14.0 | +368.6 |

## Summary

- **Euler**: 4/15 refs have negative dissipation (unphysical)
- **RK4**: 0/15 refs have negative dissipation

- **Euler mean velocity error**: 11.6%
- **RK4 mean velocity error**: 16.9%

## Interpretation

Dissipation = W_xfrc - dE. Physically, friction and damping always remove energy,
so dissipation must be positive. Negative dissipation means the integrator is
injecting phantom energy into the system through the contact solver.

### Key findings

1. **RK4 fixes the energy budget**: 0/15 refs have negative dissipation (vs 4/15 for Euler).
   W_ext is always positive under RK4 — the field consistently injects energy, as physics demands.

2. **W_int ≈ 0 under RK4**: Inter-joint coupling is genuinely conservative (range: -10 to +1 uJ).
   Under Euler, it drifts up to -25 uJ — numerical artifact from the integrator.

3. **W_joint >> W_ext under Euler**: The joint-projected formula gives 3-10× more energy than the
   naive formula under Euler (e.g., scene1_f10: W_joint=+2100 vs W_ext=-511). Under RK4 the gap
   shrinks to ~2× (W_joint=+363 vs W_ext=+174). The base-coupling artifact is much smaller with RK4.

4. **Velocity error increases with RK4**: Mean error goes from 11.6% → 16.9% because the params
   were optimized for Euler dynamics. Some refs improve (scene1_f20: 25%→7%), others degrade
   badly (scene4_f50: 28%→66%). A re-optimization with RK4 would be needed.

5. **COT under RK4 would use W_ext (naive)**: Since W_ext is always positive and the energy budget
   closes correctly, `COT = W_ext / (m·g·d)` is the physically correct formulation under RK4.
   No need for the joint-projected workaround.

---

## Rough Terrain Analysis

**Params**: rough_v2 best (results/zzz_rough_v2)
**Terrain**: rough (seed=42, 1mm std, 3 tiles, 5mm flat lead)
**Refs**: 7 references (scene1/2/4 × f10/f30 + scene2 f50)
**Sim**: 2s duration, settle=0.1s, dt=0.5ms

### Euler (semi-implicit) — Rough Terrain

| Ref | vx (mm/s) | target | err% | W_ext | W_int | W_xfrc | W_joint | dE | Dissip |
|-----|-----------|--------|------|-------|-------|--------|---------|-----|--------|
| scene1_f10 | 31.2 | 42.9 | 27.3% | -182.9 | -19.9 | -202.8 | +997.7 | +0.4 | -203.1 |
| scene1_f20 | 80.0 | — | — | -197.4 | -32.3 | -229.7 | +1837.3 | -1.3 | -228.4 |
| scene1_f30 | 77.9 | 81.6 | 4.5% | +38.7 | -9.9 | +28.8 | +2119.4 | +5.3 | +23.5 |
| scene1_f50 | 84.9 | — | — | +547.5 | -19.3 | +528.2 | +3482.6 | +8.0 | +520.1 |
| scene2_f10 | 87.2 | 65.6 | 33.0% | +64.7 | -7.9 | +56.8 | +408.1 | +3.3 | +53.5 |
| scene2_f20 | 97.3 | — | — | +172.0 | -1.8 | +170.2 | +744.6 | +0.1 | +170.1 |
| scene2_f30 | 123.1 | 128.9 | 4.5% | +258.0 | -0.8 | +257.2 | +1413.4 | +0.0 | +257.2 |
| scene2_f50 | 115.9 | 106.2 | 9.1% | +743.0 | -9.0 | +734.0 | +2549.8 | -0.2 | +734.3 |
| scene4_f10 | 76.3 | 85.7 | 11.0% | +57.8 | -0.5 | +57.3 | +424.8 | -0.7 | +58.0 |
| scene4_f20 | 123.1 | — | — | +173.4 | -3.9 | +169.5 | +804.7 | +2.0 | +167.5 |
| scene4_f30 | 24.7 | 146.0 | 83.1% | +423.6 | +7.2 | +430.7 | +1541.0 | +2.2 | +428.6 |
| scene4_f50 | 238.8 | — | — | +817.7 | -2.1 | +815.6 | +2092.0 | +13.2 | +802.4 |
| scene_wheel_f10 | 39.6 | — | — | -417.7 | -32.4 | -450.1 | +1415.3 | +2.3 | -452.5 |
| scene_wheel_f20 | 40.1 | — | — | -494.6 | -45.0 | -539.6 | +2186.7 | +1.5 | -541.1 |
| scene_wheel_f30 | 14.2 | — | — | -622.8 | -44.2 | -667.0 | +3386.0 | +1.6 | -668.6 |

### RK4 — Rough Terrain

| Ref | vx (mm/s) | target | err% | W_ext | W_int | W_xfrc | W_joint | dE | Dissip |
|-----|-----------|--------|------|-------|-------|--------|---------|-----|--------|
| scene1_f10 | 36.1 | 42.9 | 15.8% | +118.9 | -1.5 | +117.4 | +161.9 | +0.4 | +117.0 |
| scene1_f20 | 101.1 | — | — | +355.6 | -3.3 | +352.4 | +566.2 | -2.7 | +355.1 |
| scene1_f30 | 97.1 | 81.6 | 18.9% | +624.6 | -6.0 | +618.6 | +1072.8 | +1.5 | +617.1 |
| scene1_f50 | 52.1 | — | — | +1295.0 | -3.8 | +1291.3 | +1968.4 | +3.9 | +1287.4 |
| scene2_f10 | 93.2 | 65.6 | 42.2% | +188.8 | -3.2 | +185.6 | +261.8 | -1.8 | +187.4 |
| scene2_f20 | 36.7 | — | — | +309.0 | -1.8 | +307.2 | +473.0 | +1.3 | +305.9 |
| scene2_f30 | 106.9 | 128.9 | 17.1% | +637.8 | -2.9 | +634.9 | +986.1 | +0.7 | +634.2 |
| scene2_f50 | 29.0 | 106.2 | 72.7% | +1336.1 | -1.3 | +1334.9 | +1931.3 | +3.8 | +1331.1 |
| scene4_f10 | 85.1 | 85.7 | 0.6% | +161.3 | +0.9 | +162.2 | +253.8 | -1.8 | +164.0 |
| scene4_f20 | 145.5 | — | — | +371.5 | -1.1 | +370.5 | +568.9 | -1.0 | +371.5 |
| scene4_f30 | 244.3 | 146.0 | 67.3% | +741.1 | -1.3 | +739.9 | +1125.8 | +8.1 | +731.8 |
| scene4_f50 | 19.3 | — | — | +1397.7 | +0.1 | +1397.8 | +1847.7 | +0.3 | +1397.5 |
| scene_wheel_f10 | 13.4 | — | — | +103.3 | -2.4 | +100.9 | +121.0 | +0.3 | +100.5 |
| scene_wheel_f20 | 13.5 | — | — | +288.0 | -2.1 | +285.9 | +366.6 | +0.4 | +285.4 |
| scene_wheel_f30 | 30.0 | — | — | +548.8 | +0.4 | +549.2 | +770.3 | +1.8 | +547.4 |

### Rough Terrain Summary

- **Euler**: 5/15 refs have negative dissipation (scene1_f10, scene1_f20, scene_wheel_f10/f20/f30)
- **RK4**: 0/15 refs have negative dissipation

### Rough Terrain Observations

1. **Wheel morphology worst under Euler**: All 3 wheel refs have massive negative W_ext (-418 to -623 uJ)
   and dissipation (-453 to -669 uJ). The contact solver injects enormous phantom energy into
   the wheel robot on rough terrain. RK4 fixes all three: W_ext is +103 to +549 uJ (positive).

2. **Scene1 (single leg) also affected**: f10 and f20 both have negative dissipation under Euler
   (-203 and -228 uJ). RK4 fixes both.

3. **W_int near zero under RK4**: Range -6.0 to +0.9 uJ (conservative, as expected).
   Under Euler, up to -45.0 uJ drift — much worse than flat terrain, especially for wheel.

4. **W_joint/W_ext gap shrinks under RK4**: E.g., scene_wheel_f30: Euler ratio undefined (W_ext<0),
   RK4 ratio = 770.3/548.8 ≈ 1.4×. Scene4_f10: Euler 424.8/57.8 ≈ 7.3×, RK4 253.8/161.3 ≈ 1.6×.

5. **Rough terrain amplifies the integrator problem**: 5/15 negative dissipation (vs 4/15 on flat
   for the same integrator), and the magnitudes are larger (up to -669 uJ vs -533 uJ on flat).

6. **RK4 videos**: saved to `results/zzz_rough_v2/rk4_videos/` (all 15 refs).
