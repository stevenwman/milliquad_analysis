# COT Bounds from Experimental Kinematics

Bounding the Cost of Transport using only kinematics data — no direct torque or power measurement available at this scale.

## Upper Bound: Max Torque x Angular Velocity

### Key insight
Each leg has 1 magnet. The magnetic torque on a leg is `tau = m x B`, with magnitude `|tau| = m*B*sin(angle)`. The maximum possible torque is `tau_max = m * B` (when magnet perpendicular to field). We don't know the instantaneous angle between magnet and field, but we know it can't exceed this.

### Formula
```
tau_max = MAGNETIC_MOMENT * MAGNETIC_FIELD_MAGNITUDE   (per leg)
P_max   = 4 * tau_max * |omega|                        (4 legs, same drive)
COT_ub  = mean(P_max) / (m * g * v_avg)
```

This overestimates because:
- `sin(angle) < 1` most of the time — actual torque is lower
- The torque-velocity product includes phases where torque does negative work (braking), which we count as positive here

### Why it's a valid upper bound
The magnetic field is externally imposed (Helmholtz coils), so no energy is stored in the field by the robot. The only energy input channel is magnetic torque x angular velocity. Since `|tau| <= tau_max` at every instant, `P_max` is a strict upper bound on instantaneous power input.

## Lower Bound: Kinetic Energy from Rest

### Key insight
The robot starts from rest. At the measurement window, it has translational and rotational kinetic energy. This energy must have been supplied by the actuator (conservation of energy). Any additional energy went into friction/dissipation.

### Formula
```
KE_trans    = 0.5 * m * v^2                           (translational, endpoint snapshot)
KE_rot_legs = 4 * 0.5 * I_leg * omega^2               (4 legs spinning)
KE_rot_body = 0.5 * I_body_pitch * (d_pitch/dt)^2     (body pitching)
E_lower     = KE_trans + KE_rot_legs + KE_rot_body
COT_lb      = E_lower / (m * g * d_forward)
```

PE is excluded: on flat terrain, CSV y is lateral (top-down camera), not height. On step terrain, height gain reflects climbing geometry, not locomotion efficiency.

### Why it's a valid lower bound
By conservation of energy: `E_actuator = KE_final + E_friction + E_vibration + ...`. Since all dissipation terms are non-negative, `KE_final <= E_actuator`. Therefore `COT_lb <= COT_true`.

### Why it's very loose
The lower bound only captures *net* KE at a single instant. It misses:
- Friction losses (dominant energy sink — sliding, torsional, rolling contact friction)
- Cyclic acceleration/deceleration within each leg rotation (KE gained and lost every cycle)
- Vibrational energy dissipated into the substrate
- Any work done against internal damping

At this scale, friction dominates. The robot's KE is uJ-scale while the upper bound power is uW-scale sustained over seconds -> mJ-scale total energy. The lower bound captures <1% of actual energy input.

## Physical Constants

| Parameter | Value | Source |
|-----------|-------|--------|
| MAGNETIC_MOMENT | 1.13e-3 A*m^2 | `milliquad_opt/config.py:47` |
| MAGNETIC_FIELD_MAGNITUDE | 2e-3 T | `milliquad_opt/config.py:48` |
| tau_max | 2.26 uN*m per leg | m x B |
| N_LEGS | 4 | All morphologies have 4 legs (1/2/4 = spokes per leg, not leg count) |
| Robot mass (leg/L1) | 0.103 mg | From MJCF |
| Robot mass (2leg/L2) | 0.105 mg | From MJCF |
| Robot mass (4leg/L4) | 0.109 mg | From MJCF |
| Robot mass (wheel/WR) | 0.091 mg | From MJCF |
| I_leg (L1) | 4.19e-12 kg*m^2 | `model.body_inertia[leg][2]` (Izz, hinge axis) |
| I_leg (L2) | 3.48e-12 kg*m^2 | |
| I_leg (L4) | 6.99e-12 kg*m^2 | |
| I_leg (WR) | 2.59e-12 kg*m^2 | |
| I_body_pitch | 3.14e-10 kg*m^2 | Iyy, same body across all morphologies |

## Experimental CSV Columns

| Col | Group | Label | Used for |
|-----|-------|-------|----------|
| 0 | -- | t | Timestep (s, dt=1ms) |
| 1 | mass_A | x | Forward position (negated, camera frame) |
| 2 | mass_A | y | Lateral (flat, top-down) or height (step, side-view) |
| 3 | mass_A | vx | Forward velocity (negated, camera frame) |
| 4 | mass_A | vy | Lateral/vertical velocity |
| 5-8 | mass_C | x,y,vx,vy | Same as mass_A, second body marker |
| 9 | mass_B | theta | Cumulative leg rotation (deg) |
| 10 | mass_B | omega | Leg angular velocity (deg/s) |
| 11 | mass_C | theta | Body pitch (deg) |

- `omega` verified: 10Hz drive -> ~3600 deg/s mean (exact match to 10x360)
- Velocity: `v_fwd = -0.5 * (vx_A + vx_C)` (avg of 2 body markers, negated)
- Flat CSVs have NaN velocity near end (tracker loses marker) — lower bound code scans backwards to find last valid row

### Windowing
- **Flat**: last 50% of recording
- **Step**: q60 index window [45%-75%]

## Results

### Upper bound (flat, mean across 3 trials)

| Morph | 10 Hz | 20 Hz | 30 Hz | 50 Hz |
|-------|-------|-------|-------|-------|
| L1 | 12.1 | 11.8 | 14.5 | 16.9 |
| L2 | 8.2 | 7.2 | 9.9 | 11.1 |
| L4 | 5.4 | 4.6 | 6.1 | 8.2 |
| WR | 3.0 | 3.3 | 4.2 | 5.0 |

### Upper bound (step, mean across 3 trials)

| Morph | 10 Hz | 20 Hz | 30 Hz |
|-------|-------|-------|-------|
| L1 | 39.3 | 28.1 | 67.7 |
| L2 | 19.6 | 15.0 | 17.1 |
| L4 | 9.0 | 12.4 | 21.6 |
| WR | -- | -- | 19.5 |

Higher than flat due to lower velocities on steps (COT denominator shrinks).

### Lower bound (flat, mean across 3 trials)

| Morph | 10 Hz | 20 Hz | 30 Hz | 50 Hz |
|-------|-------|-------|-------|-------|
| L1 | 0.003 | 0.031 | 0.017 | 0.042 |
| L2 | 0.007 | 0.031 | 0.031 | 0.058 |
| L4 | 0.011 | 0.034 | 0.044 | 0.126 |
| WR | 0.013 | 0.044 | 0.137 | 0.210 |

### Lower bound (step, mean across 3 trials)

| Morph | 10 Hz | 20 Hz | 30 Hz |
|-------|-------|-------|-------|
| L1 | 0.015 | 0.092 | 0.120 |
| L2 | 0.042 | 0.107 | 0.153 |
| L4 | 0.065 | 0.058 | 0.155 |
| WR | -- | -- | 0.032 |

### Comparison

| | Lower bound | Sim COT | Upper bound |
|---|---|---|---|
| Flat range | 0.003-0.21 | 0.5-11.7 | 3.0-16.9 |
| Step range | 0.015-0.16 | -- | 9.0-67.7 |
| Ratio (UB/LB) | | | 60-1400x |

The sim COT (from `_common.py`, tau*omega with RK4 integrator) sits within the bounds, closer to the upper bound. This is expected: most energy goes into friction/dissipation (captured by upper bound's P_max integral), not net KE accumulation (captured by lower bound).

The upper bound is 2-10x the sim COT on flat terrain, consistent with `sin(angle)` averaging ~0.5-0.7 and some braking torque cancellation.

## Energy Scale Context

At this robot scale (~0.1 mg, ~50 mm/s):
- Translational KE: ~0.1-25 uJ (dominates lower bound at high freq)
- Leg rotational KE: ~0.001-3.7 uJ (significant for L4 with higher I_leg)
- Body pitch KE: <0.3 uJ (always negligible — slow pitch oscillation)
- Upper bound power: ~0.01-0.1 uW sustained over 1-2s -> 10-200 uJ total energy
- The gap between bounds reflects the dominance of friction at this scale

## Scripts

```bash
# Upper bound
python cot_upper_bound.py --terrain flat
python cot_upper_bound.py --terrain step

# Lower bound
python cot_lower_bound.py --terrain flat
python cot_lower_bound.py --terrain step

# Plots
python plot_cot_upper_bound.py
python plot_cot_lower_bound.py
```
