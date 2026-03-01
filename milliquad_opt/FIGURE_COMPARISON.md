# Figure Comparison: Old vs New Parameters

**Old**: `rk4_step_cold` (q75, cost=0.243) + `rk4_rough` (time-gated, cost=0.362)
**New**: `step_q60_rk-warm` (q60, cost=0.210) + `rough_spatial_rk4` (spatial-gated, cost=0.184)
**Flat**: Unchanged (`rk4_flat`, cost=0.188)

---

## Velocity vs Frequency

### Step Terrain

<table>
<tr><th>Old (rk4_step_cold, q75)</th><th>New (step_q60_rk-warm)</th></tr>
<tr>
<td><img src="velocity_vs_freq_step_old.png" width="450"/></td>
<td><img src="velocity_vs_freq_step.png" width="450"/></td>
</tr>
</table>

### Rough Terrain

<table>
<tr><th>Old (rk4_rough, time-gated)</th><th>New (rough_spatial_rk4)</th></tr>
<tr>
<td><img src="velocity_vs_freq_rough_old.png" width="450"/></td>
<td><img src="velocity_vs_freq_rough.png" width="450"/></td>
</tr>
</table>

### Flat Terrain (unchanged)

![Flat velocity](velocity_vs_freq_flat.png)

---

## COT vs Frequency

### Step Terrain

<table>
<tr><th>Old (rk4_step_cold, q75)</th><th>New (step_q60_rk-warm)</th></tr>
<tr>
<td><img src="cot_vs_freq_step_old.png" width="450"/></td>
<td><img src="cot_vs_freq_step.png" width="450"/></td>
</tr>
</table>

### Rough Terrain

<table>
<tr><th>Old (rk4_rough, time-gated)</th><th>New (rough_spatial_rk4)</th></tr>
<tr>
<td><img src="cot_vs_freq_rough_old.png" width="450"/></td>
<td><img src="cot_vs_freq_rough.png" width="450"/></td>
</tr>
</table>

### Flat Terrain (unchanged)

![Flat COT](cot_vs_freq_flat.png)

---

## Experimental vs Simulation: Velocity

<table>
<tr><th>Old (rk4_step_cold)</th><th>New (step_q60_rk-warm)</th></tr>
<tr>
<td><img src="exp_vs_sim_velocity_old2.png" width="450"/></td>
<td><img src="exp_vs_sim_velocity.png" width="450"/></td>
</tr>
</table>

## Experimental vs Simulation: Pitch

<table>
<tr><th>Old (rk4_step_cold)</th><th>New (step_q60_rk-warm)</th></tr>
<tr>
<td><img src="exp_vs_sim_pitch_old2.png" width="450"/></td>
<td><img src="exp_vs_sim_pitch.png" width="450"/></td>
</tr>
</table>

---

## Composite Figures (New)

### Sim-Only: 3×3 (flat / step / rough × velocity / COT / pitch)

All three winner params. Invalid trials excluded via gate-clearing (rough: max_x < 155mm, step: max_x < 101.5mm) or inverted (pitch > 30°); scene-colored X where all trials failed. Pitch is yaw-invariant (`arctan2(dz, sqrt(dx²+dy²))` between FL/BL legs).

![Sim composite](sim_composite.png)

### Exp vs Sim: Flat + Step (velocity + pitch)

Step experimental data uses q60 windowing (matching sim optimization targets). Flat sim includes WR f50 (~719 mm/s; exp-only failure — robot self-destructs at 50Hz experimentally).

![Exp vs sim composite](exp_vs_sim_composite.png)

### Pitch vs Frequency (new — individual panels)

<table>
<tr><th>Flat</th><th>Step</th><th>Rough</th></tr>
<tr>
<td><img src="pitch_vs_freq_flat.png" width="300"/></td>
<td><img src="pitch_vs_freq_step.png" width="300"/></td>
<td><img src="pitch_vs_freq_rough.png" width="300"/></td>
</tr>
</table>
