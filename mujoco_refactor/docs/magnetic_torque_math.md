# Magnetic Torque Model

This document describes the two magnetic torque mechanisms used in the milliquad MuJoCo simulation. Both torques are computed every timestep and applied as world-frame torques on each leg body via MuJoCo's `xfrc_applied`.

Implementation: `simulation.py` (readable) and `simulation_fast.py` (vectorized, same physics).

---

## 1. External Drive Torque

The robot is actuated by a spatially-uniform, time-varying external magnetic field that rotates in the *xz*-plane. Each leg has an embedded permanent magnet with dipole moment **m**_i.

### Drive field direction

The goal direction rotates about the *y*-axis at frequency *f*_drive:

$$\theta(t) = 2\pi \, f_{\text{drive}} \, (t - t_{\text{settle}}) \mod 2\pi$$

$$\hat{\mathbf{B}}_{\text{goal}} = \begin{pmatrix} \sin\theta \\ 0 \\ \cos\theta \end{pmatrix}$$

The field starts along +*z* and sweeps through the *xz*-plane.

### Torque on each magnet

The alignment torque on a magnetic dipole in a uniform field is **τ** = **m** × **B**. Here it is implemented as a proportional torque:

$$\boldsymbol{\tau}_{\text{ext},i} = k_p \left( \hat{\mathbf{n}}_i \times \hat{\mathbf{B}}_{\text{goal}} \right)$$

where:
- **n̂**_i is the unit north direction of magnet *i* in world frame, obtained by rotating the body-frame magnet axis by the body's orientation quaternion.
- *k*_p is a scalar gain with units of torque (N·m):

$$k_p = m \cdot B_{\text{field}} \cdot f_{\text{moment}} \cdot f_{\text{field}}$$

with *m* = `MAGNETIC_MOMENT` (1.13 × 10⁻³ A·m²), *B*_field = `MAGNETIC_FIELD_MAGNITUDE` (2 × 10⁻³ T), and *f*_moment, *f*_field being dimensionless fudge factors tuned during optimization.

The cross product gives zero torque when aligned and maximum torque when perpendicular, so this acts as a proportional controller driving each magnet's north pole toward the rotating goal direction.

### Magnet polarity convention

Legs 0 and 2 (FR, BR) have their magnet north along body +*x*; legs 1 and 3 (FL, BL) along body −*x*. Diagonally opposite legs are co-aligned while left-right neighbors are anti-aligned, producing the alternating leg swing needed for a walking gait.

---

## 2. Inter-Joint (Dipole–Dipole) Torque

Each magnet also produces a field that acts on every other magnet. This uses the standard magnetic dipole–dipole interaction.

### Dipole moment vectors

$$\mathbf{m}_i = m_{\text{mag}} \, \hat{\mathbf{n}}_i$$

where *m*_mag = `MAGNETIC_MOMENT` × `magnetic_moment_fudge` (scalar magnitude, ~1.13 × 10⁻³ A·m²).

### Field from dipole *j* at dipole *i*

$$\mathbf{B}_{j \to i} = \frac{\mu_0}{4\pi} \frac{1}{|\mathbf{r}|^3} \left[ 3(\mathbf{m}_j \cdot \hat{\mathbf{r}})\,\hat{\mathbf{r}} - \mathbf{m}_j \right]$$

where **r** = **x**_i − **x**_j is the displacement from source *j* to target *i*, and μ₀/(4π) = 10⁻⁷ N/A².

### Total field and torque at magnet *i*

$$\mathbf{B}_i = \sum_{j \neq i} \mathbf{B}_{j \to i}$$

$$\boldsymbol{\tau}_{\text{int},i} = \mathbf{m}_i \times \mathbf{B}_i$$

This is the exact point-dipole formula with no approximations beyond clamping |**r**| ≥ 10⁻⁶ m to prevent the 1/*r*³ singularity.

---

## Summary

| | External | Inter-joint |
|---|---|---|
| **Physics** | Uniform rotating field aligns magnets | Dipole–dipole coupling between magnets |
| **Formula** | τ = *k*_p (n̂ × B̂_goal) | τ = **m** × Σ **B**_dipole |
| **Role** | Drives locomotion gait | Couples leg motion (parasitic/assistive) |
| **Scales as** | Independent of geometry | 1/*r*³ — strong at close range |

---

## Code Reference

| Concept | `simulation.py` | `simulation_fast.py` |
|---|---|---|
| Magnet north direction | `_get_magnet_state()` (per-leg, scipy) | `_get_all_magnet_states()` (batched, inline quat rotation) |
| External torque | `_compute_external_torques()` L140–155 | `_compute_external_torques()` L158–169 |
| Dipole field | `_dipole_field()` L133–137 | Inline vectorized, L191–220 |
| Inter-joint torque | `_compute_interjoint_torques()` L158–181 | `_compute_interjoint_torques()` L176–220 |
| Torque application | `_apply_magnetic_forces()` L184–227 | `_apply_magnetic_forces()` L223–265 |
| Constants | `config.py`: `MAGNETIC_MOMENT`, `MAGNETIC_FIELD_MAGNITUDE`, `MU0_OVER_4PI`, `R_EPS` | same |
| Fudge → physical | `config.py` L338–339: `m_mag`, `kp_mag` | same |
