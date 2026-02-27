# Cost of Transport: Energy Computation Methodology

## Problem Statement

The milliquad robots are driven by an external rotating magnetic field that applies torques to permanent magnets embedded in each leg. Unlike motor-driven robots where actuator power is directly measurable, the energy input to these field-driven systems requires careful definition because the external torque acts on rigid bodies that simultaneously participate in joint rotation and whole-body motion.

## Power Formulation

The external magnetic torque $\boldsymbol{\tau}_{\text{ext},j}$ (in the world frame) is applied to each leg body $j$. The instantaneous angular velocity of each leg in the world frame decomposes as:

$$\boldsymbol{\omega}_{\text{leg},j} = \boldsymbol{\omega}_{\text{base}} + \dot{q}_j \, \hat{\mathbf{a}}_j$$

where $\boldsymbol{\omega}_{\text{base}}$ is the floating-base angular velocity, $\dot{q}_j$ is the scalar hinge-joint velocity, and $\hat{\mathbf{a}}_j$ is the joint axis in the world frame.

The total instantaneous power delivered by the field to leg $j$ is:

$$P_j^{\text{total}} = \boldsymbol{\tau}_{\text{ext},j} \cdot \boldsymbol{\omega}_{\text{leg},j}$$

which decomposes as:

$$P_j^{\text{total}} = \underbrace{\boldsymbol{\tau}_{\text{ext},j} \cdot \boldsymbol{\omega}_{\text{base}}}_{P_j^{\text{base}}} + \underbrace{(\boldsymbol{\tau}_{\text{ext},j} \cdot \hat{\mathbf{a}}_j)\, \dot{q}_j}_{P_j^{\text{joint}}}$$

The first term represents field energy that flows through the leg into base-body rotation via the joint constraint. The second term is the field energy that drives joint-relative motion, producing the ground-contact forces responsible for locomotion.

## Choice of Energy Metric

We define the mechanical energy input for COT using the **joint-projected power**:

$$P_{\text{ext}}(t) = \sum_{j=1}^{N_{\text{legs}}} (\boldsymbol{\tau}_{\text{ext},j} \cdot \hat{\mathbf{a}}_j)\, \dot{q}_j$$

$$E_{\text{ext}} = \int P_{\text{ext}}(t) \, dt$$

This measures the energy delivered by the external field through the joint degrees of freedom — the pathway by which the field produces locomotion. The base-rotation term $P_j^{\text{base}}$ is excluded because it represents energy exchanged between the field and whole-body rotation (primarily yaw and pitch oscillations induced by ground contact), which does not contribute to forward locomotion.

### Justification

Using the total field power $P^{\text{total}} = \sum_j \boldsymbol{\tau}_{\text{ext},j} \cdot \boldsymbol{\omega}_{\text{leg},j}$ is problematic for two reasons:

1. **Sign instability.** The base angular velocity $\boldsymbol{\omega}_{\text{base}}$ fluctuates rapidly due to contact dynamics. The net torque $\sum_j \boldsymbol{\tau}_{\text{ext},j}$ may oppose $\boldsymbol{\omega}_{\text{base}}$ over extended intervals, causing the field to act as a brake on base rotation. In our flat-terrain simulations, $\int P^{\text{total}} \, dt$ evaluates to $-498 \, \mu\text{J}$ (net negative), while the joint-projected integral gives $+804 \, \mu\text{J}$. The negative total arises because the field extracts more energy from base rotation ($-1302 \, \mu\text{J}$) than it injects through the joints.

2. **Physical interpretation.** The base-rotation power does not represent a deliberate actuation cost. It is an incidental consequence of applying body-frame torques to legs that share the base's motion. Excluding it isolates the energetic cost of the locomotion mechanism itself, analogous to reporting actuator shaft power rather than total power dissipation in a motor-driven system.

### Verification

The joint-projected power was verified against MuJoCo's rotational Jacobian. For each joint DOF $j$, the Jacobian column $\mathbf{J}_r(:, j)$ equals the joint axis $\hat{\mathbf{a}}_j$ in the world frame, and the generalized joint torque $\mathbf{J}_r^T \boldsymbol{\tau}_{\text{ext}}$ matches $(\boldsymbol{\tau}_{\text{ext}} \cdot \hat{\mathbf{a}}_j)$ to machine precision at every timestep.

## Cost of Transport

$$\text{COT} = \frac{E_{\text{ext}}}{m \, g \, d}$$

where $m$ is the total robot mass, $g = 9.81 \, \text{m/s}^2$, and $d$ is the cumulative horizontal (2D) path length over the measurement interval. We use signed energy $E_{\text{ext}}$ (net positive work by the drive field) rather than absolute energy $\int |P| \, dt$, which would inflate the estimate by double-counting oscillatory energy exchange within each drive cycle.

For step terrain, the gravitational potential energy gain $m g \Delta h$ is reported separately. At $\Delta h \approx 7 \, \text{mm}$ and robot masses of $91$–$109 \, \text{mg}$, the gravitational contribution is $6$–$8 \, \mu\text{J}$, representing $2$–$6\%$ of the total joint energy — confirming that the elevated COT on steps is dominated by contact losses rather than gravitational work.
