# Step Optimizer Cost Function

## Per-Reference Cost

Measured only in the step region (x >= step_start_x), capped at 90% of step length to avoid cliff-fall artifacts.

$$C = 5 \cdot C_{vel} + 1 \cdot C_{tumble} + 1 \cdot C_{lateral} + 0 \cdot C_{yaw} + 2 \cdot C_{progress}$$

### Velocity Error (no deadzone)

For normal refs (target > 0):

$$C_{vel} = \left(\frac{v_{sim} - v_{target}}{v_{target}}\right)^2$$

For failure-mode refs (target = 0, e.g. wheel f20 on steps):

$$C_{vel} = \left(\frac{v_{sim}}{0.05}\right)^2$$

### Tumble (threshold = 0.0, always active)

$$C_{tumble} = \frac{1}{N} \sum_{t=1}^{N} 0.1 \cdot (1 - \hat{z}_{body}(t) \cdot \hat{z}_{world})$$

### Lateral Displacement

$$C_{lateral} = (\Delta y)^2 = (y_{exit} - y_{enter})^2$$

### Yaw

Disabled (weight = 0) due to cliff-fall artifact corrupting yaw at end of staircase.

$$C_{yaw} = \begin{cases} 0 & \text{if } \Delta\psi \leq 60° \\ \left(\frac{\Delta\psi - 60°}{90°}\right)^2 & \text{otherwise} \end{cases}$$

where $\Delta\psi = \arccos(\hat{h}_{start} \cdot \hat{h}_{end})$, unsigned [0°, 180°].

### Progress Penalty

Penalizes not reaching end of step region:

$$C_{progress} = \left(1 - \text{clamp}\left(\frac{x_{final} - x_{start}}{x_{end} - x_{start}},\ 0,\ 1\right)\right)^2$$

## Aggregation

- **Jitter trials**: 2 trials per reference, select best (argmin cost)
- **Across references**: weighted sum of best-trial costs
- Failure-mode refs (target = 0) have weight = 2.0; all others weight = 1.0

## Config Values

| Parameter | Value |
|-----------|-------|
| VELOCITY_COST_WEIGHT | 5.0 |
| TUMBLE_COST_WEIGHT | 1.0 |
| LATERAL_COST_WEIGHT | 1.0 |
| YAW_COST_WEIGHT | 0.0 |
| PROGRESS_COST_WEIGHT | 2.0 |
| TUMBLE_THRESHOLD | 0.0 |
| TUMBLE_PENALTY_SCALE | 0.1 |
| YAW_THRESHOLD_DEG | 60.0 |
| FAILURE_MODE_VEL_SCALE | 0.05 m/s |
| VELOCITY_DEADZONE | False |
