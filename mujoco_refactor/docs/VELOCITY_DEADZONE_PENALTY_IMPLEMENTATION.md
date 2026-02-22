# Velocity Deadzone Penalty Implementation Plan

## Goal
Change velocity cost behavior so that:
- inside deadzone (`|error| <= speed_std`): zero cost
- outside deadzone: cost reflects total miss magnitude more strongly (not tiny excess-only growth)

This matches the intent: with `std=10%`, `9%` gets `0`, `11%` gets some penalty, and `15%` is meaningfully worse than `11%`.

## Current Behavior (today)
In `mujoco_refactor/optimizer.py` (`calculate_cost`), velocity error is:

```python
if abs(vel_deviation) <= speed_std:
    velocity_error = 0.0
else:
    excess = abs(vel_deviation) - speed_std
    velocity_error = (excess / target_velocity) ** 2
```

Then total row cost includes:

```python
VELOCITY_COST_WEIGHT * velocity_error
```

This is quadratic on **excess beyond sigma**, which makes just-outside-bound penalties small.

## Recommended New Behavior
Use deadzone + total-relative-error outside zone:

```python
rel_err = abs(vel_deviation) / target_velocity
rel_std = speed_std / target_velocity

if rel_err <= rel_std:
    velocity_error = 0.0
else:
    velocity_error = rel_err
```

Interpretation:
- still zero inside deadzone
- once outside, penalty is proportional to total percent error
- example (`std=10%`): `11% -> 0.11`, `15% -> 0.15` (about 36% worse)

If you want steeper growth (closer to "15% is ~50%+ worse than 11%"), use:

```python
velocity_error = rel_err ** 1.5
```

outside the deadzone.

## Implementation Steps
1. Edit `mujoco_refactor/optimizer.py` in `calculate_cost`.
2. Replace the current deadzone/excess block with the new deadzone/total-error block.
3. Keep `VELOCITY_COST_WEIGHT` in `config.py` as the global knob. Re-tune if needed after formula change.

## Optional (Cleaner) Config Knobs
Add to `mujoco_refactor/config.py`:

```python
VELOCITY_OUTSIDE_MODE = "total_linear"   # or "total_power"
VELOCITY_OUTSIDE_POWER = 1.5             # used for total_power
```

Then in `calculate_cost`, branch by mode for quick experimentation without code edits.

## Recommended Printout Update (Deadzone Visibility)
To make optimizer output match the deadzone logic, update `_print_ref_table` in
`mujoco_refactor/optimizer.py` to show whether each row is inside/outside `+-1σ`.

Add a `1σ` column:
- `IN` if `abs(sim_v - target) <= speed_std`
- `OUT` otherwise

Suggested implementation sketch:

```python
speed_std = row.get("speed_std", 0.0)
in_sigma = abs(sim_v - target) <= speed_std if speed_std > 0 else False
sigma_flag = "IN" if in_sigma else "OUT"
```

Then print `sigma_flag` in the table next to `Δ%`.

Why:
- `Δ%` alone is misleading when `speed_std/target` differs a lot by reference.
- This immediately shows whether velocity error is currently penalized by the deadzone term.

## Validation Checklist
1. Unit sanity (single reference):
   - set `target=1.0`, `speed_std=0.1`
   - verify `rel_err=0.09 -> 0`
   - verify `rel_err=0.11 -> >0`
   - verify `rel_err=0.15` larger than at `0.11` by intended ratio
2. Run a short optimizer smoke test (small `N_CALLS`) and confirm:
   - no crashes
   - cost monotonicity for controlled synthetic velocities behaves as expected
3. Compare best rows before/after:
   - fewer accepted high-% outliers should appear if other terms are similar
   - verify this in `optimization_bests.csv`

## Notes for the Implementing Agent
- Do not edit historical run snapshots under `mujoco_refactor/results/*/config.py` unless intentionally reproducing prior runs.
- Edit the active source config at `mujoco_refactor/config.py` for future runs.
