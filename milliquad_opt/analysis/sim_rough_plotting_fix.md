# Fixing Simulation Rough Terrain Velocity Plots

## The Problem

When plotting the `megacomposite_nocot` figure, we noticed that simulation trials for rough terrain (specifically `scene_wheel` at 30Hz) were plotting very slow trials (e.g., a trial that only achieved **10.2 mm/s**) as normal dots.

This behavior differed from the experimental rough terrain plots (`random_terrain_raw.csv`), where individual failed/stalled trials are plotted as `X` markers on the zero line.

## The Investigation

1. **Why was the slow trial selected?**
   In `validate_params.py`, the optimizer usually picks the "top N" trials by velocity error relative to a target speed. However, for "exploratory" rough terrain (like the wheel scene), there is no target speed. Therefore, the script simply performs a random selection of 5 out of 10 jittered trials.

   In our specific 30Hz WR case, the random selection picked three fast trials (94.5, 99.1, 100.8 mm/s) and two very slow/stalled trials (35.4, 10.2 mm/s).

2. **Why didn't it trigger a crash/failure?**
   A simulation `crash` is strictly reserved for MuJoCo C-level numerical instabilities (e.g., NaN/Inf values causing an exception) or early aborts. A robot that just crawls slowly but doesn't blow up the physics engine will register `crash=False`.

3. **Why wasn't the stall detected?**
   The codebase **does** have logic to detect stalled trials! The function `_is_valid_trial()` explicitly checks if a robot reached the `GATE_END` distance (which is `0.155m` for rough terrain).

   When we checked the 10.2 mm/s trial, its `max_x` was only `0.0507m`, which comprehensively failed the `0.155m` gate check. The trial was correctly returning `False` from `_is_valid_trial()`.

4. **The Root Cause in the Plotting Logic**
   If the trial was correctly flagged as invalid, why did it still plot as a dot?

   The issue was inside `analysis.plot_validation.build_plot_data()`. The code was doing this:

   ```python
   if exclude_invalid:
       valid = [r for r in valid if _is_valid_trial(r, gate_end)]
   ```

   If a trial failed the gate check, it was **silently dropped** from the dataset before plotting.

   Furthermore, the plotting capability (`scatter_only=True` mode) only draws an `X` marker for trials whose velocity value is `<= 0`. Since the stalled trials were being completely removed from the dataset, the plotter never saw them to draw the X's. (The only time X's were drawn was if `build_all_failed_freqs()` detected that _every single trial_ at a frequency failed).

## The Fix

We amended `build_plot_data()` in `plot_validation.py` to match the experimental plotting logic.

Instead of filtering out invalid trials entirely, we now check `_is_valid_trial()` inside the grouping loop. If a trial fails the validity check, we still append it to our list of velocities to plot, but we **force its value to 0.0**:

```python
for r in freq_rows:
    is_valid = _is_valid_trial(r, gate_end)
    if exclude_invalid and not is_valid:
        # Plot as failure (X marker at 0.0) but don't include in means
        trial_freqs.append(freq)
        trial_vals.append(0.0)
        continue

    # ... process normal valid trials ...
```

By injecting `0.0`, the `plot_panel()` function's `scatter_only` mode naturally picks these up and draws them as `X` markers on the x-axis, perfectly mirroring how the `n/a` entries are handled for the experimental physical trials.
