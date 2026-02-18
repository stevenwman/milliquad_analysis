#!/usr/bin/env python3
"""Test that every search dimension actually affects simulation output.

Uses the REAL simulation_fast.run_simulation() — same code path as optimizer.
condim=6 is now set inside simulation_fast.py.

For each of the 13 search parameters, perturbs it ±10% from baseline while
holding all others fixed. Compares final position to baseline.
If BOTH perturbations produce BIT-IDENTICAL output, the param is dead.

Baseline chosen to produce real locomotion at condim=6
(rolling_friction lowered from old best since >1e-3 kills sim at condim=6).
"""

import numpy as np

from config import space, sim_params_from_point, MJCF_PATHS
from simulation_fast import run_simulation

SCENE = "scene4"
FREQ = 30.0
DURATION = 3.0

# Baseline: same as best scene4 params but rolling_friction=1e-4
# (old value 0.087 kills locomotion at condim=6)
BASELINE = {
    "sliding_friction": 0.61545244,
    "torsional_friction": 0.00028306636,
    "rolling_friction": 1e-4,
    "solref_timeconst": 0.00037381967,
    "solref_dampratio": 5.9944763,
    "solimp_dmin": 0.8877443,
    "solimp_delta_d": 0.24089394,
    "solimp_width": 0.0041534584,
    "solimp_midpoint": 0.11378585,
    "solimp_power": 4.2305298,
    "magnetic_moment_fudge": 0.80040943,
    "magnetic_field_fudge": 1.1993118,
    "dof_damping": 1.4228549e-09,
}


def make_point(**overrides):
    params = dict(BASELINE)
    params.update(overrides)
    return [params[dim.name] for dim in space]


def run_once(point):
    """Run a single simulation via simulation_fast.run_simulation()."""
    params = sim_params_from_point(point)
    params["drive_freq"] = FREQ
    traj = run_simulation(
        params,
        mjcf_path=MJCF_PATHS[SCENE],
        sim_duration=DURATION,
        visualize=False,
    )
    if traj is None or len(traj) == 0:
        return None
    return np.array(traj[-1]["pos"])


def main():
    base_point = make_point()
    print("Running baseline (simulation_fast.run_simulation with condim=6)...")
    pos_base = run_once(base_point)
    if pos_base is None:
        print("  BASELINE SIM FAILED — cannot proceed")
        return
    vel = pos_base[0] / DURATION
    print(f"  Baseline final pos: {pos_base}")
    print(f"  Baseline avg velocity: {vel:.4f} m/s\n")

    if abs(vel) < 0.01:
        print("  WARNING: baseline velocity near zero — results unreliable\n")

    print(f"{'Param':<26} {'Base val':>14} {'Lo val':>14} {'Hi val':>14}  "
          f"{'Lo result':>14} {'Hi result':>14}  Verdict")
    print("-" * 130)

    for dim in space:
        base_val = BASELINE[dim.name]

        # ±10% perturbation, clamped to bounds
        lo_val = max(base_val * 0.9, dim.low)
        hi_val = min(base_val * 1.1, dim.high)

        # If clamping ate the perturbation, push harder
        if lo_val == base_val:
            lo_val = dim.low
        if hi_val == base_val:
            hi_val = dim.high

        lo_point = make_point(**{dim.name: lo_val})
        hi_point = make_point(**{dim.name: hi_val})

        pos_lo = run_once(lo_point)
        pos_hi = run_once(hi_point)

        lo_identical = pos_lo is not None and np.array_equal(pos_lo, pos_base)
        hi_identical = pos_hi is not None and np.array_equal(pos_hi, pos_base)

        if pos_lo is None:
            lo_str = "SIM FAIL"
        elif lo_identical:
            lo_str = "IDENTICAL"
        else:
            lo_str = f"diff={np.linalg.norm(pos_lo - pos_base):.2e}"

        if pos_hi is None:
            hi_str = "SIM FAIL"
        elif hi_identical:
            hi_str = "IDENTICAL"
        else:
            hi_str = f"diff={np.linalg.norm(pos_hi - pos_base):.2e}"

        if lo_identical and hi_identical:
            verdict = "!! DEAD !!"
        elif lo_identical or hi_identical:
            verdict = "PARTIAL (one direction dead)"
        else:
            verdict = "ACTIVE"

        print(f"  {dim.name:<24} {base_val:>14.6g} {lo_val:>14.6g} {hi_val:>14.6g}  "
              f"{lo_str:>14} {hi_str:>14}  {verdict}")

    print()


if __name__ == "__main__":
    main()
