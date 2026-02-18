#!/usr/bin/env python3
"""Find the exact dead-zone boundary for solref_timeconst."""
import numpy as np
from config import space, sim_params_from_point, MJCF_PATHS
from simulation_fast import run_simulation

BASELINE = {
    "sliding_friction": 0.61545244,
    "torsional_friction": 0.00028306636,
    "rolling_friction": 0.087311067,
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

def run_pos(point):
    params = sim_params_from_point(point)
    params["drive_freq"] = 30.0
    traj = run_simulation(params, mjcf_path=MJCF_PATHS["scene4"],
                          sim_duration=3.0, visualize=False)
    if traj is None:
        return None
    return np.array(traj[-1]["pos"])

pos_base = run_pos(make_point())
print(f"Baseline solref_timeconst = {BASELINE['solref_timeconst']:.6e}")
print(f"SIM_TIMESTEP = 0.0005 (2 kHz)")
print()

# Sweep between 1e-3 and 5e-3 to find the boundary
for tc in [1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3, 4.5e-3, 5e-3]:
    pos = run_pos(make_point(solref_timeconst=tc))
    if pos is None:
        print(f"  tc={tc:.4f} → SIM FAIL")
    else:
        identical = np.array_equal(pos, pos_base)
        diff = np.linalg.norm(pos - pos_base)
        tag = "IDENTICAL" if identical else f"diff={diff:.6e}"
        print(f"  tc={tc:.4f} ({tc/0.0005:.1f}×dt) → {tag}")
