#!/usr/bin/env python3
"""Test condim=6: torsional with rolling held low, and finer rolling sweep."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np
from config import space, sim_params_from_point, MJCF_PATHS
from simulation_fast import run_simulation

SCENE = "scene4"
FREQ = 30.0
DURATION = 3.0

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


def run_vel_condim6(point):
    original_from_xml = mujoco.MjModel.from_xml_path
    def patched_from_xml(path):
        model = original_from_xml(path)
        for i in range(model.ngeom):
            model.geom_condim[i] = 6
        return model
    mujoco.MjModel.from_xml_path = patched_from_xml
    try:
        params = sim_params_from_point(point)
        params["drive_freq"] = FREQ
        traj = run_simulation(params, mjcf_path=MJCF_PATHS[SCENE],
                              sim_duration=DURATION, visualize=False)
        if traj is None or len(traj) == 0:
            return None
        return (traj[-1]["pos"][0] - traj[0]["pos"][0]) / (traj[-1]["time"] - traj[0]["time"])
    finally:
        mujoco.MjModel.from_xml_path = original_from_xml


# --- Torsional sweep with rolling held at 1e-6 ---
print("=" * 70)
print("Torsional sweep at condim=6 (rolling=1e-6 so it doesn't mask)")
print("=" * 70)
for tc in [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]:
    v = run_vel_condim6(make_point(torsional_friction=tc, rolling_friction=1e-6))
    vs = f"{v:.6f}" if v is not None else "FAIL"
    print(f"  torsional={tc:.1e} → velocity={vs}")

# --- Finer rolling sweep ---
print()
print("=" * 70)
print("Fine rolling sweep at condim=6 (torsional=1e-6)")
print("=" * 70)
for rf in [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]:
    v = run_vel_condim6(make_point(rolling_friction=rf, torsional_friction=1e-6))
    vs = f"{v:.6f}" if v is not None else "FAIL"
    print(f"  rolling={rf:.1e} → velocity={vs}")
