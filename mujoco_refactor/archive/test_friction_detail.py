#!/usr/bin/env python3
"""Detailed friction diagnostic: sweep each friction component independently
and also check the actual o_friction values that MuJoCo sees at contact time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np
from config import space, sim_params_from_point, MJCF_PATHS, SIM_TIMESTEP
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
    """Build a point from baseline with overrides."""
    params = dict(BASELINE)
    params.update(overrides)
    return [params[dim.name] for dim in space]


def run_vel(point):
    """Run sim, return velocity."""
    params = sim_params_from_point(point)
    params["drive_freq"] = FREQ
    traj = run_simulation(params, mjcf_path=MJCF_PATHS[SCENE],
                          sim_duration=DURATION, visualize=False)
    if traj is None or len(traj) == 0:
        return None
    return (traj[-1]["pos"][0] - traj[0]["pos"][0]) / (traj[-1]["time"] - traj[0]["time"])


# --- Test 1: Verify o_friction is set correctly ---
print("=" * 70)
print("TEST 1: Verify o_friction values reach MuJoCo")
print("=" * 70)

for torsional in [1e-6, 0.1, 1.0, 10.0]:
    for rolling in [1e-6, 0.1, 1.0, 10.0]:
        point = make_point(torsional_friction=torsional, rolling_friction=rolling)
        params = sim_params_from_point(point)
        model = mujoco.MjModel.from_xml_path(MJCF_PATHS[SCENE])
        gf = params['ground_friction']
        model.opt.o_friction[:] = [gf[0], gf[0], gf[1], gf[2], gf[2]]
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
        # Read back
        of = model.opt.o_friction
        print(f"  torsional={torsional:.1e}, rolling={rolling:.1e} "
              f"→ o_friction={[f'{x:.6g}' for x in of]}")

# --- Test 2: Check contact friction after step ---
print()
print("=" * 70)
print("TEST 2: Check actual contact friction after mj_step")
print("=" * 70)

for torsional in [1e-6, 1.0, 10.0]:
    point = make_point(torsional_friction=torsional)
    params = sim_params_from_point(point)
    model = mujoco.MjModel.from_xml_path(MJCF_PATHS[SCENE])
    gf = params['ground_friction']
    model.opt.o_friction[:] = [gf[0], gf[0], gf[1], gf[2], gf[2]]
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
    model.opt.timestep = SIM_TIMESTEP
    data = mujoco.MjData(model)
    # Initialize and step a few times to generate contacts
    data.qpos[2] = 0.002
    for _ in range(100):
        mujoco.mj_step(model, data)
    if data.ncon > 0:
        c = data.contact[0]
        print(f"  torsional={torsional:.1e} → contact.friction={c.friction}, "
              f"condim={c.dim}, ncon={data.ncon}")
    else:
        print(f"  torsional={torsional:.1e} → no contacts after 100 steps")

# --- Test 3: What is condim for these contacts? ---
print()
print("=" * 70)
print("TEST 3: Contact dimensionality analysis")
print("=" * 70)

point = make_point()
params = sim_params_from_point(point)
model = mujoco.MjModel.from_xml_path(MJCF_PATHS[SCENE])
gf = params['ground_friction']
model.opt.o_friction[:] = [gf[0], gf[0], gf[1], gf[2], gf[2]]
model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
model.opt.timestep = SIM_TIMESTEP
data = mujoco.MjData(model)
data.qpos[2] = 0.002
for _ in range(200):
    mujoco.mj_step(model, data)

print(f"  Number of contacts: {data.ncon}")
print(f"  Default condim from model: {model.opt.o_margin}")
seen_pairs = set()
for i in range(data.ncon):
    c = data.contact[i]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    pair = (g1, g2)
    if pair not in seen_pairs:
        seen_pairs.add(pair)
        print(f"  Contact: {g1} ↔ {g2}, condim={c.dim}, "
              f"friction={[f'{x:.4g}' for x in c.friction]}")

# --- Test 4: Focused torsional/rolling sweep ---
print()
print("=" * 70)
print("TEST 4: Velocity sweep — torsional and rolling friction")
print("=" * 70)

for name, values in [
    ("torsional_friction", [1e-6, 1e-3, 0.1, 1.0, 5.0, 10.0]),
    ("rolling_friction", [1e-6, 1e-3, 0.1, 1.0, 5.0, 10.0]),
]:
    print(f"\n  Sweeping {name}:")
    for val in values:
        point = make_point(**{name: val})
        v = run_vel(point)
        vs = f"{v:.6f}" if v is not None else "FAIL"
        print(f"    {name}={val:.1e} → velocity={vs}")
