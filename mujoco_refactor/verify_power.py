#!/usr/bin/env python3
"""Verify power computation by comparing 3 methods at individual timesteps.

Method A (OLD):   P = sum_j dot(tau_ext[j], cvel[j, :3])
Method B (NEW):   P = sum_j dot(tau_ext[j], axis_world[j]) * joint_vel[j]
Method C (JAC):   P = sum_j (jacr[:, dof_j].T @ tau_ext[j]) * joint_vel[j]

Method C uses MuJoCo's mj_jacBody to get the rotational Jacobian for each leg
body, then projects tau_ext through J^T to get the generalized joint torque.
For a hinge joint, jacr[:, dof_j] IS the joint axis in world frame — so B and C
should match exactly.

Usage:
    uv run python verify_power.py
"""

from __future__ import annotations

import csv
import pathlib
import sys

import mujoco
import numpy as np

REFACTOR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REFACTOR))

from config_new import (
    LEG_BODY_OFFSET,
    MJCF_PATHS,
    SETTLE_TIME,
    SIM_TIMESTEP,
    sim_params_from_point,
    space,
)
from simulation_fast_new import (
    _apply_magnetic_forces,
    _initialize_pose,
    _LEG_BODY_SLICE,
    _quat_rotate_vec_batch,
)

PARAM_NAMES = [dim.name for dim in space]
FLAT_DIR = REFACTOR / "results" / "20260225T122342_flat_10_30_50"

# Joint axis in body-local frame (all 4 hinge joints use axis="0 0 1" in MJCF)
_JOINT_AXIS_BODY = np.array([[0, 0, 1]] * 4, dtype=float)


def load_best_point(run_dir: pathlib.Path) -> list[float]:
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    best_id = rows[-1]["id"]
    with open(run_dir / "multi_optimization_results.csv") as f:
        for row in csv.DictReader(f):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    raise ValueError(f"id {best_id!r} not found")


def get_joint_torque_via_jacobian(model, data, body_id, tau_world):
    """Compute generalized joint torques from world-frame torque using mj_jacBody.

    Returns array of generalized forces for DOFs 6-9 (the 4 hinge joints).
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    # Generalized force = J_rot^T @ tau_world
    # For joint DOFs 6-9: jacr[:, dof].T @ tau_world
    qfrc = jacr.T @ tau_world  # (nv,) generalized forces on all DOFs
    return qfrc[6:10]  # just the 4 hinge joint DOFs


def main():
    point = load_best_point(FLAT_DIR)
    sp = sim_params_from_point(point)
    sp["drive_freq"] = 10.0

    scene = "scene1"
    model = mujoco.MjModel.from_xml_path(MJCF_PATHS[scene])

    # Apply params
    model.dof_damping[-4:] = sp["dof_damping"]
    model.opt.o_solref = sp["solref"]
    model.opt.o_solimp = sp["solimp"]
    gf = sp["ground_friction"]
    model.opt.o_friction[:] = [gf[0], gf[0], gf[1], gf[2], gf[2]]
    if "noslip_iterations" in sp:
        model.opt.noslip_iterations = int(sp["noslip_iterations"])
    if "noslip_tolerance" in sp:
        model.opt.noslip_tolerance = float(sp["noslip_tolerance"])
    if "margin" in sp:
        model.opt.o_margin = float(sp["margin"])
    model.opt.timestep = SIM_TIMESTEP
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
    model.geom_condim[:] = 6

    data = mujoco.MjData(model)
    rng = np.random.default_rng(77777)
    _initialize_pose(data, init_yaw_jitter_deg=2.0, rng=rng)

    # Confirm joint axes
    print("Joint axes from model.jnt_axis:")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        print(f"  {name}: axis={model.jnt_axis[j]}  body={model.jnt_bodyid[j]}  dof={model.jnt_dofadr[j]}")
    print()

    # Identify which body IDs correspond to legs
    leg_body_ids = list(range(LEG_BODY_OFFSET, LEG_BODY_OFFSET + 4))
    print(f"Leg body IDs: {leg_body_ids}")
    print(f"Joint DOF addresses: {[model.jnt_dofadr[j] for j in range(model.njnt)]}")
    print()

    mujoco.mj_step(model, data)

    sample_times = [0.2, 0.5, 1.0, 1.5, 2.0]
    sample_idx = 0

    while data.time < 2.5:
        step_cache = {}
        _apply_magnetic_forces(
            data, sp["kp_mag"], sp["drive_freq"], SETTLE_TIME,
            sp["mag_params"], step_cache,
        )

        tau_ext = step_cache["tau_ext"]
        omega_cvel = step_cache["omega"]
        joint_vel = data.qvel[6:10].copy()
        leg_xquat = data.xquat[_LEG_BODY_SLICE].copy()

        # Method B: manual projection
        axis_world = _quat_rotate_vec_batch(leg_xquat, _JOINT_AXIS_BODY)

        # Method C: MuJoCo Jacobian projection
        # For each leg body, compute J^T @ tau_ext to get generalized joint torques
        qfrc_jac = np.zeros(4)
        for j in range(4):
            body_id = leg_body_ids[j]
            qfrc_all = get_joint_torque_via_jacobian(model, data, body_id, tau_ext[j])
            qfrc_jac[j] = qfrc_all[j]  # This leg's own joint DOF

        # Also check: does jacr column match our axis_world?
        if sample_idx < len(sample_times) and data.time >= sample_times[sample_idx]:
            print(f"=== t={data.time:.3f}s ===")

            for j in range(4):
                body_id = leg_body_ids[j]
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

                # The rotational Jacobian column for this joint's DOF
                dof_idx = 6 + j
                jacr_col = jacr[:, dof_idx]

                # Compare with our axis_world
                print(f"  Leg {j} (body {body_id}):")
                print(f"    axis_world (quat rot): {axis_world[j]}")
                print(f"    jacr[:, dof={dof_idx}]:     {jacr_col}")
                print(f"    match: {np.allclose(axis_world[j], jacr_col, atol=1e-10)}")

                # Torque projections
                tau_proj_manual = np.dot(tau_ext[j], axis_world[j])
                tau_proj_jac = np.dot(tau_ext[j], jacr_col)
                print(f"    tau_proj (manual): {tau_proj_manual:.6e}")
                print(f"    tau_proj (jac):    {tau_proj_jac:.6e}")

            # Power comparison
            P_old = [np.dot(tau_ext[j], omega_cvel[j]) for j in range(4)]
            P_new = [np.dot(tau_ext[j], axis_world[j]) * joint_vel[j] for j in range(4)]
            P_jac = [qfrc_jac[j] * joint_vel[j] for j in range(4)]

            print(f"\n  Power comparison:")
            print(f"    OLD  (cvel):  total={sum(P_old):>+12.6e}  per-leg={[f'{p:+.4e}' for p in P_old]}")
            print(f"    NEW  (proj):  total={sum(P_new):>+12.6e}  per-leg={[f'{p:+.4e}' for p in P_new]}")
            print(f"    JAC  (truth): total={sum(P_jac):>+12.6e}  per-leg={[f'{p:+.4e}' for p in P_jac]}")
            print(f"    NEW==JAC: {np.allclose(P_new, P_jac, atol=1e-15)}")
            print(f"    OLD==JAC: {np.allclose(P_old, P_jac, atol=1e-15)}")
            print()
            sample_idx += 1

        mujoco.mj_step(model, data)

    # --- Cumulative energy comparison ---
    print("\n=== Cumulative energy (t > SETTLE_TIME, 3s sim) ===")
    data2 = mujoco.MjData(model)
    rng2 = np.random.default_rng(77777)
    _initialize_pose(data2, init_yaw_jitter_deg=2.0, rng=rng2)
    mujoco.mj_step(model, data2)

    E_old_abs = 0.0
    E_new_abs = 0.0
    E_jac_abs = 0.0
    E_old_signed = 0.0
    E_new_signed = 0.0
    E_jac_signed = 0.0
    n_steps = 0

    while data2.time < 3.0:
        step_cache2 = {}
        _apply_magnetic_forces(
            data2, sp["kp_mag"], sp["drive_freq"], SETTLE_TIME,
            sp["mag_params"], step_cache2,
        )
        tau2 = step_cache2["tau_ext"]
        ocvel2 = step_cache2["omega"]
        jvel2 = data2.qvel[6:10].copy()
        lxq2 = data2.xquat[_LEG_BODY_SLICE].copy()
        ax2 = _quat_rotate_vec_batch(lxq2, _JOINT_AXIS_BODY)

        # Jacobian-based
        qfrc2 = np.zeros(4)
        for j in range(4):
            qfrc2[j] = get_joint_torque_via_jacobian(model, data2, leg_body_ids[j], tau2[j])[j]

        P_old2 = sum(np.dot(tau2[j], ocvel2[j]) for j in range(4))
        P_new2 = sum(np.dot(tau2[j], ax2[j]) * jvel2[j] for j in range(4))
        P_jac2 = sum(qfrc2[j] * jvel2[j] for j in range(4))

        mujoco.mj_step(model, data2)

        if data2.time > SETTLE_TIME:
            dt = SIM_TIMESTEP
            E_old_abs += abs(P_old2) * dt
            E_new_abs += abs(P_new2) * dt
            E_jac_abs += abs(P_jac2) * dt
            E_old_signed += P_old2 * dt
            E_new_signed += P_new2 * dt
            E_jac_signed += P_jac2 * dt
            n_steps += 1

    print(f"Steps: {n_steps}")
    print(f"\n{'Method':<20} {'E_abs (uJ)':>12} {'E_signed (uJ)':>14}")
    print("-" * 50)
    print(f"{'OLD  (cvel)':<20} {E_old_abs * 1e6:>12.2f} {E_old_signed * 1e6:>+14.2f}")
    print(f"{'NEW  (proj)':<20} {E_new_abs * 1e6:>12.2f} {E_new_signed * 1e6:>+14.2f}")
    print(f"{'JAC  (truth)':<20} {E_jac_abs * 1e6:>12.2f} {E_jac_signed * 1e6:>+14.2f}")
    print()
    print(f"NEW vs JAC |E_abs| relative error:    {abs(E_new_abs - E_jac_abs) / E_jac_abs:.2e}")
    print(f"NEW vs JAC E_signed relative error:    {abs(E_new_signed - E_jac_signed) / abs(E_jac_signed):.2e}")
    print(f"OLD vs JAC |E_abs| relative error:    {abs(E_old_abs - E_jac_abs) / E_jac_abs:.2e}")
    print(f"OLD vs JAC E_signed relative error:    {abs(E_old_signed - E_jac_signed) / abs(E_jac_signed):.2e}")


if __name__ == "__main__":
    main()
