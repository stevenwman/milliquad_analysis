#!/usr/bin/env python3
"""Evaluate Cost of Transport (COT) on flat terrain — corrected power formula.

Same as eval_cot.py but fixes the power computation:
  OLD (wrong): P = sum_legs(tau_ext[j] . omega_body[j])
    omega_body = data.cvel[leg, :3] includes floating base rotation → spurious terms
  NEW (correct): P = sum_legs(tau_ext[j] . axis_world[j]) * joint_vel[j]
    Projects world-frame torque onto joint axis, multiplies by scalar joint velocity

All 4 hinge joints have axis=[0,0,1] in body frame (from MJCF).

Usage:
    uv run python eval_cot_v2.py results/20260225T122342_flat_10_30_50
    uv run python eval_cot_v2.py results/20260225T225248_step_argmin_progress
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import mujoco
import numpy as np

from config_new import (
    MJCF_PATHS,
    REFERENCE_DATA,
    SETTLE_TIME,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    sim_params_from_point,
    space,
)
import simulation_fast_new as sim_module

PARAM_NAMES = [dim.name for dim in space]


# ---------------------------------------------------------------------------
# Joint axis helper
# ---------------------------------------------------------------------------

def _joint_axis_world(leg_xquat: np.ndarray) -> np.ndarray:
    """Compute joint axis in world frame from leg body quaternions.

    All 4 joints have body-local axis [0, 0, 1].  Rotating [0,0,1] by
    quaternion (w,x,y,z) gives the 3rd column of the rotation matrix:
        [2(xz + wy), 2(yz - wx), 1 - 2(x² + y²)]

    Args:
        leg_xquat: (4, 4) array of (w, x, y, z) quaternions.

    Returns:
        (4, 3) array of joint axes in world frame.
    """
    w = leg_xquat[:, 0]
    x = leg_xquat[:, 1]
    y = leg_xquat[:, 2]
    z = leg_xquat[:, 3]
    return np.column_stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ])


# ---------------------------------------------------------------------------
# COT computation — corrected power formula
# ---------------------------------------------------------------------------

def compute_locomotion_metrics(
    trajectory: list[dict],
    robot_mass: float,
    g: float = 9.81,
) -> dict[str, float] | None:
    """Compute energy, average power, and cost of transport from a trajectory.

    Power: P = sum_j (tau_ext[j] · axis_world[j]) * joint_vel[j]
    Energy: E_ext = integral(P_ext) dt   (signed — drive torque does positive work)
    COT = E_ext / (m g d)
    """
    start_idx = 0
    for i, state in enumerate(trajectory):
        if state["time"] >= SETTLE_TIME:
            start_idx = i
            break

    active = trajectory[start_idx:]
    if len(active) < 2:
        return None

    if "tau_ext" not in active[0] or "leg_xquat" not in active[0] or "joint_vel" not in active[0]:
        return None

    n = len(active) - 1
    dt = np.empty(n)
    power_ext = np.empty(n)
    dist_increments = np.empty(n)

    for i in range(n):
        s = active[i]
        dt[i] = active[i + 1]["time"] - s["time"]

        tau_ext = s["tau_ext"]          # (4, 3) world frame
        axis = _joint_axis_world(s["leg_xquat"])  # (4, 3) world frame
        jvel = s["joint_vel"]           # (4,) scalar rad/s

        # P = sum_j (tau_ext[j] · axis[j]) * jvel[j]
        power_ext[i] = sum(np.dot(tau_ext[j], axis[j]) * jvel[j] for j in range(4))

        p1 = active[i]["pos"][:2]
        p2 = active[i + 1]["pos"][:2]
        dist_increments[i] = np.linalg.norm(p2 - p1)

    total_time = active[-1]["time"] - active[0]["time"]
    total_distance = dist_increments.sum()

    # Signed energy: drive field does net positive work on joints
    energy_ext_signed = np.sum(power_ext * dt)
    # Absolute energy (for comparison with v1)
    energy_ext_abs = np.sum(np.abs(power_ext) * dt)

    avg_power_ext = energy_ext_signed / total_time
    mgd = robot_mass * g * total_distance
    cot_signed = energy_ext_signed / mgd if mgd > 1e-12 else float("inf")
    cot_abs = energy_ext_abs / mgd if mgd > 1e-12 else float("inf")

    return {
        "total_time": total_time,
        "total_distance": total_distance,
        "energy_ext": energy_ext_signed,
        "energy_ext_abs": energy_ext_abs,
        "avg_power_ext": avg_power_ext,
        "cot": cot_signed,
        "cot_abs": cot_abs,
        "robot_mass": robot_mass,
    }


JITTER_DEG = 2.0
BASE_SEED = 77777
N_TRIALS = 10
TOP_K = 3

from morphology_style import MORPH_LABELS as SCENE_LABELS


def load_best_point(run_dir: pathlib.Path) -> list[float]:
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    if not rows:
        sys.exit(f"ERROR: no rows in {bests_csv}")
    best_id = rows[-1]["id"]
    with open(run_dir / "multi_optimization_results.csv") as f:
        for row in csv.DictReader(f):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    raise ValueError(f"id {best_id!r} not found")


def extract_flat_velocity(traj: list[dict]) -> float:
    """Average forward velocity after settle time."""
    start = None
    for s in traj:
        if s["time"] >= SETTLE_TIME:
            start = s
            break
    if start is None:
        return 0.0
    end = traj[-1]
    dt = end["time"] - start["time"]
    if dt < 1e-6:
        return 0.0
    return (end["pos"][0] - start["pos"][0]) / dt


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=str, help="Results directory with optimization CSVs")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--freqs", nargs="+", type=float, default=None)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--jitter-deg", type=float, default=JITTER_DEG)
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"ERROR: {run_dir} is not a directory")

    point = load_best_point(run_dir)
    sim_params = sim_params_from_point(point)

    refs: list[dict] = []
    for row in REFERENCE_DATA:
        s = row["scene"]
        f = float(row.get("ctrl_freq", 30.0))
        if args.scenes and s not in args.scenes:
            continue
        if args.freqs and f not in args.freqs:
            continue
        refs.append({
            "scene": s,
            "freq": f,
            "target": float(row["speed"]),
            "speed_std": float(row.get("speed_std", 0.0)),
            "id": f"{s}_f{int(f)}",
        })

    robot_masses: dict[str, float] = {}
    for scene, xml_path in MJCF_PATHS.items():
        model = mujoco.MjModel.from_xml_path(xml_path)
        robot_masses[scene] = float(sum(model.body_mass))

    print(f"Params from: {run_dir}")
    print(f"  {len(refs)} flat references, {args.n_trials} trials each, top-{args.top_k} selection")
    print(f"  Jitter: +/-{args.jitter_deg} deg, base seed: {BASE_SEED}")
    print(f"  Sim: {SIM_DURATION}s, settle: {SETTLE_TIME}s")
    print(f"  Power: P = sum(tau_ext·axis_world * joint_vel)  [joint-frame projection]")
    print()

    csv_rows: list[dict] = []
    summaries: list[dict] = []

    for ri, ref in enumerate(refs):
        sp = dict(sim_params)
        sp["drive_freq"] = ref["freq"]
        mass = robot_masses[ref["scene"]]

        print(f"{ref['id']} (target={ref['target'] * 1000:.1f} mm/s, mass={mass * 1e6:.1f} mg, {args.n_trials} trials)")

        trials = []
        for t in range(args.n_trials):
            seed = BASE_SEED + ri * 100 + t
            traj = sim_module.run_simulation(
                sp, mjcf_path=MJCF_PATHS[ref["scene"]],
                sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
                init_yaw_jitter_deg=args.jitter_deg, rng_seed=seed,
            )
            if traj is None:
                print(f"  trial {t}: FAIL")
                trials.append({"fail": True, "seed": seed, "trial_idx": t})
                continue

            vel = extract_flat_velocity(traj)
            delta = abs(vel - ref["target"])
            metrics = compute_locomotion_metrics(traj, mass)

            if metrics is None:
                print(f"  trial {t}: vel={vel * 1000:>7.1f} mm/s  NO METRICS")
                trials.append({"fail": True, "seed": seed, "trial_idx": t})
                continue

            cot = metrics["cot"]
            print(f"  trial {t}: vel={vel * 1000:>7.1f} mm/s  delta={delta * 1000:>5.1f}  "
                  f"COT={cot:.1f}  COT_abs={metrics['cot_abs']:.1f}  "
                  f"E_signed={metrics['energy_ext'] * 1e6:.2f}uJ  E_abs={metrics['energy_ext_abs'] * 1e6:.2f}uJ")
            trials.append({
                "fail": False,
                "seed": seed,
                "trial_idx": t,
                "vel": vel,
                "delta": delta,
                "cot": cot,
                "cot_abs": metrics["cot_abs"],
                "energy_ext": metrics["energy_ext"],
                "energy_ext_abs": metrics["energy_ext_abs"],
                "distance": metrics["total_distance"],
                "total_time": metrics["total_time"],
                "avg_power_ext": metrics["avg_power_ext"],
            })

        # Select top-K by smallest delta
        valid = [x for x in trials if not x["fail"]]
        if len(valid) < args.top_k:
            print(f"  -> Only {len(valid)} valid trials (need {args.top_k})")
            selected = valid
        else:
            valid_sorted = sorted(valid, key=lambda x: x["delta"])
            selected = valid_sorted[:args.top_k]

        if selected:
            sel_indices = [x["trial_idx"] for x in selected]
            mean_vel = np.mean([x["vel"] for x in selected])
            mean_cot = np.mean([x["cot"] for x in selected])
            std_cot = np.std([x["cot"] for x in selected], ddof=1) if len(selected) > 1 else 0.0
            mean_cot_abs = np.mean([x["cot_abs"] for x in selected])
            mean_dist = np.mean([x["distance"] for x in selected])
            mean_energy = np.mean([x["energy_ext"] for x in selected])
            print(f"  -> Top {len(selected)}: trials {sel_indices}  "
                  f"mean_vel={mean_vel * 1000:.1f} mm/s  "
                  f"COT_signed={mean_cot:.1f} +/- {std_cot:.1f}  COT_abs={mean_cot_abs:.1f}")
        else:
            mean_vel = mean_cot = std_cot = mean_cot_abs = mean_dist = mean_energy = 0.0
            sel_indices = []
            print(f"  -> ALL FAILED")

        summaries.append({
            "id": ref["id"],
            "scene": ref["scene"],
            "freq": ref["freq"],
            "target": ref["target"],
            "mean_vel": mean_vel,
            "mean_cot": mean_cot,
            "std_cot": std_cot,
            "mean_cot_abs": mean_cot_abs,
            "mean_distance": mean_dist,
            "mean_energy": mean_energy,
            "n_selected": len(selected),
            "mass": mass,
        })

        for x in selected:
            csv_rows.append({
                "scene": ref["scene"],
                "freq": ref["freq"],
                "trial": x["trial_idx"],
                "seed": x["seed"],
                "velocity_mps": x["vel"],
                "target_mps": ref["target"],
                "cot_signed": x["cot"],
                "cot_abs": x["cot_abs"],
                "energy_ext_signed_J": x["energy_ext"],
                "energy_ext_abs_J": x["energy_ext_abs"],
                "distance_m": x["distance"],
                "robot_mass_kg": mass,
                "total_time_s": x["total_time"],
                "avg_power_ext_W": x["avg_power_ext"],
            })
        print()

    # --- Summary table ---
    print("=" * 110)
    print(f"  {'ref':<20} {'target':>8} {'mean_vel':>9} {'COT_sign':>9} {'COT_abs':>8} {'std_COT':>8} {'dist(mm)':>9} {'E_sign(uJ)':>11}")
    print("  " + "-" * 105)
    for s in summaries:
        if s["n_selected"] == 0:
            print(f"  {s['id']:<20} {s['target'] * 1000:>7.1f} {'FAIL':>9}")
        else:
            print(f"  {s['id']:<20} {s['target'] * 1000:>7.1f} {s['mean_vel'] * 1000:>8.1f} "
                  f"{s['mean_cot']:>9.1f} {s['mean_cot_abs']:>7.1f} {s['std_cot']:>7.1f} "
                  f"{s['mean_distance'] * 1e3:>8.1f} {s['mean_energy'] * 1e6:>10.2f}")
    print()

    # --- Save CSV ---
    csv_path = run_dir / "cot_results_v2.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scene", "freq", "trial", "seed", "velocity_mps", "target_mps",
            "cot_signed", "cot_abs", "energy_ext_signed_J", "energy_ext_abs_J",
            "distance_m", "robot_mass_kg", "total_time_s", "avg_power_ext_W",
        ])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
