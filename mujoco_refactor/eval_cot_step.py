#!/usr/bin/env python3
"""Evaluate Cost of Transport (COT) on step terrain — corrected power formula.

Adapted from eval_cot_v2.py for step terrain. Key differences:
  - Uses config_step.py references (12 step refs, incl. failure modes)
  - Injects step geometry into MJCFs (same as optimizer_step.py)
  - Energy/power measured in step region only (position-gated, not time-gated)
  - Reports gravitational PE alongside COT
  - Skips failure-mode refs (target=0) for COT (distance ≈ 0)

Power formula (same corrected v2):
  P = sum_j (tau_ext[j] · axis_world[j]) * joint_vel[j]

Usage:
    uv run python eval_cot_step.py results/20260225T225248_step_argmin_progress
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from config_step import (
    MJCF_PATHS,
    REFERENCE_DATA,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    STEP_END_X,
    STEP_PRESET,
    STEP_START_X,
    sim_params_from_point,
    space,
)
import simulation_fast_new as sim_module

PARAM_NAMES = [dim.name for dim in space]


# ---------------------------------------------------------------------------
# Step XML injection (copied from optimizer_step.py — not modified)
# ---------------------------------------------------------------------------

def _inject_steps(xml_path: str, preset: dict, out_xml: str) -> str:
    """Add step box geoms to the MJCF and write to out_xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    flat_lead = preset["flat_lead"]
    step_height = preset["step_height"]
    step_length = preset["step_length"]
    step_count = preset["step_count"]
    final_step_length = preset["final_step_length"]
    step_width = preset["step_width"]

    for i in range(step_count):
        is_final = (i == step_count - 1)
        length = final_step_length if is_final else step_length

        if is_final:
            pos_x = flat_lead + (step_count - 1) * step_length + length / 2.0
        else:
            pos_x = flat_lead + i * step_length + length / 2.0
        pos_z = (i + 1) * step_height - step_height / 2.0

        geom = ET.SubElement(worldbody, "geom")
        geom.set("name", f"step_{i}")
        geom.set("type", "box")
        geom.set("size", f"{length/2.0} {step_width/2.0} {step_height/2.0}")
        geom.set("pos", f"{pos_x} 0.0 {pos_z}")
        geom.set("rgba", "0.5 0.5 0.5 1")

    tree.write(out_xml)
    return out_xml


# ---------------------------------------------------------------------------
# Joint axis helper (copied from eval_cot_v2.py — not modified)
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
# COT computation — step region only (position-gated)
# ---------------------------------------------------------------------------

def compute_step_locomotion_metrics(
    trajectory: list[dict],
    robot_mass: float,
    step_start_x: float,
    step_end_x: float,
    g: float = 9.81,
) -> dict[str, float] | None:
    """Compute energy, power, COT, and gravitational PE in the step region.

    Position-gated: only counts data where robot x >= step_start_x and
    x <= 90% of step region (same window as optimizer_step.calculate_cost).

    Power: P = sum_j (tau_ext[j] · axis_world[j]) * joint_vel[j]
    Energy: E_ext = integral(P_ext) dt   (signed)
    COT = E_ext / (m g d_horizontal)
    Grav_PE = m * g * delta_h
    """
    # Find enter state: first state where x >= step_start_x
    enter_idx = None
    for i, state in enumerate(trajectory):
        if state["pos"][0] >= step_start_x:
            enter_idx = i
            break

    if enter_idx is None:
        return None

    # Find exit state: last state within 90% of step region
    active_cutoff = step_start_x + 0.9 * (step_end_x - step_start_x)
    exit_idx = enter_idx
    for i in range(enter_idx, len(trajectory)):
        if trajectory[i]["pos"][0] > active_cutoff:
            break
        exit_idx = i

    active = trajectory[enter_idx:exit_idx + 1]
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

    # Signed energy
    energy_ext_signed = np.sum(power_ext * dt)
    # Absolute energy
    energy_ext_abs = np.sum(np.abs(power_ext) * dt)

    avg_power_ext = energy_ext_signed / total_time if total_time > 1e-12 else 0.0
    mgd = robot_mass * g * total_distance
    cot_signed = energy_ext_signed / mgd if mgd > 1e-12 else float("inf")
    cot_abs = energy_ext_abs / mgd if mgd > 1e-12 else float("inf")

    # Gravitational PE: m * g * delta_h
    delta_h = active[-1]["pos"][2] - active[0]["pos"][2]
    grav_pe = robot_mass * g * delta_h

    # Step velocity (same as optimizer_step.calculate_cost)
    step_velocity = 0.0
    if total_time > 1e-6:
        forward_displacement = active[-1]["pos"][0] - active[0]["pos"][0]
        step_velocity = forward_displacement / total_time

    return {
        "total_time": total_time,
        "total_distance": total_distance,
        "energy_ext": energy_ext_signed,
        "energy_ext_abs": energy_ext_abs,
        "avg_power_ext": avg_power_ext,
        "cot": cot_signed,
        "cot_abs": cot_abs,
        "robot_mass": robot_mass,
        "delta_h": delta_h,
        "grav_pe": grav_pe,
        "grav_pe_fraction": grav_pe / energy_ext_signed if abs(energy_ext_signed) > 1e-15 else 0.0,
        "step_velocity": step_velocity,
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


def extract_step_velocity(traj: list[dict]) -> float:
    """Average forward velocity in the step region (position-gated)."""
    enter_state = None
    for s in traj:
        if s["pos"][0] >= STEP_START_X:
            enter_state = s
            break
    if enter_state is None:
        return 0.0

    active_cutoff = STEP_START_X + 0.9 * (STEP_END_X - STEP_START_X)
    exit_state = enter_state
    for s in traj:
        if s["pos"][0] > active_cutoff:
            break
        exit_state = s

    dt = exit_state["time"] - enter_state["time"]
    if dt < 1e-6:
        return 0.0
    return (exit_state["pos"][0] - enter_state["pos"][0]) / dt


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

    # Build step terrain XMLs
    h_mm = STEP_PRESET["step_height"] * 1000
    l_mm = STEP_PRESET["step_length"] * 1000
    n_steps = STEP_PRESET["step_count"]
    lead_mm = STEP_PRESET["flat_lead"] * 1000
    step_tag = f"step_{n_steps}x{h_mm:.0f}mm_{l_mm:.1f}L_{lead_mm:.0f}lead"

    mjcf_step_paths: dict[str, str] = {}
    print(f"Building step terrain XMLs ({step_tag}) ...")
    for scene, base_xml in MJCF_PATHS.items():
        src_dir = pathlib.Path(base_xml).parent
        stem = pathlib.Path(base_xml).stem
        out_xml = str(src_dir / f"{stem}_{step_tag}.xml")
        _inject_steps(base_xml, STEP_PRESET, out_xml)
        mjcf_step_paths[scene] = out_xml
        print(f"  {scene}: {out_xml}")

    # Build references (skip failure modes for COT)
    refs: list[dict] = []
    for row in REFERENCE_DATA:
        s = row["scene"]
        f = float(row.get("ctrl_freq", 30.0))
        if args.scenes and s not in args.scenes:
            continue
        if args.freqs and f not in args.freqs:
            continue
        target = float(row["speed"])
        is_failure = target < 1e-6
        refs.append({
            "scene": s,
            "freq": f,
            "target": target,
            "speed_std": float(row.get("speed_std", 0.0)),
            "id": f"{s}_f{int(f)}",
            "is_failure": is_failure,
        })

    robot_masses: dict[str, float] = {}
    for scene, xml_path in mjcf_step_paths.items():
        model = mujoco.MjModel.from_xml_path(xml_path)
        robot_masses[scene] = float(sum(model.body_mass))

    n_active = sum(1 for r in refs if not r["is_failure"])
    n_failure = sum(1 for r in refs if r["is_failure"])
    print(f"\nParams from: {run_dir}")
    print(f"  {len(refs)} step references ({n_active} active, {n_failure} failure modes skipped for COT)")
    print(f"  {args.n_trials} trials each, top-{args.top_k} selection")
    print(f"  Jitter: +/-{args.jitter_deg} deg, base seed: {BASE_SEED}")
    print(f"  Sim: {SIM_DURATION}s, step region: x=[{STEP_START_X*1000:.0f}, {STEP_END_X*1000:.0f}]mm")
    print(f"  Power: P = sum(tau_ext·axis_world * joint_vel)  [joint-frame projection]")
    print(f"  Energy/COT measured in step region only (position-gated)")
    print()

    csv_rows: list[dict] = []
    summaries: list[dict] = []

    for ri, ref in enumerate(refs):
        sp = dict(sim_params)
        sp["drive_freq"] = ref["freq"]
        mass = robot_masses[ref["scene"]]

        if ref["is_failure"]:
            print(f"{ref['id']} (FAILURE MODE — target=0, skipping COT)")
            summaries.append({
                "id": ref["id"], "scene": ref["scene"], "freq": ref["freq"],
                "target": 0.0, "mean_vel": 0.0, "mean_cot": float("nan"),
                "std_cot": 0.0, "mean_cot_abs": float("nan"),
                "mean_distance": 0.0, "mean_energy": 0.0,
                "mean_delta_h": 0.0, "mean_grav_pe": 0.0,
                "n_selected": 0, "mass": mass, "is_failure": True,
            })
            print()
            continue

        print(f"{ref['id']} (target={ref['target'] * 1000:.1f} mm/s, mass={mass * 1e6:.1f} mg, {args.n_trials} trials)")

        trials = []
        for t in range(args.n_trials):
            seed = BASE_SEED + ri * 100 + t
            traj = sim_module.run_simulation(
                sp, mjcf_path=mjcf_step_paths[ref["scene"]],
                sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
                init_yaw_jitter_deg=args.jitter_deg, rng_seed=seed,
            )
            if traj is None:
                print(f"  trial {t}: FAIL")
                trials.append({"fail": True, "seed": seed, "trial_idx": t})
                continue

            vel = extract_step_velocity(traj)
            delta = abs(vel - ref["target"])
            metrics = compute_step_locomotion_metrics(
                traj, mass, STEP_START_X, STEP_END_X,
            )

            if metrics is None:
                print(f"  trial {t}: vel={vel * 1000:>7.1f} mm/s  NO METRICS")
                trials.append({"fail": True, "seed": seed, "trial_idx": t})
                continue

            cot = metrics["cot"]
            print(f"  trial {t}: vel={vel * 1000:>7.1f} mm/s  delta={delta * 1000:>5.1f}  "
                  f"COT={cot:.1f}  COT_abs={metrics['cot_abs']:.1f}  "
                  f"E_signed={metrics['energy_ext'] * 1e6:.2f}uJ  "
                  f"Δh={metrics['delta_h'] * 1e3:.2f}mm  "
                  f"PE_grav={metrics['grav_pe'] * 1e6:.2f}uJ ({metrics['grav_pe_fraction'] * 100:.1f}%)")
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
                "delta_h": metrics["delta_h"],
                "grav_pe": metrics["grav_pe"],
                "grav_pe_fraction": metrics["grav_pe_fraction"],
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
            mean_delta_h = np.mean([x["delta_h"] for x in selected])
            mean_grav_pe = np.mean([x["grav_pe"] for x in selected])
            print(f"  -> Top {len(selected)}: trials {sel_indices}  "
                  f"mean_vel={mean_vel * 1000:.1f} mm/s  "
                  f"COT_signed={mean_cot:.1f} +/- {std_cot:.1f}  COT_abs={mean_cot_abs:.1f}  "
                  f"Δh={mean_delta_h * 1e3:.2f}mm  PE_grav={mean_grav_pe * 1e6:.2f}uJ")
        else:
            mean_vel = mean_cot = std_cot = mean_cot_abs = 0.0
            mean_dist = mean_energy = mean_delta_h = mean_grav_pe = 0.0
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
            "mean_delta_h": mean_delta_h,
            "mean_grav_pe": mean_grav_pe,
            "n_selected": len(selected),
            "mass": mass,
            "is_failure": False,
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
                "delta_h_m": x["delta_h"],
                "grav_pe_J": x["grav_pe"],
                "grav_pe_fraction": x["grav_pe_fraction"],
            })
        print()

    # --- Summary table ---
    print("=" * 130)
    print(f"  {'ref':<20} {'target':>8} {'mean_vel':>9} {'COT_sign':>9} {'COT_abs':>8} "
          f"{'std_COT':>8} {'dist(mm)':>9} {'E_sign(uJ)':>11} {'Δh(mm)':>7} {'PE_g(uJ)':>9}")
    print("  " + "-" * 125)
    for s in summaries:
        if s.get("is_failure"):
            print(f"  {s['id']:<20} {s['target'] * 1000:>7.1f} {'SKIP':>9}  (failure mode)")
        elif s["n_selected"] == 0:
            print(f"  {s['id']:<20} {s['target'] * 1000:>7.1f} {'FAIL':>9}")
        else:
            print(f"  {s['id']:<20} {s['target'] * 1000:>7.1f} {s['mean_vel'] * 1000:>8.1f} "
                  f"{s['mean_cot']:>9.1f} {s['mean_cot_abs']:>7.1f} {s['std_cot']:>7.1f} "
                  f"{s['mean_distance'] * 1e3:>8.1f} {s['mean_energy'] * 1e6:>10.2f} "
                  f"{s['mean_delta_h'] * 1e3:>6.2f} {s['mean_grav_pe'] * 1e6:>8.2f}")
    print()

    # --- Save CSV ---
    csv_path = run_dir / "cot_step_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scene", "freq", "trial", "seed", "velocity_mps", "target_mps",
            "cot_signed", "cot_abs", "energy_ext_signed_J", "energy_ext_abs_J",
            "distance_m", "robot_mass_kg", "total_time_s", "avg_power_ext_W",
            "delta_h_m", "grav_pe_J", "grav_pe_fraction",
        ])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
