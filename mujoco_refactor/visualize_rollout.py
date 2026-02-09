"""
Visualize or record a rollout from multi_optimization_results.csv.

Usage:
    uv run python visualize_rollout.py              # best result, interactive
    uv run python visualize_rollout.py --rank 3     # 3rd best
    uv run python visualize_rollout.py --record out.mp4
    uv run python visualize_rollout.py --scene scene2
"""

import argparse
import csv

import mujoco
import numpy as np

from config import (
    CSV_PATH,
    MAGNETIC_FIELD_MAGNITUDE,
    MAGNETIC_MOMENT,
    MJCF_PATHS,
    SETTLE_TIME,
)
import simulation


def _sim_params_from_csv_row(row: dict[str, str]) -> dict:
    """Reconstruct sim_params from a CSV row.

    Handles both current format (with fudges) and older formats.
    """
    sim_params = {
        "ground_friction": [
            float(row["sliding_friction"]),
            float(row["torsional_friction"]),
            float(row["rolling_friction"]),
        ],
        "solref": [
            float(row["solref_timeconst"]),
            float(row["solref_dampratio"]),
        ],
        "solimp": [
            float(row["solimp_dmin"]),
            float(row["solimp_dmax"]),
            float(row["solimp_width"]),
            float(row.get("solimp_midpoint", 0.5)),
            float(row.get("solimp_power", 1.0)),
        ],
        "dof_damping": float(row.get("dof_damping", 7e-10)),
    }

    if "magnetic_moment_fudge" in row and "magnetic_field_fudge" in row:
        m_mag = MAGNETIC_MOMENT * float(row["magnetic_moment_fudge"])
        kp_mag = m_mag * MAGNETIC_FIELD_MAGNITUDE * float(row["magnetic_field_fudge"])
    elif "kp_mag" in row:
        kp_mag = float(row["kp_mag"])
        m_mag = kp_mag / MAGNETIC_FIELD_MAGNITUDE
    else:
        m_mag = MAGNETIC_MOMENT
        kp_mag = m_mag * MAGNETIC_FIELD_MAGNITUDE

    sim_params["kp_mag"] = kp_mag
    sim_params["mag_params"] = {"m_mag": m_mag}
    return sim_params


# ---------------------------------------------------------------------------
# Cost of Transport analysis
# ---------------------------------------------------------------------------

def compute_locomotion_metrics(
    trajectory: list[dict],
    robot_mass: float,
    g: float = 9.81,
) -> dict[str, float] | None:
    """
    Compute energy, average power, and cost of transport from a trajectory.

    Power at each step: P = Σ_legs τ · ω  (dot product of torque and angular velocity).
    Total distance is cumulative 2D path length (not straight-line).
    COT = E / (m g d).

    Args:
        trajectory: list of state dicts with tau_ext, tau_int, omega keys.
        robot_mass: total robot mass in kg.
        g: gravitational acceleration (m/s²).

    Returns:
        dict with metrics, or None if trajectory lacks torque/omega data.
    """
    # Find start of active locomotion (after settle)
    start_idx = 0
    for i, state in enumerate(trajectory):
        if state["time"] >= SETTLE_TIME:
            start_idx = i
            break

    active = trajectory[start_idx:]
    if len(active) < 2:
        return None

    # Check that we have the needed fields
    if "tau_ext" not in active[0] or "omega" not in active[0]:
        print("Warning: Trajectory missing tau_ext/omega — was it run without step_cache?")
        return None

    n = len(active) - 1
    dt = np.empty(n)
    power_ext = np.empty(n)
    power_int = np.empty(n)
    dist_increments = np.empty(n)

    for i in range(n):
        s = active[i]
        dt[i] = active[i + 1]["time"] - s["time"]

        tau_ext = s["tau_ext"]
        tau_int = s["tau_int"]
        omega = s["omega"]

        # P = Σ_legs τ · ω
        power_ext[i] = sum(np.dot(tau_ext[j], omega[j]) for j in range(4))
        power_int[i] = sum(np.dot(tau_int[j], omega[j]) for j in range(4))

        # 2D distance increment (x, y plane)
        p1 = active[i]["pos"][:2]
        p2 = active[i + 1]["pos"][:2]
        dist_increments[i] = np.linalg.norm(p2 - p1)

    total_time = active[-1]["time"] - active[0]["time"]
    total_distance = dist_increments.sum()

    # Energy from external drive = ∫|P_ext| dt
    # Uses absolute value: both driving and braking phases cost energy to maintain the field.
    # Inter-joint coupling is conservative (internal) and excluded from COT.
    energy_ext = np.sum(np.abs(power_ext) * dt)

    # Average power (signed — net energy flow from drive field)
    avg_power_ext = np.sum(power_ext * dt) / total_time

    # COT = E_ext / (m g d)
    mgd = robot_mass * g * total_distance
    cot = energy_ext / mgd if mgd > 1e-12 else float("inf")

    return {
        "total_time": total_time,
        "total_distance": total_distance,
        "energy_ext": energy_ext,
        "avg_power_ext": avg_power_ext,
        "cot": cot,
        "robot_mass": robot_mass,
    }


def print_locomotion_metrics(metrics: dict[str, float] | None) -> None:
    """Pretty-print the output of compute_locomotion_metrics."""
    if metrics is None:
        print("\n  Could not compute locomotion metrics.")
        return

    print("\n--- Locomotion Metrics ---")
    print(f"  Duration (active):   {metrics['total_time']:.2f} s")
    print(f"  Distance (2D path):  {metrics['total_distance'] * 1e3:.2f} mm")
    print(f"  Robot mass:          {metrics['robot_mass'] * 1e6:.2f} mg")
    print(f"  Avg power (ext):     {metrics['avg_power_ext']:.4e} W")
    print(f"  Energy (ext drive):  {metrics['energy_ext']:.4e} J")
    print(f"  COT:                 {metrics['cot']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize or record a rollout from optimization results."
    )
    parser.add_argument(
        "--rank", type=int, default=1,
        help="Rank of the result to visualize (1 = best).",
    )
    parser.add_argument(
        "--record", type=str, default=None,
        help="Path to save a video recording, e.g. 'rollout.mp4'.",
    )
    parser.add_argument(
        "--drive_freq", type=float, default=30.0,
        help="Drive frequency for the simulation (Hz).",
    )
    parser.add_argument(
        "--scene", type=str, default="scene4", choices=list(MJCF_PATHS.keys()),
        help="Which scene to visualize.",
    )
    parser.add_argument(
        "--csv", type=str, default=CSV_PATH,
        help=f"Path to results CSV (default: {CSV_PATH}).",
    )
    args = parser.parse_args()

    # Read CSV
    try:
        with open(args.csv, "r") as f:
            results = sorted(csv.DictReader(f), key=lambda r: float(r["cost"]))
    except FileNotFoundError:
        print(f"Error: {args.csv} not found. Run optimizer.py first.")
        return

    if not results or args.rank > len(results):
        print(f"Error: Rank {args.rank} is out of bounds ({len(results)} results).")
        return

    selected = results[args.rank - 1]

    print(f"--- Visualizing Rank #{args.rank} ---")
    print(f"  ID: {selected['id']}")
    print(f"  Cost: {float(selected['cost']):.6f}")
    for scene in MJCF_PATHS:
        vel_key = f"velocity_{scene}"
        if vel_key in selected:
            print(f"  Avg Velocity ({scene}): {float(selected[vel_key]):.4f} m/s")

    # Reconstruct sim params
    try:
        sim_params = _sim_params_from_csv_row(selected)
    except (KeyError, ValueError) as e:
        print(f"Error parsing CSV row: {e}")
        return

    sim_params["drive_freq"] = args.drive_freq
    if args.drive_freq != 30.0:
        print(f"  Using manual drive frequency: {args.drive_freq} Hz")

    mjcf_path = MJCF_PATHS[args.scene]

    # Get robot mass from model
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    robot_mass = sum(model.body_mass)

    if args.record:
        print(f"\nRecording rollout to {args.record}...")
        traj = simulation.run_simulation(
            sim_params,
            mjcf_path=mjcf_path,
            sim_duration=10.0,
            record_path=args.record,
            ignore_stuck_detection=True,
        )
    else:
        # Headless run for metrics, then interactive for viewing
        print("\nRunning headless sim for COT analysis...")
        traj = simulation.run_simulation(
            sim_params,
            mjcf_path=mjcf_path,
            sim_duration=10.0,
            visualize=False,
            ignore_stuck_detection=True,
            progress=True,
        )

    # Compute and print locomotion metrics
    if traj:
        metrics = compute_locomotion_metrics(traj, robot_mass)
        print_locomotion_metrics(metrics)
    else:
        print("\nSimulation failed — no trajectory to analyze.")

    # Launch interactive viewer after analysis (if not recording)
    if not args.record:
        print("\nLaunching viewer... (Press SPACE to play/pause)")
        simulation.run_simulation(
            sim_params,
            mjcf_path=mjcf_path,
            sim_duration=10.0,
            visualize=True,
            ignore_stuck_detection=True,
        )


if __name__ == "__main__":
    main()
