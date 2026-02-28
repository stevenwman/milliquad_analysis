#!/usr/bin/env python3
"""Energy analysis for optimized params on a given terrain.

Computes W_ext, W_int, W_xfrc, W_joint, dE, Dissipation for each reference.
RK4 is baked into XMLs, so this just runs once per ref.

Usage:
    uv run python energy_analysis.py results/20260228T013353_rk4_flat -t flat
"""

from __future__ import annotations

import argparse
import csv
import importlib
import pathlib
import sys

import numpy as np

from config import space, sim_params_from_point, reference_rows, SETTLE_TIME


PARAM_NAMES = [dim.name for dim in space]


def load_best_point(run_dir: pathlib.Path) -> list[float]:
    """Load best parameter point from a completed run."""
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    if not rows:
        sys.exit(f"ERROR: no rows in {bests_csv}")
    best = rows[-1]
    # Try multi CSV for full precision
    multi_csv = run_dir / "multi_optimization_results.csv"
    if multi_csv.exists():
        best_id = best["id"]
        for row in csv.DictReader(open(multi_csv)):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    return [float(best[name]) for name in PARAM_NAMES]


def _joint_axis_world(leg_xquat: np.ndarray) -> np.ndarray:
    """Joint axis [0,0,1] rotated into world frame by leg body quaternion."""
    w = leg_xquat[:, 0]
    x = leg_xquat[:, 1]
    y = leg_xquat[:, 2]
    z = leg_xquat[:, 3]
    return np.column_stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ])


def compute_energy_breakdown(traj: list[dict], settle_time: float) -> dict | None:
    """Compute W_ext, W_int, W_xfrc, W_joint, dE, Dissipation from trajectory."""
    start_idx = 0
    for i, s in enumerate(traj):
        if s["time"] >= settle_time:
            start_idx = i
            break
    active = traj[start_idx:]
    if len(active) < 2:
        return None
    if "tau_ext" not in active[0] or "omega" not in active[0]:
        return None

    n = len(active) - 1
    dt = np.empty(n)
    p_ext = np.empty(n)
    p_int = np.empty(n)
    p_joint = np.empty(n)

    for i in range(n):
        s = active[i]
        dt[i] = active[i + 1]["time"] - s["time"]

        tau_ext = s["tau_ext"]    # (4, 3)
        tau_int = s["tau_int"]    # (4, 3)
        omega = s["omega"]        # (4, 3)
        axis = _joint_axis_world(s["leg_xquat"])  # (4, 3)
        jvel = s["joint_vel"]     # (4,)

        # Naive: P = Σ τ · ω
        p_ext[i] = np.sum(tau_ext * omega)
        p_int[i] = np.sum(tau_int * omega)

        # Joint-projected: P = Σ (τ · â) * q̇
        p_joint[i] = sum(np.dot(tau_ext[j], axis[j]) * jvel[j] for j in range(4))

    W_ext = np.sum(p_ext * dt) * 1e6    # J -> uJ
    W_int = np.sum(p_int * dt) * 1e6
    W_xfrc = W_ext + W_int
    W_joint = np.sum(p_joint * dt) * 1e6

    # dE from MuJoCo's data.energy
    if "energy" in active[0] and "energy" in active[-1]:
        E_start = sum(active[0]["energy"])
        E_end = sum(active[-1]["energy"])
        dE = (E_end - E_start) * 1e6
    else:
        dE = float("nan")

    dissip = W_xfrc - dE

    return {
        "W_ext": W_ext,
        "W_int": W_int,
        "W_xfrc": W_xfrc,
        "W_joint": W_joint,
        "dE": dE,
        "Dissip": dissip,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--terrain", "-t", required=True,
                        help="Terrain config (flat, step, rough)")
    args = parser.parse_args()

    # Import simulation and monkey-patch to capture data.energy
    import simulation as sim

    _orig_record = sim._record_state
    def _patched_record(trajectory, data, step_cache=None):
        _orig_record(trajectory, data, step_cache)
        trajectory[-1]["energy"] = data.energy.copy()
    sim._record_state = _patched_record

    # Load terrain config and params
    config_mod = importlib.import_module(f"config_{args.terrain}")
    MJCF_PATHS = config_mod.MJCF_PATHS
    SIM_DURATION = config_mod.SIM_DURATION
    ref_rows = list(reference_rows(config_mod.REFERENCE_DATA))

    point = load_best_point(args.run_dir)
    sim_params_base = sim_params_from_point(point)
    print(f"Loaded best params from {args.run_dir}")
    print(f"Terrain: {args.terrain}, Refs: {len(ref_rows)}, Duration: {SIM_DURATION}s\n")

    # Spawn offset for rough terrain
    extra_kwargs = {}
    if args.terrain in ("rough", "rough_cold", "rough_centered"):
        extra_kwargs["spawn_offset"] = (
            config_mod.SPAWN_X, 0.0, config_mod.SPAWN_Z_RAISE
        )

    # Header
    print(f"| {'Ref':<18} | {'vx mm/s':>7} | {'target':>6} | {'err%':>5} "
          f"| {'W_ext':>7} | {'W_int':>7} | {'W_xfrc':>8} | {'W_joint':>8} "
          f"| {'dE':>6} | {'Dissip':>7} |")
    print(f"|{'-'*20}|{'-'*9}|{'-'*8}|{'-'*7}"
          f"|{'-'*9}|{'-'*9}|{'-'*10}|{'-'*10}"
          f"|{'-'*8}|{'-'*9}|")

    n_negative = 0
    for ref_row in ref_rows:
        scene = ref_row["scene"]
        ref_id = ref_row["id"]
        target = ref_row["speed"]

        sp = dict(sim_params_base)
        sp["drive_freq"] = ref_row.get("ctrl_freq", 10.0)

        try:
            traj = sim.run_simulation(
                sp,
                mjcf_path=MJCF_PATHS[scene],
                sim_duration=SIM_DURATION,
                visualize=False,
                progress=False,
                ignore_stuck_detection=True,
                **extra_kwargs,
            )
        except Exception as e:
            print(f"| {ref_id:<18} | CRASH: {e}")
            continue

        if traj is None:
            print(f"| {ref_id:<18} | FAILED")
            continue

        # Velocity
        settle_idx = next((i for i, s in enumerate(traj) if s["time"] >= SETTLE_TIME), 0)
        dt_total = traj[-1]["time"] - traj[settle_idx]["time"]
        vx = (traj[-1]["pos"][0] - traj[settle_idx]["pos"][0]) / dt_total if dt_total > 1e-6 else 0.0

        if target > 1e-9:
            err_pct = abs(vx - target) / target * 100
            err_str = f"{err_pct:4.1f}%"
            target_str = f"{target*1000:6.1f}"
        else:
            err_str = "   —"
            target_str = "   0.0"

        # Energy
        eb = compute_energy_breakdown(traj, SETTLE_TIME)
        if eb is None:
            print(f"| {ref_id:<18} | {vx*1000:7.1f} | {target_str} | {err_str} | NO ENERGY DATA")
            continue

        flag = " <<<" if eb["Dissip"] < 0 else ""
        if eb["Dissip"] < 0:
            n_negative += 1

        print(f"| {ref_id:<18} | {vx*1000:7.1f} | {target_str} | {err_str} "
              f"| {eb['W_ext']:+7.1f} | {eb['W_int']:+7.1f} | {eb['W_xfrc']:+8.1f} "
              f"| {eb['W_joint']:+8.1f} | {eb['dE']:+6.1f} | {eb['Dissip']:+7.1f} |{flag}")

    print(f"\nNegative dissipation: {n_negative}/{len(ref_rows)} refs")
    if n_negative == 0:
        print("Energy budget closes for all refs — RK4 is working correctly.")
    else:
        print("WARNING: negative dissipation detected — integrator injecting phantom energy.")


if __name__ == "__main__":
    main()
