#!/usr/bin/env python3
"""Compare FL angular velocity (omega) between experimental and sim across all refs.

Uses best-matching seeds from cot_results.csv (closest velocity to target).

Usage:
    uv run python compare_omega_table.py
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP_CSV_DIR = ROOT / "experimental_data" / "csv" / "flat"
REFACTOR = pathlib.Path(__file__).resolve().parent

FLAT_DIR = REFACTOR / "results" / "20260225T122342_flat_10_30_50"
STEP_DIR = REFACTOR / "results" / "20260225T225248_step_argmin_progress"

sys.path.insert(0, str(REFACTOR))
from config_new import MJCF_PATHS, SETTLE_TIME, SIM_DURATION, SIMULATION_TIMEOUT, sim_params_from_point, space
import simulation_fast_new as sim_module

PARAM_NAMES = [dim.name for dim in space]
FL_IDX = 1  # FL joint index


# ── Experimental conditions ──
# (freq, morph, scene, files, trial_indices_1based, points, steady_t)
CONDITIONS = [
    (10, "leg",   "scene1",      ["f10leg1-1.csv","f10leg2-2.csv","f10leg3-3.csv","f10leg4-4.csv"], [1,2,3], 2500, 0.3),
    (10, "2leg",  "scene2",      ["f102leg1-1.csv","f102leg2-2.csv","f102leg3-3.csv","f102leg4-4.csv"], [1,2,4], 1480, 0.3),
    (10, "4leg",  "scene4",      ["f104leg1-1.csv","f104leg2-2.csv","f104leg3-3.csv","f104leg4-4.csv"], [1,2,4], 1199, 0.3),
    (10, "wheel", "scene_wheel", ["f10w1-1.csv","f10w2-2.csv","f10w3-3.csv","f10w4-4.csv"], [1,2,3], 910, 0.3),
    (20, "leg",   "scene1",      ["f20leg1-1.csv","f20leg2-2.csv","f20leg3-3.csv"], [1,2,3], None, None),
    (20, "2leg",  "scene2",      ["f202leg1-1.csv","f202leg2-2.csv","f202leg3-3.csv"], [1,2,3], None, None),
    (20, "4leg",  "scene4",      ["f204leg1-1.csv","f204leg2-2.csv","f204leg3-3.csv"], [1,2,3], None, None),
    (20, "wheel", "scene_wheel", ["f20w1-1.csv","f20w2-2.csv","f20w3-3.csv"], [1,2,3], None, None),
    (30, "leg",   "scene1",      ["f30leg1-1.csv","f30leg2-2.csv","f30leg3-3.csv","f30leg4-4.csv"], [1,2,3,4], 1100, 0.15),
    (30, "2leg",  "scene2",      ["f302leg1-1.csv","f302leg2-2.csv","f302leg3-3.csv","f302leg4-4.csv"], [1,2,3], 760, 0.3),
    (30, "4leg",  "scene4",      ["f304leg1-1.csv","f304leg2-2.csv","f304leg3-3.csv","f304leg4-4.csv"], [1,2,3,4], 550, 0.3),
    (30, "wheel", "scene_wheel", ["f30w1-1.csv","f30w2-2.csv","f30w3-3.csv","f30w4-4.csv"], [1,2,3,4], 350, 0.3),
    (50, "leg",   "scene1",      ["f50leg1-1.csv","f50leg2-2.csv","f50leg3-3.csv"], [1,2,3], 1960, 0.35),
    (50, "2leg",  "scene2",      ["f502leg1-1.csv","f502leg2-2.csv","f502leg3-3.csv"], [1,2,3], 1280, 0.35),
    (50, "4leg",  "scene4",      ["f504leg1-1.csv","f504leg2-2.csv","f504leg3-3.csv"], [1,2,3], 1060, 0.35),
    (50, "wheel", "scene_wheel", ["f50w1-1.csv","f50w2-2.csv","f50w3-3.csv"], [1,2,3], 620, 0.25),
]


def find_omega_col(csv_path: pathlib.Path) -> int:
    """Parse header to find the ω column index."""
    with open(csv_path) as f:
        f.readline()  # row 0: mass labels
        col_names = f.readline().strip().split(",")
    for i, name in enumerate(col_names):
        if name.strip() == "ω":
            return i
    raise ValueError(f"No ω column in {csv_path}")


def load_exp_omega(csv_path: pathlib.Path, points: int | None, steady_t: float | None) -> np.ndarray:
    """Load experimental omega (deg/s) for steady-state portion."""
    omega_col = find_omega_col(csv_path)
    dat = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    if points is not None:
        dat = dat[:points, :]
    t = dat[:, 0]
    omega = dat[:, omega_col]

    if steady_t is not None:
        mask = t > steady_t
    else:
        # 20Hz: use last 50%
        mid = len(t) // 2
        mask = np.zeros(len(t), dtype=bool)
        mask[mid:] = True

    omega_ss = omega[mask]
    return omega_ss[~np.isnan(omega_ss)]


def load_best_point(run_dir: pathlib.Path) -> list[float]:
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    best_id = rows[-1]["id"]
    with open(run_dir / "multi_optimization_results.csv") as f:
        for row in csv.DictReader(f):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    raise ValueError(f"id {best_id!r} not found")


def load_cot_seeds(run_dir: pathlib.Path) -> dict[tuple[str, float], list[int]]:
    """Load best seeds per (scene, freq) from cot_results.csv.

    Returns dict mapping (scene, freq) -> list of seeds (sorted by smallest vel error).
    """
    cot_csv = run_dir / "cot_results.csv"
    if not cot_csv.exists():
        return {}

    seeds: dict[tuple[str, float], list[int]] = {}
    for row in csv.DictReader(open(cot_csv)):
        key = (row["scene"], float(row["freq"]))
        seeds.setdefault(key, []).append(int(row["seed"]))
    return seeds


def run_sim_omega(sim_params: dict, scene: str, freq: float, seed: int) -> tuple[np.ndarray, float]:
    """Run sim, return (FL omega in deg/s after settle, forward velocity in m/s)."""
    sp = dict(sim_params)
    sp["drive_freq"] = freq

    traj = sim_module.run_simulation(
        sp, mjcf_path=MJCF_PATHS[scene],
        sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
        init_yaw_jitter_deg=2.0, rng_seed=seed,
    )
    if traj is None:
        return np.array([]), 0.0

    t = np.array([s["time"] for s in traj])
    theta_rad = np.array([s["joint_pos"][FL_IDX] for s in traj])
    theta_deg = np.degrees(theta_rad)

    # d(theta)/dt
    dt = np.diff(t)
    omega = np.diff(theta_deg) / dt  # deg/s

    # After settle
    t_mid = 0.5 * (t[:-1] + t[1:])
    mask = t_mid >= SETTLE_TIME

    # Forward velocity
    start_idx = next(i for i, s in enumerate(traj) if s["time"] >= SETTLE_TIME)
    vel = (traj[-1]["pos"][0] - traj[start_idx]["pos"][0]) / (traj[-1]["time"] - traj[start_idx]["time"])

    return omega[mask], vel


def main():
    print("Loading params...")
    flat_point = load_best_point(FLAT_DIR)
    flat_sp = sim_params_from_point(flat_point)
    step_point = load_best_point(STEP_DIR)
    step_sp = sim_params_from_point(step_point)

    flat_seeds = load_cot_seeds(FLAT_DIR)
    step_seeds = load_cot_seeds(STEP_DIR)

    header = (f"{'Condition':<15} {'Exp mean':>9} {'Exp std':>9} "
              f"{'Flat mean':>10} {'Flat std':>9} {'F ratio':>7} {'F v_err':>7} "
              f"{'Step mean':>10} {'Step std':>9} {'S ratio':>7} {'S v_err':>7}")
    print(f"\n{header}")
    print("-" * len(header))

    # Target velocities from REFERENCE_DATA
    from config_new import REFERENCE_DATA
    targets = {}
    for row in REFERENCE_DATA:
        key = (row["scene"], float(row.get("ctrl_freq", 30.0)))
        targets[key] = float(row["speed"])

    for freq, morph, scene, files, trial_idx, points, steady_t in CONDITIONS:
        label = f"{morph} f{freq}"
        key = (scene, float(freq))

        # Experimental: aggregate omega across trials
        trial_files = [files[i - 1] for i in trial_idx]
        all_exp_omega = []
        for f in trial_files:
            path = EXP_CSV_DIR / f
            omega = load_exp_omega(path, points, steady_t)
            all_exp_omega.append(omega)
        exp_omega = np.concatenate(all_exp_omega)
        exp_mean = np.mean(exp_omega)
        exp_std = np.std(exp_omega)

        target = targets.get(key, 0.0)
        parts = [f"{label:<15} {exp_mean:>9.0f} {exp_std:>9.0f}"]

        for sp_name, sp, seed_map in [("flat", flat_sp, flat_seeds), ("step", step_sp, step_seeds)]:
            # Use first (best) seed from COT results, fall back to 42
            seeds = seed_map.get(key, [42])
            seed = seeds[0]

            sim_omega, vel = run_sim_omega(sp, scene, freq, seed)
            if len(sim_omega) == 0:
                parts.append(f"{'FAIL':>10} {'':>9} {'':>7} {'':>7}")
            else:
                sim_mean = np.mean(sim_omega)
                sim_std = np.std(sim_omega)
                ratio = sim_std / exp_std if exp_std > 0 else float("inf")
                v_err = (vel - target) / target * 100 if target > 1e-6 else 0
                parts.append(f"{sim_mean:>10.0f} {sim_std:>9.0f} {ratio:>6.1f}x {v_err:>+6.0f}%")

        print(" ".join(parts))

    print()


if __name__ == "__main__":
    main()
