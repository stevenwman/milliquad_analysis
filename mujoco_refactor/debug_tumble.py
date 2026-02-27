#!/usr/bin/env python3
"""Re-run specific sweep trials and dump frame-by-frame uprightness to diagnose tumble.

Picks a few trials from the 0.5mm sweep CSV covering low/mid/high tumble values,
re-runs each with the same y_offset, and prints min/max uprightness + timestamps
where the robot crosses key thresholds.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as R

import eval_rough_terrain as ert
import simulation_fast_new as sim_module
from config_new import MJCF_PATHS, SIM_DURATION, SIMULATION_TIMEOUT, SETTLE_TIME, sim_params_from_point

_BODY_Z_LOCAL = np.array([0.0, 0.0, 1.0])
_NOMINAL_BODY_Z_WORLD = np.array([0.0, 0.0, -1.0])

RUN_DIR = pathlib.Path("results/20260225T122342_flat_10_30_50")
HEIGHT_STD = 0.0005  # 0.5mm — matches the sweep

# Trials to diagnose: (scene, freq, trial_idx, y_off_mm, reported_tmb)
# Picked from 0.5mm CSV covering range of tumble values
TRIALS = [
    # Low tumble
    ("scene4", 30.0, 0, -8.42, 0.0000, "4leg f30 — tmb=0, vel=211"),
    ("scene4", 30.0, 4, -6.71, 0.0000, "4leg f30 — tmb=0, vel=191"),
    # Mid tumble
    ("scene1", 10.0, 2, 4.81, 0.0475, "1leg f10 — tmb=0.048, vel=37"),
    ("scene1", 10.0, 4, -7.40, 0.0101, "1leg f10 — tmb=0.010, vel=32"),
    # Higher tumble
    ("scene1", 10.0, 6, -8.06, 0.1470, "1leg f10 — tmb=0.147, vel=17"),
    ("scene1", 30.0, 3, -2.13, 0.1273, "1leg f30 — tmb=0.127, vel=67"),
    # Wheel with tumble
    ("scene_wheel", 10.0, 2, 0.42, 0.1934, "wheel f10 — tmb=0.193, vel=1.2"),
]


def analyze_trajectory(traj, label):
    """Compute uprightness stats from trajectory."""
    times = []
    uprights = []
    for state in traj:
        t = state["time"]
        quat = state["quat"]
        body_z = R.from_quat(quat, scalar_first=True).apply(_BODY_Z_LOCAL)
        u = np.dot(body_z, _NOMINAL_BODY_Z_WORLD)
        times.append(t)
        uprights.append(u)

    times = np.array(times)
    uprights = np.array(uprights)

    # Only active period (after settle)
    active = times >= SETTLE_TIME
    t_act = times[active]
    u_act = uprights[active]

    # Recompute tumble penalty exactly as optimizer does
    tmb = 0.0
    for u in uprights:
        if u < 0.0:  # TUMBLE_THRESHOLD = 0.0
            tmb += (1 - u) * 0.1  # TUMBLE_PENALTY_SCALE = 0.1
    tmb /= max(len(uprights), 1)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Frames: {len(uprights)} total, {len(u_act)} active")
    print(f"  Uprightness — min: {u_act.min():.4f}, max: {u_act.max():.4f}, mean: {u_act.mean():.4f}")
    print(f"  Recomputed tumble penalty: {tmb:.6f}")

    # Thresholds
    for thresh, desc in [(0.0, "past 90° (threshold)"), (-0.5, "past 120°"), (-0.9, "nearly inverted")]:
        below = u_act < thresh
        n_below = below.sum()
        pct = n_below / len(u_act) * 100
        if n_below > 0:
            first_t = t_act[below][0]
            last_t = t_act[below][-1]
            print(f"  uprightness < {thresh:+.1f} ({desc}): {n_below} frames ({pct:.1f}%), "
                  f"first@{first_t:.3f}s, last@{last_t:.3f}s")
        else:
            print(f"  uprightness < {thresh:+.1f} ({desc}): never")

    # Time trace: sample every 0.1s
    print(f"  Time trace (every 0.1s):")
    for sample_t in np.arange(0.0, t_act[-1] + 0.05, 0.1):
        idx = np.argmin(np.abs(t_act - sample_t))
        print(f"    t={t_act[idx]:.2f}s  uprightness={u_act[idx]:+.4f}")


def main():
    point = ert.load_best_point(RUN_DIR)
    sim_params = sim_params_from_point(point)

    ert.TERRAIN_HEIGHT_MEAN = HEIGHT_STD * 2
    ert.TERRAIN_HEIGHT_STD = HEIGHT_STD

    with tempfile.TemporaryDirectory(prefix="debug_tumble_") as tmp_dir:
        for scene, freq, trial_idx, y_off_mm, reported_tmb, label in TRIALS:
            ert._y_offset = y_off_mm / 1000.0

            mjcf = ert.inject_tiled_rough(MJCF_PATHS[scene], tmp_dir)
            sp = dict(sim_params)
            sp["drive_freq"] = freq

            traj = sim_module.run_simulation(
                sp, mjcf_path=mjcf,
                sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
                ignore_stuck_detection=True,
            )
            ert.cleanup_temp_xmls()

            if traj is None:
                print(f"\n{label}: SIMULATION FAILED (not reproducible)")
                continue

            analyze_trajectory(traj, f"{label} [reported tmb={reported_tmb:.4f}]")

    ert._y_offset = 0.0


if __name__ == "__main__":
    main()
