#!/usr/bin/env python3
"""Per-reference metrics for a completed optimization run.

Runs best params on their native terrain, reports velocity, pitch RMS, COT,
tumble, lateral displacement, and yaw per reference.

Usage:
    uv run python -m analysis.eval_metrics results/20260228T013353_rk4_flat
    uv run python -m analysis.eval_metrics results/20260228T102010_rk4_rough --csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import pathlib
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from analysis._common import (
    PARAM_NAMES,
    load_best_point,
    detect_terrain,
    compute_pitch_rms,
    compute_cot,
    extract_velocity,
    SETTLE_TIME,
)
from config import space, sim_params_from_point, reference_rows

_BODY_Z_LOCAL = np.array([0.0, 0.0, 1.0])
_NOMINAL_BODY_Z_WORLD = np.array([0.0, 0.0, -1.0])
_BODY_X_LOCAL = np.array([1.0, 0.0, 0.0])


def _compute_tumble(traj: list[dict], threshold: float = 0.0, scale: float = 0.1) -> float:
    penalty = 0.0
    for s in traj:
        body_z = R.from_quat(s["quat"], scalar_first=True).apply(_BODY_Z_LOCAL)
        uprightness = np.dot(body_z, _NOMINAL_BODY_Z_WORLD)
        if uprightness < threshold:
            penalty += (1 - uprightness) * scale
    return penalty / max(len(traj), 1)


def _compute_lateral_yaw(traj: list[dict], settle_time: float) -> tuple[float, float]:
    start = traj[0]
    for s in traj:
        if s["time"] >= settle_time:
            start = s
            break
    end = traj[-1]

    lateral = abs(end["pos"][1] - start["pos"][1])

    start_x = R.from_quat(start["quat"], scalar_first=True).apply(_BODY_X_LOCAL)[:2]
    end_x = R.from_quat(end["quat"], scalar_first=True).apply(_BODY_X_LOCAL)[:2]
    s_norm, e_norm = np.linalg.norm(start_x), np.linalg.norm(end_x)
    if s_norm > 1e-6 and e_norm > 1e-6:
        cos_yaw = np.clip(np.dot(start_x / s_norm, end_x / e_norm), -1.0, 1.0)
        yaw_deg = float(np.degrees(np.arccos(cos_yaw)))
    else:
        yaw_deg = 0.0
    return lateral, yaw_deg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--terrain", "-t", default=None)
    parser.add_argument("--csv", dest="write_csv", action="store_true")
    args = parser.parse_args()

    terrain = args.terrain or detect_terrain(args.run_dir)
    config_mod = importlib.import_module(f"config_{terrain}")
    MJCF_PATHS = config_mod.MJCF_PATHS
    SIM_DURATION = config_mod.SIM_DURATION
    ref_rows = list(reference_rows(config_mod.REFERENCE_DATA))

    import simulation as sim

    point = load_best_point(args.run_dir)
    sim_params_base = sim_params_from_point(point)

    # Step terrain spatial gating
    step_start_x = getattr(config_mod, "STEP_START_X", None)
    step_end_x = getattr(config_mod, "STEP_END_X", None)

    # Rough terrain spawn
    extra_kwargs = {}
    if terrain.startswith("rough"):
        extra_kwargs["spawn_offset"] = (
            config_mod.SPAWN_X, 0.0, config_mod.SPAWN_Z_RAISE
        )

    # Robot masses (cached per scene)
    masses: dict[str, float] = {}
    for scene, xml_path in MJCF_PATHS.items():
        model = mujoco.MjModel.from_xml_path(xml_path)
        masses[scene] = float(sum(model.body_mass))

    tumble_threshold = getattr(config_mod, "TUMBLE_THRESHOLD", 0.0)

    print(f"Run: {args.run_dir.name}")
    print(f"Terrain: {terrain}, Refs: {len(ref_rows)}\n")

    # Header
    print(f"  {'Ref':<18} {'vx mm/s':>7} {'target':>7} {'err%':>6} "
          f"{'pitch':>6} {'COT':>6} {'tumble':>7} {'lateral':>8} {'yaw':>5}")
    print(f"  {'-' * 80}")

    csv_rows = []

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
            print(f"  {ref_id:<18} CRASH: {e}")
            continue

        if traj is None:
            print(f"  {ref_id:<18} FAILED")
            continue

        vx = extract_velocity(traj, SETTLE_TIME, step_start_x, step_end_x)
        pitch = compute_pitch_rms(traj, SETTLE_TIME)
        cot = compute_cot(traj, masses[scene], SETTLE_TIME)
        tumble = _compute_tumble(traj, tumble_threshold)
        lateral, yaw = _compute_lateral_yaw(traj, SETTLE_TIME)

        if target > 1e-9:
            err_pct = abs(vx - target) / target * 100
            err_str = f"{err_pct:5.1f}%"
        else:
            err_str = "    —"

        cot_str = f"{cot:6.1f}" if cot is not None else "     —"

        print(f"  {ref_id:<18} {vx*1000:7.1f} {target*1000:7.1f} {err_str} "
              f"{pitch:6.2f} {cot_str} {tumble:7.4f} {lateral*100:6.1f}cm {yaw:5.1f}")

        csv_rows.append({
            "ref_id": ref_id, "scene": scene,
            "freq_hz": ref_row.get("ctrl_freq", 10.0),
            "velocity_mps": vx, "target_mps": target,
            "error_pct": abs(vx - target) / target * 100 if target > 1e-9 else 0.0,
            "pitch_rms_deg": pitch,
            "cot": cot if cot is not None else float("nan"),
            "tumble": tumble, "lateral_m": lateral, "yaw_deg": yaw,
        })

    if args.write_csv and csv_rows:
        csv_path = args.run_dir / "eval_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n  CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
