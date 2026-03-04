#!/usr/bin/env python3
"""Validate optimized params with jittered trials.

For each reference condition, runs N_TRIALS jittered simulations and saves
all trajectories to NPZ + metadata to CSV. No trial selection or velocity
computation — that belongs in downstream plotting code.

Flat/Step: yaw jitter (matching optimizer's INIT_YAW_JITTER_DEG)
Rough: Y-position jitter (matching optimizer's Y_JITTER)

Uses different base seeds than the optimizer to test generalization.

Usage:
    uv run python -m analysis.validate_params results/20260228T013353_rk4_flat
    uv run python -m analysis.validate_params results/20260303T151416_step_065gate --terrain step_065 --csv
    uv run python -m analysis.validate_params results/20260303T224229_rough_tg --csv --record
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import pathlib
import sys
from datetime import datetime

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from analysis._common import (
    load_best_point,
    detect_terrain,
    compute_pitch_series,
    SETTLE_TIME,
)
from config import sim_params_from_point, reference_rows

N_TRIALS_DEFAULT = 5
# Different from optimizer seeds (12345 yaw, 77777 Y) to test generalization
BASE_SEED = 99999
# Experimental CSV morphology → scene name mapping
_MORPH_TO_SCENE = {"leg": "scene1", "2-leg": "scene2", "4-leg": "scene4", "wheel": "scene_wheel"}

# Rough terrain XMLs for scenes not in config_rough.MJCF_PATHS (exploratory only)
_EXTRA_ROUGH_XMLS: dict[str, str] = {
    "scene_wheel": str(pathlib.Path(__file__).resolve().parent.parent / "robots" / "wheel" / "scene_wheel_rough.xml"),
}


def _get_robot_mass(mjcf_path: str) -> float:
    """Load model just to read total mass."""
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    return float(np.sum(model.body_mass))


def _store_trajectory_arrays(
    traj: list[dict], traj_key: str, traj_arrays: dict[str, np.ndarray],
) -> None:
    """Extract all raw timeseries from trajectory and store in traj_arrays dict.

    Stores under keys: {traj_key}_{field} for each field.
    Float arrays stored as float32 (~6e-8 relative error, halves file size).
    """
    f32 = np.float32

    # Kinematics (always present)
    traj_arrays[f"{traj_key}_time"] = np.array([s["time"] for s in traj], dtype=f32)
    traj_arrays[f"{traj_key}_pos_x"] = np.array([s["pos"][0] for s in traj], dtype=f32)
    traj_arrays[f"{traj_key}_pos_y"] = np.array([s["pos"][1] for s in traj], dtype=f32)
    traj_arrays[f"{traj_key}_pos_z"] = np.array([s["pos"][2] for s in traj], dtype=f32)
    traj_arrays[f"{traj_key}_vel_x"] = np.array([s["vel"][0] for s in traj], dtype=f32)
    traj_arrays[f"{traj_key}_vel_y"] = np.array([s["vel"][1] for s in traj], dtype=f32)
    traj_arrays[f"{traj_key}_vel_z"] = np.array([s["vel"][2] for s in traj], dtype=f32)
    traj_arrays[f"{traj_key}_pitch"] = compute_pitch_series(traj).astype(f32)
    traj_arrays[f"{traj_key}_joint_pos"] = np.array(
        [s["joint_pos"] for s in traj], dtype=f32)  # (T, 4)

    # Drive angle (from step_cache — present when magnetic forces applied)
    if "drive_angle" in traj[0]:
        traj_arrays[f"{traj_key}_drive_angle"] = np.array(
            [s["drive_angle"] for s in traj], dtype=f32)

    # External torques + angular velocity (for power/COT recomputation)
    if "tau_ext" in traj[0]:
        traj_arrays[f"{traj_key}_tau_ext"] = np.array(
            [s["tau_ext"] for s in traj], dtype=f32)  # (T, 4, 3)
    if "omega" in traj[0]:
        traj_arrays[f"{traj_key}_omega"] = np.array(
            [s["omega"] for s in traj], dtype=f32)  # (T, 4, 3)

    # Energy fields (leg_xquat, joint_vel, leg_xpos — kept for axis-projection if needed)
    if "leg_xquat" in traj[0]:
        traj_arrays[f"{traj_key}_leg_xquat"] = np.array(
            [s["leg_xquat"] for s in traj], dtype=f32)  # (T, 4, 4)
        traj_arrays[f"{traj_key}_joint_vel"] = np.array(
            [s["joint_vel"] for s in traj], dtype=f32)  # (T, 4)
        traj_arrays[f"{traj_key}_leg_xpos"] = np.array(
            [s["leg_xpos"] for s in traj], dtype=f32)  # (T, 4, 3)

    # Contact data (from _extract_contact_data)
    if "leg_in_contact" in traj[0]:
        traj_arrays[f"{traj_key}_leg_in_contact"] = np.array(
            [s["leg_in_contact"] for s in traj], dtype=bool)  # (T, 4)
        traj_arrays[f"{traj_key}_leg_normal_force"] = np.array(
            [s["leg_normal_force"] for s in traj], dtype=f32)  # (T, 4)
        traj_arrays[f"{traj_key}_leg_tangent_force"] = np.array(
            [s["leg_tangent_force"] for s in traj], dtype=f32)  # (T, 4)
        traj_arrays[f"{traj_key}_leg_contact_pos"] = np.array(
            [s["leg_contact_pos"] for s in traj], dtype=f32)  # (T, 4, 3)
        traj_arrays[f"{traj_key}_total_ncon"] = np.array(
            [s["total_ncon"] for s in traj], dtype=np.int16)  # (T,)

    # Chassis-terrain contact (from _extract_contact_data)
    if "body_in_contact" in traj[0]:
        traj_arrays[f"{traj_key}_body_in_contact"] = np.array(
            [s["body_in_contact"] for s in traj], dtype=bool)  # (T,)
        traj_arrays[f"{traj_key}_body_normal_force"] = np.array(
            [s["body_normal_force"] for s in traj], dtype=f32)  # (T,)
        traj_arrays[f"{traj_key}_body_tangent_force"] = np.array(
            [s["body_tangent_force"] for s in traj], dtype=f32)  # (T,)


def _load_exploratory_rough_conditions(
    ref_data: list[dict],
    available_xmls: dict[str, str],
) -> list[dict]:
    """Parse random_terrain_raw.csv and return conditions not in ref_data.

    Returns list of dicts with keys: scene, ctrl_freq, exp_speed, exp_success_pct.
    Only includes conditions whose scene has an available XML.
    """
    csv_path = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "experimental_data" / "csv" / "random_terrain_raw.csv"
    )
    if not csv_path.exists():
        return []

    # Build set of (scene, freq) already in REFERENCE_DATA
    ref_keys = {(r["scene"], float(r["ctrl_freq"])) for r in ref_data}

    conditions = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = _MORPH_TO_SCENE.get(row["morphology"])
            if scene is None or scene not in available_xmls:
                continue
            freq = float(row["freq_hz"])
            if (scene, freq) in ref_keys:
                continue  # already in REFERENCE_DATA
            exp_speed = float(row["ave"]) / 1000.0  # mm/s → m/s
            success_pct = float(row["success_rate_pct"])
            conditions.append({
                "scene": scene,
                "ctrl_freq": freq,
                "exp_speed": exp_speed,
                "exp_success_pct": success_pct,
                "id": f"{scene}_f{freq:.0f}",
            })
    return conditions


def _run_trial(sim_module, sim_params, mjcf_path, sim_duration, freq,
               seed, is_rough, y_jitter, spawn_x, spawn_z, yaw_jitter_deg):
    """Run a single jittered trial. Returns (traj, jitter_type, jitter_value, extra_kw)."""
    sp = dict(sim_params)
    sp["drive_freq"] = freq

    extra_kw = {}
    if is_rough:
        rng = np.random.default_rng(seed)
        y_offset = rng.uniform(-y_jitter, y_jitter)
        jitter_value = y_offset * 1000  # mm
        jitter_type = "y_mm"
        extra_kw["spawn_offset"] = (spawn_x, y_offset, spawn_z)
    else:
        rng = np.random.default_rng(seed)
        jitter_value = rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)
        jitter_type = "yaw_deg"
        extra_kw["init_yaw_jitter_deg"] = yaw_jitter_deg
        extra_kw["rng_seed"] = seed

    try:
        traj = sim_module.run_simulation(
            sp, mjcf_path=mjcf_path,
            sim_duration=sim_duration, visualize=False, progress=False,
            ignore_stuck_detection=True, **extra_kw,
        )
    except Exception as e:
        return None, jitter_type, jitter_value, extra_kw, str(e)

    return traj, jitter_type, jitter_value, extra_kw, None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--terrain", type=str, default=None,
                        help="Override auto-detected terrain (e.g. 'step_065' to cross-eval)")
    parser.add_argument("--csv", action="store_true",
                        help="Write per-trial CSV + NPZ to run_dir/")
    parser.add_argument("--record", action="store_true",
                        help="Record video of all non-crash trials")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                        help=f"Number of jittered trials per ref (default: {N_TRIALS_DEFAULT})")
    args = parser.parse_args()

    N_TRIALS = args.n_trials

    terrain = args.terrain or detect_terrain(args.run_dir)
    config_mod = importlib.import_module(f"config_{terrain}")

    point = load_best_point(args.run_dir)
    sim_params = sim_params_from_point(point)

    import simulation as sim_module

    is_rough = terrain.startswith("rough")
    is_step = terrain.startswith("step")

    MJCF_PATHS = dict(config_mod.MJCF_PATHS)
    SIM_DURATION = config_mod.SIM_DURATION
    ref_rows = list(reference_rows(config_mod.REFERENCE_DATA))

    # For rough terrain: merge in extra XMLs for exploratory conditions
    exploratory_conditions: list[dict] = []
    if is_rough:
        MJCF_PATHS.update(_EXTRA_ROUGH_XMLS)
        exploratory_conditions = _load_exploratory_rough_conditions(
            config_mod.REFERENCE_DATA, MJCF_PATHS,
        )

    if is_rough:
        y_jitter = config_mod.Y_JITTER
        spawn_x = config_mod.SPAWN_X
        spawn_z = config_mod.SPAWN_Z_RAISE
        yaw_jitter_deg = 0.0
    else:
        y_jitter = 0.0
        spawn_x = 0.0
        spawn_z = 0.0
        yaw_jitter_deg = config_mod.INIT_YAW_JITTER_DEG

    print(f"\nValidating: {args.run_dir.name}")
    print(f"Terrain: {terrain}  |  Trials: {N_TRIALS}")
    jitter_desc = f"Y ±{y_jitter*1000:.0f}mm" if is_rough else f"yaw ±{yaw_jitter_deg:.0f}°"
    print(f"Jitter: {jitter_desc}  |  Base seed: {BASE_SEED}")
    print(f"{'='*70}\n")

    all_trial_rows: list[dict] = []
    replay_specs: list[dict] = []
    traj_arrays: dict[str, np.ndarray] = {}

    # --- Reference conditions ---
    all_conditions = []
    for ref_idx, ref_row in enumerate(ref_rows):
        scene = ref_row["scene"]
        if scene not in MJCF_PATHS:
            continue
        all_conditions.append({
            "scene": scene,
            "freq": ref_row.get("ctrl_freq", 10.0),
            "rid": ref_row.get("id", f"{scene}_f{ref_row.get('ctrl_freq', 10.0):.0f}"),
            "target": ref_row["speed"],
            "seed_offset": ref_idx * N_TRIALS,
            "is_exploratory": False,
        })

    # --- Exploratory conditions (rough only) ---
    expl_seed_base = len(ref_rows) * N_TRIALS
    for expl_idx, expl in enumerate(exploratory_conditions):
        if expl["scene"] not in MJCF_PATHS:
            continue
        all_conditions.append({
            "scene": expl["scene"],
            "freq": expl["ctrl_freq"],
            "rid": expl["id"],
            "target": 0.0,
            "seed_offset": expl_seed_base + expl_idx * N_TRIALS,
            "is_exploratory": True,
            "exp_speed": expl["exp_speed"],
            "exp_success_pct": expl["exp_success_pct"],
        })

    for cond in all_conditions:
        scene = cond["scene"]
        freq = cond["freq"]
        rid = cond["rid"]
        target = cond["target"]

        if cond["is_exploratory"]:
            print(f"  {rid}  f={freq:.0f}Hz  (exploratory, exp={cond['exp_speed']*100:.1f}cm/s)")
        elif target < 1e-9:
            print(f"  {rid}  f={freq:.0f}Hz  (failure mode, target=0)")
        else:
            print(f"  {rid}  f={freq:.0f}Hz  target={target*100:.1f}cm/s")

        for t in range(N_TRIALS):
            seed = BASE_SEED + cond["seed_offset"] + t

            print(f"    trial {t+1}/{N_TRIALS} (seed={seed}) ...", end="", flush=True)

            traj, jitter_type, jitter_value, extra_kw, err_msg = _run_trial(
                sim_module, sim_params, MJCF_PATHS[scene], SIM_DURATION, freq,
                seed, is_rough, y_jitter, spawn_x, spawn_z, yaw_jitter_deg,
            )

            crash = traj is None
            if crash:
                print(f" CRASH ({err_msg})")
            else:
                traj_key = f"{rid}_t{t}"
                _store_trajectory_arrays(traj, traj_key, traj_arrays)
                max_x = float(traj_arrays[f"{traj_key}_pos_x"].max())
                print(f" done (max_x={max_x*1000:.1f}mm)")

                if args.record:
                    replay_specs.append({
                        "rid": rid, "scene": scene, "freq": freq,
                        "trial": t, "seed": seed, "extra_kw": extra_kw,
                    })

            all_trial_rows.append({
                "ref_id": rid, "scene": scene, "ctrl_freq": freq,
                "target_speed": target if not cond["is_exploratory"] else "",
                "trial": t, "rng_seed": seed,
                "jitter_type": jitter_type, "jitter_value": jitter_value,
                "crash": crash, "selected": True,
            })

    # --- Summary ---
    n_total = len(all_trial_rows)
    n_crash = sum(1 for r in all_trial_rows if r["crash"])
    print(f"\n{'='*70}")
    print(f"Done: {n_total} trials, {n_crash} crashes, {n_total - n_crash} valid")
    print(f"{'='*70}")

    # --- Video recording pass ---
    if args.record and replay_specs:
        video_dir = args.run_dir / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Recording {len(replay_specs)} trials to {video_dir}/")
        for i, spec in enumerate(replay_specs):
            video_name = f"{spec['rid']}_t{spec['trial']}_s{spec['seed']}.mp4"
            video_path = video_dir / video_name
            print(f"    ({i+1}/{len(replay_specs)}) {video_name} ...", end="", flush=True)
            sp = dict(sim_params)
            sp["drive_freq"] = spec["freq"]
            try:
                sim_module.run_simulation(
                    sp, mjcf_path=MJCF_PATHS[spec["scene"]],
                    sim_duration=SIM_DURATION, visualize=False, progress=False,
                    ignore_stuck_detection=True,
                    record_path=str(video_path),
                    **spec["extra_kw"],
                )
                print(" done")
            except Exception as e:
                print(f" FAILED ({e.__class__.__name__})")
        print(f"  Videos: {video_dir}/")

    # --- Trajectory data ---
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.csv and traj_arrays:
        npz_path = args.run_dir / f"{ts}_validation_trajectories.npz"
        np.savez_compressed(npz_path, **traj_arrays)
        trial_prefixes = set(k.rsplit("_", 1)[0] for k in traj_arrays if "_time" in k)
        print(f"\n  Trajectories: {npz_path} ({len(trial_prefixes)} trials, {len(traj_arrays)} arrays)")

    # --- CSV output ---
    if args.csv and all_trial_rows:
        csv_path = args.run_dir / f"{ts}_validation_trials.csv"
        fieldnames = [
            "ref_id", "scene", "ctrl_freq", "target_speed", "trial",
            "rng_seed", "jitter_type", "jitter_value", "crash", "selected",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_trial_rows)
        print(f"\n  CSV: {csv_path}")

    # --- Trajectory overview plot ---
    if args.csv and traj_arrays:
        if is_step:
            step_start_x = getattr(config_mod, "STEP_START_X", None)
            step_end_x = getattr(config_mod, "STEP_END_X", None)
        elif is_rough:
            step_start_x = config_mod.FLAT_LEAD
            step_end_x = config_mod.FLAT_LEAD + 2 * config_mod._X_HALF
        else:
            step_start_x = None
            step_end_x = None

        # Build per-ref trial_duration map from REFERENCE_DATA (if present)
        trial_duration_map: dict[str, float] = {}
        for rd in config_mod.REFERENCE_DATA:
            if "trial_duration" in rd:
                from config import _make_ref_id
                rid = rd.get("id", _make_ref_id(rd["scene"], rd["ctrl_freq"]))
                trial_duration_map[rid] = rd["trial_duration"]

        from analysis.plot_trajectories import plot_trajectory_overview
        plot_trajectory_overview(args.run_dir, step_start_x, step_end_x,
                                npz_path=npz_path, csv_path=csv_path,
                                trial_duration_map=trial_duration_map)


if __name__ == "__main__":
    main()
