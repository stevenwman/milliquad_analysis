#!/usr/bin/env python3
"""Validate optimized params with jittered trials.

For each reference condition, runs N_TRIALS jittered simulations and selects
the top N_SELECT by velocity match. Reports mean +/- std velocity error and COT.

Flat/Step: yaw jitter (matching optimizer's INIT_YAW_JITTER_DEG)
Rough: Y-position jitter (matching optimizer's Y_JITTER)
Step failure refs (target=0): verify robot doesn't move, report pass/fail.

Uses different base seeds than the optimizer to test generalization.

Usage:
    uv run python -m analysis.validate_params results/20260228T013353_rk4_flat
    uv run python -m analysis.validate_params results/20260228T093833_rk4_step_cold --csv
    uv run python -m analysis.validate_params results/20260228T102010_rk4_rough --csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import pathlib
import sys

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from analysis._common import (
    load_best_point,
    detect_terrain,
    extract_velocity,
    compute_cot,
    compute_pitch_rms,
    min_window_velocity,
    SETTLE_TIME,
)
from config import sim_params_from_point, reference_rows

N_TRIALS_DEFAULT = 5
N_SELECT_DEFAULT = 3
# Different from optimizer seeds (12345 yaw, 77777 Y) to test generalization
BASE_SEED = 99999
FAILURE_VX_THRESHOLD = 0.005  # m/s — below this, robot is "not moving"

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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--terrain", type=str, default=None,
                        help="Override auto-detected terrain (e.g. 'step' to cross-eval)")
    parser.add_argument("--csv", action="store_true",
                        help="Write per-trial CSV to run_dir/validation_trials.csv")
    parser.add_argument("--record", action="store_true",
                        help="Record video of selected (top-N) trials")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                        help=f"Number of jittered trials per ref (default: {N_TRIALS_DEFAULT})")
    parser.add_argument("--n-select", type=int, default=N_SELECT_DEFAULT,
                        help=f"Select top N by velocity match (default: {N_SELECT_DEFAULT})")
    args = parser.parse_args()

    N_TRIALS = args.n_trials
    N_SELECT = args.n_select

    terrain = args.terrain or detect_terrain(args.run_dir)
    config_mod = importlib.import_module(f"config_{terrain}")

    point = load_best_point(args.run_dir)
    sim_params = sim_params_from_point(point)

    import simulation as sim_module

    is_rough = terrain.startswith("rough")
    is_step = terrain.startswith("step")

    MJCF_PATHS = dict(config_mod.MJCF_PATHS)  # copy so we can extend for rough
    SIM_DURATION = config_mod.SIM_DURATION
    ref_rows = list(reference_rows(config_mod.REFERENCE_DATA))

    # For rough terrain: merge in extra XMLs for exploratory conditions
    exploratory_conditions: list[dict] = []
    if is_rough:
        MJCF_PATHS.update(_EXTRA_ROUGH_XMLS)
        exploratory_conditions = _load_exploratory_rough_conditions(
            config_mod.REFERENCE_DATA, MJCF_PATHS,
        )

    if is_step:
        step_start_x = getattr(config_mod, "STEP_START_X", None)
        step_end_x = getattr(config_mod, "STEP_END_X", None)
    elif is_rough:
        step_start_x = config_mod.FLAT_LEAD
        step_end_x = config_mod.FLAT_LEAD + 2 * config_mod._X_HALF
    else:
        step_start_x = None
        step_end_x = None

    if is_rough:
        y_jitter = config_mod.Y_JITTER
        spawn_x = config_mod.SPAWN_X
        spawn_z = config_mod.SPAWN_Z_RAISE
        yaw_jitter_deg = 0.0
    else:
        y_jitter = 0.0
        yaw_jitter_deg = config_mod.INIT_YAW_JITTER_DEG

    # Robot mass per scene (for COT)
    scene_mass: dict[str, float] = {}
    for scene, mjcf_path in MJCF_PATHS.items():
        scene_mass[scene] = _get_robot_mass(mjcf_path)

    print(f"\nValidating: {args.run_dir.name}")
    print(f"Terrain: {terrain}  |  Trials: {N_TRIALS}  |  Select top: {N_SELECT}")
    jitter_desc = f"Y ±{y_jitter*1000:.0f}mm" if is_rough else f"yaw ±{yaw_jitter_deg:.0f}°"
    print(f"Jitter: {jitter_desc}  |  Base seed: {BASE_SEED}")
    print(f"{'='*70}\n")

    all_trial_rows: list[dict] = []
    ref_summaries: list[dict] = []
    failure_mode_results: list[dict] = []
    replay_specs: list[dict] = []  # for --record: re-run selected trials with video
    traj_arrays: dict[str, np.ndarray] = {}  # {rid}_t{trial}_{time,pos_x} → 1D array

    for ref_idx, ref_row in enumerate(ref_rows):
        scene = ref_row["scene"]
        target = ref_row["speed"]
        freq = ref_row.get("ctrl_freq", 10.0)
        rid = ref_row.get("id", f"{scene}_f{freq:.0f}")

        if scene not in MJCF_PATHS:
            continue

        mass = scene_mass[scene]

        # --- Failure mode refs (target=0): single trial, no jitter ---
        if target < 1e-9:
            sp = dict(sim_params)
            sp["drive_freq"] = freq

            extra_kw: dict = {}
            if is_rough:
                extra_kw["spawn_offset"] = (spawn_x, 0.0, spawn_z)

            print(f"  {rid} (failure mode, target=0) ...", end="", flush=True)

            try:
                traj = sim_module.run_simulation(
                    sp, mjcf_path=MJCF_PATHS[scene],
                    sim_duration=SIM_DURATION, visualize=False, progress=False,
                    ignore_stuck_detection=True, **extra_kw,
                )
            except Exception:
                traj = None

            if traj is None:
                print(" CRASH")
                failure_mode_results.append({"id": rid, "result": "CRASH", "vx": None})
            else:
                vx = extract_velocity(traj, SETTLE_TIME, step_start_x, step_end_x)
                passed = abs(vx) < FAILURE_VX_THRESHOLD
                status = "PASS" if passed else f"FAIL (vx={vx*100:.2f}cm/s)"
                print(f" vx={vx*100:.2f}cm/s  {status}")
                failure_mode_results.append({"id": rid, "result": "PASS" if passed else "FAIL", "vx": vx})
            continue

        # --- Normal refs: N_TRIALS jittered trials ---
        print(f"  {rid}  f={freq:.0f}Hz  target={target*100:.1f}cm/s")
        trials: list[dict] = []

        for t in range(N_TRIALS):
            seed = BASE_SEED + ref_idx * N_TRIALS + t
            sp = dict(sim_params)
            sp["drive_freq"] = freq

            extra_kw = {}
            if is_rough:
                rng = np.random.default_rng(seed)
                y_offset = rng.uniform(-y_jitter, y_jitter)
                jitter_value = y_offset * 1000  # store in mm
                jitter_type = "y_mm"
                extra_kw["spawn_offset"] = (spawn_x, y_offset, spawn_z)
            else:
                # Compute actual yaw for CSV (matches what simulation draws internally)
                rng = np.random.default_rng(seed)
                jitter_value = rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)
                jitter_type = "yaw_deg"
                extra_kw["init_yaw_jitter_deg"] = yaw_jitter_deg
                extra_kw["rng_seed"] = seed

            print(f"    trial {t+1}/{N_TRIALS} (seed={seed}, {jitter_type}={jitter_value:+.3f}) ...",
                  end="", flush=True)

            try:
                traj = sim_module.run_simulation(
                    sp, mjcf_path=MJCF_PATHS[scene],
                    sim_duration=SIM_DURATION, visualize=False, progress=False,
                    ignore_stuck_detection=True, **extra_kw,
                )
            except Exception as e:
                print(f" CRASH ({e.__class__.__name__})")
                trials.append({"trial": t, "seed": seed, "jitter_type": jitter_type,
                               "jitter_value": jitter_value, "vx": None,
                               "err": float("inf"), "cot": None, "crash": True,
                               "min_window_vx": 0.0, "pitch_rms": None})
                continue

            if traj is None:
                print(" CRASH (None)")
                trials.append({"trial": t, "seed": seed, "jitter_type": jitter_type,
                               "jitter_value": jitter_value, "vx": None,
                               "err": float("inf"), "cot": None, "crash": True,
                               "min_window_vx": 0.0, "pitch_rms": None})
                continue

            vx = extract_velocity(traj, SETTLE_TIME, step_start_x, step_end_x)
            traj_key = f"{rid}_t{t}"
            traj_arrays[f"{traj_key}_time"] = np.array([s["time"] for s in traj])
            traj_arrays[f"{traj_key}_pos_x"] = np.array([s["pos"][0] for s in traj])
            err = abs(vx - target) / target * 100
            cot = compute_cot(traj, mass, SETTLE_TIME,
                              step_start_x=step_start_x, step_end_x=step_end_x)
            mwv = min_window_velocity(traj, freq, SETTLE_TIME,
                                      step_start_x=step_start_x, step_end_x=step_end_x)
            pitch_rms = compute_pitch_rms(traj, SETTLE_TIME,
                                         step_start_x=step_start_x, step_end_x=step_end_x)

            cot_str = f"  COT={cot:.2f}" if cot is not None else ""
            print(f" vx={vx*100:.1f}cm/s  err={err:.1f}%{cot_str}  mwv={mwv*100:.1f}mm/s  pitch={pitch_rms:.1f}°")

            trials.append({"trial": t, "seed": seed, "jitter_type": jitter_type,
                           "jitter_value": jitter_value, "vx": vx,
                           "err": err, "cot": cot, "crash": False,
                           "min_window_vx": mwv, "pitch_rms": pitch_rms})

        # Select top N_SELECT by velocity error
        valid = [tr for tr in trials if not tr["crash"]]
        valid.sort(key=lambda tr: tr["err"])
        selected_set = set(id(tr) for tr in valid[:N_SELECT])

        for tr in trials:
            tr["selected"] = id(tr) in selected_set

        selected = [tr for tr in trials if tr["selected"]]

        if selected:
            sel_errs = [tr["err"] for tr in selected]
            sel_cots = [tr["cot"] for tr in selected if tr["cot"] is not None]
            mean_err = float(np.mean(sel_errs))
            std_err = float(np.std(sel_errs))
            mean_cot = float(np.mean(sel_cots)) if sel_cots else None

            cot_str = f"  COT={mean_cot:.2f}" if mean_cot is not None else ""
            print(f"    >> top {N_SELECT}: err={mean_err:.1f} +/- {std_err:.1f}%{cot_str}")

            ref_summaries.append({
                "id": rid, "scene": scene, "freq": freq, "target": target,
                "mean_err": mean_err, "std_err": std_err, "mean_cot": mean_cot,
            })

            # Store replay specs for recording
            if args.record:
                for tr in selected:
                    replay_kw: dict = {}
                    if is_rough:
                        rng = np.random.default_rng(tr["seed"])
                        y_off = rng.uniform(-y_jitter, y_jitter)
                        replay_kw["spawn_offset"] = (spawn_x, y_off, spawn_z)
                    else:
                        replay_kw["init_yaw_jitter_deg"] = yaw_jitter_deg
                        replay_kw["rng_seed"] = tr["seed"]
                    replay_specs.append({
                        "rid": rid, "scene": scene, "freq": freq,
                        "trial": tr["trial"], "seed": tr["seed"],
                        "extra_kw": replay_kw,
                    })
        else:
            print(f"    >> ALL CRASHED")

        # Accumulate for CSV
        for tr in trials:
            all_trial_rows.append({
                "ref_id": rid, "scene": scene, "ctrl_freq": freq,
                "target_speed": target, "trial": tr["trial"], "rng_seed": tr["seed"],
                "jitter_type": tr["jitter_type"], "jitter_value": tr["jitter_value"],
                "vx": tr["vx"] if tr["vx"] is not None else "",
                "velocity_error_pct": tr["err"] if not tr["crash"] else "",
                "cot": tr["cot"] if tr["cot"] is not None else "",
                "crash": tr["crash"], "selected": tr["selected"],
                "min_window_vx": tr.get("min_window_vx", ""),
                "pitch_rms": tr["pitch_rms"] if tr["pitch_rms"] is not None else "",
            })

    # --- Exploratory conditions (rough only, no optimization target) ---
    exploratory_summaries: list[dict] = []
    if exploratory_conditions:
        print(f"\n  --- Exploratory (no optimization target, random selection) ---")
        # Seed offset: after all ref trials
        expl_seed_base = BASE_SEED + len(ref_rows) * N_TRIALS

        for expl_idx, expl in enumerate(exploratory_conditions):
            scene = expl["scene"]
            freq = expl["ctrl_freq"]
            rid = expl["id"]
            exp_speed = expl["exp_speed"]
            success_pct = expl["exp_success_pct"]

            if scene not in MJCF_PATHS:
                continue

            mass = scene_mass.setdefault(scene, _get_robot_mass(MJCF_PATHS[scene]))

            print(f"  {rid}  f={freq:.0f}Hz  exp={exp_speed*100:.1f}cm/s ({success_pct:.0f}% success)")
            trials: list[dict] = []

            for t in range(N_TRIALS):
                seed = expl_seed_base + expl_idx * N_TRIALS + t
                sp = dict(sim_params)
                sp["drive_freq"] = freq

                rng = np.random.default_rng(seed)
                y_offset = rng.uniform(-y_jitter, y_jitter)
                jitter_value = y_offset * 1000
                jitter_type = "y_mm"
                extra_kw = {"spawn_offset": (spawn_x, y_offset, spawn_z)}

                print(f"    trial {t+1}/{N_TRIALS} (seed={seed}, y_mm={jitter_value:+.3f}) ...",
                      end="", flush=True)

                try:
                    traj = sim_module.run_simulation(
                        sp, mjcf_path=MJCF_PATHS[scene],
                        sim_duration=SIM_DURATION, visualize=False, progress=False,
                        ignore_stuck_detection=True, **extra_kw,
                    )
                except Exception as e:
                    print(f" CRASH ({e.__class__.__name__})")
                    trials.append({"trial": t, "seed": seed, "jitter_type": jitter_type,
                                   "jitter_value": jitter_value, "vx": None,
                                   "cot": None, "crash": True, "min_window_vx": 0.0,
                                   "pitch_rms": None})
                    continue

                if traj is None:
                    print(" CRASH (None)")
                    trials.append({"trial": t, "seed": seed, "jitter_type": jitter_type,
                                   "jitter_value": jitter_value, "vx": None,
                                   "cot": None, "crash": True, "min_window_vx": 0.0,
                                   "pitch_rms": None})
                    continue

                vx = extract_velocity(traj, SETTLE_TIME)
                traj_key = f"{rid}_t{t}"
                traj_arrays[f"{traj_key}_time"] = np.array([s["time"] for s in traj])
                traj_arrays[f"{traj_key}_pos_x"] = np.array([s["pos"][0] for s in traj])
                cot = compute_cot(traj, mass, SETTLE_TIME)
                mwv = min_window_velocity(traj, freq, SETTLE_TIME)
                pitch_rms = compute_pitch_rms(traj, SETTLE_TIME)

                cot_str = f"  COT={cot:.2f}" if cot is not None else ""
                print(f" vx={vx*100:.1f}cm/s{cot_str}  mwv={mwv*100:.1f}mm/s  pitch={pitch_rms:.1f}°")

                trials.append({"trial": t, "seed": seed, "jitter_type": jitter_type,
                               "jitter_value": jitter_value, "vx": vx,
                               "cot": cot, "crash": False, "min_window_vx": mwv,
                               "pitch_rms": pitch_rms})

            # Random selection (no sorting by error)
            valid = [tr for tr in trials if not tr["crash"]]
            if len(valid) > N_SELECT:
                rng_sel = np.random.default_rng(expl_seed_base + expl_idx)
                sel_indices = rng_sel.choice(len(valid), size=N_SELECT, replace=False)
                selected = [valid[i] for i in sel_indices]
            else:
                selected = valid
            selected_set = set(id(tr) for tr in selected)

            for tr in trials:
                tr["selected"] = id(tr) in selected_set

            if selected:
                sel_vxs = [tr["vx"] * 100 for tr in selected]  # cm/s
                sel_cots = [tr["cot"] for tr in selected if tr["cot"] is not None]
                mean_vx = float(np.mean(sel_vxs))
                std_vx = float(np.std(sel_vxs))
                mean_cot = float(np.mean(sel_cots)) if sel_cots else None

                cot_str = f"  COT={mean_cot:.2f}" if mean_cot is not None else ""
                print(f"    >> random {len(selected)}: vx={mean_vx:.1f} +/- {std_vx:.1f} cm/s{cot_str}")

                exploratory_summaries.append({
                    "id": rid, "scene": scene, "freq": freq,
                    "exp_speed": exp_speed, "mean_vx": mean_vx / 100,
                    "std_vx": std_vx / 100, "mean_cot": mean_cot,
                })
            else:
                print(f"    >> ALL CRASHED")

            # Store replay specs for recording
            if args.record:
                for tr in selected:
                    rng_r = np.random.default_rng(tr["seed"])
                    y_off = rng_r.uniform(-y_jitter, y_jitter)
                    replay_specs.append({
                        "rid": rid, "scene": scene, "freq": freq,
                        "trial": tr["trial"], "seed": tr["seed"],
                        "extra_kw": {"spawn_offset": (spawn_x, y_off, spawn_z)},
                    })

            # CSV rows
            for tr in trials:
                all_trial_rows.append({
                    "ref_id": rid, "scene": scene, "ctrl_freq": freq,
                    "target_speed": "", "trial": tr["trial"], "rng_seed": tr["seed"],
                    "jitter_type": tr["jitter_type"], "jitter_value": tr["jitter_value"],
                    "vx": tr["vx"] if tr["vx"] is not None else "",
                    "velocity_error_pct": "",
                    "cot": tr["cot"] if tr["cot"] is not None else "",
                    "crash": tr["crash"], "selected": tr["selected"],
                    "min_window_vx": tr.get("min_window_vx", ""),
                    "pitch_rms": tr["pitch_rms"] if tr["pitch_rms"] is not None else "",
                })

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"Summary: {args.run_dir.name} ({terrain})")
    print(f"{'='*70}")

    if ref_summaries:
        all_errs = [r["mean_err"] for r in ref_summaries]
        all_cots = [r["mean_cot"] for r in ref_summaries if r["mean_cot"] is not None]
        worst = ref_summaries[int(np.argmax(all_errs))]

        print(f"  Refs evaluated:      {len(ref_summaries)}")
        print(f"  Mean velocity error: {np.mean(all_errs):.1f}%")
        print(f"  Max velocity error:  {max(all_errs):.1f}% ({worst['id']})")
        if all_cots:
            print(f"  Mean COT:            {np.mean(all_cots):.2f}")
            print(f"  COT range:           {min(all_cots):.2f} - {max(all_cots):.2f}")

    if failure_mode_results:
        n_pass = sum(1 for fm in failure_mode_results if fm["result"] == "PASS")
        print(f"\n  Failure mode refs:   {n_pass}/{len(failure_mode_results)} passed")
        for fm in failure_mode_results:
            vx_str = f"vx={fm['vx']*100:.2f}cm/s" if fm["vx"] is not None else "CRASH"
            print(f"    {fm['id']}: {fm['result']}  ({vx_str})")

    if exploratory_summaries:
        print(f"\n  Exploratory (no target, random selection):")
        for es in exploratory_summaries:
            cot_str = f"  COT={es['mean_cot']:.2f}" if es["mean_cot"] is not None else ""
            print(f"    {es['id']}: vx={es['mean_vx']*100:.1f}cm/s  "
                  f"(exp={es['exp_speed']*100:.1f}cm/s){cot_str}")

    # --- Video recording pass ---
    if args.record and replay_specs:
        video_dir = pathlib.Path("videos") / terrain
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Recording {len(replay_specs)} selected trials to {video_dir}/")
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
    if args.csv and traj_arrays:
        npz_path = args.run_dir / "validation_trajectories.npz"
        np.savez_compressed(npz_path, **traj_arrays)
        print(f"\n  Trajectories: {npz_path} ({len(traj_arrays)//2} trials)")

    # --- CSV output ---
    if args.csv and all_trial_rows:
        csv_path = args.run_dir / "validation_trials.csv"
        fieldnames = [
            "ref_id", "scene", "ctrl_freq", "target_speed", "trial",
            "rng_seed", "jitter_type", "jitter_value", "vx",
            "velocity_error_pct", "cot", "crash", "selected", "min_window_vx",
            "pitch_rms",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_trial_rows)
        print(f"\n  CSV: {csv_path}")


if __name__ == "__main__":
    main()
