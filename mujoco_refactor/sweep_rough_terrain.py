#!/usr/bin/env python3
"""Sweep rough terrain: run multiple morphology/freq combos with Y jitter, record videos.

Usage:
    uv run python sweep_rough_terrain.py results/20260225T225248_step_argmin_progress
    uv run python sweep_rough_terrain.py results/XXXXX --height-std 0.001 --freqs 10 30 50
    uv run python sweep_rough_terrain.py results/XXXXX --n-trials 20 --n-record 3 --y-jitter 0.01125
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import tempfile

import numpy as np

import eval_rough_terrain as ert
import simulation_fast_new as sim_module
from config_new import MJCF_PATHS, SIM_DURATION, SIMULATION_TIMEOUT, sim_params_from_point
from optimizer_new import calculate_cost


# Ranking cost weights: reward forward velocity, penalize tumble + yaw, ignore lateral
RANK_VEL_WEIGHT = 5.0
RANK_TUMBLE_WEIGHT = 1.0
RANK_YAW_WEIGHT = 1.0


def rank_cost(trial: dict) -> float:
    """Lower = better. Rewards forward velocity, penalizes tumble + yaw."""
    if trial.get("fail"):
        return 1e6
    # vel is in mm/s, convert to m/s for comparable scale with penalties
    return (-RANK_VEL_WEIGHT * trial["vel"] / 1000.0
            + RANK_TUMBLE_WEIGHT * trial["tmb"]
            + RANK_YAW_WEIGHT * trial["yaw_pen"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=str, help="Results directory with optimization CSVs")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Morphologies to test (default: all)")
    parser.add_argument("--freqs", nargs="+", type=float, default=[10.0, 50.0],
                        help="Frequencies to test (default: 10 50)")
    parser.add_argument("--height-std", type=float, default=0.00025,
                        help="Terrain height std in meters (default: 0.25mm)")
    parser.add_argument("--y-jitter", type=float, default=0.005,
                        help="Max Y jitter in meters (default: ±5mm)")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Number of Y-jitter trials per config (default: 1)")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--no-record", action="store_true", help="Skip video recording")
    parser.add_argument("--n-record", type=int, default=1,
                        help="Number of top trials to record per config (default: 1)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for videos + CSV (default: run_dir/rough_terrain_videos)")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        return

    ert.TERRAIN_HEIGHT_MEAN = args.height_std * 2
    ert.TERRAIN_HEIGHT_STD = args.height_std

    # Clamp y-jitter to hfield half-width minus robot margin (~3mm)
    y_half = ert.TERRAIN_NY * ert.TERRAIN_SL / 2.0
    max_y = y_half - 0.003  # 3mm margin for robot body
    if args.y_jitter > max_y:
        print(f"WARNING: y-jitter {args.y_jitter*1000:.1f}mm > max safe {max_y*1000:.1f}mm, clamping")
        args.y_jitter = max_y

    scenes = args.scenes or list(MJCF_PATHS.keys())
    combos = [(s, f) for s in scenes for f in args.freqs]

    point = ert.load_best_point(run_dir)
    sim_params = sim_params_from_point(point)

    rng = np.random.default_rng(args.seed)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else run_dir / "rough_terrain_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    h_mm = args.height_std * 1000
    print(f"Terrain: std={h_mm:.2f}mm, mean={h_mm*2:.2f}mm, y_jitter=±{args.y_jitter*1000:.1f}mm")
    print(f"Params from: {run_dir}")
    print(f"{len(combos)} combos × {args.n_trials} trials, record={'no' if args.no_record else f'top {args.n_record}'}")
    print(f"Rank cost: vel_w={RANK_VEL_WEIGHT}, tmb_w={RANK_TUMBLE_WEIGHT}, yaw_w={RANK_YAW_WEIGHT}\n")

    # All trial data for CSV export
    all_trials = []
    results = []

    with tempfile.TemporaryDirectory(prefix="sweep_rough_") as tmp_dir:
        # ── Phase 1: run all trials ──
        combo_trials = {}  # label -> list of trial dicts
        for scene, freq in combos:
            label = f"{scene}_f{int(freq)}"
            sp = dict(sim_params)
            sp["drive_freq"] = freq

            print(f"{label} ({args.n_trials} trials):")
            trials = []
            for t in range(args.n_trials):
                y_off = rng.uniform(-args.y_jitter, args.y_jitter)
                ert._y_offset = y_off

                mjcf = ert.inject_tiled_rough(MJCF_PATHS[scene], tmp_dir)
                traj = sim_module.run_simulation(
                    sp, mjcf_path=mjcf,
                    sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
                    ignore_stuck_detection=True,
                )
                ert.cleanup_temp_xmls()

                if traj is None:
                    print(f"  trial {t:>2}: y={y_off*1000:+6.1f}mm  FAIL")
                    trial_data = {"fail": True, "y_off": y_off, "trial_idx": t,
                                  "scene": scene, "freq": freq, "label": label}
                    trials.append(trial_data)
                else:
                    cd = calculate_cost(traj, target_velocity=1.0, verbose=False)
                    v = cd["avg_forward_velocity"] * 1000
                    lat = cd["lateral_displacement"] * 100
                    yaw_deg = cd["yaw_deviation_deg"]
                    tmb = cd["tumble_penalty"]
                    # Compute yaw penalty (same thresholded formula as optimizer)
                    from config_new import YAW_THRESHOLD_DEG
                    yaw_pen = 0.0
                    if yaw_deg > YAW_THRESHOLD_DEG:
                        excess = yaw_deg - YAW_THRESHOLD_DEG
                        yaw_pen = (excess / 90.0) ** 2

                    trial_data = {
                        "fail": False, "vel": v, "lat": lat, "yaw_deg": yaw_deg,
                        "yaw_pen": yaw_pen, "tmb": tmb, "y_off": y_off,
                        "trial_idx": t, "scene": scene, "freq": freq, "label": label,
                    }
                    trial_data["rank_cost"] = rank_cost(trial_data)
                    print(f"  trial {t:>2}: y={y_off*1000:+6.1f}mm  vel={v:>6.1f}mm/s  "
                          f"tmb={tmb:.3f}  yaw={yaw_deg:>5.1f}°  cost={trial_data['rank_cost']:>7.3f}")
                    trials.append(trial_data)

                all_trials.append(trial_data)

            # Rank: lowest tumble first, break ties by highest velocity
            valid = [x for x in trials if not x["fail"]]
            if valid:
                ranked = sorted(valid, key=lambda x: (x["tmb"], -x["vel"]))
                print(f"  -> recording {min(args.n_record, len(ranked))} best:")
                for ri, r in enumerate(ranked[:args.n_record]):
                    ti = trials.index(r)
                    print(f"     #{ri+1} trial {ti}: {r['vel']:.1f} mm/s, tmb={r['tmb']:.4f}, yaw={r['yaw_deg']:.1f}°")
                results.append({"id": label, **ranked[0]})
            else:
                ranked = []
                print(f"  -> ALL FAILED")
                results.append({"id": label, "fail": True})

            combo_trials[label] = (trials, ranked, scene, freq)
            print()

        # ── Phase 2: record top-N videos ──
        if not args.no_record:
            print(f"Recording top {args.n_record} per config ...\n")
            for scene, freq in combos:
                label = f"{scene}_f{int(freq)}"
                trials, ranked, _, _ = combo_trials[label]
                sp = dict(sim_params)
                sp["drive_freq"] = freq

                to_record = ranked[:args.n_record] if ranked else [trials[0]]
                for rank_i, rec_trial in enumerate(to_record):
                    ert._y_offset = rec_trial["y_off"]
                    mjcf = ert.inject_tiled_rough(MJCF_PATHS[scene], tmp_dir)
                    suffix = f"_t{rank_i}" if args.n_record > 1 else ""
                    vid_path = str(out_dir / f"{label}_rough_{h_mm:.2f}mm{suffix}.mp4")
                    sim_module.run_simulation(
                        sp, mjcf_path=mjcf,
                        sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
                        ignore_stuck_detection=True,
                        record_path=vid_path,
                    )
                    ert.cleanup_temp_xmls()
                    if rec_trial.get("fail"):
                        print(f"  {vid_path} (FAIL)")
                    else:
                        print(f"  {vid_path} ({rec_trial['vel']:.1f}mm/s, cost={rec_trial['rank_cost']:.3f})")

    ert._y_offset = 0.0

    # ── Save CSV ──
    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "freq", "trial_idx", "y_off_mm", "fail",
                         "vel_mm_s", "lat_cm", "yaw_deg", "yaw_pen", "tumble", "rank_cost"])
        for t in all_trials:
            if t.get("fail"):
                writer.writerow([t["scene"], t["freq"], t["trial_idx"],
                                 repr(t['y_off']*1000), "TRUE",
                                 "", "", "", "", "", ""])
            else:
                writer.writerow([t["scene"], t["freq"], t["trial_idx"],
                                 repr(t['y_off']*1000), "FALSE",
                                 f"{t['vel']:.2f}", f"{t['lat']:.3f}",
                                 f"{t['yaw_deg']:.1f}", f"{t['yaw_pen']:.4f}",
                                 f"{t['tmb']:.4f}", f"{t['rank_cost']:.4f}"])
    print(f"\nSaved: {csv_path}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  {'config':<25} {'vel(mm/s)':>9} {'tmb':>6} {'yaw°':>6} {'cost':>8}  {'n_ok/n':>6}")
    print(f"  {'-'*65}")
    for scene, freq in combos:
        label = f"{scene}_f{int(freq)}"
        trials, ranked, _, _ = combo_trials[label]
        n_ok = len([x for x in trials if not x.get("fail")])
        if ranked:
            b = ranked[0]
            print(f"  {label:<25} {b['vel']:>8.1f} {b['tmb']:>6.3f} {b['yaw_deg']:>5.1f}° {b['rank_cost']:>8.3f}  {n_ok:>2}/{len(trials)}")
        else:
            print(f"  {label:<25} {'FAIL':>9} {'':>6} {'':>6} {'':>8}  {n_ok:>2}/{len(trials)}")
    print()


if __name__ == "__main__":
    main()
