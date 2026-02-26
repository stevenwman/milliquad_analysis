#!/usr/bin/env python3
"""Sweep rough terrain: run multiple morphology/freq combos with Y jitter, record videos.

Usage:
    uv run python sweep_rough_terrain.py results/20260225T225248_step_argmin_progress
    uv run python sweep_rough_terrain.py results/XXXXX --height-std 0.0005 --freqs 10 30 50
    uv run python sweep_rough_terrain.py results/XXXXX --scenes scene4 scene_wheel --no-record
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile

import numpy as np

import eval_rough_terrain as ert
import simulation_fast_new as sim_module
from config_new import MJCF_PATHS, SIM_DURATION, SIMULATION_TIMEOUT, sim_params_from_point
from optimizer_new import calculate_cost


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
    vid_dir = run_dir / "rough_terrain_videos"
    vid_dir.mkdir(exist_ok=True)

    h_mm = args.height_std * 1000
    print(f"Terrain: std={h_mm:.2f}mm, mean={h_mm*2:.2f}mm, y_jitter=±{args.y_jitter*1000:.1f}mm")
    print(f"Params from: {run_dir}")
    print(f"{len(combos)} combos × {args.n_trials} trials, record={'no' if args.no_record else 'yes'}\n")

    results = []

    with tempfile.TemporaryDirectory(prefix="sweep_rough_") as tmp_dir:
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
                    print(f"  trial {t}: y={y_off*1000:+.1f}mm  FAIL")
                    trials.append({"fail": True, "y_off": y_off})
                else:
                    cd = calculate_cost(traj, target_velocity=1.0, verbose=False)
                    v = cd["avg_forward_velocity"] * 1000
                    lat = cd["lateral_displacement"] * 100
                    yaw = cd["yaw_deviation_deg"]
                    tmb = cd["tumble_penalty"]
                    print(f"  trial {t}: y={y_off*1000:+.1f}mm  vel={v:>6.1f}mm/s  lat={lat:.2f}cm  yaw={yaw:.1f}°  tmb={tmb:.3f}")
                    trials.append({"fail": False, "vel": v, "lat": lat, "yaw": yaw, "tmb": tmb,
                                   "y_off": y_off, "seed_idx": t})

            valid = [x for x in trials if not x["fail"]]
            if valid:
                best = max(valid, key=lambda x: x["vel"])
                bi = trials.index(best)
                print(f"  -> BEST trial {bi}: {best['vel']:.1f} mm/s (y={best['y_off']*1000:+.1f}mm)")
                results.append({"id": label, **best})

                # Record best trial video
                if not args.no_record:
                    ert._y_offset = best["y_off"]
                    mjcf = ert.inject_tiled_rough(MJCF_PATHS[scene], tmp_dir)
                    vid_path = str(vid_dir / f"{label}_rough_{h_mm:.2f}mm.mp4")
                    sim_module.run_simulation(
                        sp, mjcf_path=mjcf,
                        sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
                        ignore_stuck_detection=True,
                        record_path=vid_path,
                    )
                    ert.cleanup_temp_xmls()
                    print(f"  Saved: {vid_path}")
            else:
                print(f"  -> ALL FAILED")
                results.append({"id": label, "fail": True})
            print()

    ert._y_offset = 0.0

    # Summary
    print(f"{'='*70}")
    print(f"  {'config':<25} {'vel(mm/s)':>9} {'lat(cm)':>7} {'yaw':>5} {'tmb':>5}")
    print(f"  {'-'*55}")
    for r in results:
        if r.get("fail"):
            print(f"  {r['id']:<25} {'FAIL':>9}")
        else:
            print(f"  {r['id']:<25} {r['vel']:>8.1f} {r['lat']:>7.2f} {r['yaw']:>5.1f} {r['tmb']:>5.3f}")
    print()


if __name__ == "__main__":
    main()
