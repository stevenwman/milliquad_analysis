#!/usr/bin/env python3
"""Record videos for the best result in a completed optimization run.

Usage:
    uv run python record_best.py --terrain step results/20260228T011613_rk4_step/
"""
import argparse
import csv
import importlib
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import simulation  # noqa: E402 — must come after MUJOCO_GL is set
from config import space, sim_params_from_point, reference_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", required=True, choices=["flat", "step", "rough"])
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("--top", type=int, default=1, help="Record top-N results")
    args = parser.parse_args()

    config_mod = importlib.import_module(f"config_{args.terrain}")
    ref_rows = list(reference_rows(config_mod.REFERENCE_DATA))
    mjcf_paths = config_mod.MJCF_PATHS
    sim_duration = config_mod.SIM_DURATION

    multi_csv = args.results_dir / "multi_optimization_results.csv"
    if not multi_csv.exists():
        sys.exit(f"Not found: {multi_csv}")

    rows = list(csv.DictReader(open(multi_csv)))
    sorted_rows = sorted(rows, key=lambda r: float(r["cost"]))
    top_rows = sorted_rows[: args.top]

    for rank, row in enumerate(top_rows, start=1):
        cost = float(row["cost"])
        rid = row["id"]
        print(f"\n#{rank}: cost={cost:.6f}  id={rid}")

        point = [float(row[dim.name]) for dim in space]
        sim_params = sim_params_from_point(point)

        for ref_row in ref_rows:
            scene = ref_row["scene"]
            ref_id = ref_row["id"]
            video_path = args.results_dir / f"rank_{rank:02d}_{ref_id}.mp4"
            print(f"  Recording {ref_id} → {video_path}...")

            sim_params_scene = dict(sim_params)
            sim_params_scene["drive_freq"] = ref_row.get("ctrl_freq", 10.0)

            extra_kwargs = {}
            if args.terrain == "rough":
                from config_rough import SPAWN_X, SPAWN_Z_RAISE  # noqa: F401
                extra_kwargs["spawn_offset"] = (SPAWN_X, 0.0, SPAWN_Z_RAISE)

            simulation.run_simulation(
                sim_params_scene,
                mjcf_path=mjcf_paths[scene],
                sim_duration=sim_duration + 2.0,
                record_path=str(video_path),
                **extra_kwargs,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
