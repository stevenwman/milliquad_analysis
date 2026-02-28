"""Record videos of all 15 flat references using RK4 integrator.

Injects <option integrator="RK4"/> into temporary MJCF copies,
runs simulation with flat_10_30_50 best params, records MP4s.

Usage:
    uv run python record_rk4_refs.py results/20260225T122342_flat_10_30_50
    uv run python record_rk4_refs.py results/20260225T122342_flat_10_30_50 --scenes scene1 scene4
    uv run python record_rk4_refs.py results/20260225T122342_flat_10_30_50 --freqs 10 30
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import xml.etree.ElementTree as ET

from config_new import (
    MJCF_PATHS,
    REFERENCE_DATA,
    SIM_DURATION,
    sim_params_from_point,
    space,
)
import simulation_fast_new as sim

PARAM_NAMES = [dim.name for dim in space]


def load_best_point(run_dir: pathlib.Path) -> list[float]:
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    if not rows:
        sys.exit(f"ERROR: no rows in {bests_csv}")
    best_id = rows[-1]["id"]
    multi_csv = run_dir / "multi_optimization_results.csv"
    if multi_csv.exists():
        for row in csv.DictReader(open(multi_csv)):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    return [float(rows[-1][name]) for name in PARAM_NAMES]


def make_rk4_xml(xml_path: str) -> str:
    """Create a temporary MJCF copy with integrator="RK4"."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("integrator", "RK4")
    # Also enable energy tracking
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("energy", "enable")
    out_path = xml_path.replace(".xml", "_rk4_tmp.xml")
    tree.write(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--freqs", nargs="+", type=int, default=None)
    parser.add_argument("--duration", type=float, default=SIM_DURATION)
    args = parser.parse_args()

    point = load_best_point(args.run_dir)
    sim_params = sim_params_from_point(point)
    print(f"Loaded best params from {args.run_dir}")

    # Build reference list
    refs = []
    for row in REFERENCE_DATA:
        scene = row["scene"]
        freq = int(row["ctrl_freq"])
        if args.scenes and scene not in args.scenes:
            continue
        if args.freqs and freq not in args.freqs:
            continue
        refs.append({"scene": scene, "freq": freq, "target": row["speed"],
                      "id": f"{scene}_f{freq}"})

    # Generate RK4 XMLs (one per scene)
    rk4_paths: dict[str, str] = {}
    for scene, xml_path in MJCF_PATHS.items():
        rk4_paths[scene] = make_rk4_xml(xml_path)

    out_dir = args.run_dir / "rk4_videos"
    out_dir.mkdir(exist_ok=True)

    try:
        for ref in refs:
            sp = dict(sim_params)
            sp["drive_freq"] = float(ref["freq"])
            video_path = out_dir / f"{ref['id']}_rk4.mp4"
            print(f"\n{ref['id']} (target={ref['target']*1000:.1f} mm/s) -> {video_path}")

            traj = sim.run_simulation(
                sp,
                mjcf_path=rk4_paths[ref["scene"]],
                sim_duration=args.duration,
                record_path=str(video_path),
            )
            if traj is None:
                print(f"  FAILED")
                continue

            # Report velocity
            settle_idx = next(i for i, s in enumerate(traj) if s["time"] >= 0.1)
            vx = (traj[-1]["pos"][0] - traj[settle_idx]["pos"][0]) / (traj[-1]["time"] - traj[settle_idx]["time"])
            err = abs(vx - ref["target"]) / ref["target"] * 100
            print(f"  vx={vx*1000:.1f} mm/s  target={ref['target']*1000:.1f}  err={err:.1f}%")

    finally:
        # Cleanup temp XMLs
        for p in rk4_paths.values():
            path = pathlib.Path(p)
            if path.exists():
                path.unlink()
        print("\nCleaned up temp RK4 XMLs")

    print(f"\nVideos saved to: {out_dir}")


if __name__ == "__main__":
    main()
