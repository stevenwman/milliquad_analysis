#!/usr/bin/env python3
"""Run N jitter trials per reference, pick closest velocity to target, optionally record.

Evaluates step-optimized params on BOTH flat and step terrain references.
For each (scene, freq, terrain) triple, runs N trials with yaw jitter,
picks the trial whose simulated velocity is closest to the experimental
target, and optionally records that trial as video.

Usage:
    uv run python eval_best_trial.py results/20260225T... --record
    uv run python eval_best_trial.py results/20260225T... --scenes scene4 --freqs 30 --n-trials 5
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

from config_new import (
    MJCF_PATHS,
    SETTLE_TIME,
    SIMULATION_TIMEOUT,
    sim_params_from_point,
    space,
)
from config_new import REFERENCE_DATA as FLAT_REFS
from config_new import SIM_DURATION as FLAT_DURATION
from config_step import (
    REFERENCE_DATA as STEP_REFS,
    SIM_DURATION as STEP_DURATION,
    STEP_END_X,
    STEP_PRESET,
    STEP_START_X,
)

import simulation_fast_new as sim_module

PARAM_NAMES = [dim.name for dim in space]
JITTER_DEG = 2.0
BASE_SEED = 77777
N_TRIALS_FLAT = 3
N_TRIALS_STEP = 10


# ---------------------------------------------------------------------------
# Param loading
# ---------------------------------------------------------------------------

def load_best_point(run_dir: pathlib.Path) -> list[float]:
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    if not rows:
        sys.exit(f"ERROR: no rows in {bests_csv}")
    best_id = rows[-1]["id"]
    with open(run_dir / "multi_optimization_results.csv") as f:
        for row in csv.DictReader(f):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    raise ValueError(f"id {best_id!r} not found")


# ---------------------------------------------------------------------------
# Velocity extraction (matches optimizer conventions)
# ---------------------------------------------------------------------------

def extract_flat_velocity(traj: list[dict]) -> float:
    """Average forward velocity after settle time (same as optimizer_new)."""
    start = None
    for s in traj:
        if s["time"] >= SETTLE_TIME:
            start = s
            break
    if start is None:
        return 0.0
    end = traj[-1]
    dt = end["time"] - start["time"]
    if dt < 1e-6:
        return 0.0
    return (end["pos"][0] - start["pos"][0]) / dt


def extract_step_velocity(traj: list[dict]) -> float:
    """Average forward velocity in step region (same as optimizer_step)."""
    enter = None
    for s in traj:
        if s["pos"][0] >= STEP_START_X:
            enter = s
            break
    if enter is None:
        return 0.0
    cutoff = STEP_START_X + 0.9 * (STEP_END_X - STEP_START_X)
    exit_s = enter
    for s in traj:
        if s["pos"][0] > cutoff:
            break
        exit_s = s
    dt = exit_s["time"] - enter["time"]
    if dt < 1e-6:
        return 0.0
    return (exit_s["pos"][0] - enter["pos"][0]) / dt


# ---------------------------------------------------------------------------
# Step XML generation (matches optimizer_step._inject_steps)
# ---------------------------------------------------------------------------

_temp_xmls: list[str] = []


def inject_steps(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    wb = tree.getroot().find("worldbody")
    p = STEP_PRESET
    for i in range(p["step_count"]):
        is_final = (i == p["step_count"] - 1)
        length = p["final_step_length"] if is_final else p["step_length"]
        if is_final:
            pos_x = p["flat_lead"] + (p["step_count"] - 1) * p["step_length"] + length / 2.0
        else:
            pos_x = p["flat_lead"] + i * p["step_length"] + length / 2.0
        pos_z = (i + 1) * p["step_height"] - p["step_height"] / 2.0
        g = ET.SubElement(wb, "geom")
        g.set("name", f"step_{i}")
        g.set("type", "box")
        g.set("size", f"{length / 2.0} {p['step_width'] / 2.0} {p['step_height'] / 2.0}")
        g.set("pos", f"{pos_x} 0.0 {pos_z}")
        g.set("rgba", "0.5 0.5 0.5 1")
    src_dir = pathlib.Path(xml_path).parent
    out = str(src_dir / f"{pathlib.Path(xml_path).stem}_step_eval_tmp.xml")
    tree.write(out)
    _temp_xmls.append(out)
    return out


def cleanup():
    for p in _temp_xmls:
        try:
            pathlib.Path(p).unlink(missing_ok=True)
        except OSError:
            pass
    _temp_xmls.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=str, help="Results directory with optimization CSVs")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--freqs", nargs="+", type=float, default=None)
    parser.add_argument("--n-trials-flat", type=int, default=N_TRIALS_FLAT)
    parser.add_argument("--n-trials-step", type=int, default=N_TRIALS_STEP)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--terrain", choices=["flat", "step", "both"], default="both")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"ERROR: {run_dir} is not a directory")

    point = load_best_point(run_dir)
    sim_params = sim_params_from_point(point)

    # Build reference list: flat + step
    refs: list[dict] = []
    if args.terrain in ("flat", "both"):
        for row in FLAT_REFS:
            s, f = row["scene"], float(row.get("ctrl_freq", 30.0))
            if args.scenes and s not in args.scenes:
                continue
            if args.freqs and f not in args.freqs:
                continue
            refs.append({"id": f"{s}_f{int(f)}_flat", "scene": s, "freq": f,
                          "target": float(row["speed"]), "terrain": "flat"})

    if args.terrain in ("step", "both"):
        for row in STEP_REFS:
            s, f = row["scene"], float(row.get("ctrl_freq", 30.0))
            if args.scenes and s not in args.scenes:
                continue
            if args.freqs and f not in args.freqs:
                continue
            refs.append({"id": f"{s}_f{int(f)}_step", "scene": s, "freq": f,
                          "target": float(row["speed"]), "terrain": "step"})

    # Generate step XMLs (one per scene)
    step_xmls: dict[str, str] = {}
    for scene, xml_path in MJCF_PATHS.items():
        step_xmls[scene] = inject_steps(xml_path)

    print(f"Loaded params from {run_dir}")
    print(f"  {len(refs)} references, flat={args.n_trials_flat} step={args.n_trials_step} trials")
    print()

    try:
        results: dict[str, dict] = {}
        for ri, ref in enumerate(refs):
            sp = dict(sim_params)
            sp["drive_freq"] = ref["freq"]
            is_step = ref["terrain"] == "step"
            mjcf = step_xmls[ref["scene"]] if is_step else MJCF_PATHS[ref["scene"]]
            duration = STEP_DURATION if is_step else FLAT_DURATION
            extract = extract_step_velocity if is_step else extract_flat_velocity
            n_trials = args.n_trials_step if is_step else args.n_trials_flat

            print(f"{ref['id']} (target={ref['target'] * 1000:.1f} mm/s, {n_trials} trials)")
            trials = []
            for t in range(n_trials):
                seed = BASE_SEED + ri * 100 + t
                traj = sim_module.run_simulation(
                    sp, mjcf_path=mjcf, sim_duration=duration,
                    wall_timeout=SIMULATION_TIMEOUT,
                    init_yaw_jitter_deg=JITTER_DEG, rng_seed=seed,
                )
                if traj is None:
                    trials.append({"vel": 0.0, "delta": abs(ref["target"]),
                                   "seed": seed, "fail": True})
                    print(f"  trial {t}: FAIL")
                    continue
                vel = extract(traj)
                delta = abs(vel - ref["target"])
                trials.append({"vel": vel, "delta": delta, "seed": seed, "fail": False})
                print(f"  trial {t}: {vel * 1000:>7.1f} mm/s  delta={delta * 1000:>5.1f}")

            valid = [x for x in trials if not x["fail"]]
            if valid:
                best = min(valid, key=lambda x: x["delta"])
                bi = trials.index(best)
                print(f"  -> BEST trial {bi}: {best['vel'] * 1000:.1f} mm/s, "
                      f"delta={best['delta'] * 1000:.1f} mm/s")
            else:
                best = None
                print(f"  -> ALL FAILED")
            results[ref["id"]] = {"ref": ref, "best": best}
            print()

        # --- Summary table ---
        print("=" * 85)
        print(f"  {'ref':<25} {'target':>8} {'best':>8} {'delta':>8} {'%err':>6} {'terrain':>7}")
        print("  " + "-" * 83)
        for ref in refs:
            r = results[ref["id"]]
            tgt = ref["target"] * 1000
            if r["best"]:
                v = r["best"]["vel"] * 1000
                d = r["best"]["delta"] * 1000
                pct = (r["best"]["delta"] / ref["target"] * 100) if ref["target"] > 1e-6 else 0
                print(f"  {ref['id']:<25} {tgt:>7.1f} {v:>7.1f} {d:>7.1f} {pct:>5.1f}%  {ref['terrain']:>5}")
            else:
                print(f"  {ref['id']:<25} {tgt:>7.1f} {'FAIL':>8}{'':>8}{'':>6}  {ref['terrain']:>5}")
        print()

        # --- Save results CSV ---
        csv_path = run_dir / "validation_best_trials.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ref_id", "scene", "freq", "terrain", "target_velocity",
                         "best_velocity", "delta", "seed"])
            for ref in refs:
                r = results[ref["id"]]
                if r["best"]:
                    w.writerow([ref["id"], ref["scene"], ref["freq"], ref["terrain"],
                                f"{ref['target']:.10f}",
                                f"{r['best']['vel']:.10f}",
                                f"{r['best']['delta']:.10f}",
                                r["best"]["seed"]])
                else:
                    w.writerow([ref["id"], ref["scene"], ref["freq"], ref["terrain"],
                                f"{ref['target']:.10f}", "", "", ""])
        print(f"Saved: {csv_path}\n")

        # --- Record best trial videos ---
        if args.record:
            vid_dir = run_dir / "validation_videos"
            vid_dir.mkdir(exist_ok=True)
            print(f"Recording to {vid_dir} ...")
            for ref in refs:
                r = results[ref["id"]]
                if r["best"] is None:
                    continue
                sp = dict(sim_params)
                sp["drive_freq"] = ref["freq"]
                is_step = ref["terrain"] == "step"
                mjcf = step_xmls[ref["scene"]] if is_step else MJCF_PATHS[ref["scene"]]
                duration = STEP_DURATION if is_step else FLAT_DURATION
                vid_path = str(vid_dir / f"{ref['id']}.mp4")
                sim_module.run_simulation(
                    sp, mjcf_path=mjcf, sim_duration=duration,
                    wall_timeout=SIMULATION_TIMEOUT,
                    init_yaw_jitter_deg=JITTER_DEG, rng_seed=r["best"]["seed"],
                    record_path=vid_path,
                )
                print(f"  Saved: {vid_path}")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
