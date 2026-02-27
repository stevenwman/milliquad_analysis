#!/usr/bin/env python3
"""Evaluate Cost of Transport (COT) on flat terrain for optimized params.

Runs N yaw-jitter trials per flat reference, selects top-K by closest
velocity to experimental target, computes COT for each selected trial.

COT = E_ext / (m * g * d)
  E_ext = integral(|P_ext|) dt   (absolute power from external drive field)
  P_ext = sum_legs(tau_ext . omega)
  d = cumulative 2D path length (not straight-line)

Usage:
    uv run python eval_cot.py results/20260225T122342_flat_10_30_50
    uv run python eval_cot.py results/20260225T225248_step_argmin_progress
    uv run python eval_cot.py results/XXXXX --scenes scene4 --freqs 30 --n-trials 10 --top-k 3
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import mujoco
import numpy as np

from config_new import (
    MJCF_PATHS,
    REFERENCE_DATA,
    SETTLE_TIME,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    sim_params_from_point,
    space,
)
import simulation_fast_new as sim_module

PARAM_NAMES = [dim.name for dim in space]


# ---------------------------------------------------------------------------
# COT computation (from visualize_rollout.py — inlined to avoid import chain)
# ---------------------------------------------------------------------------

def compute_locomotion_metrics(
    trajectory: list[dict],
    robot_mass: float,
    g: float = 9.81,
) -> dict[str, float] | None:
    """Compute energy, average power, and cost of transport from a trajectory.

    COT = E_ext / (m g d), where E_ext = integral(|P_ext|) dt.
    """
    start_idx = 0
    for i, state in enumerate(trajectory):
        if state["time"] >= SETTLE_TIME:
            start_idx = i
            break

    active = trajectory[start_idx:]
    if len(active) < 2:
        return None

    if "tau_ext" not in active[0] or "omega" not in active[0]:
        return None

    n = len(active) - 1
    dt = np.empty(n)
    power_ext = np.empty(n)
    dist_increments = np.empty(n)

    for i in range(n):
        s = active[i]
        dt[i] = active[i + 1]["time"] - s["time"]
        tau_ext = s["tau_ext"]
        omega = s["omega"]
        power_ext[i] = sum(np.dot(tau_ext[j], omega[j]) for j in range(4))
        p1 = active[i]["pos"][:2]
        p2 = active[i + 1]["pos"][:2]
        dist_increments[i] = np.linalg.norm(p2 - p1)

    total_time = active[-1]["time"] - active[0]["time"]
    total_distance = dist_increments.sum()
    energy_ext = np.sum(np.abs(power_ext) * dt)
    avg_power_ext = np.sum(power_ext * dt) / total_time
    mgd = robot_mass * g * total_distance
    cot = energy_ext / mgd if mgd > 1e-12 else float("inf")

    return {
        "total_time": total_time,
        "total_distance": total_distance,
        "energy_ext": energy_ext,
        "avg_power_ext": avg_power_ext,
        "cot": cot,
        "robot_mass": robot_mass,
    }
JITTER_DEG = 2.0
BASE_SEED = 77777
N_TRIALS = 10
TOP_K = 3

# Scene name → display label (matches experimental_data/plot_velocity_vs_freq.py)
SCENE_LABELS = {"scene1": "1-leg", "scene2": "2-leg", "scene4": "4-leg", "scene_wheel": "wheel"}


# ---------------------------------------------------------------------------
# Param loading (same pattern as eval_best_trial.py)
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
# Velocity extraction (same as eval_best_trial.py / optimizer_new.py)
# ---------------------------------------------------------------------------

def extract_flat_velocity(traj: list[dict]) -> float:
    """Average forward velocity after settle time."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=str, help="Results directory with optimization CSVs")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--freqs", nargs="+", type=float, default=None)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--jitter-deg", type=float, default=JITTER_DEG)
    parser.add_argument("--record", action="store_true", help="Record video of single best trial per ref")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"ERROR: {run_dir} is not a directory")

    point = load_best_point(run_dir)
    sim_params = sim_params_from_point(point)

    # Build reference list from REFERENCE_DATA (flat only)
    refs: list[dict] = []
    for row in REFERENCE_DATA:
        s = row["scene"]
        f = float(row.get("ctrl_freq", 30.0))
        if args.scenes and s not in args.scenes:
            continue
        if args.freqs and f not in args.freqs:
            continue
        refs.append({
            "scene": s,
            "freq": f,
            "target": float(row["speed"]),
            "speed_std": float(row.get("speed_std", 0.0)),
            "id": f"{s}_f{int(f)}",
        })

    # Load robot mass per scene
    robot_masses: dict[str, float] = {}
    for scene, xml_path in MJCF_PATHS.items():
        model = mujoco.MjModel.from_xml_path(xml_path)
        robot_masses[scene] = float(sum(model.body_mass))

    print(f"Params from: {run_dir}")
    print(f"  {len(refs)} flat references, {args.n_trials} trials each, top-{args.top_k} selection")
    print(f"  Jitter: +/-{args.jitter_deg} deg, base seed: {BASE_SEED}")
    print(f"  Sim: {SIM_DURATION}s, settle: {SETTLE_TIME}s")
    print()

    # Per-trial CSV rows
    csv_rows: list[dict] = []

    # Per-reference summaries
    summaries: list[dict] = []

    for ri, ref in enumerate(refs):
        sp = dict(sim_params)
        sp["drive_freq"] = ref["freq"]
        mass = robot_masses[ref["scene"]]

        print(f"{ref['id']} (target={ref['target'] * 1000:.1f} mm/s, mass={mass * 1e6:.1f} mg, {args.n_trials} trials)")

        trials = []
        for t in range(args.n_trials):
            seed = BASE_SEED + ri * 100 + t
            traj = sim_module.run_simulation(
                sp, mjcf_path=MJCF_PATHS[ref["scene"]],
                sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
                init_yaw_jitter_deg=args.jitter_deg, rng_seed=seed,
            )
            if traj is None:
                print(f"  trial {t}: FAIL")
                trials.append({"fail": True, "seed": seed, "trial_idx": t})
                continue

            vel = extract_flat_velocity(traj)
            delta = abs(vel - ref["target"])
            metrics = compute_locomotion_metrics(traj, mass)

            if metrics is None:
                print(f"  trial {t}: vel={vel * 1000:>7.1f} mm/s  NO METRICS")
                trials.append({"fail": True, "seed": seed, "trial_idx": t})
                continue

            cot = metrics["cot"]
            print(f"  trial {t}: vel={vel * 1000:>7.1f} mm/s  delta={delta * 1000:>5.1f}  "
                  f"COT={cot:.1f}  dist={metrics['total_distance'] * 1e3:.1f}mm  "
                  f"E={metrics['energy_ext'] * 1e6:.2f}uJ")
            trials.append({
                "fail": False,
                "seed": seed,
                "trial_idx": t,
                "vel": vel,
                "delta": delta,
                "cot": cot,
                "energy_ext": metrics["energy_ext"],
                "distance": metrics["total_distance"],
                "total_time": metrics["total_time"],
                "avg_power_ext": metrics["avg_power_ext"],
            })

        # Select top-K by smallest delta
        valid = [x for x in trials if not x["fail"]]
        if len(valid) < args.top_k:
            print(f"  -> Only {len(valid)} valid trials (need {args.top_k})")
            selected = valid
        else:
            valid_sorted = sorted(valid, key=lambda x: x["delta"])
            selected = valid_sorted[:args.top_k]

        if selected:
            sel_indices = [x["trial_idx"] for x in selected]
            mean_vel = np.mean([x["vel"] for x in selected])
            mean_cot = np.mean([x["cot"] for x in selected])
            std_cot = np.std([x["cot"] for x in selected], ddof=1) if len(selected) > 1 else 0.0
            mean_dist = np.mean([x["distance"] for x in selected])
            mean_energy = np.mean([x["energy_ext"] for x in selected])
            print(f"  -> Top {len(selected)}: trials {sel_indices}  "
                  f"mean_vel={mean_vel * 1000:.1f} mm/s  mean_COT={mean_cot:.1f} +/- {std_cot:.1f}")
        else:
            mean_vel = mean_cot = std_cot = mean_dist = mean_energy = 0.0
            sel_indices = []
            print(f"  -> ALL FAILED")

        summaries.append({
            "id": ref["id"],
            "scene": ref["scene"],
            "freq": ref["freq"],
            "target": ref["target"],
            "mean_vel": mean_vel,
            "mean_cot": mean_cot,
            "std_cot": std_cot,
            "mean_distance": mean_dist,
            "mean_energy": mean_energy,
            "n_selected": len(selected),
            "mass": mass,
        })

        # Add selected trials to CSV
        for x in selected:
            csv_rows.append({
                "scene": ref["scene"],
                "freq": ref["freq"],
                "trial": x["trial_idx"],
                "seed": x["seed"],
                "velocity_mps": x["vel"],
                "target_mps": ref["target"],
                "cot": x["cot"],
                "energy_ext_J": x["energy_ext"],
                "distance_m": x["distance"],
                "robot_mass_kg": mass,
                "total_time_s": x["total_time"],
                "avg_power_ext_W": x["avg_power_ext"],
            })
        print()

    # --- Summary table ---
    print("=" * 95)
    print(f"  {'ref':<20} {'target':>8} {'mean_vel':>9} {'mean_COT':>9} {'std_COT':>8} {'dist(mm)':>9} {'E(uJ)':>8}")
    print("  " + "-" * 90)
    for s in summaries:
        if s["n_selected"] == 0:
            print(f"  {s['id']:<20} {s['target'] * 1000:>7.1f} {'FAIL':>9}")
        else:
            print(f"  {s['id']:<20} {s['target'] * 1000:>7.1f} {s['mean_vel'] * 1000:>8.1f} "
                  f"{s['mean_cot']:>9.1f} {s['std_cot']:>7.1f} "
                  f"{s['mean_distance'] * 1e3:>8.1f} {s['mean_energy'] * 1e6:>8.2f}")
    print()

    # --- Save CSV ---
    csv_path = run_dir / "cot_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scene", "freq", "trial", "seed", "velocity_mps", "target_mps",
            "cot", "energy_ext_J", "distance_m", "robot_mass_kg", "total_time_s", "avg_power_ext_W",
        ])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Saved: {csv_path}")

    # --- Save summary .md ---
    md_path = run_dir / "cot_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# COT Evaluation Summary\n\n")
        f.write(f"## Parameters\n\n")
        f.write(f"- **Source**: `{run_dir}`\n")
        f.write(f"- **Terrain**: flat\n")
        f.write(f"- **References**: {len(refs)} flat conditions from `config_new.REFERENCE_DATA`\n")
        f.write(f"- **Trials per ref**: {args.n_trials}\n")
        f.write(f"- **Top-K selection**: {args.top_k} (by smallest |v_sim - v_target|)\n")
        f.write(f"- **Yaw jitter**: +/-{args.jitter_deg} deg\n")
        f.write(f"- **Base seed**: {BASE_SEED} (seed = {BASE_SEED} + ref_idx * 100 + trial_idx)\n")
        f.write(f"- **Sim duration**: {SIM_DURATION}s, settle: {SETTLE_TIME}s\n")
        f.write(f"- **COT formula**: `COT = integral(|P_ext|) dt / (m * g * d)`\n")
        f.write(f"  - P_ext = sum_legs(tau_ext . omega)\n")
        f.write(f"  - d = cumulative 2D path length\n\n")

        f.write(f"## Robot Masses\n\n")
        f.write(f"| Scene | Mass (mg) |\n|-------|----------|\n")
        for scene in sorted(robot_masses.keys()):
            f.write(f"| {scene} | {robot_masses[scene] * 1e6:.2f} |\n")
        f.write(f"\n")

        f.write(f"## Results\n\n")
        f.write(f"| Ref | Freq | Target (mm/s) | Sim vel (mm/s) | Vel err % | COT | COT std | Dist (mm) | Energy (uJ) |\n")
        f.write(f"|-----|------|--------------|---------------|----------|-----|---------|-----------|-------------|\n")
        for s in summaries:
            if s["n_selected"] == 0:
                f.write(f"| {s['id']} | {int(s['freq'])} | {s['target'] * 1000:.1f} | FAIL | - | - | - | - | - |\n")
            else:
                vel_err = (s['mean_vel'] - s['target']) / s['target'] * 100 if s['target'] > 1e-6 else 0
                f.write(f"| {s['id']} | {int(s['freq'])} | {s['target'] * 1000:.1f} | "
                        f"{s['mean_vel'] * 1000:.1f} | {vel_err:+.0f}% | "
                        f"{s['mean_cot']:.1f} | {s['std_cot']:.1f} | "
                        f"{s['mean_distance'] * 1e3:.1f} | {s['mean_energy'] * 1e6:.2f} |\n")
        f.write(f"\n")

        # Per-trial breakdown
        f.write(f"## Per-Trial Breakdown (top-{args.top_k} selected trials)\n\n")
        f.write(f"| Ref | Freq | Target | Trial velocities (mm/s) | Trial errors | Trial COTs |\n")
        f.write(f"|-----|------|--------|------------------------|-------------|------------|\n")
        for s in summaries:
            ref_trials = [r for r in csv_rows if r["scene"] == s["scene"] and r["freq"] == s["freq"]]
            if not ref_trials:
                f.write(f"| {s['id']} | {int(s['freq'])} | {s['target'] * 1000:.1f} | FAIL | - | - |\n")
                continue
            vels = ", ".join(f"{r['velocity_mps'] * 1000:.1f}" for r in ref_trials)
            errs = ", ".join(f"{(r['velocity_mps'] - r['target_mps']) / r['target_mps'] * 100:+.0f}%"
                            for r in ref_trials)
            cots = ", ".join(f"{r['cot']:.1f}" for r in ref_trials)
            f.write(f"| {s['id']} | {int(s['freq'])} | {s['target'] * 1000:.1f} | {vels} | {errs} | {cots} |\n")
        f.write(f"\n")

        # Per-morphology summary
        f.write(f"## Per-Morphology Summary\n\n")
        f.write(f"| Morphology | Mean COT | Freq range |\n")
        f.write(f"|------------|----------|------------|\n")
        for scene in ["scene1", "scene2", "scene4", "scene_wheel"]:
            scene_sums = [s for s in summaries if s["scene"] == scene and s["n_selected"] > 0]
            if scene_sums:
                cots = [s["mean_cot"] for s in scene_sums]
                freqs = [int(s["freq"]) for s in scene_sums]
                f.write(f"| {SCENE_LABELS.get(scene, scene)} | {np.mean(cots):.1f} | {min(freqs)}-{max(freqs)} Hz |\n")
        f.write(f"\n")

    print(f"Saved: {md_path}")

    # --- Record best trial videos ---
    if args.record:
        vid_dir = run_dir / "cot_videos"
        vid_dir.mkdir(exist_ok=True)
        print(f"\nRecording to {vid_dir} ...")
        for ri, ref in enumerate(refs):
            # Find best trial (smallest delta) from csv_rows
            ref_rows = [r for r in csv_rows if r["scene"] == ref["scene"]
                        and r["freq"] == ref["freq"]]
            if not ref_rows:
                continue
            best = min(ref_rows, key=lambda r: abs(r["velocity_mps"] - r["target_mps"]))
            sp = dict(sim_params)
            sp["drive_freq"] = ref["freq"]
            vid_path = str(vid_dir / f"{ref['id']}_cot.mp4")
            sim_module.run_simulation(
                sp, mjcf_path=MJCF_PATHS[ref["scene"]],
                sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
                init_yaw_jitter_deg=args.jitter_deg, rng_seed=best["seed"],
                record_path=vid_path,
            )
            print(f"  Saved: {vid_path}")


if __name__ == "__main__":
    main()
