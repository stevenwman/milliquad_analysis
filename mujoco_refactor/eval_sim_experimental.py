#!/usr/bin/env python3
"""Run flat-ground sims, pick best jitter trial, output CSVs in experimental format.

For each (scene, freq) reference, runs N jitter trials, picks the trial whose
simulated velocity is closest to the experimental target, then writes a CSV
mimicking the experimental tracking data format (front/back magnet positions,
velocities via finite difference, and geometric body pitch).

The CSVs can be read by the existing experimental plotting pipeline
(experimental_data/scripts/flat_pipeline.py) without modification.

Usage:
    uv run python eval_sim_experimental.py results/20260225T122342_flat_10_30_50/
    uv run python eval_sim_experimental.py results/20260225T225248_step_argmin_progress/
    uv run python eval_sim_experimental.py results/... --scenes scene4 --freqs 30 --n-trials 5
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

from config_new import (
    MJCF_PATHS,
    REFERENCE_DATA,
    SETTLE_TIME,
    SIM_DURATION,
    SIM_TIMESTEP,
    SIMULATION_TIMEOUT,
    sim_params_from_point,
    space,
)
import simulation_fast_new as sim_module

PARAM_NAMES = [dim.name for dim in space]
JITTER_DEG = 2.0
BASE_SEED = 77777
N_TRIALS_DEFAULT = 3
DOWNSAMPLE_FACTOR = 2  # 2kHz sim → 1kHz CSV (match experimental sampling)

# Leg body indices within leg_xpos array (slice order: FR=0, FL=1, BR=2, BL=3)
FL_IDX = 1  # front-left → "mass_A" (front magnet)
BL_IDX = 3  # back-left  → "mass_C" (back magnet)

SCENE_TO_MORPH = {
    "scene1": "leg",
    "scene2": "2legged",
    "scene4": "4legged",
    "scene_wheel": "wheel",
}

# CSV header matching experimental format exactly
CSV_HEADER_ROW1 = ",mass_A,,,,mass_C,,,,mass_B,,mass_C"
CSV_HEADER_ROW2 = "t,x,y,vx,vy,x,y,vx,vy,\u03b8,\u03c9,\u03b8"


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
    raise ValueError(f"id {best_id!r} not found in multi_optimization_results.csv")


# ---------------------------------------------------------------------------
# Velocity extraction (same as eval_best_trial.py / optimizer_new)
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
# Trajectory → experimental CSV data
# ---------------------------------------------------------------------------

def trajectory_to_csv_data(traj: list[dict]) -> np.ndarray:
    """Convert a sim trajectory to a (N, 12) array matching experimental CSV format.

    Columns: t, mass_A(x,y,vx,vy), mass_C(x,y,vx,vy), mass_B_theta, mass_B_omega, mass_C_theta
    """
    # Downsample: take every DOWNSAMPLE_FACTOR-th step
    traj_ds = traj[::DOWNSAMPLE_FACTOR]
    n = len(traj_ds)

    # Extract FL (front) and BL (back) leg body world positions
    t = np.array([s["time"] for s in traj_ds])
    fl_pos = np.array([s["leg_xpos"][FL_IDX] for s in traj_ds])  # (n, 3) xyz
    bl_pos = np.array([s["leg_xpos"][BL_IDX] for s in traj_ds])  # (n, 3) xyz

    # Map sim coordinates → experimental CSV convention:
    #   CSV x = -sim_x  (experimental camera: forward = -x)
    #   CSV y = sim_z    (experimental camera: up = +y; sim: up = +z)
    fl_x = -fl_pos[:, 0]
    fl_y = fl_pos[:, 2]
    bl_x = -bl_pos[:, 0]
    bl_y = bl_pos[:, 2]

    # Velocities via finite difference (mimics camera tracker)
    dt_arr = np.diff(t)
    fl_vx = np.empty(n)
    fl_vy = np.empty(n)
    bl_vx = np.empty(n)
    bl_vy = np.empty(n)

    fl_vx[1:] = np.diff(fl_x) / dt_arr
    fl_vy[1:] = np.diff(fl_y) / dt_arr
    bl_vx[1:] = np.diff(bl_x) / dt_arr
    bl_vy[1:] = np.diff(bl_y) / dt_arr

    # First row: use forward difference (same as second row)
    fl_vx[0] = fl_vx[1] if n > 1 else 0.0
    fl_vy[0] = fl_vy[1] if n > 1 else 0.0
    bl_vx[0] = bl_vx[1] if n > 1 else 0.0
    bl_vy[0] = bl_vy[1] if n > 1 else 0.0

    # Theta: geometric pitch from front-back marker positions
    # Use sim coordinates (before sign flip) for the geometry
    dx = fl_pos[:, 0] - bl_pos[:, 0]  # forward separation
    dz = fl_pos[:, 2] - bl_pos[:, 2]  # height difference
    theta_raw = np.degrees(np.arctan2(dz, dx))

    # Unwrap to avoid ±180 discontinuities, then subtract initial value
    theta_unwrapped = np.degrees(np.unwrap(np.radians(theta_raw)))
    theta = theta_unwrapped - theta_unwrapped[0]

    # Placeholders for mass_B theta and omega (not used by pipeline)
    zeros = np.zeros(n)

    # Assemble (N, 12) array
    data = np.column_stack([
        t,       # col 0
        fl_x,    # col 1: mass_A x
        fl_y,    # col 2: mass_A y
        fl_vx,   # col 3: mass_A vx
        fl_vy,   # col 4: mass_A vy
        bl_x,    # col 5: mass_C x
        bl_y,    # col 6: mass_C y
        bl_vx,   # col 7: mass_C vx
        bl_vy,   # col 8: mass_C vy
        zeros,   # col 9: mass_B theta (placeholder)
        zeros,   # col 10: mass_B omega (placeholder)
        theta,   # col 11: mass_C theta (body pitch)
    ])
    return data


def write_experimental_csv(path: pathlib.Path, data: np.ndarray) -> None:
    """Write a (N, 12) array as a CSV with experimental-format headers."""
    with open(path, "w", newline="") as f:
        f.write(CSV_HEADER_ROW1 + "\n")
        f.write(CSV_HEADER_ROW2 + "\n")
        for row in data:
            f.write(",".join(f"{v:.6E}" for v in row) + "\n")


# ---------------------------------------------------------------------------
# Plotly HTML generation
# ---------------------------------------------------------------------------

def generate_html_plots(csv_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    """Generate per-(freq, morphology) HTML plots from sim CSVs."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("WARNING: plotly not installed, skipping HTML plots")
        return

    MM_SCALE = 1000.0

    # Collect all sim CSVs and group by (freq, morphology)
    csv_files = sorted(csv_dir.glob("sim_*.csv"))
    groups: dict[str, list[pathlib.Path]] = {}
    for csv_path in csv_files:
        # Parse filename: sim_<scene>_f<freq>.csv
        stem = csv_path.stem  # e.g. "sim_scene1_f10"
        parts = stem.split("_")
        # Extract freq and scene from parts
        freq_part = parts[-1]  # e.g. "f10"
        scene_parts = parts[1:-1]  # e.g. ["scene1"] or ["scene", "wheel"]
        scene = "_".join(scene_parts)
        morph = SCENE_TO_MORPH.get(scene, scene)
        freq_hz = freq_part[1:]  # strip "f" prefix
        key = f"{freq_hz}hz_{morph}"
        groups.setdefault(key, []).append(csv_path)

    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for group_name, paths in sorted(groups.items()):
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=(
                f"Forward Speed vs. Time - {group_name}",
                f"Body Height vs. Time - {group_name}",
                f"Body Angle vs. Time - {group_name}",
                f"Angular Velocity vs. Time - {group_name}",
            ),
        )

        for i, csv_path in enumerate(paths):
            dat = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
            t = dat[:, 0]
            # Same extraction as flat_pipeline._extract_flat
            vx = 0.5 * ((-dat[:, 3] * MM_SCALE) + (-dat[:, 7] * MM_SCALE))
            y_raw = 0.5 * (dat[:, 2] + dat[:, 6])
            y = (y_raw - y_raw[min(2, len(y_raw) - 1)] + np.min(y_raw)) * MM_SCALE
            theta = dat[:, 11]
            omega = dat[:, 10]

            color = colors[i % len(colors)]
            name = csv_path.stem
            fig.add_trace(go.Scatter(x=t, y=vx, mode="lines", name=name,
                                     line=dict(color=color)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", showlegend=False,
                                     line=dict(color=color)), row=2, col=1)
            fig.add_trace(go.Scatter(x=t, y=theta, mode="lines", showlegend=False,
                                     line=dict(color=color)), row=3, col=1)
            fig.add_trace(go.Scatter(x=t, y=omega, mode="lines", showlegend=False,
                                     line=dict(color=color)), row=4, col=1)

        fig.update_yaxes(title_text="v_x [mm/s]", row=1, col=1)
        fig.update_yaxes(title_text="height [mm]", row=2, col=1)
        fig.update_yaxes(title_text="theta [deg]", row=3, col=1)
        fig.update_yaxes(title_text="omega", row=4, col=1)
        fig.update_xaxes(title_text="Time [s]", row=4, col=1)
        fig.update_layout(
            template="plotly_white", width=1280, height=1150,
            legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
            margin=dict(l=70, r=220, t=80, b=60),
        )
        out_path = plot_dir / f"{group_name}.html"
        fig.write_html(out_path, include_plotlyjs=True)
        print(f"  Plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=str, help="Results directory with optimization CSVs")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--freqs", nargs="+", type=float, default=None)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--record", action="store_true", help="Record best trial videos")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"ERROR: {run_dir} is not a directory")

    point = load_best_point(run_dir)
    sim_params = sim_params_from_point(point)

    # Build reference list (flat only)
    refs: list[dict] = []
    for row in REFERENCE_DATA:
        s = row["scene"]
        f = float(row.get("ctrl_freq", 30.0))
        if args.scenes and s not in args.scenes:
            continue
        if args.freqs and f not in args.freqs:
            continue
        refs.append({
            "id": f"{s}_f{int(f)}",
            "scene": s,
            "freq": f,
            "target": float(row["speed"]),
        })

    csv_dir = run_dir / "sim_csvs"
    csv_dir.mkdir(exist_ok=True)

    print(f"Loaded params from {run_dir}")
    print(f"  {len(refs)} references, {args.n_trials} trials each")
    print()

    # Collect results for summary
    summary_rows: list[dict] = []

    for ri, ref in enumerate(refs):
        sp = dict(sim_params)
        sp["drive_freq"] = ref["freq"]
        mjcf = MJCF_PATHS[ref["scene"]]

        print(f"{ref['id']} (target={ref['target'] * 1000:.1f} mm/s, {args.n_trials} trials)")

        # Run jitter trials, keep all trajectories
        trials = []
        for t_idx in range(args.n_trials):
            seed = BASE_SEED + ri * 100 + t_idx
            traj = sim_module.run_simulation(
                sp, mjcf_path=mjcf, sim_duration=SIM_DURATION,
                wall_timeout=SIMULATION_TIMEOUT,
                init_yaw_jitter_deg=JITTER_DEG, rng_seed=seed,
            )
            if traj is None:
                trials.append({"vel": 0.0, "delta": abs(ref["target"]),
                                "seed": seed, "traj": None})
                print(f"  trial {t_idx}: FAIL")
                continue
            vel = extract_flat_velocity(traj)
            delta = abs(vel - ref["target"])
            trials.append({"vel": vel, "delta": delta, "seed": seed, "traj": traj})
            print(f"  trial {t_idx}: {vel * 1000:>7.1f} mm/s  delta={delta * 1000:>5.1f}")

        # Pick best trial
        valid = [x for x in trials if x["traj"] is not None]
        if not valid:
            print(f"  -> ALL FAILED\n")
            summary_rows.append({
                "id": ref["id"], "target": ref["target"], "vel": None,
                "err_pct": None, "mean_pitch": None, "seed": None,
            })
            continue

        best = min(valid, key=lambda x: x["delta"])
        bi = trials.index(best)
        print(f"  -> BEST trial {bi}: {best['vel'] * 1000:.1f} mm/s, "
              f"delta={best['delta'] * 1000:.1f} mm/s")

        # Convert best trajectory to CSV data and write
        csv_data = trajectory_to_csv_data(best["traj"])
        csv_path = csv_dir / f"sim_{ref['id']}.csv"
        write_experimental_csv(csv_path, csv_data)
        print(f"  -> CSV: {csv_path}")

        # Compute summary stats
        err_pct = (best["delta"] / ref["target"] * 100) if ref["target"] > 1e-6 else 0.0
        mean_pitch = float(np.mean(np.abs(csv_data[:, 11])))
        summary_rows.append({
            "id": ref["id"], "target": ref["target"], "vel": best["vel"],
            "err_pct": err_pct, "mean_pitch": mean_pitch, "seed": best["seed"],
        })
        print()

    # --- Summary table to stdout ---
    print("=" * 80)
    print(f"  {'ref':<22} {'target':>8} {'sim':>8} {'err%':>7} {'pitch':>7} {'seed':>7}")
    print("  " + "-" * 78)
    for r in summary_rows:
        tgt = r["target"] * 1000
        if r["vel"] is not None:
            v = r["vel"] * 1000
            print(f"  {r['id']:<22} {tgt:>7.1f} {v:>7.1f} {r['err_pct']:>6.1f}% "
                  f"{r['mean_pitch']:>6.2f}° {r['seed']:>7}")
        else:
            print(f"  {r['id']:<22} {tgt:>7.1f} {'FAIL':>8}")
    print()

    # --- Summary .md ---
    run_label = run_dir.name
    md_path = run_dir / "sim_experimental_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Sim Experimental-Format Summary\n\n")
        f.write(f"**Params**: `{run_label}`\n\n")
        f.write(f"| Ref | Target (mm/s) | Sim (mm/s) | Err% | Mean |Pitch| (deg) | Seed |\n")
        f.write(f"|-----|---------------|------------|------|-------|------|\n")
        for r in summary_rows:
            tgt = r["target"] * 1000
            if r["vel"] is not None:
                v = r["vel"] * 1000
                f.write(f"| {r['id']} | {tgt:.1f} | {v:.1f} | {r['err_pct']:.1f}% "
                        f"| {r['mean_pitch']:.2f} | {r['seed']} |\n")
            else:
                f.write(f"| {r['id']} | {tgt:.1f} | FAIL | - | - | - |\n")
    print(f"Summary: {md_path}")

    # --- Record best trial videos ---
    if args.record:
        vid_dir = run_dir / "sim_videos"
        vid_dir.mkdir(exist_ok=True)
        print(f"Recording videos to {vid_dir} ...")
        for ref, r in zip(refs, summary_rows):
            if r["seed"] is None:
                continue
            sp = dict(sim_params)
            sp["drive_freq"] = ref["freq"]
            vid_path = str(vid_dir / f"sim_{ref['id']}.mp4")
            sim_module.run_simulation(
                sp, mjcf_path=MJCF_PATHS[ref["scene"]], sim_duration=SIM_DURATION,
                wall_timeout=SIMULATION_TIMEOUT,
                init_yaw_jitter_deg=JITTER_DEG, rng_seed=r["seed"],
                record_path=vid_path,
            )
            print(f"  {vid_path}")
        print()

    # --- HTML plots ---
    exp_root = pathlib.Path(__file__).resolve().parent.parent / "experimental_data"
    plot_dir = exp_root / "plots" / "fake_exp" / run_label
    print(f"\nGenerating HTML plots...")
    generate_html_plots(csv_dir, plot_dir)
    print(f"Done. Plots: {plot_dir}")


if __name__ == "__main__":
    main()
