"""
Friction sensitivity analysis — L2 (scene2) on rough terrain.

Sweeps each friction parameter independently while holding the other 13 params
at the rough-optimized baseline. Measures success rate and average velocity
across jittered trials.

Usage:
    uv run python friction_sensitivity.py [--n-trials 10] [--n-values 5] [--workers 16]

Output in results/friction_sensitivity/:
    friction_sweep.csv           — per-trial results
    friction_sweep_summary.csv   — aggregated per (param, value, freq)
    trajectories.npz             — body state time series (pos, vel, quat, omega)
    overview_{param}.png         — x-pos vs time grid plot per swept parameter
"""

import argparse
import csv
import multiprocessing
import pathlib
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

from config import (
    PACKAGE_DIR,
    SETTLE_TIME,
    SIM_TIMESTEP,
    SIMULATION_TIMEOUT,
    sim_params_from_point,
)
from config_rough_tg import (
    SPAWN_X,
    SPAWN_Z_RAISE,
    Y_JITTER,
    Y_JITTER_SEED,
    SIM_DURATION,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENE = "scene2"
MJCF_PATH = str(PACKAGE_DIR / "robots" / "quad" / "scene_2_rough.xml")
FREQS = [10.0, 30.0, 50.0]

# Rough terrain spatial gate
ROUGH_START_X = 0.005   # m
ROUGH_END_X = 0.155     # m

FRICTION_PARAMS = {
    "sliding_friction":   {"lo": 0.01,  "hi": 2.0},
    "torsional_friction": {"lo": 1e-6,  "hi": 1.0},
    "rolling_friction":   {"lo": 1e-6,  "hi": 1e-3},
}


def _build_sweep_global(lo: float, hi: float, n: int) -> np.ndarray:
    """Log-spaced sweep across full search space [lo, hi]."""
    return np.geomspace(lo, hi, n)


def _build_sweep_local(baseline: float, lo: float, hi: float, n: int) -> np.ndarray:
    """Log-spaced sweep centered on baseline, spanning [lo, hi].

    With n=5: 2 points below baseline, baseline, 2 points above.
    Each side uses the full available range to that bound independently.
    """
    n_side = (n - 1) // 2
    log_bl = np.log10(baseline)
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    below = np.linspace(log_lo, log_bl, n_side + 1)[:-1]
    above = np.linspace(log_bl, log_hi, n_side + 1)[1:]
    pts = np.concatenate([below, [log_bl], above])
    return 10.0 ** pts

# ---------------------------------------------------------------------------
# Load baseline params
# ---------------------------------------------------------------------------

def load_baseline_point(run_dir: pathlib.Path) -> dict[str, float]:
    """Load the final best point from optimization_bests.csv."""
    csv_path = run_dir / "optimization_bests.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    from config import space
    return {dim.name: float(last[dim.name]) for dim in space}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Task:
    __slots__ = (
        "param_name", "param_value", "ctrl_freq", "trial_idx",
        "rng_seed", "jitter_y", "point",
    )
    def __init__(self, param_name, param_value, ctrl_freq, trial_idx,
                 rng_seed, jitter_y, point):
        self.param_name = param_name
        self.param_value = param_value
        self.ctrl_freq = ctrl_freq
        self.trial_idx = trial_idx
        self.rng_seed = rng_seed
        self.jitter_y = jitter_y
        self.point = point


def _run_one(task: Task) -> dict[str, Any]:
    """Run a single simulation trial. Returns result dict."""
    from simulation import run_simulation

    params = sim_params_from_point(task.point)
    params["drive_freq"] = task.ctrl_freq

    spawn_offset = (SPAWN_X, task.jitter_y, SPAWN_Z_RAISE)

    trajectory = run_simulation(
        params=params,
        mjcf_path=MJCF_PATH,
        sim_duration=SIM_DURATION,
        spawn_offset=spawn_offset,
        rng_seed=task.rng_seed,
        wall_timeout=SIMULATION_TIMEOUT,
    )

    crash = trajectory is None
    result = {
        "param_name": task.param_name,
        "param_value": task.param_value,
        "ctrl_freq": task.ctrl_freq,
        "trial": task.trial_idx,
        "rng_seed": task.rng_seed,
        "jitter_y": task.jitter_y,
        "crash": crash,
    }

    # --- extract body trajectory arrays ---
    traj_arrays = {}
    if not crash and len(trajectory) > 0:
        n = len(trajectory)
        t_arr = np.array([s["time"] for s in trajectory])
        pos = np.array([s["pos"] for s in trajectory])       # (N, 3)
        vel = np.array([s["vel"] for s in trajectory])       # (N, 3)
        quat = np.array([s["quat"] for s in trajectory])     # (N, 4)
        # omega = chassis angular velocity from cvel (3-component)
        # not directly in trajectory — use qvel[3:6] via vel stores only translational
        # Actually: data.qvel[:3] is translational, data.qvel[3:6] is rotational
        # but _record_state stores data.qvel[:3] as "vel"
        # We need angular velocity — it's not stored in the trajectory dict.
        # However data.cvel[1, :3] (chassis body) would be it.
        # For now store what we have; omega can be added later.

        traj_arrays["time"] = t_arr
        traj_arrays["pos"] = pos
        traj_arrays["vel"] = vel
        traj_arrays["quat"] = quat

        # Compute scalar metrics via spatial gating
        max_x = float(pos[:, 0].max())
        success = max_x >= ROUGH_END_X

        vx = 0.0
        if success:
            enter_mask = pos[:, 0] >= ROUGH_START_X
            exit_mask = pos[:, 0] >= ROUGH_END_X
            if enter_mask.any() and exit_mask.any():
                enter_idx = int(np.argmax(enter_mask))
                exit_idx = int(np.argmax(exit_mask))
                if exit_idx > enter_idx:
                    dt = t_arr[exit_idx] - t_arr[enter_idx]
                    if dt > 1e-6:
                        vx = (pos[exit_idx, 0] - pos[enter_idx, 0]) / dt

        result["max_x"] = max_x
        result["success"] = success
        result["vx_m_s"] = vx
    else:
        traj_arrays = {}
        result["max_x"] = 0.0
        result["success"] = False
        result["vx_m_s"] = 0.0

    result["_traj_arrays"] = traj_arrays
    return result


# ---------------------------------------------------------------------------
# Overview plot
# ---------------------------------------------------------------------------

def _plot_overview(
    out_path: pathlib.Path,
    param_name: str,
    sweep_values: np.ndarray,
    baseline_value: float,
    results: list[dict],
    freqs: list[float],
):
    """Grid plot: rows=sweep values, cols=frequencies. Each cell = x-pos vs time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "TeX Gyre Pagella", "font.size": 8})

    TRIAL_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
                    "#e6ab02", "#a6761d", "#666666", "#1f78b4", "#b2df8a"]

    nrows = len(sweep_values)
    ncols = len(freqs)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.5, nrows * 2.2),
        constrained_layout=True,
        squeeze=False,
    )

    for ri, val in enumerate(sweep_values):
        for ci, freq in enumerate(freqs):
            ax = axes[ri][ci]
            trials = [r for r in results
                      if r["param_name"] == param_name
                      and np.isclose(r["param_value"], val, rtol=1e-6)
                      and r["ctrl_freq"] == freq]

            n_success = sum(1 for t in trials if t["success"])
            n_total = len(trials)

            for t in trials:
                if t["crash"] or "time" not in t.get("_traj_arrays", {}):
                    continue
                tarr = t["_traj_arrays"]["time"]
                xmm = t["_traj_arrays"]["pos"][:, 0] * 1000.0
                color = TRIAL_COLORS[t["trial"] % len(TRIAL_COLORS)]
                ls = "-" if t["success"] else "--"
                alpha = 1.0 if t["success"] else 0.4
                ax.plot(tarr, xmm, color=color, lw=0.9, ls=ls, alpha=alpha)

            # Gate lines
            ax.axhline(ROUGH_START_X * 1000, color="tab:blue", ls=":", lw=0.7, alpha=0.5)
            ax.axhline(ROUGH_END_X * 1000, color="tab:red", ls=":", lw=0.7, alpha=0.5)
            ax.axvline(SETTLE_TIME, color="k", ls="--", lw=0.6, alpha=0.4)

            # Title
            is_baseline = np.isclose(val, baseline_value, rtol=0.05)
            star = " *" if is_baseline else ""
            ax.set_title(
                f"{val:.2e}{star}  [{n_success}/{n_total}]",
                fontsize=7, fontweight="bold" if is_baseline else "normal",
            )

            if ri == nrows - 1:
                ax.set_xlabel("Time (s)", fontsize=7)
            if ci == 0:
                ax.set_ylabel("x (mm)", fontsize=7)

    # Column headers
    for ci, freq in enumerate(freqs):
        axes[0][ci].text(
            0.5, 1.25, f"f = {int(freq)} Hz",
            transform=axes[0][ci].transAxes, ha="center", fontsize=9, fontweight="bold",
        )

    fig.suptitle(
        f"Friction sensitivity: {param_name} — scene2 on rough",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Overview plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Friction sensitivity sweep")
    parser.add_argument("--n-trials", type=int, default=10, help="Jitter trials per point")
    parser.add_argument("--n-values", type=int, default=5, help="Sweep points per param")
    parser.add_argument("--mode", choices=["local", "global"], default="local",
                        help="local = centered on baseline, global = full search space")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="results/20260303T224229_rough_tg",
        help="Rough optimization run dir for baseline params",
    )
    args = parser.parse_args()

    run_dir = PACKAGE_DIR / args.run_dir
    baseline_point = load_baseline_point(run_dir)
    print(f"Baseline params from: {run_dir}")
    print(f"  sliding_friction:   {baseline_point['sliding_friction']:.6f}")
    print(f"  torsional_friction: {baseline_point['torsional_friction']:.2e}")
    print(f"  rolling_friction:   {baseline_point['rolling_friction']:.2e}")

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = PACKAGE_DIR / "results" / f"{timestamp}_friction_sensitivity_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Build tasks
    tasks: list[Task] = []
    jitter_rng = np.random.default_rng(Y_JITTER_SEED)

    for param_name, bounds in FRICTION_PARAMS.items():
        bl = baseline_point[param_name]
        if args.mode == "local":
            sweep_values = _build_sweep_local(bl, bounds["lo"], bounds["hi"], args.n_values)
        else:
            sweep_values = _build_sweep_global(bounds["lo"], bounds["hi"], args.n_values)
        print(f"  {param_name}: {[f'{v:.2e}' for v in sweep_values]}  (baseline={bl:.2e})")

        for val in sweep_values:
            point = dict(baseline_point)
            point[param_name] = float(val)

            for freq in FREQS:
                # Pre-generate jitter offsets (deterministic per freq)
                trial_jitters = jitter_rng.uniform(-Y_JITTER, Y_JITTER, args.n_trials)
                for ti in range(args.n_trials):
                    seed = hash((param_name, val, freq, ti)) % (2**31)
                    tasks.append(Task(
                        param_name=param_name,
                        param_value=float(val),
                        ctrl_freq=freq,
                        trial_idx=ti,
                        rng_seed=seed,
                        jitter_y=float(trial_jitters[ti]),
                        point=point,
                    ))

    print(f"Total simulations: {len(tasks)}")
    print(f"Workers: {args.workers}")

    # Run with progress counter
    t0 = time.perf_counter()
    results = []
    n_total = len(tasks)
    with multiprocessing.Pool(args.workers, maxtasksperchild=4) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, tasks), 1):
            results.append(result)
            status = "ok" if result["success"] else ("CRASH" if result["crash"] else "fail")
            print(
                f"  [{i}/{n_total}] {result['param_name']}={result['param_value']:.2e} "
                f"f{int(result['ctrl_freq'])} t{result['trial']} → {status}"
                f"  vx={result['vx_m_s']*1000:.0f}mm/s" if not result["crash"] else "",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # --- Save per-trial CSV ---
    csv_path = out_dir / "friction_sweep.csv"
    csv_fields = [
        "param_name", "param_value", "ctrl_freq", "trial", "rng_seed",
        "jitter_y", "crash", "success", "max_x", "vx_m_s",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  Per-trial CSV: {csv_path}")

    # --- Save trajectories as NPZ ---
    npz_path = out_dir / "trajectories.npz"
    npz_dict = {}
    for r in results:
        if r["crash"] or not r.get("_traj_arrays"):
            continue
        prefix = f"{r['param_name']}__{r['param_value']:.6e}__f{int(r['ctrl_freq'])}_t{r['trial']}"
        for key, arr in r["_traj_arrays"].items():
            npz_dict[f"{prefix}_{key}"] = arr
    np.savez_compressed(npz_path, **npz_dict)
    print(f"  Trajectories NPZ: {npz_path} ({len(npz_dict)} arrays)")

    # --- Summary CSV ---
    summary_path = out_dir / "friction_sweep_summary.csv"
    summary_fields = [
        "param_name", "param_value", "ctrl_freq",
        "n_trials", "n_success", "success_rate",
        "mean_vx_mm_s", "std_vx_mm_s",
    ]
    # Group results
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (r["param_name"], r["param_value"], r["ctrl_freq"])
        groups[key].append(r)

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for (pname, pval, freq), trials in sorted(groups.items()):
            n = len(trials)
            successes = [t for t in trials if t["success"]]
            ns = len(successes)
            vxs = [t["vx_m_s"] * 1000.0 for t in successes]  # m/s -> mm/s
            w.writerow({
                "param_name": pname,
                "param_value": pval,
                "ctrl_freq": freq,
                "n_trials": n,
                "n_success": ns,
                "success_rate": ns / n if n > 0 else 0.0,
                "mean_vx_mm_s": float(np.mean(vxs)) if vxs else 0.0,
                "std_vx_mm_s": float(np.std(vxs)) if vxs else 0.0,
            })
    print(f"  Summary CSV: {summary_path}")

    # --- Overview plots ---
    for param_name, bounds in FRICTION_PARAMS.items():
        bl = baseline_point[param_name]
        if args.mode == "local":
            sweep_values = _build_sweep_local(bl, bounds["lo"], bounds["hi"], args.n_values)
        else:
            sweep_values = _build_sweep_global(bounds["lo"], bounds["hi"], args.n_values)
        param_results = [r for r in results if r["param_name"] == param_name]
        plot_path = out_dir / f"overview_{param_name}.png"
        _plot_overview(
            plot_path, param_name, sweep_values,
            baseline_point[param_name], param_results, FREQS,
        )

    print(f"\nAll done. Results in {out_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
