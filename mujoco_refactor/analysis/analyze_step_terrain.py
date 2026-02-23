"""
Extract steady-state forward velocities from experimental step terrain CSVs
and compare with simulation results.

CSV naming convention:
  s10leg1-1.csv   = steps, 10Hz, 1-leg (scene1), trial 1
  s102leg1-1.csv  = steps, 10Hz, 2-leg (scene2), trial 1
  s104leg1-1.csv  = steps, 10Hz, 4-leg (scene4), trial 1
  s30w1-1.csv     = steps, 30Hz, wheel, trial 1

Column format: t, x, y, vx, vy (mass_A), x, y, vx, vy (mass_C), theta, omega (mass_B), theta (mass_C)

Usage:
  uv run python mujoco_refactor/analyze_step_terrain.py
"""

import numpy as np
import os
import re
from collections import defaultdict

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "experimental_data", "csv", "steps"
)

# Simulation results: step terrain (step_default preset: 8 steps, 1mm, 4.5mm long)
SIM_STEP = {
    "scene1_f10": 43.2, "scene1_f20": 73.4, "scene1_f30": 69.7,
    "scene2_f10": 68.1, "scene2_f20": 101.4, "scene2_f30": 159.3,
    "scene4_f10": 100.3, "scene4_f20": 151.6, "scene4_f30": 202.9,
    "wheel_f10": 15.9, "wheel_f20": 17.8, "wheel_f30": 13.8,
}

# Simulation flat baselines
SIM_FLAT = {
    "scene1_f10": 50.2, "scene1_f20": 93.0, "scene1_f30": 116.3,
    "scene2_f10": 82.8, "scene2_f20": 132.5, "scene2_f30": 162.0,
    "scene4_f10": 107.6, "scene4_f20": 186.4, "scene4_f30": 268.7,
    "wheel_f10": 146.1, "wheel_f20": 308.2, "wheel_f30": 448.3,
}

# Experimental flat baselines (from VELOCITY_SUMMARY_FLAT.md)
EXP_FLAT = {
    "scene1_f10": 51.18, "scene1_f20": 126.4, "scene1_f30": 118.77,
    "scene2_f10": 82.44, "scene2_f20": 113.1, "scene2_f30": 179.61,
    "scene4_f10": 112.10, "scene4_f20": 184.1, "scene4_f30": 274.43,
    "wheel_f10": 143.11, "wheel_f20": 305.8, "wheel_f30": 450.27,
}


def parse_filename(fname: str):
    """Parse step CSV filename -> (scene, freq, trial) or (None, None, None)."""
    base = fname.replace(".csv", "")
    # wheel: s{freq}w{trial}-{trial}
    m = re.match(r"^s(\d+)w(\d+)-\d+$", base)
    if m:
        return "wheel", int(m.group(1)), int(m.group(2))
    # 4-leg: s{freq}4leg{trial}-{trial}
    m = re.match(r"^s(\d+)4leg(\d+)-\d+$", base)
    if m:
        return "scene4", int(m.group(1)), int(m.group(2))
    # 2-leg: s{freq}2leg{trial}-{trial}
    m = re.match(r"^s(\d+)2leg(\d+)-\d+$", base)
    if m:
        return "scene2", int(m.group(1)), int(m.group(2))
    # 1-leg: s{freq}leg{trial}-{trial}
    m = re.match(r"^s(\d+)leg(\d+)-\d+$", base)
    if m:
        return "scene1", int(m.group(1)), int(m.group(2))
    return None, None, None


def compute_trial_velocity(fpath: str) -> float:
    """Compute steady-state forward velocity for a single trial CSV."""
    data = np.genfromtxt(fpath, delimiter=",", skip_header=2)
    vx_A = data[:, 3]  # mass_A vx
    vx_C = data[:, 7]  # mass_C vx
    # Average of two tracking points, negated (camera convention), m/s -> mm/s
    vx = 0.5 * (-vx_A * 1000 + -vx_C * 1000)
    # Last 50% for steady-state
    n = len(vx)
    vx_ss = vx[n // 2:]
    vx_ss = vx_ss[~np.isnan(vx_ss)]
    return float(np.mean(vx_ss))


def main():
    data_dir = os.path.abspath(DATA_DIR)
    cond_trials = defaultdict(list)

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        scene, freq, trial = parse_filename(fname)
        if scene is None:
            print(f"WARNING: could not parse {fname}")
            continue
        fpath = os.path.join(data_dir, fname)
        trial_mean = compute_trial_velocity(fpath)
        cond_trials[f"{scene}_f{freq}"].append(trial_mean)

    # Per-condition statistics
    cond_stats = {}
    for key, trials in sorted(cond_trials.items()):
        arr = np.array(trials)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        cond_stats[key] = (mean, std, arr)

    # --- Main comparison table ---
    W = 120
    print()
    print("=" * W)
    print("STEP TERRAIN: Experimental vs Simulation Velocity Comparison")
    print("=" * W)
    print(
        f"{'Config':<14} {'Exp vel':>10} {'Exp std':>10} {'Sim vel':>10} "
        f"{'Sim ratio':>10} {'Exp ratio':>10} {'Error%':>10} {'Trials':>8}"
    )
    print(
        f"{'':>14} {'(mm/s)':>10} {'(mm/s)':>10} {'(mm/s)':>10} "
        f"{'step/flat':>10} {'step/flat':>10} {'':>10} {'':>8}"
    )
    print("-" * W)

    scenes = ["scene1", "scene2", "scene4", "wheel"]
    freqs = [10, 20, 30]

    for scene in scenes:
        for freq in freqs:
            key = f"{scene}_f{freq}"
            sim_s = SIM_STEP.get(key)
            sim_f = SIM_FLAT.get(key)
            exp_f = EXP_FLAT.get(key)

            if key in cond_stats:
                mean, std, trials = cond_stats[key]
                sim_ratio = sim_s / sim_f if sim_f else None
                exp_ratio = mean / exp_f if exp_f else None
                error_pct = (sim_s - mean) / mean * 100 if mean != 0 else None
                print(
                    f"{key:<14} {mean:>10.2f} {std:>10.2f} {sim_s:>10.1f} "
                    f"{sim_ratio:>10.3f} {exp_ratio:>10.3f} {error_pct:>+10.1f} "
                    f"{len(trials):>8d}"
                )
            else:
                sim_ratio = sim_s / sim_f if (sim_s and sim_f) else None
                r_str = f"{sim_ratio:.3f}" if sim_ratio else "N/A"
                print(
                    f"{key:<14} {'N/A':>10} {'N/A':>10} {sim_s:>10.1f} "
                    f"{r_str:>10} {'N/A':>10} {'N/A':>10} {'0':>8}"
                )
    print("-" * W)

    # --- Summary ---
    matched_keys = [k for k in cond_stats if k in SIM_STEP]
    errors = np.array(
        [(SIM_STEP[k] - cond_stats[k][0]) / cond_stats[k][0] * 100 for k in matched_keys]
    )
    print(f"\nSummary (n={len(errors)} conditions with experimental data):")
    print(f"  Mean absolute error:   {np.mean(np.abs(errors)):.1f}%")
    print(f"  Median absolute error: {np.median(np.abs(errors)):.1f}%")
    print(f"  Mean signed error:     {np.mean(errors):+.1f}%")
    worst = matched_keys[int(np.argmax(np.abs(errors)))]
    print(f"  Max absolute error:    {np.max(np.abs(errors)):.1f}% ({worst})")

    # --- Per-trial detail ---
    print()
    print("=" * W)
    print("PER-TRIAL DETAIL")
    print("=" * W)
    print(f"{'Config':<14} {'Trial 1':>10} {'Trial 2':>10} {'Trial 3':>10} {'Mean':>10} {'Std':>10}")
    print("-" * W)
    for scene in scenes:
        for freq in freqs:
            key = f"{scene}_f{freq}"
            if key in cond_stats:
                mean, std, trials = cond_stats[key]
                t_strs = [f"{t:>10.2f}" for t in trials]
                while len(t_strs) < 3:
                    t_strs.append(f"{'':>10}")
                print(f"{key:<14} {t_strs[0]} {t_strs[1]} {t_strs[2]} {mean:>10.2f} {std:>10.2f}")
    print("-" * W)

    # --- Terrain traversal ratio comparison ---
    print()
    print("=" * W)
    print("TERRAIN TRAVERSAL RATIO (step_vel / flat_vel)")
    print("=" * W)
    print(f"{'Config':<14} {'Sim ratio':>12} {'Exp ratio':>12} {'Ratio diff':>12}")
    print("-" * W)
    for scene in scenes:
        for freq in freqs:
            key = f"{scene}_f{freq}"
            sim_s = SIM_STEP.get(key)
            sim_f = SIM_FLAT.get(key)
            exp_f = EXP_FLAT.get(key)
            sim_ratio = sim_s / sim_f if (sim_s and sim_f) else None

            if key in cond_stats:
                mean = cond_stats[key][0]
                exp_ratio = mean / exp_f if exp_f else None
                diff = (sim_ratio - exp_ratio) if (sim_ratio and exp_ratio) else None
                print(f"{key:<14} {sim_ratio:>12.3f} {exp_ratio:>12.3f} {diff:>+12.3f}")
            else:
                r_str = f"{sim_ratio:.3f}" if sim_ratio else "N/A"
                print(f"{key:<14} {r_str:>12} {'N/A':>12} {'N/A':>12}")
    print("-" * W)


if __name__ == "__main__":
    main()
