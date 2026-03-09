"""Overlay sim vs exp for scene2 (2-leg) at 20Hz: pitch + velocity.

Picks the (sim, exp) trial pair with closest average velocity.
Sim trajectory trimmed to exp trial duration.

Usage:
    uv run python -m analysis.investigative.plot_pitch_overlay_2leg_f20 \
        --run-dir results/20260303T192801_flat_tg
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETTLE_TIME = 0.1
SIM_TIMESTEP = 1.0 / 2000.0
VX_SMOOTH_SIM = 100  # 50ms window at 2kHz
VX_SMOOTH_EXP = 10   # ~10ms window at ~1kHz

EXP_CSV_DIR = Path(__file__).resolve().parent.parent.parent.parent / "experimental_data" / "csv" / "flat"
EXP_FILES = ["f202leg1-1.csv", "f202leg2-2.csv", "f202leg3-3.csv"]

SIM_COLOR = "#1f77b4"
EXP_COLOR = "#d62728"


def _body_theta_col(csv_path: Path) -> int:
    with open(csv_path) as f:
        row1 = f.readline().strip().split(",")
        row2 = f.readline().strip().split(",")
    groups, current = [], ""
    for label in row1:
        if label:
            current = label
        groups.append(current)
    theta_cols = [i for i in range(len(row2))
                  if row2[i] == "\u03b8" and i < len(groups) and groups[i] == "mass_C"]
    if theta_cols:
        return theta_cols[-1]
    raise ValueError(f"No mass_C \u03b8 column in {csv_path}")


def _smooth_vx(vx: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(vx) < window:
        return vx
    kernel = np.ones(window) / window
    return np.convolve(vx, kernel, mode="same")


def _load_exp_trial(csv_path: Path) -> dict:
    """Load time, pitch, position from an exp CSV."""
    theta_col = _body_theta_col(csv_path)
    dat = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    t = dat[:, 0] - dat[0, 0]
    pitch = dat[:, theta_col]
    # Position: avg of mass_A (cols 1,2) and mass_C (cols 5,6)
    avg_x = (dat[:, 1] + dat[:, 5]) / 2
    # Exp x decreases forward, flip sign
    x_forward = avg_x[0] - avg_x
    dt = np.diff(t)
    dt[dt == 0] = 1e-6
    avg_vel = x_forward[-1] / t[-1] if t[-1] > 0 else 0.0
    return {"time": t, "pitch": pitch, "x_forward": x_forward, "avg_vel": avg_vel, "dt": dt}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir

    # Discover sim data
    npz_candidates = sorted(run_dir.glob("*_validation_trajectories.npz"))
    csv_candidates = sorted(run_dir.glob("*_validation_trials.csv"))
    if not npz_candidates or not csv_candidates:
        raise FileNotFoundError(f"No validation NPZ/CSV in {run_dir}")
    npz_path, csv_path = npz_candidates[-1], csv_candidates[-1]
    print(f"NPZ: {npz_path.name}")
    print(f"CSV: {csv_path.name}")

    d = np.load(str(npz_path), allow_pickle=True)

    # Get sim trial keys for scene2_f20
    sim_tkeys = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["ref_id"] == "scene2_f20" and row.get("crash", "").lower() != "true":
                sim_tkeys.append(f"scene2_f20_t{row['trial']}")
    if not sim_tkeys:
        raise ValueError("No scene2_f20 sim trials found in CSV")
    print(f"Sim trials: {sim_tkeys}")

    # Compute sim avg velocities (post-settle, full duration)
    sim_trials = []
    for tkey in sim_tkeys:
        pos_x = d[f"{tkey}_pos_x"]
        time = d[f"{tkey}_time"]
        pitch = d[f"{tkey}_pitch"]
        settle_idx = int(np.searchsorted(time, SETTLE_TIME))
        dt = time[-1] - time[settle_idx]
        avg_vel = (pos_x[-1] - pos_x[settle_idx]) / dt if dt > 1e-6 else 0.0
        sim_trials.append({"tkey": tkey, "pos_x": pos_x, "time": time, "pitch": pitch,
                           "settle_idx": settle_idx, "avg_vel": avg_vel})

    # Load exp trials
    exp_trials = []
    for fname in EXP_FILES:
        csv_p = EXP_CSV_DIR / fname
        try:
            exp_trials.append(_load_exp_trial(csv_p))
        except Exception as e:
            print(f"  WARN: {fname}: {e}")
    if not exp_trials:
        raise ValueError("No exp trials loaded")

    # Find best (sim, exp) pair: minimize |sim_avg_vel - exp_avg_vel|
    best_diff = float("inf")
    best_si, best_ei = 0, 0
    for si, st in enumerate(sim_trials):
        for ei, et in enumerate(exp_trials):
            diff = abs(st["avg_vel"] - et["avg_vel"])
            if diff < best_diff:
                best_diff = diff
                best_si, best_ei = si, ei

    sim = sim_trials[best_si]
    exp = exp_trials[best_ei]
    print(f"Best pair: sim {sim['tkey']} (vel={sim['avg_vel']*100:.2f} cm/s) "
          f"+ exp trial {best_ei+1} (vel={exp['avg_vel']*100:.2f} cm/s), "
          f"diff={best_diff*100:.2f} cm/s")

    # Trim sim: start where instantaneous velocity first crosses ~0, working
    # backward from settle_idx.  Use smoothed vx to avoid noise.
    full_vx = np.diff(sim["pos_x"]) / SIM_TIMESTEP * 100  # cm/s
    full_vx_smooth = _smooth_vx(full_vx, VX_SMOOTH_SIM)
    sim_start_idx = 0
    for i in range(sim["settle_idx"], -1, -1):
        if i < len(full_vx_smooth) and abs(full_vx_smooth[i]) < 0.5:  # < 0.5 cm/s ≈ zero
            sim_start_idx = i
            break
    sim_start_time = sim["time"][sim_start_idx]
    print(f"Sim start: t={sim_start_time*1000:.1f} ms (first near-zero vel before settle)")
    exp_dur = exp["time"][-1]
    sim_end_time = SETTLE_TIME + exp_dur
    sim_end_idx = min(int(np.searchsorted(sim["time"], sim_end_time)), len(sim["time"]))
    sim_t = sim["time"][sim_start_idx:sim_end_idx] - sim_start_time
    sim_pitch = sim["pitch"][sim_start_idx:sim_end_idx]
    sim_px = sim["pos_x"][sim_start_idx:sim_end_idx]

    # Sim velocity (trimmed)
    sim_vx_raw = np.diff(sim_px) / SIM_TIMESTEP * 100  # cm/s
    sim_vx = _smooth_vx(sim_vx_raw, VX_SMOOTH_SIM)

    # Exp velocity
    exp_vx_raw = np.diff(exp["x_forward"]) / exp["dt"] * 100  # cm/s
    exp_vx = _smooth_vx(exp_vx_raw, VX_SMOOTH_EXP)

    # --- Plot ---
    fig, (ax_pitch, ax_vel) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"2-Leg @ 20 Hz: Sim vs Exp (best-matched pair)", fontsize=13, fontweight="bold")

    # Pitch overlay
    ax_pitch.plot(exp["time"], exp["pitch"], color=EXP_COLOR, linewidth=1.0,
                  alpha=0.9, label=f"Exp trial {best_ei+1}")
    ax_pitch.plot(sim_t, sim_pitch, color=SIM_COLOR, linewidth=1.0,
                  alpha=0.9, label=f"Sim {sim['tkey'].split('_t')[-1]}")
    ax_pitch.set_xlabel("Time (s)")
    ax_pitch.set_ylabel("Pitch (\u00b0)")
    ax_pitch.set_title("Pitch")
    ax_pitch.legend(fontsize=9)
    ax_pitch.grid(True, alpha=0.3)

    # Velocity overlay
    ax_vel.plot(exp["time"][:-1], exp_vx, color=EXP_COLOR, linewidth=1.0,
                alpha=0.9, label=f"Exp trial {best_ei+1}")
    ax_vel.plot(sim_t[:-1], sim_vx, color=SIM_COLOR, linewidth=1.0,
                alpha=0.9, label=f"Sim {sim['tkey'].split('_t')[-1]}")
    ax_vel.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("vx (cm/s)")
    ax_vel.set_title("Forward Velocity")
    ax_vel.legend(fontsize=9)
    ax_vel.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = run_dir / f"{ts}_pitch_overlay_2leg_f20.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
