"""3-way overlay: Default params vs Optimized (flat_tg) vs Experiment for scene2 (2-leg) at 20Hz.

Picks one exp trial, then finds closest-velocity sim trial from each run dir.
All trajectories trimmed to exp duration.

Usage:
    uv run python -m analysis.investigative.plot_pitch_overlay_3way
"""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 8
import matplotlib.pyplot as plt
import numpy as np

SETTLE_TIME = 0.1
SIM_TIMESTEP = 1.0 / 2000.0
VX_SMOOTH_SIM = 100  # 50ms window at 2kHz
VX_SMOOTH_EXP = 10   # ~10ms window at ~1kHz

EXP_CSV_DIR = Path(__file__).resolve().parent.parent.parent.parent / "experimental_data" / "csv" / "flat"
EXP_FILES = ["f202leg1-1.csv", "f202leg2-2.csv", "f202leg3-3.csv"]

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
DEFAULT_DIR = RESULTS_DIR / "mujoco_defaults"
FLAT_TG_DIR = RESULTS_DIR / "20260303T192801_flat_tg"

EXP_COLOR = "#1f77b4"       # blue
OPT_COLOR = "#2ca02c"       # green
DEFAULT_COLOR = "#d62728"   # red


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
    theta_col = _body_theta_col(csv_path)
    dat = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    t = dat[:, 0] - dat[0, 0]
    pitch = dat[:, theta_col]
    avg_x = (dat[:, 1] + dat[:, 5]) / 2
    x_forward = avg_x[0] - avg_x
    dt = np.diff(t)
    dt[dt == 0] = 1e-6
    avg_vel = x_forward[-1] / t[-1] if t[-1] > 0 else 0.0
    return {"time": t, "pitch": pitch, "x_forward": x_forward, "avg_vel": avg_vel, "dt": dt}


def _load_sim_trials(run_dir: Path) -> list[dict]:
    npz_candidates = sorted(run_dir.glob("*_validation_trajectories.npz"))
    csv_candidates = sorted(run_dir.glob("*_validation_trials.csv"))
    if not npz_candidates or not csv_candidates:
        raise FileNotFoundError(f"No validation NPZ/CSV in {run_dir}")
    d = np.load(str(npz_candidates[-1]), allow_pickle=True)
    tkeys = []
    with open(csv_candidates[-1]) as f:
        for row in csv.DictReader(f):
            if row["ref_id"] == "scene2_f20" and row.get("crash", "").lower() != "true":
                tkeys.append(f"scene2_f20_t{row['trial']}")
    trials = []
    for tkey in tkeys:
        pos_x = d[f"{tkey}_pos_x"]
        time = d[f"{tkey}_time"]
        pitch = d[f"{tkey}_pitch"]
        settle_idx = int(np.searchsorted(time, SETTLE_TIME))
        dt = time[-1] - time[settle_idx]
        avg_vel = (pos_x[-1] - pos_x[settle_idx]) / dt if dt > 1e-6 else 0.0
        trials.append({"tkey": tkey, "pos_x": pos_x, "time": time, "pitch": pitch,
                        "settle_idx": settle_idx, "avg_vel": avg_vel})
    return trials


def _best_match(sim_trials: list[dict], target_vel: float) -> dict:
    best_idx = min(range(len(sim_trials)),
                   key=lambda i: abs(sim_trials[i]["avg_vel"] - target_vel))
    return sim_trials[best_idx]


def _trim_sim(sim: dict, exp_dur: float):
    """Trim sim trajectory: find start (near-zero vel before settle), end at settle + exp_dur."""
    full_vx = np.diff(sim["pos_x"]) / SIM_TIMESTEP * 100
    full_vx_smooth = _smooth_vx(full_vx, VX_SMOOTH_SIM)
    start_idx = 0
    for i in range(sim["settle_idx"], -1, -1):
        if i < len(full_vx_smooth) and abs(full_vx_smooth[i]) < 0.5:
            start_idx = i
            break
    start_time = sim["time"][start_idx]
    end_time = SETTLE_TIME + exp_dur
    end_idx = min(int(np.searchsorted(sim["time"], end_time)), len(sim["time"]))
    t = sim["time"][start_idx:end_idx] - start_time
    pitch = sim["pitch"][start_idx:end_idx]
    px = sim["pos_x"][start_idx:end_idx]
    vx_raw = np.diff(px) / SIM_TIMESTEP * 100
    vx = _smooth_vx(vx_raw, VX_SMOOTH_SIM)
    return t, pitch, vx


def main():
    # Load exp
    exp_trials = []
    for fname in EXP_FILES:
        try:
            exp_trials.append(_load_exp_trial(EXP_CSV_DIR / fname))
        except Exception as e:
            print(f"  WARN: {fname}: {e}")

    # Load sim from both runs
    print("Loading default params...")
    default_trials = _load_sim_trials(DEFAULT_DIR)
    print(f"  {len(default_trials)} trials")
    print("Loading flat_tg params...")
    flat_tg_trials = _load_sim_trials(FLAT_TG_DIR)
    print(f"  {len(flat_tg_trials)} trials")

    # Pick exp trial with median velocity
    exp_vels = [e["avg_vel"] for e in exp_trials]
    median_idx = int(np.argsort(exp_vels)[len(exp_vels) // 2])
    exp = exp_trials[median_idx]
    print(f"Exp trial {median_idx+1}: vel={exp['avg_vel']*100:.2f} cm/s (median)")

    # Match each sim run to this exp trial's velocity
    sim_default = _best_match(default_trials, exp["avg_vel"])
    sim_opt = _best_match(flat_tg_trials, exp["avg_vel"])
    print(f"Default: {sim_default['tkey']} vel={sim_default['avg_vel']*100:.2f} cm/s")
    print(f"Flat_tg: {sim_opt['tkey']} vel={sim_opt['avg_vel']*100:.2f} cm/s")

    exp_dur = exp["time"][-1]
    t_def, pitch_def, vx_def = _trim_sim(sim_default, exp_dur)
    t_opt, pitch_opt, vx_opt = _trim_sim(sim_opt, exp_dur)

    exp_vx_raw = np.diff(exp["x_forward"]) / exp["dt"] * 100
    exp_vx = _smooth_vx(exp_vx_raw, VX_SMOOTH_EXP)

    # --- Plot (vertical stack: velocity on top, pitch on bottom) ---
    fig, (ax_vel, ax_pitch) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)

    for ax in (ax_pitch, ax_vel):
        ax.tick_params(axis="both", labelsize=12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Velocity (top) — no x-axis label/ticks, has legend
    ax_vel.plot(t_def[:-1], vx_def, color=DEFAULT_COLOR, linewidth=1.0, alpha=0.8, label="Default")
    ax_vel.plot(t_opt[:-1], vx_opt, color=OPT_COLOR, linewidth=1.0, alpha=0.9, label="Optimized")
    ax_vel.plot(exp["time"][:-1], exp_vx, color=EXP_COLOR, linewidth=1.0, alpha=0.9, label="Experimental")
    ax_vel.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax_vel.set_ylabel("Vx (cm/s)", fontsize=14)
    ax_vel.yaxis.set_label_coords(-0.09, 0.5)
    ax_vel.legend(fontsize=12, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.3),
                  frameon=False)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.tick_params(axis="x", labelbottom=False)

    # Pitch (bottom) — has x-axis label
    ax_pitch.plot(t_def, pitch_def, color=DEFAULT_COLOR, linewidth=1.0, alpha=0.8)
    ax_pitch.plot(t_opt, pitch_opt, color=OPT_COLOR, linewidth=1.0, alpha=0.9)
    ax_pitch.plot(exp["time"], exp["pitch"], color=EXP_COLOR, linewidth=1.0, alpha=0.9)
    ax_pitch.set_xlabel("Time (s)", fontsize=14)
    ax_pitch.set_ylabel("Pitch (\u00b0)", fontsize=14)
    ax_pitch.yaxis.set_label_coords(-0.09, 0.5)
    ax_pitch.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax_pitch.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"{ts}_pitch_overlay_3way.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
