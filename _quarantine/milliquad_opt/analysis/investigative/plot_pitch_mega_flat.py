"""Mega figure: sim vs exp pitch time series for flat terrain.

Rows = (scene, freq) conditions.
Columns = [Sim pitch, Exp pitch, Sim XY, Exp XY, Velocity (sim+exp overlay)].
Sim pitch/XY trimmed to longest exp trial duration (settle stripped).
Velocity column shows full untrimmed trajectories for both.
Y-axis matched across sim/exp per row.

Usage:
    uv run python -m analysis.investigative.plot_pitch_mega_flat
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Sim data ---
NPZ_PATH = Path(__file__).resolve().parent.parent.parent / (
    "results/20260228T013353_rk4_flat/20260302T181101_validation_trajectories.npz"
)
CSV_PATH = NPZ_PATH.parent / "20260302T181101_validation_trials.csv"

# --- Exp data ---
EXP_CSV_DIR = Path(__file__).resolve().parent.parent.parent.parent / "experimental_data" / "csv" / "flat"

SETTLE_TIME = 0.1  # seconds (from config.py)
SIM_TIMESTEP = 1.0 / 2000.0  # 2 kHz

# Exp flat conditions: (freq, morph, [files], [trial_indices])
EXP_FLAT_CONDITIONS = [
    (10, "leg",   ["f10leg1-1.csv", "f10leg2-2.csv", "f10leg3-3.csv"], [1, 2, 3]),
    (10, "2leg",  ["f102leg1-1.csv", "f102leg2-2.csv", "f102leg3-3.csv"], [1, 2, 3]),
    (10, "4leg",  ["f104leg1-1.csv", "f104leg2-2.csv", "f104leg3-3.csv"], [1, 2, 3]),
    (10, "wheel", ["f10w1-1.csv", "f10w2-2.csv", "f10w3-3.csv"], [1, 2, 3]),
    (20, "leg",   ["f20leg1-1.csv", "f20leg2-2.csv", "f20leg3-3.csv"], [1, 2, 3]),
    (20, "2leg",  ["f202leg1-1.csv", "f202leg2-2.csv", "f202leg3-3.csv"], [1, 2, 3]),
    (20, "4leg",  ["f204leg1-1.csv", "f204leg2-2.csv", "f204leg3-3.csv"], [1, 2, 3]),
    (20, "wheel", ["f20w1-1.csv", "f20w2-2.csv", "f20w3-3.csv"], [1, 2, 3]),
    (30, "leg",   ["f30leg1-1.csv", "f30leg2-2.csv", "f30leg3-3.csv"], [1, 2, 3]),
    (30, "2leg",  ["f302leg1-1.csv", "f302leg2-2.csv", "f302leg3-3.csv"], [1, 2, 3]),
    (30, "4leg",  ["f304leg1-1.csv", "f304leg2-2.csv", "f304leg3-3.csv"], [1, 2, 3]),
    (30, "wheel", ["f30w1-1.csv", "f30w2-2.csv", "f30w3-3.csv"], [1, 2, 3]),
    (50, "leg",   ["f50leg1-1.csv", "f50leg2-2.csv", "f50leg3-3.csv"], [1, 2, 3]),
    (50, "2leg",  ["f502leg1-1.csv", "f502leg2-2.csv", "f502leg3-3.csv"], [1, 2, 3]),
    (50, "4leg",  ["f504leg1-1.csv", "f504leg2-2.csv", "f504leg3-3.csv"], [1, 2, 3]),
    (50, "wheel", ["f50w1-1.csv", "f50w2-2.csv", "f50w3-3.csv"], [1, 2, 3]),
]

MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}
COLORS_SIM = ["#1f77b4", "#2ca02c", "#9467bd"]
COLORS_EXP = ["#d62728", "#ff7f0e", "#8c564b"]

# Rolling window for vx smoothing (in samples)
VX_SMOOTH_SIM = 100   # 100 samples at 2kHz = 50ms
VX_SMOOTH_EXP = 10    # 10 samples at ~1kHz = ~10ms


def _body_theta_col(csv_path: Path) -> int:
    with open(csv_path) as f:
        row1 = f.readline().strip().split(",")
        row2 = f.readline().strip().split(",")
    groups, current = [], ""
    for label in row1:
        if label:
            current = label
        groups.append(current)
    theta_cols = [i for i in range(len(row2)) if row2[i] == "\u03b8" and i < len(groups) and groups[i] == "mass_C"]
    if theta_cols:
        return theta_cols[-1]
    raise ValueError(f"No mass_C \u03b8 column in {csv_path}")


def load_exp_pitch(csv_name: str) -> tuple[np.ndarray, np.ndarray]:
    csv_path = EXP_CSV_DIR / csv_name
    theta_col = _body_theta_col(csv_path)
    dat = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    return dat[:, 0], dat[:, theta_col]


def _smooth_vx(vx: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(vx) < window:
        return vx
    kernel = np.ones(window) / window
    return np.convolve(vx, kernel, mode="same")


def main():
    d = np.load(str(NPZ_PATH), allow_pickle=True)

    selected = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["selected"] == "True":
                rid = row["ref_id"]
                selected[rid].append(f"{rid}_t{row['trial']}")

    exp_data: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    exp_file_map: dict[tuple[str, int], list[str]] = {}
    for freq, morph, files, idx in EXP_FLAT_CONDITIONS:
        scene = MORPH_TO_SCENE[morph]
        trial_files = [files[i - 1] for i in idx]
        trials = []
        for fname in trial_files:
            try:
                trials.append(load_exp_pitch(fname))
            except Exception as e:
                print(f"  WARN: {fname}: {e}")
        exp_data[(scene, freq)] = trials
        exp_file_map[(scene, freq)] = trial_files

    # Build row list: only conditions that have both sim and exp data
    rows = []
    for rid in sorted(selected):
        parts = rid.split("_")
        if "wheel" in rid:
            scene = "scene_wheel"
            freq = int(parts[-1][1:])
        else:
            scene = parts[0]
            freq = int(parts[1][1:])
        if (scene, freq) in exp_data and exp_data[(scene, freq)]:
            rows.append((rid, scene, freq))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 5, figsize=(32, 2.8 * n_rows), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 3, 2, 2, 3]})
    fig.suptitle("Flat: Sim vs Exp  |  Pitch + Position + Velocity", fontsize=14, fontweight="bold")

    for r, (rid, scene, freq) in enumerate(rows):
        ax_sim, ax_exp, ax_xsim, ax_xexp, ax_vel = axes[r]

        # Find max exp trial duration to trim sim pitch/XY
        max_exp_dur = 0.0
        for t_exp, _ in exp_data.get((scene, freq), []):
            dur = t_exp[-1] - t_exp[0]
            if dur > max_exp_dur:
                max_exp_dur = dur

        # --- Sim: strip settle, trim pitch/XY to longest exp trial duration ---
        rms_list = []
        sim_xy_traces = []
        for ci, tkey in enumerate(selected[rid]):
            pitch_full = d[f"{tkey}_pitch"]
            pos_x_full = d[f"{tkey}_pos_x"]
            pos_z_full = d[f"{tkey}_pos_z"]
            time_full = d[f"{tkey}_time"]
            color = COLORS_SIM[ci % len(COLORS_SIM)]
            label = f"t{tkey.split('_t')[-1]}"

            settle_idx = int(np.searchsorted(time_full, SETTLE_TIME))

            # --- Trimmed for pitch/XY ---
            end_time = SETTLE_TIME + max_exp_dur if max_exp_dur > 0 else time_full[-1]
            end_idx = min(int(np.searchsorted(time_full, end_time)), len(time_full))

            pitch = pitch_full[settle_idx:end_idx]
            pos_x = pos_x_full[settle_idx:end_idx]
            pos_z = pos_z_full[settle_idx:end_idx]
            time_trim = time_full[settle_idx:end_idx] - SETTLE_TIME

            ax_sim.plot(time_trim, pitch, color=color, linewidth=0.6, alpha=0.8, label=label)
            rms_list.append(np.std(pitch - pitch[0]))

            x0 = np.mean(pos_x[:50])
            z0 = np.mean(pos_z[:50])
            x_all = (pos_x - x0) * 1000
            z_all = (pos_z - z0) * 1000
            sim_xy_traces.append((x_all, z_all, color, label))

            # --- Untrimmed vx for velocity column ---
            px_active = pos_x_full[settle_idx:]
            t_active = time_full[settle_idx:] - SETTLE_TIME
            vx_raw = np.diff(px_active) / SIM_TIMESTEP * 100  # m/s -> cm/s
            vx_smooth = _smooth_vx(vx_raw, VX_SMOOTH_SIM)
            ax_vel.plot(t_active[:-1], vx_smooth, color=color, linewidth=0.6, alpha=0.7,
                        label=f"sim {label}")

        rms_mean = np.mean(rms_list) if rms_list else 0
        ax_sim.set_title(f"{rid}  SIM pitch  (RMS={rms_mean:.1f}\u00b0)", fontsize=9)
        ax_sim.set_ylabel("Pitch (\u00b0)")
        ax_sim.legend(fontsize=7, loc="upper right")
        ax_sim.grid(True, alpha=0.3)

        # Sim XY path
        for x_all, z_all, col, lbl in sim_xy_traces:
            ax_xsim.plot(x_all, z_all, color=col, linewidth=1.0, alpha=0.8, label=lbl)
            ax_xsim.plot(x_all[0], z_all[0], "o", color=col, ms=3, alpha=0.7)
        ax_xsim.set_title(f"{rid}  SIM XY", fontsize=9)
        ax_xsim.set_xlabel("x (mm)")
        ax_xsim.set_ylabel("y (mm)")
        ax_xsim.legend(fontsize=6, loc="upper left")
        ax_xsim.grid(True, alpha=0.3)

        # --- Exp: full recording ---
        exp_pitches = []
        exp_xy_traces = []
        trial_files = exp_file_map.get((scene, freq), [])

        for ci, (t_exp, theta_exp) in enumerate(exp_data[(scene, freq)]):
            t0_e = t_exp[0]
            color_e = COLORS_EXP[ci % len(COLORS_EXP)]

            ax_exp.plot(t_exp - t0_e, theta_exp, color=color_e,
                        linewidth=0.6, alpha=0.8, label=f"trial {ci+1}")
            exp_pitches.append(np.std(theta_exp))

            # XY + velocity from CSV
            if ci < len(trial_files):
                try:
                    csv_p = EXP_CSV_DIR / trial_files[ci]
                    dat = np.genfromtxt(csv_p, delimiter=",", skip_header=2)
                    t_csv = dat[:, 0] - dat[0, 0]
                    avg_x = (dat[:, 1] + dat[:, 5]) / 2
                    avg_y = (dat[:, 2] + dat[:, 6]) / 2

                    # XY path
                    x0_e = np.mean(avg_x[:50])
                    y0_e = np.mean(avg_y[:50])
                    x_e = (x0_e - avg_x) * 1000
                    y_e = (avg_y - y0_e) * 1000
                    exp_xy_traces.append((x_e, y_e, color_e, f"t{ci+1}"))

                    # Velocity: dx/dt from position, sign-flipped (exp x decreases forward)
                    dt_e = np.diff(t_csv)
                    dt_e[dt_e == 0] = 1e-6
                    vx_e = -np.diff(avg_x) / dt_e * 100  # cm/s, flipped
                    vx_e_smooth = _smooth_vx(vx_e, VX_SMOOTH_EXP)
                    ax_vel.plot(t_csv[:-1], vx_e_smooth, color=color_e, linewidth=0.6,
                                alpha=0.7, label=f"exp t{ci+1}")
                except Exception:
                    pass

        rms_exp = np.mean(exp_pitches) if exp_pitches else 0
        ax_exp.set_title(f"{rid}  EXP pitch  (RMS={rms_exp:.1f}\u00b0)", fontsize=9)
        ax_exp.set_ylabel("Pitch (\u00b0)")
        ax_exp.legend(fontsize=7, loc="upper right")
        ax_exp.grid(True, alpha=0.3)

        # Exp XY path
        for xf, yf, col, lbl in exp_xy_traces:
            ax_xexp.plot(xf, yf, color=col, linewidth=1.0, alpha=0.8, label=lbl)
            ax_xexp.plot(xf[0], yf[0], "o", color=col, ms=3, alpha=0.7)
        ax_xexp.set_title(f"{rid}  EXP XY", fontsize=9)
        ax_xexp.set_xlabel("x (mm)")
        ax_xexp.set_ylabel("y (mm)")
        ax_xexp.legend(fontsize=6, loc="upper left")
        ax_xexp.grid(True, alpha=0.3)

        # Velocity column labels
        ax_vel.set_title(f"{rid}  vx (untrimmed)", fontsize=9)
        ax_vel.set_ylabel("vx (cm/s)")
        ax_vel.legend(fontsize=5, loc="upper right", ncol=2)
        ax_vel.grid(True, alpha=0.3)
        ax_vel.axhline(0, color="grey", lw=0.5, alpha=0.5)

        # Match pitch y-limits across sim/exp
        ymin = min(ax_sim.get_ylim()[0], ax_exp.get_ylim()[0])
        ymax = max(ax_sim.get_ylim()[1], ax_exp.get_ylim()[1])
        ax_sim.set_ylim(ymin, ymax)
        ax_exp.set_ylim(ymin, ymax)

        # Match pitch time x-limits
        xmax_t = max(ax_sim.get_xlim()[1], ax_exp.get_xlim()[1])
        ax_sim.set_xlim(0, xmax_t)
        ax_exp.set_xlim(0, xmax_t)

        # Match XY limits across sim/exp
        xy_xmin = min(ax_xsim.get_xlim()[0], ax_xexp.get_xlim()[0])
        xy_xmax = max(ax_xsim.get_xlim()[1], ax_xexp.get_xlim()[1])
        ax_xsim.set_xlim(xy_xmin, xy_xmax)
        ax_xexp.set_xlim(xy_xmin, xy_xmax)

        xy_ymin = min(ax_xsim.get_ylim()[0], ax_xexp.get_ylim()[0])
        xy_ymax = max(ax_xsim.get_ylim()[1], ax_xexp.get_ylim()[1])
        ax_xsim.set_ylim(xy_ymin, xy_ymax)
        ax_xexp.set_ylim(xy_ymin, xy_ymax)

        if r == n_rows - 1:
            ax_sim.set_xlabel("Time (s)")
            ax_exp.set_xlabel("Time (s)")
            ax_vel.set_xlabel("Time (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(__file__).resolve().parent / "pitch_mega_sim_vs_exp_flat.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
