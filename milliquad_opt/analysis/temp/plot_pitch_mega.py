"""Mega figure: sim vs exp pitch time series for step terrain.

Rows = (scene, freq) conditions. Columns = [Sim, Exp].
Sim shows full spatial gate with vertical lines marking q60 sub-window.
Exp shows q60 index window [0.45n : 0.75n].
Y-axis matched across sim/exp per row.

Usage:
    uv run python -m analysis.temp.plot_pitch_mega
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
    "results/20260228T230022_step_q60_rk-warm/20260302T181206_validation_trajectories.npz"
)
CSV_PATH = NPZ_PATH.parent / "20260302T181206_validation_trials.csv"

# --- Exp data ---
EXP_CSV_DIR = Path(__file__).resolve().parent.parent.parent.parent / "experimental_data" / "csv" / "steps"

# Step geometry
STEP_START_X = 0.05
STEP_END_X = 0.05 + 7 * 0.0045 + 0.02  # 0.1015
CUTOFF_X = STEP_START_X + 0.65 * (STEP_END_X - STEP_START_X)  # RMS gate end
TRIM_X = STEP_END_X  # trajectory trim end (100%)

# Exp step conditions: (freq, morph, [files], [trial_indices])
EXP_STEP_CONDITIONS = [
    (10, "leg",   ["s10leg1-1.csv", "s10leg2-2.csv", "s10leg3-3.csv"], [1, 2, 3]),
    (10, "2leg",  ["s102leg1-1.csv", "s102leg2-2.csv", "s102leg3-3.csv"], [1, 2, 3]),
    (10, "4leg",  ["s104leg1-1.csv", "s104leg2-2.csv", "s104leg3-3.csv"], [1, 2, 3]),
    (20, "leg",   ["s20leg1-1.csv", "s20leg2-2.csv", "s20leg3-3.csv"], [1, 2, 3]),
    (20, "2leg",  ["s202leg1-1.csv", "s202leg2-2.csv", "s202leg3-3.csv"], [1, 2, 3]),
    (20, "4leg",  ["s204leg1-1.csv", "s204leg2-2.csv", "s204leg3-3.csv"], [1, 2, 3]),
    (30, "leg",   ["s30leg1-1.csv", "s30leg2-2.csv", "s30leg3-3.csv"], [1, 2, 3]),
    (30, "2leg",  ["s302leg1-1.csv", "s302leg2-2.csv", "s302leg3-3.csv"], [1, 2, 3]),
    (30, "4leg",  ["s304leg1-1.csv", "s304leg2-2.csv", "s304leg3-3.csv"], [1, 2, 3]),
    (30, "wheel", ["s30w1-1.csv", "s30w2-2.csv", "s30w3-3.csv"], [1, 2, 3]),
]

MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}
COLORS_SIM = ["#1f77b4", "#2ca02c", "#9467bd"]
COLORS_EXP = ["#d62728", "#ff7f0e", "#8c564b"]


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


def main():
    d = np.load(str(NPZ_PATH), allow_pickle=True)

    selected = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["selected"] == "True":
                rid = row["ref_id"]
                selected[rid].append(f"{rid}_t{row['trial']}")

    exp_data: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for freq, morph, files, idx in EXP_STEP_CONDITIONS:
        scene = MORPH_TO_SCENE[morph]
        trial_files = [files[i - 1] for i in idx]
        trials = []
        for fname in trial_files:
            try:
                trials.append(load_exp_pitch(fname))
            except Exception as e:
                print(f"  WARN: {fname}: {e}")
        exp_data[(scene, freq)] = trials

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
    fig, axes = plt.subplots(n_rows, 4, figsize=(26, 2.8 * n_rows), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 3, 2, 2]})
    fig.suptitle("Step Pitch: Sim vs Exp  |  Pitch (left) + Position (right)", fontsize=14, fontweight="bold")

    for r, (rid, scene, freq) in enumerate(rows):
        ax_sim, ax_exp, ax_xsim, ax_xexp = axes[r]

        # --- Sim: full trajectory with spatial gate shading ---
        rms_full_list = []
        sim_gate_bands = []  # (t_enter, t_exit, color) for deferred shading
        sim_xy_traces = []   # (x_all, z_all, x_gate, z_gate, color, label)
        for ci, tkey in enumerate(selected[rid]):
            pitch = d[f"{tkey}_pitch"]
            pos_x = d[f"{tkey}_pos_x"]
            pos_z = d[f"{tkey}_pos_z"]  # MuJoCo: z = lateral (top-down view)
            time = d[f"{tkey}_time"]
            color = COLORS_SIM[ci % len(COLORS_SIM)]
            label = f"t{tkey.split('_t')[-1]}"

            enter_idx = int(np.searchsorted(pos_x, STEP_START_X))
            gate_indices = np.where(pos_x >= CUTOFF_X)[0]
            if len(gate_indices) == 0:
                continue
            gate_idx = int(gate_indices[0])
            if gate_idx <= enter_idx or (gate_idx - enter_idx) < 10:
                continue

            # Trim point: 100% step end (or end of data if robot didn't reach)
            trim_indices = np.where(pos_x >= TRIM_X)[0]
            trim_idx = int(trim_indices[0]) if len(trim_indices) > 0 else len(pos_x) - 1

            # Plot trajectory up to step end (trim cliff-fall)
            ax_sim.plot(time[:trim_idx+1], pitch[:trim_idx+1], color=color, linewidth=0.6, alpha=0.8, label=label)

            # Pitch RMS over 75% spatial gate
            p_gate = pitch[enter_idx:gate_idx+1]
            rms_full_list.append(np.std(p_gate - p_gate[0]))

            # Store gate band for shading
            sim_gate_bands.append((time[enter_idx], time[gate_idx], color))

            # XY: traj up to step end + gated portion highlighted
            x_all = pos_x[:trim_idx+1] * 1000
            z_all = pos_z[:trim_idx+1] * 1000
            x_gate = pos_x[enter_idx:gate_idx+1] * 1000
            z_gate = pos_z[enter_idx:gate_idx+1] * 1000
            sim_xy_traces.append((x_all, z_all, x_gate, z_gate, color, label))

        # Add spatial gate shading bands on pitch (stacked in bottom 1/3)
        n_bands = len(sim_gate_bands)
        for bi, (t_lo, t_hi, color) in enumerate(sim_gate_bands):
            y_lo_frac = bi / max(n_bands, 1) / 3
            y_hi_frac = (bi + 1) / max(n_bands, 1) / 3
            ax_sim.axvspan(t_lo, t_hi, ymin=y_lo_frac, ymax=y_hi_frac,
                           color=color, alpha=0.2)

        rms_full = np.mean(rms_full_list) if rms_full_list else 0
        ax_sim.set_title(f"{rid}  SIM pitch  (spatial gate={rms_full:.1f}\u00b0)", fontsize=9)
        ax_sim.set_ylabel("Pitch (\u00b0)")
        ax_sim.legend(fontsize=7, loc="upper right")
        ax_sim.grid(True, alpha=0.3)

        # Sim XY path (top-down): thin=full traj, thick=gated portion
        for x_all, z_all, x_gate, z_gate, col, lbl in sim_xy_traces:
            ax_xsim.plot(x_all, z_all, color=col, linewidth=0.4, alpha=0.3)
            ax_xsim.plot(x_gate, z_gate, color=col, linewidth=1.0, alpha=0.8, label=lbl)
            ax_xsim.plot(x_gate[0], z_gate[0], "o", color=col, ms=3, alpha=0.7)
        ax_xsim.axvline(STEP_START_X * 1000, color="grey", ls="--", lw=0.8, alpha=0.5)
        ax_xsim.axvline(CUTOFF_X * 1000, color="grey", ls="-", lw=0.8, alpha=0.5)
        ax_xsim.set_title(f"{rid}  SIM XY", fontsize=9)
        ax_xsim.set_xlabel("x (mm)")
        ax_xsim.set_ylabel("y (mm)")
        ax_xsim.legend(fontsize=6, loc="upper left")
        # ax_xsim.set_aspect("equal")  # removed: lateral << forward squishes panel
        ax_xsim.grid(True, alpha=0.3)

        # --- Exp: full recording with q60 shading ---
        exp_pitches = []
        exp_q60_bands = []  # (t_lo, t_hi, color) for deferred shading on pitch
        exp_xy_traces = []  # (x, y, x_q60, y_q60, color, label)
        exp_cond = [(f, m, fls, idx) for f, m, fls, idx in EXP_STEP_CONDITIONS
                     if MORPH_TO_SCENE[m] == scene and f == freq]
        exp_trial_files = []
        if exp_cond:
            _, morph, files, idx = exp_cond[0]
            exp_trial_files = [files[i - 1] for i in idx]

        for ci, (t_exp, theta_exp) in enumerate(exp_data[(scene, freq)]):
            ne = len(theta_exp)
            lo_e = int(0.45 * ne)
            hi_e = int(0.75 * ne)
            t0_e = t_exp[0]
            color_e = COLORS_EXP[ci % len(COLORS_EXP)]

            # Pitch
            ax_exp.plot(t_exp - t0_e, theta_exp, color=color_e,
                        linewidth=0.6, alpha=0.8, label=f"trial {ci+1}")
            exp_pitches.append(np.std(theta_exp[lo_e:hi_e]))
            exp_q60_bands.append((t_exp[lo_e] - t0_e, t_exp[min(hi_e, ne-1)] - t0_e, color_e))

            # XY path from CSV: avg of mass_A (cols 1,2) and mass_C (cols 5,6)
            if ci < len(exp_trial_files):
                try:
                    csv_p = EXP_CSV_DIR / exp_trial_files[ci]
                    dat = np.genfromtxt(csv_p, delimiter=",", skip_header=2)
                    avg_x = (dat[:, 1] + dat[:, 5]) / 2
                    avg_y = (dat[:, 2] + dat[:, 6]) / 2
                    x_e = (avg_x[0] - avg_x) * 1000  # flip + zero-offset → mm forward
                    y_e = (avg_y - avg_y[0]) * 1000   # zero-offset, mm lateral
                    exp_xy_traces.append((x_e, y_e, x_e[lo_e:hi_e], y_e[lo_e:hi_e],
                                          color_e, f"t{ci+1}"))
                except Exception:
                    pass

        # Add q60 shading bands on pitch panel only
        n_bands_e = len(exp_q60_bands)
        for bi, (t_lo, t_hi, color_e) in enumerate(exp_q60_bands):
            y_lo_frac = bi / max(n_bands_e, 1) / 3
            y_hi_frac = (bi + 1) / max(n_bands_e, 1) / 3
            ax_exp.axvspan(t_lo, t_hi, ymin=y_lo_frac, ymax=y_hi_frac,
                           color=color_e, alpha=0.2)

        rms_exp = np.mean(exp_pitches) if exp_pitches else 0
        ax_exp.set_title(f"{rid}  EXP pitch  (q60={rms_exp:.1f}\u00b0)", fontsize=9)
        ax_exp.set_ylabel("Pitch (\u00b0)")
        ax_exp.legend(fontsize=7, loc="upper right")
        ax_exp.grid(True, alpha=0.3)

        # Exp XY path (top-down)
        for xf, yf, xq, yq, col, lbl in exp_xy_traces:
            ax_xexp.plot(xf, yf, color=col, linewidth=0.5, alpha=0.4)
            ax_xexp.plot(xq, yq, color=col, linewidth=1.5, alpha=0.9, label=lbl)
            ax_xexp.plot(xf[0], yf[0], "o", color=col, ms=3, alpha=0.7)
        ax_xexp.set_title(f"{rid}  EXP XY", fontsize=9)
        ax_xexp.set_xlabel("x (mm)")
        ax_xexp.set_ylabel("y (mm)")
        ax_xexp.legend(fontsize=6, loc="upper left")
        # ax_xexp.set_aspect("equal")
        ax_xexp.grid(True, alpha=0.3)

        # Match pitch y-limits across sim/exp
        ymin = min(ax_sim.get_ylim()[0], ax_exp.get_ylim()[0])
        ymax = max(ax_sim.get_ylim()[1], ax_exp.get_ylim()[1])
        ax_sim.set_ylim(ymin, ymax)
        ax_exp.set_ylim(ymin, ymax)

        # Match pitch time x-limits
        xmax_t = max(ax_sim.get_xlim()[1], ax_exp.get_xlim()[1])
        ax_sim.set_xlim(0, xmax_t)
        ax_exp.set_xlim(0, xmax_t)

        # Match XY x-limits across sim/exp
        xy_xmin = min(ax_xsim.get_xlim()[0], ax_xexp.get_xlim()[0])
        xy_xmax = max(ax_xsim.get_xlim()[1], ax_xexp.get_xlim()[1])
        ax_xsim.set_xlim(xy_xmin, xy_xmax)
        ax_xexp.set_xlim(xy_xmin, xy_xmax)

        if r == n_rows - 1:
            ax_sim.set_xlabel("Time (s)")
            ax_exp.set_xlabel("Time (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(__file__).resolve().parent / "pitch_mega_sim_vs_exp_0.65.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
