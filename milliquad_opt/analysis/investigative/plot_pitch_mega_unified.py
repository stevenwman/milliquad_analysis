"""Unified pitch mega figure: sim vs exp pitch/XY/velocity for flat or step terrain.

Auto-discovers NPZ + CSV from a run dir. Supports trial selection (top N closest
to target velocity) or all trials.

Usage:
    uv run python -m analysis.investigative.plot_pitch_mega_unified \
        --terrain flat --run-dir results/20260303T192801_flat_tg --n-select 3

    uv run python -m analysis.investigative.plot_pitch_mega_unified \
        --terrain step --run-dir results/20260303T151416_step_065gate
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SETTLE_TIME = 0.1
SIM_TIMESTEP = 1.0 / 2000.0

# Step geometry
STEP_START_X = 0.05
STEP_END_X = 0.05 + 7 * 0.0045 + 0.02  # 0.1015
CUTOFF_X = STEP_START_X + 0.65 * (STEP_END_X - STEP_START_X)

# Flat trial durations (mean exp recording length, seconds)
_FLAT_TRIAL_DURATION: dict[tuple[str, float], float] = {
    ("scene1", 10.0): 2.625, ("scene1", 20.0): 1.093, ("scene1", 30.0): 1.197,
    ("scene1", 50.0): 1.023,
    ("scene2", 10.0): 1.567, ("scene2", 20.0): 1.021, ("scene2", 30.0): 0.827,
    ("scene2", 50.0): 0.663,
    ("scene4", 10.0): 1.245, ("scene4", 20.0): 0.712, ("scene4", 30.0): 0.589,
    ("scene4", 50.0): 0.547,
    ("scene_wheel", 10.0): 0.965, ("scene_wheel", 20.0): 0.478,
    ("scene_wheel", 30.0): 0.384,
}

MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}
COLORS_SIM = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#bcbd22",
              "#17becf", "#7f7f7f", "#8c564b", "#d62728", "#ff7f0e"]
COLORS_EXP = ["#d62728", "#ff7f0e", "#8c564b"]

VX_SMOOTH_SIM = 100
VX_SMOOTH_EXP = 10

# ---------------------------------------------------------------------------
# Experimental conditions
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _load_exp_pitch(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    theta_col = _body_theta_col(csv_path)
    dat = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    return dat[:, 0], dat[:, theta_col]


def _smooth_vx(vx: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(vx) < window:
        return vx
    kernel = np.ones(window) / window
    return np.convolve(vx, kernel, mode="same")


def _discover_files(run_dir: Path) -> tuple[Path, Path]:
    """Find latest NPZ and CSV in run dir."""
    npz_candidates = sorted(run_dir.glob("*_validation_trajectories.npz"))
    csv_candidates = sorted(run_dir.glob("*_validation_trials.csv"))
    if not npz_candidates:
        raise FileNotFoundError(f"No *_validation_trajectories.npz in {run_dir}")
    if not csv_candidates:
        raise FileNotFoundError(f"No *_validation_trials.csv in {run_dir}")
    return npz_candidates[-1], csv_candidates[-1]


def _parse_rid(rid: str) -> tuple[str, int]:
    """Parse ref_id like 'scene1_f10' or 'scene_wheel_f30' into (scene, freq)."""
    parts = rid.split("_")
    if "wheel" in rid:
        scene = "scene_wheel"
        freq = int(parts[-1][1:])
    else:
        scene = parts[0]
        freq = int(parts[1][1:])
    return scene, freq


def _compute_trial_velocity_flat(npz_data, tkey: str, scene: str, freq: float) -> float:
    """Compute avg forward velocity for a flat trial over the time-gated window."""
    pos_x = npz_data[f"{tkey}_pos_x"]
    time = npz_data[f"{tkey}_time"]
    settle_idx = int(np.searchsorted(time, SETTLE_TIME))
    trial_dur = _FLAT_TRIAL_DURATION.get((scene, freq))
    if trial_dur is not None:
        end_time = SETTLE_TIME + trial_dur
        end_idx = min(int(np.searchsorted(time, end_time)), len(time) - 1)
    else:
        end_idx = len(time) - 1
    dt = time[end_idx] - time[settle_idx]
    if dt < 1e-6:
        return 0.0
    return (pos_x[end_idx] - pos_x[settle_idx]) / dt


def _compute_trial_velocity_step(npz_data, tkey: str) -> float:
    """Compute avg forward velocity for a step trial over the 65% spatial gate."""
    pos_x = npz_data[f"{tkey}_pos_x"]
    time = npz_data[f"{tkey}_time"]
    enter_indices = np.where(pos_x >= STEP_START_X)[0]
    if len(enter_indices) == 0:
        return 0.0
    enter_idx = int(enter_indices[0])
    gate_indices = np.where(pos_x >= CUTOFF_X)[0]
    if len(gate_indices) == 0:
        return 0.0
    gate_idx = int(gate_indices[0])
    dt = time[gate_idx] - time[enter_idx]
    if dt < 1e-6:
        return 0.0
    return (pos_x[gate_idx] - pos_x[enter_idx]) / dt


def _select_trials(all_tkeys: dict[str, list[str]], npz_data, terrain: str,
                   n_select: int, target_velocities: dict[str, float]) -> dict[str, list[str]]:
    """Select n_select trials per ref_id closest to target velocity."""
    selected = {}
    for rid, tkeys in all_tkeys.items():
        if len(tkeys) <= n_select:
            selected[rid] = tkeys
            continue

        scene, freq = _parse_rid(rid)
        target = target_velocities.get(rid, 0.0)

        scored = []
        for tkey in tkeys:
            if terrain == "flat":
                vel = _compute_trial_velocity_flat(npz_data, tkey, scene, freq)
            else:
                vel = _compute_trial_velocity_step(npz_data, tkey)
            scored.append((abs(vel - target), tkey))
        scored.sort(key=lambda x: x[0])
        selected[rid] = [tkey for _, tkey in scored[:n_select]]

    return selected


def _load_target_velocities(terrain: str) -> dict[str, float]:
    """Load target velocities from the appropriate config module."""
    import importlib, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    if terrain == "flat":
        cfg = importlib.import_module("config_flat_tg")
    else:
        cfg = importlib.import_module("config_step_065")
    from config import reference_rows
    rows = reference_rows(cfg.REFERENCE_DATA)
    return {row["id"]: row["speed"] for row in rows}


# ---------------------------------------------------------------------------
# Flat terrain plotting
# ---------------------------------------------------------------------------

def _plot_flat(fig, axes, rows, npz_data, trial_map, exp_data, exp_file_map, exp_csv_dir):
    """Plot flat terrain: 5 columns per row."""
    for r, (rid, scene, freq) in enumerate(rows):
        ax_sim, ax_exp, ax_xsim, ax_xexp, ax_vel = axes[r]

        # Max exp trial duration for sim trimming
        max_exp_dur = 0.0
        for t_exp, _ in exp_data.get((scene, freq), []):
            dur = t_exp[-1] - t_exp[0]
            if dur > max_exp_dur:
                max_exp_dur = dur

        # --- Sim ---
        rms_list = []
        sim_xy_traces = []
        tkeys = trial_map.get(rid, [])
        for ci, tkey in enumerate(tkeys):
            pitch_full = npz_data[f"{tkey}_pitch"]
            pos_x_full = npz_data[f"{tkey}_pos_x"]
            pos_z_full = npz_data[f"{tkey}_pos_z"]
            time_full = npz_data[f"{tkey}_time"]
            color = COLORS_SIM[ci % len(COLORS_SIM)]
            label = f"t{tkey.split('_t')[-1]}"

            settle_idx = int(np.searchsorted(time_full, SETTLE_TIME))
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
            sim_xy_traces.append(((pos_x - x0) * 1000, (pos_z - z0) * 1000, color, label))

            # Untrimmed vx
            px_active = pos_x_full[settle_idx:]
            t_active = time_full[settle_idx:] - SETTLE_TIME
            vx_raw = np.diff(px_active) / SIM_TIMESTEP * 100
            vx_smooth = _smooth_vx(vx_raw, VX_SMOOTH_SIM)
            ax_vel.plot(t_active[:-1], vx_smooth, color=color, linewidth=0.6, alpha=0.7,
                        label=f"sim {label}")

        rms_mean = np.mean(rms_list) if rms_list else 0
        ax_sim.set_title(f"{rid}  SIM pitch  (RMS={rms_mean:.1f}\u00b0)", fontsize=9)
        ax_sim.set_ylabel("Pitch (\u00b0)")
        ax_sim.legend(fontsize=7, loc="upper right")
        ax_sim.grid(True, alpha=0.3)

        for x_all, z_all, col, lbl in sim_xy_traces:
            ax_xsim.plot(x_all, z_all, color=col, linewidth=1.0, alpha=0.8, label=lbl)
            ax_xsim.plot(x_all[0], z_all[0], "o", color=col, ms=3, alpha=0.7)
        ax_xsim.set_title(f"{rid}  SIM XY", fontsize=9)
        ax_xsim.set_xlabel("x (mm)")
        ax_xsim.set_ylabel("y (mm)")
        ax_xsim.legend(fontsize=6, loc="upper left")
        ax_xsim.grid(True, alpha=0.3)

        # --- Exp ---
        exp_pitches = []
        exp_xy_traces = []
        trial_files = exp_file_map.get((scene, freq), [])

        for ci, (t_exp, theta_exp) in enumerate(exp_data.get((scene, freq), [])):
            t0_e = t_exp[0]
            color_e = COLORS_EXP[ci % len(COLORS_EXP)]

            ax_exp.plot(t_exp - t0_e, theta_exp, color=color_e,
                        linewidth=0.6, alpha=0.8, label=f"trial {ci+1}")
            exp_pitches.append(np.std(theta_exp))

            if ci < len(trial_files):
                try:
                    csv_p = exp_csv_dir / trial_files[ci]
                    dat = np.genfromtxt(csv_p, delimiter=",", skip_header=2)
                    t_csv = dat[:, 0] - dat[0, 0]
                    avg_x = (dat[:, 1] + dat[:, 5]) / 2
                    avg_y = (dat[:, 2] + dat[:, 6]) / 2

                    x0_e = np.mean(avg_x[:50])
                    y0_e = np.mean(avg_y[:50])
                    x_e = (x0_e - avg_x) * 1000
                    y_e = (avg_y - y0_e) * 1000
                    exp_xy_traces.append((x_e, y_e, color_e, f"t{ci+1}"))

                    dt_e = np.diff(t_csv)
                    dt_e[dt_e == 0] = 1e-6
                    vx_e = -np.diff(avg_x) / dt_e * 100
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

        for xf, yf, col, lbl in exp_xy_traces:
            ax_xexp.plot(xf, yf, color=col, linewidth=1.0, alpha=0.8, label=lbl)
            ax_xexp.plot(xf[0], yf[0], "o", color=col, ms=3, alpha=0.7)
        ax_xexp.set_title(f"{rid}  EXP XY", fontsize=9)
        ax_xexp.set_xlabel("x (mm)")
        ax_xexp.set_ylabel("y (mm)")
        ax_xexp.legend(fontsize=6, loc="upper left")
        ax_xexp.grid(True, alpha=0.3)

        ax_vel.set_title(f"{rid}  vx (untrimmed)", fontsize=9)
        ax_vel.set_ylabel("vx (cm/s)")
        ax_vel.legend(fontsize=5, loc="upper right", ncol=2)
        ax_vel.grid(True, alpha=0.3)
        ax_vel.axhline(0, color="grey", lw=0.5, alpha=0.5)

        # Match limits
        _match_ylim(ax_sim, ax_exp)
        _match_xlim(ax_sim, ax_exp, lo=0)
        _match_xy_lim(ax_xsim, ax_xexp)

        if r == len(rows) - 1:
            ax_sim.set_xlabel("Time (s)")
            ax_exp.set_xlabel("Time (s)")
            ax_vel.set_xlabel("Time (s)")


# ---------------------------------------------------------------------------
# Step terrain plotting
# ---------------------------------------------------------------------------

def _plot_step(fig, axes, rows, npz_data, trial_map, exp_data, exp_file_map, exp_csv_dir):
    """Plot step terrain: 4 columns per row."""
    for r, (rid, scene, freq) in enumerate(rows):
        ax_sim, ax_exp, ax_xsim, ax_xexp = axes[r]

        # --- Sim ---
        rms_list = []
        sim_gate_bands = []
        sim_xy_traces = []
        tkeys = trial_map.get(rid, [])
        for ci, tkey in enumerate(tkeys):
            pitch = npz_data[f"{tkey}_pitch"]
            pos_x = npz_data[f"{tkey}_pos_x"]
            pos_z = npz_data[f"{tkey}_pos_z"]
            time = npz_data[f"{tkey}_time"]
            color = COLORS_SIM[ci % len(COLORS_SIM)]
            label = f"t{tkey.split('_t')[-1]}"

            enter_idx = int(np.searchsorted(pos_x, STEP_START_X))
            gate_indices = np.where(pos_x >= CUTOFF_X)[0]
            if len(gate_indices) == 0:
                continue
            gate_idx = int(gate_indices[0])
            if gate_idx <= enter_idx or (gate_idx - enter_idx) < 10:
                continue

            trim_indices = np.where(pos_x >= STEP_END_X)[0]
            trim_idx = int(trim_indices[0]) if len(trim_indices) > 0 else len(pos_x) - 1

            ax_sim.plot(time[:trim_idx+1], pitch[:trim_idx+1], color=color,
                        linewidth=0.6, alpha=0.8, label=label)

            p_gate = pitch[enter_idx:gate_idx+1]
            rms_list.append(np.std(p_gate - p_gate[0]))
            sim_gate_bands.append((time[enter_idx], time[gate_idx], color))

            x_all = pos_x[:trim_idx+1] * 1000
            z_all = pos_z[:trim_idx+1] * 1000
            x_gate = pos_x[enter_idx:gate_idx+1] * 1000
            z_gate = pos_z[enter_idx:gate_idx+1] * 1000
            sim_xy_traces.append((x_all, z_all, x_gate, z_gate, color, label))

        # Gate shading
        n_bands = len(sim_gate_bands)
        for bi, (t_lo, t_hi, color) in enumerate(sim_gate_bands):
            y_lo_frac = bi / max(n_bands, 1) / 3
            y_hi_frac = (bi + 1) / max(n_bands, 1) / 3
            ax_sim.axvspan(t_lo, t_hi, ymin=y_lo_frac, ymax=y_hi_frac,
                           color=color, alpha=0.2)

        rms_full = np.mean(rms_list) if rms_list else 0
        ax_sim.set_title(f"{rid}  SIM pitch  (gate={rms_full:.1f}\u00b0)", fontsize=9)
        ax_sim.set_ylabel("Pitch (\u00b0)")
        ax_sim.legend(fontsize=7, loc="upper right")
        ax_sim.grid(True, alpha=0.3)

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
        ax_xsim.grid(True, alpha=0.3)

        # --- Exp ---
        exp_pitches = []
        exp_q60_bands = []
        exp_xy_traces = []
        trial_files = exp_file_map.get((scene, freq), [])

        for ci, (t_exp, theta_exp) in enumerate(exp_data.get((scene, freq), [])):
            ne = len(theta_exp)
            lo_e = int(0.45 * ne)
            hi_e = int(0.75 * ne)
            t0_e = t_exp[0]
            color_e = COLORS_EXP[ci % len(COLORS_EXP)]

            ax_exp.plot(t_exp - t0_e, theta_exp, color=color_e,
                        linewidth=0.6, alpha=0.8, label=f"trial {ci+1}")
            exp_pitches.append(np.std(theta_exp[lo_e:hi_e]))
            exp_q60_bands.append((t_exp[lo_e] - t0_e, t_exp[min(hi_e, ne-1)] - t0_e, color_e))

            if ci < len(trial_files):
                try:
                    csv_p = exp_csv_dir / trial_files[ci]
                    dat = np.genfromtxt(csv_p, delimiter=",", skip_header=2)
                    avg_x = (dat[:, 1] + dat[:, 5]) / 2
                    avg_y = (dat[:, 2] + dat[:, 6]) / 2
                    x_e = (avg_x[0] - avg_x) * 1000
                    y_e = (avg_y - avg_y[0]) * 1000
                    exp_xy_traces.append((x_e, y_e, x_e[lo_e:hi_e], y_e[lo_e:hi_e],
                                          color_e, f"t{ci+1}"))
                except Exception:
                    pass

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

        for xf, yf, xq, yq, col, lbl in exp_xy_traces:
            ax_xexp.plot(xf, yf, color=col, linewidth=0.5, alpha=0.4)
            ax_xexp.plot(xq, yq, color=col, linewidth=1.5, alpha=0.9, label=lbl)
            ax_xexp.plot(xf[0], yf[0], "o", color=col, ms=3, alpha=0.7)
        ax_xexp.set_title(f"{rid}  EXP XY", fontsize=9)
        ax_xexp.set_xlabel("x (mm)")
        ax_xexp.set_ylabel("y (mm)")
        ax_xexp.legend(fontsize=6, loc="upper left")
        ax_xexp.grid(True, alpha=0.3)

        # Match limits
        _match_ylim(ax_sim, ax_exp)
        _match_xlim(ax_sim, ax_exp, lo=0)
        _match_xy_lim(ax_xsim, ax_xexp)

        if r == len(rows) - 1:
            ax_sim.set_xlabel("Time (s)")
            ax_exp.set_xlabel("Time (s)")


# ---------------------------------------------------------------------------
# Axis matching helpers
# ---------------------------------------------------------------------------

def _match_ylim(ax1, ax2):
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)


def _match_xlim(ax1, ax2, lo=None):
    xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    xmin = lo if lo is not None else min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)


def _match_xy_lim(ax1, ax2):
    for getter, setter in [("get_xlim", "set_xlim"), ("get_ylim", "set_ylim")]:
        lo = min(getattr(ax1, getter)()[0], getattr(ax2, getter)()[0])
        hi = max(getattr(ax1, getter)()[1], getattr(ax2, getter)()[1])
        getattr(ax1, setter)(lo, hi)
        getattr(ax2, setter)(lo, hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terrain", required=True, choices=["flat", "step"],
                        help="Terrain type")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Path to results run directory")
    parser.add_argument("--n-select", type=int, default=None,
                        help="Select top N trials closest to target velocity (default: all)")
    args = parser.parse_args()

    run_dir = args.run_dir
    terrain = args.terrain

    # Discover data files
    npz_path, csv_path = _discover_files(run_dir)
    print(f"NPZ: {npz_path.name}")
    print(f"CSV: {csv_path.name}")

    npz_data = np.load(str(npz_path), allow_pickle=True)

    # Build trial map from CSV: {ref_id: [tkey, ...]}
    all_tkeys: dict[str, list[str]] = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("crash", "").lower() == "true":
                continue
            rid = row["ref_id"]
            tkey = f"{rid}_t{row['trial']}"
            all_tkeys[rid].append(tkey)

    # Trial selection
    if args.n_select is not None:
        target_vels = _load_target_velocities(terrain)
        trial_map = _select_trials(all_tkeys, npz_data, terrain, args.n_select, target_vels)
        suffix = ""
        print(f"Selected {args.n_select} trials per condition (closest to target velocity)")
    else:
        trial_map = dict(all_tkeys)
        suffix = "_all"
        n_total = sum(len(v) for v in trial_map.values())
        print(f"Using all {n_total} trials")

    # Load experimental data
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if terrain == "flat":
        exp_csv_dir = repo_root / "experimental_data" / "csv" / "flat"
        exp_conditions = EXP_FLAT_CONDITIONS
    else:
        exp_csv_dir = repo_root / "experimental_data" / "csv" / "steps"
        exp_conditions = EXP_STEP_CONDITIONS

    exp_data: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    exp_file_map: dict[tuple[str, int], list[str]] = {}
    for freq, morph, files, idx in exp_conditions:
        scene = MORPH_TO_SCENE[morph]
        trial_files = [files[i - 1] for i in idx]
        trials = []
        for fname in trial_files:
            try:
                trials.append(_load_exp_pitch(exp_csv_dir / fname))
            except Exception as e:
                print(f"  WARN: {fname}: {e}")
        exp_data[(scene, freq)] = trials
        exp_file_map[(scene, freq)] = trial_files

    # Build row list: conditions with both sim and exp data
    plot_rows = []
    for rid in sorted(trial_map):
        scene, freq = _parse_rid(rid)
        if (scene, freq) in exp_data and exp_data[(scene, freq)]:
            plot_rows.append((rid, scene, freq))

    if not plot_rows:
        print("ERROR: No conditions with both sim and exp data")
        return

    n_rows = len(plot_rows)
    n_cols = 5 if terrain == "flat" else 4
    width_ratios = [3, 3, 2, 2, 3] if terrain == "flat" else [3, 3, 2, 2]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 2.8 * n_rows),
                             squeeze=False, gridspec_kw={"width_ratios": width_ratios})
    terrain_label = "Flat" if terrain == "flat" else "Step"
    fig.suptitle(f"{terrain_label}: Sim vs Exp  |  Pitch + Position"
                 + (" + Velocity" if terrain == "flat" else ""),
                 fontsize=14, fontweight="bold")

    if terrain == "flat":
        _plot_flat(fig, axes, plot_rows, npz_data, trial_map, exp_data, exp_file_map, exp_csv_dir)
    else:
        _plot_step(fig, axes, plot_rows, npz_data, trial_map, exp_data, exp_file_map, exp_csv_dir)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = run_dir / f"{ts}_pitch_mega_sim_vs_exp{suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
