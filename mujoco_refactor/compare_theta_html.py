#!/usr/bin/env python3
"""Compare FL joint angle (theta) between experimental data and sim (two param sets).

Generates interactive HTML plot for 1-leg f10 case using plotly.

Usage:
    uv run python compare_theta_html.py
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Paths ---
ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP_CSV_DIR = ROOT / "experimental_data" / "csv" / "flat"
REFACTOR = pathlib.Path(__file__).resolve().parent

# Param set directories
FLAT_DIR = REFACTOR / "results" / "20260225T122342_flat_10_30_50"
STEP_DIR = REFACTOR / "results" / "20260225T225248_step_argmin_progress"

# Experimental files for 1-leg f10 (trials 1,2,3 from FLAT_CONDITIONS)
EXP_FILES = ["f10leg1-1.csv", "f10leg2-2.csv", "f10leg3-3.csv"]

# Sim config
sys.path.insert(0, str(REFACTOR))
from config_new import MJCF_PATHS, SETTLE_TIME, SIM_DURATION, SIMULATION_TIMEOUT, sim_params_from_point, space
import simulation_fast_new as sim_module

PARAM_NAMES = [dim.name for dim in space]
BASE_SEED = 77777
FL_IDX = 1  # FL joint index in joint_pos/joint_vel arrays


def load_best_point(run_dir: pathlib.Path) -> list[float]:
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    best_id = rows[-1]["id"]
    with open(run_dir / "multi_optimization_results.csv") as f:
        for row in csv.DictReader(f):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    raise ValueError(f"id {best_id!r} not found")


def load_experimental(csv_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load experimental theta (col 11, deg) and omega (col 10, deg/s)."""
    path = EXP_CSV_DIR / csv_name
    dat = np.genfromtxt(path, delimiter=",", skip_header=2)
    t = dat[:, 0]
    theta = dat[:, 9]   # degrees, cumulative FL joint angle (mass_B)
    omega = dat[:, 10]  # deg/s (angular velocity of mass_B)
    return t, theta, omega


def run_sim_trial(sim_params: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run sim, return (time, FL_theta_deg, FL_omega_deg_s)."""
    sp = dict(sim_params)
    sp["drive_freq"] = 10.0

    traj = sim_module.run_simulation(
        sp, mjcf_path=MJCF_PATHS["scene1"],
        sim_duration=SIM_DURATION, wall_timeout=SIMULATION_TIMEOUT,
        init_yaw_jitter_deg=2.0, rng_seed=seed,
    )
    if traj is None:
        raise RuntimeError("Simulation failed")

    t = np.array([s["time"] for s in traj])
    theta_rad = np.array([s["joint_pos"][FL_IDX] for s in traj])
    theta_deg = np.degrees(theta_rad)

    # Numerical derivative of theta for smooth omega (matches experimental pipeline)
    dt = np.diff(t)
    dtheta = np.diff(theta_deg)
    omega_smooth = dtheta / dt  # deg/s
    # Pad to match length
    omega_smooth = np.concatenate([[omega_smooth[0]], omega_smooth])

    return t, theta_deg, omega_smooth


def main():
    # --- Load experimental data ---
    print("Loading experimental data...")
    exp_data = []
    for f in EXP_FILES:
        t, theta, omega = load_experimental(f)
        exp_data.append({"t": t, "theta": theta, "omega": omega, "name": f})

    # --- Run sims ---
    print("Loading flat params...")
    flat_point = load_best_point(FLAT_DIR)
    flat_params = sim_params_from_point(flat_point)

    print("Loading step params...")
    step_point = load_best_point(STEP_DIR)
    step_params = sim_params_from_point(step_point)

    # Use seed that gives closest-to-target velocity (trial 4 for flat, from cot_results.csv)
    # Just use trial 0 for simplicity — deterministic seed
    seed = BASE_SEED  # scene1 is ref_index 0

    print("Running flat-params sim...")
    t_flat, theta_flat, omega_flat = run_sim_trial(flat_params, seed)

    print("Running step-params sim...")
    t_step, theta_step, omega_step = run_sim_trial(step_params, seed)

    # --- Create plotly figure ---
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["FL Joint Angle (theta)", "FL Angular Velocity (omega, from d(theta)/dt)"],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    colors = {
        "exp": ["rgba(100,100,100,0.4)", "rgba(120,120,120,0.4)", "rgba(140,140,140,0.4)"],
        "flat": "rgba(31,119,180,0.8)",
        "step": "rgba(255,127,14,0.8)",
    }

    # Experimental trials (all 3)
    for i, ed in enumerate(exp_data):
        show_legend = i == 0
        # Trim to 2500 points (as in FLAT_CONDITIONS)
        n = min(2500, len(ed["t"]))
        fig.add_trace(go.Scatter(
            x=ed["t"][:n], y=ed["theta"][:n],
            mode="lines", name="Experimental" if show_legend else None,
            line=dict(color=colors["exp"][i], width=1),
            legendgroup="exp", showlegend=show_legend,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ed["t"][:n], y=ed["omega"][:n],
            mode="lines", name="Experimental" if show_legend else None,
            line=dict(color=colors["exp"][i], width=1),
            legendgroup="exp", showlegend=False,
        ), row=2, col=1)

    # Sim flat params
    fig.add_trace(go.Scatter(
        x=t_flat, y=theta_flat,
        mode="lines", name="Flat params",
        line=dict(color=colors["flat"], width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_flat, y=omega_flat,
        mode="lines", name="Flat params",
        line=dict(color=colors["flat"], width=1.5),
        showlegend=False,
    ), row=2, col=1)

    # Sim step params
    fig.add_trace(go.Scatter(
        x=t_step, y=theta_step,
        mode="lines", name="Step params",
        line=dict(color=colors["step"], width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_step, y=omega_step,
        mode="lines", name="Step params",
        line=dict(color=colors["step"], width=1.5),
        showlegend=False,
    ), row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Angle (deg)", row=1, col=1)
    fig.update_yaxes(title_text="Omega (deg/s)", row=2, col=1)
    fig.update_layout(
        title="1-leg f10: FL Joint Angle — Experimental vs Simulation",
        height=800, width=1200,
        hovermode="x unified",
    )

    out_path = REFACTOR / "results" / "theta_comparison_1leg_f10.html"
    fig.write_html(str(out_path))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
