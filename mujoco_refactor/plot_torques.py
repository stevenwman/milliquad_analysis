#!/usr/bin/env python3
"""Plot external torque and joint diagnostics for all reference configs.

Usage:
    uv run python plot_torques.py results/20260219T142207_loose_fudge
"""

import argparse
import csv
import importlib
import pathlib
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    DEFAULT_CTRL_FREQ,
    MJCF_PATHS,
    SETTLE_TIME,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    reference_rows,
    sim_params_from_point,
    space,
)

SIM_MODULE = "simulation_fast"
LEG_NAMES = ["FR", "FL", "BR", "BL"]
LEG_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

REF_ROWS = reference_rows()
PARAM_NAMES = [dim.name for dim in space]

# ── subplot definitions ──────────────────────────────────────────────
# Each entry: (title, y-axis label, y-axis kwargs)
SUBPLOTS = [
    ("τ_ext hinge-axis projection",      "N·m",      {}),
    ("Joint angular velocity",            "rad/s",    {}),
    ("Magnet & field angle (drive plane)","rad",      dict(range=[-np.pi, np.pi])),
    ("sin(θ_field − θ_magnet)",           "sin(err)", {}),
    ("Forward velocity",                  "m/s",      {}),
    ("Body height",                       "m",        {}),
    ("Chassis pitch, roll & yaw",         "Δ deg",    {}),
]
N_ROWS = len(SUBPLOTS)


def load_best_point(run_dir: pathlib.Path) -> list[float]:
    """Load full-precision params for the final best from multi CSV."""
    bests_csv = run_dir / "optimization_bests.csv"
    bests_rows = list(csv.DictReader(open(bests_csv)))
    if not bests_rows:
        print(f"ERROR: no rows in {bests_csv}")
        sys.exit(1)
    best_id = bests_rows[-1]["id"]

    multi_csv = run_dir / "multi_optimization_results.csv"
    with open(multi_csv) as f:
        for row in csv.DictReader(f):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    raise ValueError(f"id {best_id!r} not found in {multi_csv}")


def _extract(traj, mask):
    """Extract and compute all per-leg and chassis signals from trajectory."""
    get = lambda key: np.array([s[key] for s in traj])[mask]

    tau_ext_all = get("tau_ext")          # [T, 4, 3]
    leg_xquats = get("leg_xquat")        # [T, 4, 4]
    chassis_quats = get("quat")           # [T, 4]
    north_all = get("north")              # [T, 4, 3]
    drive_angle = get("drive_angle")      # [T]

    # ── hinge-axis torque projection (per-leg body quats) ──
    lw, lx, ly, lz = [leg_xquats[:, :, i] for i in range(4)]
    axis_world = np.stack([
        2 * (lx * lz + lw * ly),
        2 * (ly * lz - lw * lx),
        1 - 2 * (lx * lx + ly * ly),
    ], axis=-1)                           # [T, 4, 3]
    tau_hinge = np.einsum("tlc,tlc->tl", tau_ext_all, axis_world)

    # Compute sign flip from actual axis data: align all legs to the
    # same reference direction (first leg's axis at t=0).
    ref_axis = axis_world[0, 0]                        # [3] — leg 0 at first timestep
    dots = np.einsum("c,lc->l", ref_axis, axis_world[0])  # [4] dot with each leg
    sign_flip = np.sign(dots)                          # +1 if same dir, -1 if opposite

    tau_hinge *= sign_flip

    # ── joint kinematics (sign-normalized) ──
    joint_vel = get("joint_vel") * sign_flip

    # ── magnet / field angles ──
    magnet_angle = np.arctan2(north_all[:, :, 0], north_all[:, :, 2])
    drive_wrapped = (drive_angle + np.pi) % (2 * np.pi) - np.pi
    sin_err = np.sin(drive_angle[:, None] - magnet_angle)

    # ── chassis translational state ──
    pos = get("pos")                             # [T, 3]
    vel = get("vel")                             # [T, 3]
    forward_vel = vel[:, 0]                      # x-axis = forward
    body_height = pos[:, 2]                      # z-axis = up

    # ── chassis Euler angles (ZYX: yaw-pitch-roll) ──
    w, x, y, z = [chassis_quats[:, i] for i in range(4)]
    roll  = np.degrees(np.unwrap(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = np.degrees(np.unwrap(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))))

    return dict(
        tau_hinge=tau_hinge, joint_vel=joint_vel,
        magnet_angle=magnet_angle, drive_wrapped=drive_wrapped,
        sin_err=sin_err, forward_vel=forward_vel, body_height=body_height,
        pitch=pitch, roll=roll, yaw=yaw,
    )


def _add_per_leg(fig, t, data_2d, row, show_legend=False):
    """Add 4 per-leg traces to a subplot row."""
    for i in range(4):
        fig.add_trace(go.Scattergl(
            x=t, y=data_2d[:, i], mode="lines",
            name=LEG_NAMES[i], legendgroup=LEG_NAMES[i],
            line=dict(color=LEG_COLORS[i], width=1),
            showlegend=show_legend,
        ), row=row, col=1)


def _build_figure(ref_id, t, d):
    """Build the subplot figure for one reference config."""
    fig = make_subplots(
        rows=N_ROWS, cols=1, shared_xaxes=True,
        row_heights=[0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.15],
        vertical_spacing=0.030,
        subplot_titles=[s[0] for s in SUBPLOTS],
    )

    # Per-leg subplots
    _add_per_leg(fig, t, d["tau_hinge"],  row=1, show_legend=True)
    _add_per_leg(fig, t, d["joint_vel"],  row=2)

    # Magnet angle (per-leg, wrapped) + drive field angle
    magnet_wrapped = (d["magnet_angle"] + np.pi) % (2 * np.pi) - np.pi
    _add_per_leg(fig, t, magnet_wrapped, row=3)
    fig.add_trace(go.Scattergl(
        x=t, y=d["drive_wrapped"], mode="lines",
        name="field angle", legendgroup="field",
        line=dict(color="#333333", width=1.5, dash="dot"),
    ), row=3, col=1)

    _add_per_leg(fig, t, d["sin_err"], row=4)

    # Forward velocity (single trace)
    fig.add_trace(go.Scattergl(
        x=t, y=d["forward_vel"], mode="lines",
        name="v_x", legendgroup="chassis",
        line=dict(color="#e66101", width=1.5),
    ), row=5, col=1)

    # Body height (single trace)
    fig.add_trace(go.Scattergl(
        x=t, y=d["body_height"], mode="lines",
        name="z", legendgroup="chassis",
        line=dict(color="#5e3c99", width=1.5),
    ), row=6, col=1)

    # Chassis RPY (mean-subtracted)
    for val, name, color in [
        (d["pitch"], "pitch", "#ff7f00"),
        (d["roll"],  "roll",  "#a65628"),
        (d["yaw"],   "yaw",   "#1b9e77"),
    ]:
        fig.add_trace(go.Scattergl(
            x=t, y=val - val.mean(), mode="lines",
            name=f"{name} (μ={val.mean():.1f}°)", legendgroup="chassis",
            line=dict(color=color, width=1.5),
        ), row=7, col=1)

    # Axis labels
    for row_idx, (_, ylabel, ykwargs) in enumerate(SUBPLOTS, 1):
        fig.update_yaxes(title_text=ylabel, row=row_idx, col=1, **ykwargs)
    fig.update_xaxes(title_text="time (s)", row=N_ROWS, col=1)

    fig.update_layout(
        title=ref_id, height=1600, width=900,
        legend=dict(orientation="h", yanchor="bottom", y=-0.04,
                    xanchor="center", x=0.5),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot torque diagnostics")
    parser.add_argument("run_dir", type=str, help="Results directory")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    point = load_best_point(run_dir)
    sim_params = sim_params_from_point(point)
    sim_module = importlib.import_module(SIM_MODULE)

    # Run sims (no jitter, single trial each)
    trajectories = {}
    for ref_row in REF_ROWS:
        ref_id = ref_row["id"]
        sim_params_scene = dict(sim_params)
        sim_params_scene["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)

        print(f"  Running {ref_id}...", end=" ", flush=True)
        traj = sim_module.run_simulation(
            sim_params_scene,
            mjcf_path=MJCF_PATHS[ref_row["scene"]],
            sim_duration=SIM_DURATION,
            wall_timeout=SIMULATION_TIMEOUT,
            init_yaw_jitter_deg=0.0,
        )
        print("FAILED" if traj is None else f"OK ({len(traj)} steps)")
        trajectories[ref_id] = traj

    # Generate one HTML per ref
    out_dir = run_dir / "torque_plots"
    out_dir.mkdir(exist_ok=True)

    for ref_row in REF_ROWS:
        ref_id = ref_row["id"]
        traj = trajectories[ref_id]
        if traj is None:
            print(f"  Skipping {ref_id} (FAILED)")
            continue

        times = np.array([s["time"] for s in traj])
        mask = times >= SETTLE_TIME
        t = times[mask]

        d = _extract(traj, mask)
        fig = _build_figure(ref_id, t, d)

        out_path = out_dir / f"{ref_id}.html"
        fig.write_html(str(out_path))
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
