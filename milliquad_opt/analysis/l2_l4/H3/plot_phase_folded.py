#!/usr/bin/env python3
"""H3+H4 Plot: Phase-folded velocity and torque/power.

Top row: vel_x (mm/s) binned by drive_angle mod 2pi.
Bottom row (default): sum of per-leg |tau_ext| (uN*m).
Bottom row (--power): signed mechanical power (uW) = sum(tau_axial * joint_vel).
Columns = frequencies. One terrain per invocation.

Usage (from milliquad_opt/):
    uv run python -m analysis.l2_l4.H3.plot_phase_folded \
        results/20260228T013353_rk4_flat
    uv run python -m analysis.l2_l4.H3.plot_phase_folded \
        results/20260228T013353_rk4_flat --normalize
    uv run python -m analysis.l2_l4.H3.plot_phase_folded \
        results/20260228T013353_rk4_flat --power
"""

from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from analysis.l2_l4._plot_style import (
    MORPH_COLORS,
    MORPH_LABELS,
    MORPH_ORDER,
    TERRAIN_TITLES,
)
from analysis.l2_l4._trial_filter import (
    active_mask,
    detect_terrain,
    find_npz,
    is_valid_trial,
    parse_key,
)

FIGURE_DIR = pathlib.Path(__file__).parent / "figures"
N_BINS = 36  # 10-degree bins


def _joint_axis_world(leg_xquat: np.ndarray) -> np.ndarray:
    """Rotate joint axis [0,0,1] into world frame via leg body quaternion.

    Args:
        leg_xquat: (T, 4, 4) — quaternions (w,x,y,z) per leg per timestep.

    Returns:
        (T, 4, 3) — world-frame joint axis per leg per timestep.
    """
    w = leg_xquat[:, :, 0]
    x = leg_xquat[:, :, 1]
    y = leg_xquat[:, :, 2]
    z = leg_xquat[:, :, 3]
    ax = 2 * (x * z + w * y)
    ay = 2 * (y * z - w * x)
    az = 1 - 2 * (x * x + y * y)
    return np.stack([ax, ay, az], axis=-1)  # (T, 4, 3)


def _collect(run_dir: pathlib.Path, normalize: bool = False, power: bool = False):
    """Phase-fold vel_x and torque (or power) for each valid trial.

    Args:
        normalize: if True, divide each trial's profile by its own mean
                   before aggregation (y-axis becomes fraction of mean).
        power: if True, compute signed mechanical power instead of |tau_ext|.

    Returns:
        data: {freq: {scene: {"vel": (mean, std), "torque": (mean, std)}}}
              where mean/std are arrays of shape (N_BINS,).
        freqs: sorted frequency list
        terrain: str
    """
    terrain = detect_terrain(run_dir)
    npz = np.load(find_npz(run_dir))

    bin_edges = np.linspace(0, 2 * np.pi, N_BINS + 1)

    # Discover trials
    trials: dict[tuple[str, float], list[int]] = {}
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[3] == "time":
            scene, freq, tidx, _ = parsed
            trials.setdefault((scene, freq), []).append(tidx)

    # Per-trial phase-folded profiles
    per_trial: dict[float, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    for (scene, freq), tidxs in trials.items():
        if scene not in MORPH_ORDER:
            continue
        for t in tidxs:
            prefix = f"{scene}_f{freq:g}_t{t}"
            time = npz[f"{prefix}_time"]
            pos_x = npz[f"{prefix}_pos_x"]
            pitch = npz[f"{prefix}_pitch"]

            if not is_valid_trial(pos_x, pitch, terrain, scene, freq):
                continue
            mask = active_mask(time, pos_x, terrain)
            if mask.sum() < 20:
                continue

            vx = npz[f"{prefix}_vel_x"][mask] * 1000  # m/s -> mm/s
            tau = npz[f"{prefix}_tau_ext"][mask]  # (T, 4, 3)
            drive = npz[f"{prefix}_drive_angle"][mask]

            if power:
                # Signed mechanical power: P = sum_legs(dot(tau, axis) * joint_vel)
                xquat = npz[f"{prefix}_leg_xquat"][mask]   # (T, 4, 4)
                jvel = npz[f"{prefix}_joint_vel"][mask]     # (T, 4)
                axis = _joint_axis_world(xquat)             # (T, 4, 3)
                # dot(tau, axis) per leg: sum over xyz
                tau_axial = (tau * axis).sum(axis=2)        # (T, 4)
                per_leg_power = tau_axial * jvel             # (T, 4)
                bottom_row = per_leg_power.sum(axis=1) * 1e6  # W -> uW
            else:
                # Net torque magnitude: sum of per-leg |tau| per timestep
                bottom_row = np.linalg.norm(tau, axis=2).sum(axis=1) * 1e6  # N*m -> uN*m

            phase = drive % (2 * np.pi)
            bin_idx = np.clip(np.digitize(phase, bin_edges) - 1, 0, N_BINS - 1)

            vel_profile = np.zeros(N_BINS)
            bottom_profile = np.zeros(N_BINS)
            for b in range(N_BINS):
                in_bin = bin_idx == b
                if in_bin.sum() > 0:
                    vel_profile[b] = vx[in_bin].mean()
                    bottom_profile[b] = bottom_row[in_bin].mean()

            if normalize:
                vel_mean = vel_profile.mean()
                bottom_mean = bottom_profile.mean()
                if abs(vel_mean) > 1e-6:
                    vel_profile = vel_profile / vel_mean
                if abs(bottom_mean) > 1e-6:
                    bottom_profile = bottom_profile / bottom_mean

            per_trial[freq][scene].append((vel_profile, bottom_profile))

    npz.close()

    # Aggregate across trials: mean +/- std
    all_freqs = sorted(per_trial.keys())
    data: dict[float, dict[str, dict]] = {}
    for freq in all_freqs:
        data[freq] = {}
        for scene in MORPH_ORDER:
            trials_list = per_trial[freq].get(scene, [])
            if not trials_list:
                continue
            vel_arr = np.array([t[0] for t in trials_list])
            torque_arr = np.array([t[1] for t in trials_list])
            data[freq][scene] = {
                "vel": (vel_arr.mean(axis=0), vel_arr.std(axis=0)),
                "torque": (torque_arr.mean(axis=0), torque_arr.std(axis=0)),
                "n": len(trials_list),
            }

    return data, all_freqs, terrain


def plot(run_dirs: list[pathlib.Path], save: bool = True, normalize: bool = False,
         power: bool = False):
    # Single terrain per invocation
    rd = run_dirs[0]
    data, freqs, terrain = _collect(rd, normalize=normalize, power=power)

    if not freqs:
        print("No data to plot.")
        return None

    n_cols = len(freqs)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 5),
                             sharex=True, squeeze=False)

    bin_centers = np.linspace(0, 360, N_BINS, endpoint=False) + 5  # bin center in degrees

    for col, freq in enumerate(freqs):
        ax_vel = axes[0, col]
        ax_bottom = axes[1, col]

        if freq not in data:
            ax_vel.set_visible(False)
            ax_bottom.set_visible(False)
            continue

        for scene in MORPH_ORDER:
            if scene not in data[freq]:
                continue
            d = data[freq][scene]
            color = MORPH_COLORS[scene]
            label = MORPH_LABELS[scene] if col == 0 else None

            v_mean, v_std = d["vel"]
            ax_vel.plot(bin_centers, v_mean, color=color, label=label, linewidth=1.5)
            ax_vel.fill_between(bin_centers, v_mean - v_std, v_mean + v_std,
                                color=color, alpha=0.15)

            t_mean, t_std = d["torque"]
            ax_bottom.plot(bin_centers, t_mean, color=color, linewidth=1.5)
            ax_bottom.fill_between(bin_centers, t_mean - t_std, t_mean + t_std,
                                   color=color, alpha=0.15)

        ax_vel.set_title(f"{freq:g} Hz")
        ax_bottom.set_xlabel("Drive angle (deg)")
        ax_bottom.set_xlim(0, 360)
        ax_bottom.set_xticks([0, 90, 180, 270, 360])

        if power:
            ax_bottom.axhline(0, color="0.5", linewidth=0.5, linestyle="--", zorder=0)

        for ax in (ax_vel, ax_bottom):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    if normalize:
        axes[0, 0].set_ylabel("vel_x / mean")
        axes[1, 0].set_ylabel("power / mean" if power else "|tau_ext| / mean")
    elif power:
        axes[0, 0].set_ylabel("vel_x (mm/s)")
        axes[1, 0].set_ylabel("power (uW)")
    else:
        axes[0, 0].set_ylabel("vel_x (mm/s)")
        axes[1, 0].set_ylabel("net |tau_ext| (uN*m)")

    fig.legend(
        [plt.Line2D([0], [0], color=MORPH_COLORS[s], lw=2) for s in MORPH_ORDER],
        [MORPH_LABELS[s] for s in MORPH_ORDER],
        loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.99),
    )

    terrain_label = TERRAIN_TITLES.get(terrain, terrain)
    tags = []
    if power:
        tags.append("power")
    if normalize:
        tags.append("normalized")
    tag_str = f" ({', '.join(tags)})" if tags else ""
    fig.suptitle(f"Phase-folded velocity & {'power' if power else 'torque'}"
                 f" — {terrain_label}{tag_str}", fontsize=12, y=1.02)
    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        suffix_parts = []
        if power:
            suffix_parts.append("power")
        if normalize:
            suffix_parts.append("norm")
        suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
        out = FIGURE_DIR / f"{ts}_phase_folded_{terrain}{suffix}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize each trial by its mean (y = fraction of mean)")
    parser.add_argument("--power", action="store_true",
                        help="Show signed mechanical power instead of |tau_ext|")
    args = parser.parse_args()
    plot(args.run_dirs, save=not args.no_save, normalize=args.normalize,
         power=args.power)


if __name__ == "__main__":
    main()
