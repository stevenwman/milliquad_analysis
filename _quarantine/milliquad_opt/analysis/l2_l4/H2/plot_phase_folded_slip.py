#!/usr/bin/env python3
"""H2 Plot: Phase-folded slip ratio per leg.

Rows = terrains, columns = frequencies. Per-leg slip ratio traces overlaid,
colored by morphology. Each leg is a separate line (same color, different phase).
Shows where in the gait cycle each leg hits the friction cone.

Usage (from milliquad_opt/):
    uv run python -m analysis.l2_l4.H2.plot_phase_folded_slip \
        results/20260228T013353_rk4_flat \
        results/20260228T230022_step_q60_rk-warm \
        results/20260228T202903_rough_spatial_rk4
"""

from __future__ import annotations

import argparse
import csv
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
SLIP_THRESHOLD = 0.95


def _load_sliding_friction(run_dir: pathlib.Path) -> float:
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    if not rows:
        raise FileNotFoundError(f"No rows in {bests_csv}")
    return float(rows[-1]["sliding_friction"])


def _collect(run_dir: pathlib.Path):
    """Phase-fold per-leg slip ratio.

    Returns:
        data: {freq: {scene: {"legs": list of 4 (mean, std) or None, "n": int}}}
        freqs: sorted list
        terrain: str
        mu: float
    """
    terrain = detect_terrain(run_dir)
    npz = np.load(find_npz(run_dir))
    mu = _load_sliding_friction(run_dir)

    bin_edges = np.linspace(0, 2 * np.pi, N_BINS + 1)

    trials: dict[tuple[str, float], list[int]] = {}
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[3] == "time":
            scene, freq, tidx, _ = parsed
            trials.setdefault((scene, freq), []).append(tidx)

    # per_trial[freq][scene] = list of (4, N_BINS) arrays (nan where no contact)
    per_trial: dict[float, dict[str, list[np.ndarray]]] = (
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

            drive = npz[f"{prefix}_drive_angle"][mask]
            contact = npz[f"{prefix}_leg_in_contact"][mask]   # (T', 4)
            fn = npz[f"{prefix}_leg_normal_force"][mask]       # (T', 4)
            ft = npz[f"{prefix}_leg_tangent_force"][mask]      # (T', 4)

            phase = drive % (2 * np.pi)
            bin_idx = np.clip(np.digitize(phase, bin_edges) - 1, 0, N_BINS - 1)

            leg_profiles = np.full((4, N_BINS), np.nan)
            for leg in range(4):
                for b in range(N_BINS):
                    in_bin = bin_idx == b
                    valid = in_bin & contact[:, leg] & (fn[:, leg] > 1e-10)
                    if valid.sum() == 0:
                        continue
                    ratio = ft[valid, leg] / (mu * fn[valid, leg])
                    leg_profiles[leg, b] = float(ratio.mean())

            per_trial[freq][scene].append(leg_profiles)

    npz.close()

    all_freqs = sorted(per_trial.keys())
    data: dict[float, dict[str, dict]] = {}
    for freq in all_freqs:
        data[freq] = {}
        for scene in MORPH_ORDER:
            trials_list = per_trial[freq].get(scene, [])
            if not trials_list:
                continue
            stacked = np.array(trials_list)  # (n_trials, 4, N_BINS)
            legs = []
            for leg in range(4):
                leg_data = stacked[:, leg, :]  # (n_trials, N_BINS)
                with np.errstate(all="ignore"):
                    mean = np.nanmean(leg_data, axis=0)
                    std = np.nanstd(leg_data, axis=0)
                if np.all(np.isnan(mean)):
                    legs.append(None)
                else:
                    legs.append((mean, std))
            data[freq][scene] = {"legs": legs, "n": len(trials_list)}

    return data, all_freqs, terrain, mu


def plot(run_dirs: list[pathlib.Path], save: bool = True):
    # Collect per-terrain
    terrain_data: dict[str, tuple] = {}
    for rd in run_dirs:
        t = detect_terrain(rd)
        terrain_data[t] = _collect(rd)

    terrain_order = [t for t in ["flat", "step", "rough"] if t in terrain_data]
    if not terrain_order:
        print("No data to plot.")
        return None

    # Union of all frequencies across terrains
    all_freqs: set[float] = set()
    for data, freqs, _, _ in terrain_data.values():
        all_freqs.update(freqs)
    freq_list = sorted(all_freqs)

    n_rows = len(terrain_order)
    n_cols = len(freq_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows),
                             sharey=True, squeeze=False)

    bin_centers = np.linspace(0, 360, N_BINS, endpoint=False) + 5

    for row, terrain in enumerate(terrain_order):
        data, freqs, _, mu = terrain_data[terrain]

        for col, freq in enumerate(freq_list):
            ax = axes[row, col]

            if freq not in data:
                ax.set_visible(False)
                continue

            for scene in MORPH_ORDER:
                if scene not in data[freq]:
                    continue
                d = data[freq][scene]
                color = MORPH_COLORS[scene]

                any_plotted = False
                for leg_data in d["legs"]:
                    if leg_data is None:
                        continue
                    mean, std = leg_data
                    label = (MORPH_LABELS[scene]
                             if (row == 0 and col == 0 and not any_plotted)
                             else None)
                    ax.plot(bin_centers, mean, color=color, linewidth=1.2,
                            alpha=0.7, label=label)
                    ax.fill_between(bin_centers, mean - std, mean + std,
                                    color=color, alpha=0.08)
                    any_plotted = True

            ax.axhline(SLIP_THRESHOLD, color="0.5", linewidth=0.5,
                        linestyle="--", zorder=0)
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_ylim(0, 1.05)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                ax.set_title(f"{freq:g} Hz")
            if row == n_rows - 1:
                ax.set_xlabel("Drive angle (deg)")

        terrain_label = TERRAIN_TITLES.get(terrain, terrain)
        axes[row, 0].set_ylabel(f"{terrain_label}\n($\\mu$ = {mu:.3f})\n\nSlip ratio")

    fig.legend(
        [plt.Line2D([0], [0], color=MORPH_COLORS[s], lw=2) for s in MORPH_ORDER],
        [MORPH_LABELS[s] for s in MORPH_ORDER],
        loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.99),
    )

    fig.suptitle("Phase-folded slip ratio", fontsize=12, y=1.02)
    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        tag = "_".join(terrain_order)
        out = FIGURE_DIR / f"{ts}_phase_folded_slip_{tag}.png"
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
    args = parser.parse_args()
    plot(args.run_dirs, save=not args.no_save)


if __name__ == "__main__":
    main()
