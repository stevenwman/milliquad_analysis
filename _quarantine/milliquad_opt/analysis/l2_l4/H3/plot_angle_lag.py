#!/usr/bin/env python3
"""H4 Plot: Angle lag distribution per morphology.

1x4 grid — one polar histogram per morphology showing the distribution of
(joint_pos[leg] - drive_angle) mod 2pi for each leg. Legs overlaid as
semi-transparent histograms.

L1's single active leg clusters at one lag value. L4's 4 legs should be
spread ~90 deg apart (pi/2 phase offsets).

Usage (from milliquad_opt/):
    uv run python -m analysis.l2_l4.H3.plot_angle_lag \
        results/20260228T013353_rk4_flat
    uv run python -m analysis.l2_l4.H3.plot_angle_lag \
        results/20260228T013353_rk4_flat --freq 10
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
N_ANGLE_BINS = 36
LEG_NAMES = ["FR", "FL", "BR", "BL"]
LEG_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def _collect_lags(run_dir: pathlib.Path, freq_filter: float | None = None):
    """Pool angle lag samples per morphology and leg across valid trials.

    Returns:
        lags: {scene: {leg_idx: array of lag values in [0, 2pi)}}
        terrain: str
        freqs_used: set of freqs included
    """
    terrain = detect_terrain(run_dir)
    npz = np.load(find_npz(run_dir))

    trials: dict[tuple[str, float], list[int]] = {}
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[3] == "time":
            scene, freq, tidx, _ = parsed
            if freq_filter is not None and freq != freq_filter:
                continue
            trials.setdefault((scene, freq), []).append(tidx)

    lags: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    freqs_used = set()

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

            joint_pos = npz[f"{prefix}_joint_pos"][mask]  # (T, 4)
            drive = npz[f"{prefix}_drive_angle"][mask]    # (T,)

            for leg in range(4):
                lag = (joint_pos[:, leg] - drive) % (2 * np.pi)
                lags[scene][leg].append(lag)

            freqs_used.add(freq)

    npz.close()

    # Concatenate per-leg arrays
    for scene in lags:
        for leg in lags[scene]:
            lags[scene][leg] = np.concatenate(lags[scene][leg])

    return lags, terrain, freqs_used


def plot(run_dirs: list[pathlib.Path], save: bool = True, freq_filter: float | None = None):
    rd = run_dirs[0]
    lags, terrain, freqs_used = _collect_lags(rd, freq_filter)

    if not lags:
        print("No data to plot.")
        return None

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), subplot_kw={"projection": "polar"})

    bin_edges = np.linspace(0, 2 * np.pi, N_ANGLE_BINS + 1)

    for i, scene in enumerate(MORPH_ORDER):
        ax = axes[i]
        if scene not in lags:
            ax.set_title(MORPH_LABELS[scene], fontsize=11)
            ax.text(0, 0, "no data", ha="center", va="center", fontsize=9, color="0.5")
            continue

        for leg in range(4):
            if leg not in lags[scene] or len(lags[scene][leg]) == 0:
                continue
            counts, _ = np.histogram(lags[scene][leg], bins=bin_edges, density=True)
            # Bar width = bin width
            theta = bin_edges[:-1] + np.pi / N_ANGLE_BINS  # bin centers
            ax.bar(theta, counts, width=2 * np.pi / N_ANGLE_BINS,
                   color=LEG_COLORS[leg], alpha=0.4, label=LEG_NAMES[leg],
                   edgecolor=LEG_COLORS[leg], linewidth=0.5)

        ax.set_title(MORPH_LABELS[scene], fontsize=11, pad=12)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.3, 1.15),
                      fontsize=7, ncol=4, frameon=False)

    freq_str = f" @ {freq_filter:g}Hz" if freq_filter else f" (all freqs)"
    terrain_label = TERRAIN_TITLES.get(terrain, terrain)
    fig.suptitle(f"Angle lag (joint - drive) — {terrain_label}{freq_str}",
                 fontsize=12, y=1.05)
    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        freq_tag = f"_f{freq_filter:g}" if freq_filter else ""
        out = FIGURE_DIR / f"{ts}_angle_lag_{terrain}{freq_tag}.png"
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
    parser.add_argument("--freq", type=float, default=None,
                        help="Filter to a single frequency (e.g. 10)")
    args = parser.parse_args()
    plot(args.run_dirs, save=not args.no_save, freq_filter=args.freq)


if __name__ == "__main__":
    main()
