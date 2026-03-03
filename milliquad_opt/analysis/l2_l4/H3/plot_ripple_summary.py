#!/usr/bin/env python3
"""H3 Plot: Ripple coefficient summary.

1x3 figure — one panel per terrain (flat, step, rough).
x-axis = morphology, grouped bars colored by frequency.
Shared y-axis for cross-terrain comparison.

Usage (from milliquad_opt/):
    uv run python -m analysis.l2_l4.H3.plot_ripple_summary \
        results/20260228T013353_rk4_flat \
        results/20260228T230022_step_q60_rk-warm \
        results/20260228T202903_rough_spatial_rk4
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


def _collect(run_dir: pathlib.Path):
    """Compute ripple coefficient per (scene, freq).

    Returns:
        data: {freq: {scene: {"ripple": (mean, std), "n": int}}}
        freqs: sorted list
        terrain: str
    """
    terrain = detect_terrain(run_dir)
    npz = np.load(find_npz(run_dir))

    trials: dict[tuple[str, float], list[int]] = {}
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[3] == "time":
            scene, freq, tidx, _ = parsed
            trials.setdefault((scene, freq), []).append(tidx)

    per_trial: dict[float, dict[str, list[float]]] = (
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

            time_a = time[mask]
            dt = float(time_a[1] - time_a[0])
            if dt < 1e-10:
                continue

            vx = npz[f"{prefix}_vel_x"][mask]
            mean_vx = float(np.mean(vx))

            period = 1.0 / freq
            steps_per_rev = max(1, int(period / dt))
            n_full_revs = len(vx) // steps_per_rev

            if n_full_revs < 2 or abs(mean_vx) < 1e-6:
                continue

            vx_revs = vx[:n_full_revs * steps_per_rev].reshape(n_full_revs, steps_per_rev)
            rev_stds = vx_revs.std(axis=1)
            ripple = float(np.mean(rev_stds) / abs(mean_vx))

            per_trial[freq][scene].append(ripple)

    npz.close()

    all_freqs = sorted(per_trial.keys())
    data: dict[float, dict[str, dict]] = {}
    for freq in all_freqs:
        data[freq] = {}
        for scene in MORPH_ORDER:
            vals = per_trial[freq].get(scene, [])
            if not vals:
                continue
            arr = np.array(vals)
            data[freq][scene] = {
                "ripple": (float(arr.mean()), float(arr.std())),
                "n": len(vals),
            }

    return data, all_freqs, terrain


def plot(run_dirs: list[pathlib.Path], save: bool = True):
    terrain_data = {}
    for rd in run_dirs:
        t = detect_terrain(rd)
        terrain_data[t] = _collect(rd)

    terrain_order = [t for t in ["flat", "step", "rough"] if t in terrain_data]
    if not terrain_order:
        print("No data to plot.")
        return None

    n_morphs = len(MORPH_ORDER)

    n_cols = len(terrain_order)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 3.5),
                             sharey=True, squeeze=False)

    # Bar geometry: groups of n_morphs bars per frequency
    bar_w = 0.7 / n_morphs

    for col, terrain in enumerate(terrain_order):
        ax = axes[0, col]
        data, freqs, _ = terrain_data[terrain]
        n_freqs = len(freqs)

        for mi, scene in enumerate(MORPH_ORDER):
            offset = (mi - (n_morphs - 1) / 2) * bar_w
            vals = []
            errs = []
            positions = []
            for fi, freq in enumerate(freqs):
                if scene in data.get(freq, {}):
                    d = data[freq][scene]
                    vals.append(d["ripple"][0])
                    errs.append(d["ripple"][1])
                else:
                    vals.append(0)
                    errs.append(0)
                positions.append(fi + offset)

            color = MORPH_COLORS[scene]
            ax.bar(positions, vals, width=bar_w * 0.9, color=color,
                   yerr=errs, edgecolor="white", linewidth=0.5,
                   capsize=2, error_kw={"linewidth": 0.7}, alpha=0.85,
                   label=MORPH_LABELS[scene] if col == 0 else None)

        ax.set_xticks(range(n_freqs))
        ax.set_xticklabels([f"{f:g}" for f in freqs])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(TERRAIN_TITLES.get(terrain, terrain))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0, 0].set_ylabel("Ripple coefficient")

    # Legend using morphology colors
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MORPH_COLORS[s], alpha=0.85)
               for s in MORPH_ORDER]
    labels = [MORPH_LABELS[s] for s in MORPH_ORDER]
    fig.legend(handles, labels, loc="upper right", frameon=False,
               fontsize=9, bbox_to_anchor=(0.99, 0.95))

    fig.suptitle("Velocity ripple coefficient", fontsize=12, y=1.02)
    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        tag = "_".join(terrain_order)
        out = FIGURE_DIR / f"{ts}_ripple_summary_{tag}.png"
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
