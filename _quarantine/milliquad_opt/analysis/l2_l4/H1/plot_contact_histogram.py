#!/usr/bin/env python3
"""H1 Plot 1b: Simultaneous contact count histogram + body contact.

Grouped bar chart — x = number of legs in contact (0–4) plus a "Body" bin
showing chassis-terrain contact fraction.  One panel per (terrain, freq).

Bars show mean across valid trials, error bars show ±1 std.
Annotated with N=valid/total trial counts per morphology.

Usage (from milliquad_opt/):
    uv run python -m analysis.l2_l4.H1.plot_contact_histogram \
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
    SETTLE_TIME,
    active_mask,
    detect_terrain,
    find_npz,
    is_valid_trial,
    parse_key,
)

FIGURE_DIR = pathlib.Path(__file__).parent / "figures"
N_LEG_BINS = 5  # 0, 1, 2, 3, 4 simultaneous leg contacts
N_BINS = N_LEG_BINS + 1  # +1 for body contact bin
BIN_LABELS = ["0", "1", "2", "3", "4", "Body"]


def _collect_histograms(run_dir: pathlib.Path, failed: bool = False):
    """Return per-trial histograms (leg bins + body bin) and trial counts.

    Args:
        failed: if True, collect FAILED trials instead of passing ones.
                Uses time-only mask (no spatial gating) since robot may not
                have reached the terrain region.

    Returns:
        hists:  {freq: {scene: (mean_hist, std_hist)}}  — shape (6,)
        counts: {freq: {scene: (n_selected, n_total)}}
        freqs:  sorted list of frequencies
    """
    terrain = detect_terrain(run_dir)
    npz = np.load(find_npz(run_dir))

    has_body = any("body_in_contact" in k for k in npz.files)

    trials: dict[tuple[str, float], list[int]] = {}
    for key in npz.files:
        parsed = parse_key(key)
        if parsed and parsed[3] == "time":
            scene, freq, tidx, _ = parsed
            trials.setdefault((scene, freq), []).append(tidx)

    per_trial_hists: dict[float, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    n_total: dict[float, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for (scene, freq), tidxs in trials.items():
        if scene not in MORPH_ORDER:
            continue
        for t in tidxs:
            n_total[freq][scene] += 1
            prefix = f"{scene}_f{freq:g}_t{t}"
            time = npz[f"{prefix}_time"]
            pos_x = npz[f"{prefix}_pos_x"]
            pitch = npz[f"{prefix}_pitch"]
            contact = npz[f"{prefix}_leg_in_contact"]

            valid = is_valid_trial(pos_x, pitch, terrain, scene, freq)
            if failed and valid:
                continue  # skip passing trials
            if not failed and not valid:
                continue  # skip failing trials

            if failed:
                mask = time >= SETTLE_TIME
            else:
                mask = active_mask(time, pos_x, terrain)
            if mask.sum() < 10:
                continue

            n_legs = contact[mask].sum(axis=1)
            leg_hist = np.bincount(n_legs, minlength=N_LEG_BINS)[:N_LEG_BINS].astype(float)
            n_active = float(leg_hist.sum())
            leg_hist /= n_active

            body_frac = 0.0
            if has_body:
                body_key = f"{prefix}_body_in_contact"
                if body_key in npz:
                    body_frac = float(npz[body_key][mask].mean())

            hist = np.append(leg_hist, body_frac)
            per_trial_hists[freq][scene].append(hist)

    npz.close()

    all_freqs = sorted(per_trial_hists.keys())
    hists = {}
    counts = {}
    for freq in all_freqs:
        hists[freq] = {}
        counts[freq] = {}
        for scene in MORPH_ORDER:
            trial_hists = per_trial_hists[freq].get(scene, [])
            n_valid = len(trial_hists)
            counts[freq][scene] = (n_valid, n_total[freq].get(scene, 0))
            if n_valid > 0:
                arr = np.array(trial_hists)
                hists[freq][scene] = (arr.mean(axis=0), arr.std(axis=0))

    return hists, counts, all_freqs


def plot(run_dirs: list[pathlib.Path], save: bool = True, failed: bool = False):
    terrain_data = {}
    for rd in run_dirs:
        t = detect_terrain(rd)
        terrain_data[t] = _collect_histograms(rd, failed=failed)

    # Drop terrains with no data (e.g. flat has zero failures)
    terrain_order = [
        t for t in ["flat", "step", "rough"]
        if t in terrain_data and terrain_data[t][2]  # has any frequencies
    ]

    if not terrain_order:
        print("No data to plot.")
        return None

    # Unified frequency grid
    all_freqs = sorted(set(
        f for t in terrain_order for f in terrain_data[t][2]
    ))

    n_rows = len(terrain_order)
    n_cols = len(all_freqs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows),
                             sharey=True, squeeze=False)

    bar_width = 0.18
    x = np.arange(N_BINS)

    for row, terrain in enumerate(terrain_order):
        hists_by_freq, counts_by_freq, freqs = terrain_data[terrain]

        for col, freq in enumerate(all_freqs):
            ax = axes[row, col]

            if freq not in hists_by_freq:
                ax.set_visible(False)
                continue

            hists = hists_by_freq[freq]
            counts = counts_by_freq[freq]

            morphs_present = [s for s in MORPH_ORDER if s in hists]
            n_morphs = len(morphs_present)
            if n_morphs == 0:
                ax.text(0.5, 0.5, "no valid\ntrials", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="0.5")
                ax.set_xticks(x)
                ax.set_xticklabels(BIN_LABELS)
                continue

            offsets = np.linspace(-(n_morphs - 1) / 2, (n_morphs - 1) / 2, n_morphs) * bar_width

            for i, scene in enumerate(morphs_present):
                mean_h, std_h = hists[scene]

                # Leg bins: solid fill
                ax.bar(
                    x[:N_LEG_BINS] + offsets[i],
                    mean_h[:N_LEG_BINS],
                    yerr=std_h[:N_LEG_BINS],
                    width=bar_width * 0.9,
                    color=MORPH_COLORS[scene],
                    label=MORPH_LABELS[scene] if row == 0 and col == 0 else None,
                    edgecolor="white",
                    linewidth=0.5,
                    capsize=2,
                    error_kw={"linewidth": 0.8},
                )

                # Body bin: hatched to distinguish from leg counts
                ax.bar(
                    x[N_LEG_BINS] + offsets[i],
                    mean_h[N_LEG_BINS],
                    yerr=std_h[N_LEG_BINS],
                    width=bar_width * 0.9,
                    color=MORPH_COLORS[scene],
                    edgecolor="white",
                    linewidth=0.5,
                    hatch="//",
                    alpha=0.7,
                    capsize=2,
                    error_kw={"linewidth": 0.8},
                )

            # Separator line between leg and body bins
            sep_x = N_LEG_BINS - 0.5
            ax.axvline(sep_x, color="0.7", linewidth=0.8, linestyle="--", zorder=0)

            # Trial count annotation
            ann_parts = []
            for scene in MORPH_ORDER:
                n_valid, n_tot = counts.get(scene, (0, 0))
                if n_tot > 0:
                    ann_parts.append(f"{MORPH_LABELS[scene]}={n_valid}/{n_tot}")
            ax.text(0.5, 1.0, "  ".join(ann_parts), transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6, color="0.4")

            ax.set_xticks(x)
            ax.set_xticklabels(BIN_LABELS)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                ax.set_title(f"{freq:g} Hz", pad=14)
            if row == n_rows - 1:
                ax.set_xlabel("Contact type")

        axes[row, 0].set_ylabel(f"{TERRAIN_TITLES.get(terrain, terrain)}\nFraction of timesteps")

    # Legend
    handles, labels = [], []
    for scene in MORPH_ORDER:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=MORPH_COLORS[scene]))
        labels.append(MORPH_LABELS[scene])
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9,
               bbox_to_anchor=(0.99, 0.99))

    mode = "FAILED trials" if failed else "passing trials"
    fig.suptitle(f"H1: Leg + body contacts by morphology & frequency ({mode})",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        terrain_tag = "_".join(terrain_order)
        suffix = "_failed" if failed else ""
        out = FIGURE_DIR / f"{ts}_contact_histogram_{terrain_tag}{suffix}.png"
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
    parser.add_argument("--failed", action="store_true",
                        help="Show failed trials instead of passing ones.")
    args = parser.parse_args()
    plot(args.run_dirs, save=not args.no_save, failed=args.failed)


if __name__ == "__main__":
    main()
