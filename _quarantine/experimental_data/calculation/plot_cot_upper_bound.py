"""Plot COT upper bound from experimental kinematics.

2-row figure (flat, step) matching the style of cot_overlay_90_65.png:
  - Scatter dots per trial, shading band (mean ± std)
  - Morphology colors: L1=blue, L2=gold, L4=green, WR=pink
  - Dodged by morphology within each frequency bracket
  - Grey gap bands between frequency zones

Usage:
    python plot_cot_upper_bound.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 6
import matplotlib.pyplot as plt
import numpy as np

from cot_upper_bound import (
    FLAT_CONDITIONS,
    STEP_CONDITIONS,
    CSV_DIR_FLAT,
    CSV_DIR_STEP,
    compute_trial_cot,
)

# Morphology display config
MORPH_ORDER = ["leg", "2leg", "4leg", "wheel"]
MORPH_LABELS = {"leg": "L1", "2leg": "L2", "4leg": "L4", "wheel": "WR"}
MORPH_COLORS = {
    "leg": "#5b9bd5",
    "2leg": "#ffc000",
    "4leg": "#2e8b57",
    "wheel": "#e91e78",
}

DODGE_WIDTH = 3.5  # Hz between morphologies


def _collect_cot(conditions, csv_dir, terrain):
    """Collect per-trial COT values into {morph: {freq: [cot_values]}}."""
    data = {}
    for freq, morph, files in conditions:
        if morph not in data:
            data[morph] = {}
        cots = []
        for fname in files:
            p = csv_dir / fname
            if not p.exists():
                continue
            r = compute_trial_cot(p, morph, terrain)
            if r is not None:
                cots.append(r["cot_upper"])
        if cots:
            data[morph][freq] = cots
    return data


def _plot_terrain(ax, data, title):
    """Plot one terrain row with dodged scatter + shading."""
    all_freqs = sorted({f for m in data for f in data[m]})
    n_morph = len(MORPH_ORDER)

    for mi, morph in enumerate(MORPH_ORDER):
        if morph not in data or not data[morph]:
            continue
        color = MORPH_COLORS[morph]
        label = MORPH_LABELS[morph]
        dx = (mi - (n_morph - 1) / 2) * DODGE_WIDTH / (n_morph - 1)

        freqs_present = sorted(data[morph].keys())
        means, stds, xs = [], [], []
        for freq in freqs_present:
            vals = data[morph][freq]
            x = freq + dx
            # Scatter individual trials
            for v in vals:
                ax.scatter(x, v, color=color, s=18, zorder=5, alpha=0.8,
                           edgecolors="none")
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            xs.append(x)

        xs = np.array(xs)
        means = np.array(means)
        stds = np.array(stds)

        # Shading band
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.2)
        ax.plot(xs, means, color=color, linewidth=1.2, label=label, zorder=4)

    # Bracket ticks + grey gap bands
    half_spread = DODGE_WIDTH / 2 + 0.75
    for fi, freq in enumerate(all_freqs):
        lo = freq - half_spread
        hi = freq + half_spread
        ax.plot([lo, lo], [0, 0], marker="|", color="black", ms=6, mew=0.8,
                clip_on=False, zorder=10)
        ax.plot([hi, hi], [0, 0], marker="|", color="black", ms=6, mew=0.8,
                clip_on=False, zorder=10)
        # Grey band in gap to next freq
        if fi < len(all_freqs) - 1:
            next_lo = all_freqs[fi + 1] - half_spread
            ax.axvspan(hi, next_lo, color="#f0f0f0", zorder=0)
        # Grey band on edges
        if fi == 0:
            ax.axvspan(freq - half_spread - 3, lo, color="#f0f0f0", zorder=0)
        if fi == len(all_freqs) - 1:
            ax.axvspan(hi, freq + half_spread + 3, color="#f0f0f0", zorder=0)

    ax.set_xticks(all_freqs)
    ax.set_xlim(all_freqs[0] - half_spread - 2, all_freqs[-1] + half_spread + 2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=-0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)


def main():
    flat_data = _collect_cot(FLAT_CONDITIONS, CSV_DIR_FLAT, "flat")
    step_data = _collect_cot(STEP_CONDITIONS, CSV_DIR_STEP, "step")

    fig, (ax_flat, ax_step) = plt.subplots(2, 1, figsize=(3.5, 5.0))

    _plot_terrain(ax_flat, flat_data, "Flat")
    _plot_terrain(ax_step, step_data, "Step")

    # Labels
    ax_flat.set_ylabel("COT upper bound", fontsize=9)
    ax_step.set_ylabel("COT upper bound", fontsize=9)
    ax_step.set_xlabel("Drive frequency (Hz)", fontsize=9)
    ax_flat.set_xlabel("Drive frequency (Hz)", fontsize=9)

    # Subplot lettering
    ax_flat.text(0.02, 0.95, "(a)", transform=ax_flat.transAxes,
                 fontsize=10, fontweight="bold", va="top")
    ax_step.text(0.02, 0.95, "(b)", transform=ax_step.transAxes,
                 fontsize=10, fontweight="bold", va="top")

    # Legend
    handles, labels = ax_flat.get_legend_handles_labels()
    ax_flat.legend().remove() if ax_flat.get_legend() else None
    ax_step.legend().remove() if ax_step.get_legend() else None

    fig.suptitle("Exp COT Upper Bound (tau_max × |ω|)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    if handles:
        plot_center = (ax_flat.get_position().x0 + ax_flat.get_position().x1) / 2
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   fontsize=7, framealpha=0.9,
                   bbox_to_anchor=(plot_center, 0.02))

    out = Path(__file__).resolve().parent / "cot_upper_bound.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
