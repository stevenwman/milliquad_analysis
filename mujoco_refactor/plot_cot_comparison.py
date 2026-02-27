#!/usr/bin/env python3
"""Side-by-side COT comparison: flat vs step terrain.

Usage:
    MPLBACKEND=Agg uv run python plot_cot_comparison.py
"""
import csv
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 14
import matplotlib.pyplot as plt
import numpy as np

from morphology_style import MORPH_COLORS, MORPH_LABELS, MORPH_ORDER

# Map scene keys to short morph keys
SCENE_ORDER = MORPH_ORDER

FLAT_CSV = "results/20260225T122342_flat_10_30_50/cot_results_v2.csv"
STEP_CSV = "results/20260225T225248_step_argmin_progress/cot_step_results.csv"


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            cot_val = r.get("cot") or r.get("cot_signed")
            rows.append({"scene": r["scene"], "freq": float(r["freq"]), "cot": float(cot_val)})
    return rows


def build_data(rows):
    data = {}
    for scene in SCENE_ORDER:
        sr = [r for r in rows if r["scene"] == scene]
        freqs = sorted(set(r["freq"] for r in sr))
        means, stds, tf, tc = [], [], [], []
        for f in freqs:
            c = [r["cot"] for r in sr if r["freq"] == f]
            means.append(np.mean(c))
            stds.append(np.std(c, ddof=1) if len(c) > 1 else 0.0)
            tf.extend([f] * len(c))
            tc.extend(c)
        data[scene] = {
            "mean_freqs": np.array(freqs),
            "means": np.array(means),
            "stds": np.array(stds),
            "trial_freqs": tf,
            "trial_cots": tc,
        }
    return data


flat_data = build_data(load_csv(FLAT_CSV))
step_data = build_data(load_csv(STEP_CSV))

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 7.2))

for scene in SCENE_ORDER:
    color = MORPH_COLORS[scene]
    label = MORPH_LABELS[scene]

    # Flat (top)
    d = flat_data[scene]
    if len(d["mean_freqs"]) > 0:
        ax_top.fill_between(d["mean_freqs"], d["means"] - d["stds"], d["means"] + d["stds"],
                            color=color, alpha=0.3, label=label)
        ax_top.scatter(d["trial_freqs"], d["trial_cots"], color=color, alpha=0.5, s=25, zorder=3)

    # Step (bottom)
    d = step_data[scene]
    if len(d["mean_freqs"]) > 0:
        ax_bot.fill_between(d["mean_freqs"], d["means"] - d["stds"], d["means"] + d["stds"],
                            color=color, alpha=0.3)
        ax_bot.scatter(d["trial_freqs"], d["trial_cots"], color=color, alpha=0.5, s=25, zorder=3)

ax_top.set_title("Flat Terrain")
ax_bot.set_title("Step Terrain")
for ax in (ax_top, ax_bot):
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cost of Transport")
    ax.grid(True, alpha=0.2)

# Auto-fit each x-axis independently
for ax, data in [(ax_top, flat_data), (ax_bot, step_data)]:
    all_f = sorted(set(f for d in data.values() for f in d["mean_freqs"]))
    if all_f:
        ax.set_xticks([int(f) for f in all_f])
        ax.set_xlim(all_f[0] - 3, all_f[-1] + 3)

# Single legend on bottom panel
ax_bot.legend(*ax_top.get_legend_handles_labels(), loc="upper left", fontsize=12, framealpha=0.9)

fig.tight_layout()
out = "results/cot_flat_vs_step.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
