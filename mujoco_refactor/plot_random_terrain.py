#!/usr/bin/env python3
"""Random terrain: success rate + speed vs frequency, side by side.

Usage:
    MPLBACKEND=Agg uv run python plot_random_terrain.py
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 16
import matplotlib.pyplot as plt
import numpy as np

# ── Raw trial data from experimental table (n/a = failed trial) ──
FREQS = [10, 30, 50]

# Each morph has 5 trials per frequency; None = failed/n/a
DATA = {
    "L1": {
        "color": "#1E88E5",
        "trials": {
            10: [44.37, 42.07, 42.48, None, 42.76],
            30: [86.09, 76.47, 71.04, 92.86, None],
            50: [None, None, 56.52, None, None],
        },
    },
    "L2": {
        "color": "#FFC107",
        "trials": {
            10: [65.66, 71.43, 65.66, 68.42, 56.77],
            30: [129.13, 128.97, 128.91, 128.49, None],
            50: [None, 70.65, 134.02, None, 114.04],
        },
    },
    "L4": {
        "color": "#007561",
        "trials": {
            10: [87.25, 75.58, 94.2, 82.8, 88.44],
            30: [144.44, 178.08, 166.67, 94.89, None],
            50: [None, None, None, None, 101.56],
        },
    },
    "WR": {
        "color": "#D81B60",
        "trials": {
            10: [None, None, None, 81.25, None],
            30: [None, 123.8, None, 178.08, 160.49],
            50: [None, None, None, None, 180.56],
        },
    },
}

N_TRIALS = 5
morphs = list(DATA.keys())
n_morphs = len(morphs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), width_ratios=[1, 2])
fig.subplots_adjust(wspace=0.3)

# ── Left: Success Rate bar chart ──
bar_width = 0.18
x = np.arange(len(FREQS))

for i, morph in enumerate(morphs):
    d = DATA[morph]
    rates = [sum(1 for v in d["trials"][f] if v is not None) / N_TRIALS * 100 for f in FREQS]
    offset = (i - (n_morphs - 1) / 2) * bar_width
    ax1.bar(x + offset, rates, bar_width, color=d["color"], label=morph, edgecolor="white", linewidth=0.5)

ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Success Rate (%, n = 5)")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{f} Hz" for f in FREQS])
ax1.set_ylim(0, 105)
ax1.set_yticks([0, 20, 40, 60, 80, 100])
ax1.grid(True, alpha=0.2, axis="y")

# ── Right: Speed vs Frequency — scatter only, categorical x-axis ──
freq_to_x = {f: i for i, f in enumerate(FREQS)}  # 10→0, 20→1, 30→2
jitter_width = 0.22  # offset per morphology from center
for i, morph in enumerate(morphs):
    d = DATA[morph]
    offset = (i - (n_morphs - 1) / 2) * jitter_width
    trial_xs = []
    trial_speeds = []
    for f in FREQS:
        valid = sorted([v for v in d["trials"][f] if v is not None])
        n = len(valid)
        for rank, v in enumerate(valid):
            # spread along a diagonal: lowest speed left, highest right
            dx = (rank / (n - 1) - 0.5) * 0.06 if n > 1 else 0.0
            trial_xs.append(freq_to_x[f] + offset + dx)
            trial_speeds.append(v)

    ax2.scatter(trial_xs, trial_speeds, color=d["color"], s=30, alpha=0.8, zorder=3)

# Alternating background bands to emphasize discrete bins
for j in range(len(FREQS)):
    if j % 2 == 0:
        ax2.axvspan(j - 0.5, j + 0.5, color="#f0f0f0", zorder=0)
    ax2.axvline(j - 0.5, color="#cccccc", linewidth=0.5, zorder=1)
ax2.axvline(len(FREQS) - 0.5, color="#cccccc", linewidth=0.5, zorder=1)

ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Speed (mm/s)")
ax2.set_xticks(range(len(FREQS)))
ax2.set_xticklabels([f"{f} Hz" for f in FREQS])
ax2.set_xlim(-0.5, len(FREQS) - 0.5)
ax2.grid(True, alpha=0.2, axis="y")

# Legend inside top-left of scatter
ax2.legend(morphs, loc="upper left", fontsize=12, framealpha=0.9)

fig.tight_layout()
from pathlib import Path
out = str(Path(__file__).resolve().parent.parent / "experimental_data" / "plots" / "random_terrain_summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
