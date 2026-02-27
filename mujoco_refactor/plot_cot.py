#!/usr/bin/env python3
"""Plot COT vs frequency from cot_results.csv.

Matches style of experimental_data/plot_velocity_vs_freq.py:
  - 4 morphology lines (1-leg, 2-leg, 4-leg, wheel)
  - Shaded std bands from top-K trials
  - Individual trial points as scatter

Usage:
    uv run python plot_cot.py results/20260225T122342_flat_10_30_50/cot_results.csv
    uv run python plot_cot.py flat/cot_results.csv step/cot_results.csv --labels "flat params" "step params"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 14
import matplotlib.pyplot as plt
import numpy as np

from morphology_style import MORPH_COLORS as COLORS, MORPH_LABELS as LABELS


def load_cot_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            # v1 has "cot", v2 has "cot_signed"
            cot_val = row.get("cot") or row.get("cot_signed")
            rows.append({
                "scene": row["scene"],
                "freq": float(row["freq"]),
                "cot": float(cot_val),
                "velocity_mps": float(row["velocity_mps"]),
            })
    return rows


def build_plot_data(rows: list[dict]) -> dict:
    """Group by morphology → {scene: {freqs, trials, mean_freqs, means, stds}}."""
    data = {}
    for scene in COLORS:
        scene_rows = [r for r in rows if r["scene"] == scene]
        freqs_all = sorted(set(r["freq"] for r in scene_rows))

        trial_freqs = []
        trial_cots = []
        mean_freqs = []
        means = []
        stds = []

        for freq in freqs_all:
            cots = [r["cot"] for r in scene_rows if r["freq"] == freq]
            trial_freqs.extend([freq] * len(cots))
            trial_cots.extend(cots)
            mean_freqs.append(freq)
            means.append(np.mean(cots))
            stds.append(np.std(cots, ddof=1) if len(cots) > 1 else 0.0)

        data[scene] = {
            "freqs": trial_freqs,
            "trials": trial_cots,
            "mean_freqs": mean_freqs,
            "means": means,
            "stds": stds,
        }
    return data


def plot_cot(ax, data: dict, title: str):
    for scene in ("scene1", "scene2", "scene4", "scene_wheel"):
        d = data[scene]
        if not d["mean_freqs"]:
            continue
        freq_arr = np.array(d["mean_freqs"])
        mean = np.array(d["means"])
        std = np.array(d["stds"])
        ax.fill_between(freq_arr, mean - std, mean + std,
                         color=COLORS[scene], alpha=0.2)
        ax.plot(freq_arr, mean, "-o", color=COLORS[scene], label=LABELS[scene],
                markersize=5, linewidth=1.5)
        ax.scatter(d["freqs"], d["trials"], color=COLORS[scene], alpha=0.4, s=20, zorder=3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cost of Transport (dimensionless)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Auto-fit x-axis to data
    all_freqs = sorted(set(f for d in data.values() for f in d["mean_freqs"]))
    if all_freqs:
        ax.set_xticks(all_freqs)
        ax.set_xlim(all_freqs[0] - 3, all_freqs[-1] + 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_files", nargs="+", help="One or two cot_results.csv paths")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each CSV (default: derive from path)")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    if len(args.csv_files) == 1:
        rows = load_cot_csv(args.csv_files[0])
        data = build_plot_data(rows)
        label = (args.labels[0] if args.labels else Path(args.csv_files[0]).parent.name)

        fig, ax = plt.subplots(figsize=(7, 5))
        plot_cot(ax, data, f"COT vs Frequency — {label}")
        fig.tight_layout()

        out = args.output or str(Path(args.csv_files[0]).parent / "cot_vs_freq.png")
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")

    elif len(args.csv_files) == 2:
        labels = args.labels or [Path(f).parent.name for f in args.csv_files]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for i, (csv_path, label) in enumerate(zip(args.csv_files, labels)):
            rows = load_cot_csv(csv_path)
            data = build_plot_data(rows)
            plot_cot(axes[i], data, f"COT vs Frequency — {label}")

        fig.tight_layout()
        out = args.output or "cot_vs_freq_comparison.png"
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")

    else:
        sys.exit("ERROR: provide 1 or 2 CSV files")

    plt.show()


if __name__ == "__main__":
    main()
