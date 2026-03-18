"""
Summary plot for friction sensitivity analysis.

3 columns (sliding, torsional, rolling) × 2 rows (success rate, mean velocity).
Each panel has 3 lines (f10, f30, f50). Vertical dashed line at baseline.

Usage:
    uv run python plot_friction_summary.py results/XXXXXXXX_friction_sensitivity
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Baseline values (from rough_tg optimization_bests.csv final row)
BASELINE = {
    "sliding_friction": 0.640039,
    "torsional_friction": 2.35e-05,
    "rolling_friction": 1.48e-06,
}

PARAM_ORDER = ["sliding_friction", "torsional_friction", "rolling_friction"]
PARAM_LABELS = ["Sliding friction", "Torsional friction", "Rolling friction"]

FREQ_COLORS = {10.0: "#1E88E5", 30.0: "#FFC107", 50.0: "#D81B60"}
FREQ_MARKERS = {10.0: "o", 30.0: "s", 50.0: "^"}


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python plot_friction_summary.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    summary_csv = results_dir / "friction_sweep_summary.csv"

    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))

    # Group: param_name -> freq -> sorted list of (value, success_rate, mean_vx, std_vx)
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pname = r["param_name"]
        freq = float(r["ctrl_freq"])
        data[pname][freq].append({
            "value": float(r["param_value"]),
            "success_rate": float(r["success_rate"]),
            "mean_vx": float(r["mean_vx_mm_s"]),
            "std_vx": float(r["std_vx_mm_s"]),
            "n_success": int(r["n_success"]),
            "n_trials": int(r["n_trials"]),
        })

    for pname in data:
        for freq in data[pname]:
            data[pname][freq].sort(key=lambda d: d["value"])

    # --- Plot ---
    plt.rcParams.update({
        "font.family": "TeX Gyre Pagella",
        "font.size": 9,
    })

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 5.5),
        constrained_layout=True,
    )

    for ci, (pname, plabel) in enumerate(zip(PARAM_ORDER, PARAM_LABELS)):
        ax_sr = axes[0, ci]  # success rate
        ax_vx = axes[1, ci]  # velocity

        for freq in sorted(FREQ_COLORS.keys()):
            if freq not in data[pname]:
                continue
            entries = data[pname][freq]
            vals = [e["value"] for e in entries]
            sr = [e["success_rate"] * 100 for e in entries]
            vx = [e["mean_vx"] for e in entries]
            vx_std = [e["std_vx"] for e in entries]

            color = FREQ_COLORS[freq]
            marker = FREQ_MARKERS[freq]
            label = f"{int(freq)} Hz"

            ax_sr.plot(vals, sr, color=color, marker=marker, ms=5, lw=1.5, label=label)
            ax_vx.plot(vals, vx, color=color, marker=marker, ms=5, lw=1.5, label=label)
            # Shade std (only where there are successes)
            vx_arr = np.array(vx)
            std_arr = np.array(vx_std)
            mask = vx_arr > 0
            if mask.any():
                v_masked = np.array(vals)[mask]
                ax_vx.fill_between(
                    v_masked,
                    vx_arr[mask] - std_arr[mask],
                    vx_arr[mask] + std_arr[mask],
                    color=color, alpha=0.15,
                )

        # Baseline vertical line
        bl = BASELINE[pname]
        ax_sr.axvline(bl, color="k", ls="--", lw=1.0, alpha=0.6, zorder=0,
                      label="Baseline" if ci == 0 else None)
        ax_vx.axvline(bl, color="k", ls="--", lw=1.0, alpha=0.6, zorder=0)

        ax_sr.set_xscale("log")
        ax_vx.set_xscale("log")

        ax_sr.set_ylim(-5, 105)
        ax_sr.set_ylabel("Success rate (%)" if ci == 0 else "")
        ax_vx.set_ylabel("Mean velocity (mm/s)" if ci == 0 else "")
        ax_vx.set_xlabel(plabel)

        ax_sr.set_title(plabel, fontsize=11, fontweight="bold")

        if ci == 0:
            ax_sr.legend(fontsize=8, loc="lower left")

        ax_sr.grid(axis="y", alpha=0.3)
        ax_vx.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Friction sensitivity — L2 (scene2) on rough terrain",
        fontsize=13, fontweight="bold",
    )

    out_path = results_dir / "friction_sensitivity_summary.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
