"""
Clean single-frequency (30 Hz) friction sensitivity figure for presentation.

Rough terrain: 2 rows (success rate + velocity).
Flat terrain: 1 row (velocity only — success is always 100%).

Reads baseline from friction_sweep.csv to auto-detect the optimization run's params.

Usage:
    uv run python plot_friction_summary_30hz.py results/XXXXXXXX_friction_sensitivity_local
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARAM_ORDER = ["sliding_friction", "torsional_friction", "rolling_friction"]
PARAM_LABELS = ["Sliding friction", "Torsional friction", "Rolling friction"]

COLOR = "#2E7D32"
COLOR_LIGHT = "#A5D6A7"
FREQ = 30.0


_BASELINES = {
    "rough": {
        "sliding_friction": 0.640039,
        "torsional_friction": 2.35e-05,
        "rolling_friction": 1.48e-06,
    },
    "flat": {
        "sliding_friction": 0.492594,
        "torsional_friction": 6.11e-04,
        "rolling_friction": 1.11e-06,
    },
}


def _get_baseline(terrain: str) -> dict[str, float]:
    """Return optimized baseline values for the given terrain."""
    return _BASELINES[terrain]


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python plot_friction_summary_30hz.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    summary_csv = results_dir / "friction_sweep_summary.csv"

    # Detect terrain
    dirname = results_dir.name
    is_flat = "flat" in dirname
    terrain_label = "flat terrain" if is_flat else "rough terrain"

    terrain_key = "flat" if is_flat else "rough"
    baseline = _get_baseline(terrain_key)

    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))

    data = defaultdict(list)
    for r in rows:
        if float(r["ctrl_freq"]) != FREQ:
            continue
        data[r["param_name"]].append({
            "value": float(r["param_value"]),
            "success_rate": float(r["success_rate"]),
            "mean_vx": float(r["mean_vx_mm_s"]),
            "std_vx": float(r["std_vx_mm_s"]),
        })
    for pname in data:
        data[pname].sort(key=lambda d: d["value"])

    plt.rcParams.update({
        "font.family": "TeX Gyre Pagella",
        "font.size": 10,
    })

    nrows = 1 if is_flat else 2
    fig, axes = plt.subplots(
        nrows, 3,
        figsize=(11, 2.8 if is_flat else 4.5),
        constrained_layout=True,
        squeeze=False,
    )

    for ci, (pname, plabel) in enumerate(zip(PARAM_ORDER, PARAM_LABELS)):
        entries = data[pname]
        vals = [e["value"] for e in entries]
        vx = [e["mean_vx"] for e in entries]
        vx_std = [e["std_vx"] for e in entries]
        bl = baseline.get(pname, 0)

        if not is_flat:
            # Success rate row
            ax_sr = axes[0, ci]
            sr = [e["success_rate"] * 100 for e in entries]
            ax_sr.plot(vals, sr, color=COLOR, marker="o", ms=6, lw=2, zorder=3)
            ax_sr.axvline(bl, color="k", ls="--", lw=1.2, alpha=0.5, zorder=2,
                          label="Optimized" if ci == 0 else None)
            ax_sr.set_xscale("log")
            ax_sr.set_ylim(-5, 105)
            ax_sr.set_title(plabel, fontsize=12, fontweight="bold")
            if ci == 0:
                ax_sr.set_ylabel("Success rate (%)")
                ax_sr.legend(fontsize=9, loc="lower left")
            ax_sr.grid(axis="y", alpha=0.25, lw=0.5)

        # Velocity row
        ax_vx = axes[-1, ci]
        vx_arr = np.array(vx)
        std_arr = np.array(vx_std)
        ax_vx.plot(vals, vx, color=COLOR, marker="o", ms=6, lw=2, zorder=3)
        mask = vx_arr > 0
        if mask.any():
            v_masked = np.array(vals)[mask]
            ax_vx.fill_between(
                v_masked,
                vx_arr[mask] - std_arr[mask],
                vx_arr[mask] + std_arr[mask],
                color=COLOR_LIGHT, alpha=0.4, zorder=1,
            )
        ax_vx.axvline(bl, color="k", ls="--", lw=1.2, alpha=0.5, zorder=2,
                      label="Optimized" if ci == 0 and is_flat else None)
        ax_vx.set_xscale("log")
        ax_vx.set_ylim(bottom=0)
        ax_vx.set_xlabel(plabel)
        if ci == 0:
            ax_vx.set_ylabel("Mean velocity (mm/s)")
            if is_flat:
                ax_vx.legend(fontsize=9, loc="lower left")
        ax_vx.grid(axis="y", alpha=0.25, lw=0.5)

        if is_flat:
            ax_vx.set_title(plabel, fontsize=12, fontweight="bold")

    fig.suptitle(
        f"Friction sensitivity — L2 on {terrain_label} (30 Hz)",
        fontsize=14, fontweight="bold",
    )

    out_path = results_dir / "friction_sensitivity_30hz.png"
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
