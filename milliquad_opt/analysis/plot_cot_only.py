"""COT-only column figure: 3 terrains × 1 column (sim COT).

Layout (3×1):
    flat    COT
    step    COT
    rough   COT

Rough panels use scatter_only mode (no shading, intra-morphology spreading).

Usage:
    uv run python -m analysis.plot_cot_only \
        results/20260228T013353_rk4_flat \
        results/20260228T230022_step_q60_rk-warm \
        results/20260228T202903_rough_spatial_rk4
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 6
import matplotlib.pyplot as plt

from analysis._common import detect_terrain
from analysis.plot_validation import (
    GATE_END,
    build_all_failed_freqs,
    build_plot_data,
    load_validation_csv,
    plot_panel,
)

_TERRAIN_ORDER = ["flat", "step", "rough"]
_ROW_LABELS = {"flat": "Flat", "step": "Step", "rough": "Rough"}

_SHARED_FAILURES: dict[str, dict[str, list[float]]] = {
    "step": {"scene_wheel": [10.0, 20.0]},
}


def _strip_failure_freqs(data: dict, failures: dict[str, list[float]]):
    for scene, fail_freqs in failures.items():
        if scene not in data:
            continue
        d = data[scene]
        for ff in fail_freqs:
            keep = [j for j, f in enumerate(d["freqs"]) if f != ff]
            d["freqs"] = [d["freqs"][j] for j in keep]
            d["trials"] = [d["trials"][j] for j in keep]
            if ff in d["mean_freqs"]:
                idx = d["mean_freqs"].index(ff)
                d["mean_freqs"].pop(idx)
                d["means"].pop(idx)
                d["stds"].pop(idx)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=pathlib.Path,
        help="Run dirs for flat, step, and/or rough (auto-detected from name)",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    terrain_data: dict[str, tuple[list[dict], pathlib.Path]] = {}
    for run_dir in args.run_dirs:
        csv_path = run_dir / "validation_trials.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        terrain = detect_terrain(run_dir)
        rows = load_validation_csv(csv_path)
        terrain_data[terrain] = (rows, run_dir)

    present = [t for t in _TERRAIN_ORDER if t in terrain_data]
    if not present:
        sys.exit("No validation CSVs found")

    n_rows = len(present)
    fig, axes = plt.subplots(n_rows, 1, figsize=(3.5, 2.0 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for i, terrain in enumerate(present):
        rows, run_dir = terrain_data[terrain]
        ge = GATE_END.get(terrain)
        sim_failures = _SHARED_FAILURES.get(terrain, {})
        all_failed = build_all_failed_freqs(rows, selected_only=True, gate_end=ge)
        so = terrain.startswith("rough")

        cot_data = build_plot_data(rows, "cot", selected_only=True,
                                   exclude_invalid=True, gate_end=ge)
        _strip_failure_freqs(cot_data, sim_failures)
        plot_panel(axes[i], cot_data, "", "", sim_failures,
                   all_failed=all_failed, scatter_only=so)

    # --- Post-hoc axis cleanup ---
    from matplotlib.collections import PathCollection
    for i in range(n_rows):
        ax = axes[i]
        terrain = present[i]
        row_label = _ROW_LABELS.get(terrain, terrain.title())

        # Shrink dots and X markers
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                coll.set_sizes([12])
        for line in ax.lines:
            if line.get_marker() == 'x':
                line.set_markersize(6)
                line.set_markeredgewidth(1.5)

        # Subplot lettering
        letter = chr(ord('a') + i)
        ax.text(0.02, 0.95, f"({letter})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")

        # Terrain label on left, y-ticks on left
        ax.set_ylabel(row_label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)

        # Remove xlabel except bottom row
        if i < n_rows - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Drive frequency (Hz)", fontsize=9)

    # Title — anchored just above top panel
    axes[0].set_title("Cost of Transport", fontsize=10, fontweight="bold", pad=8)

    # Legend below bottom panel, centered on plot area
    handles, labels = None, None
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    if handles:
        # Center legend on the plot area (use axes bbox midpoint)
        plot_left = axes[0].get_position().x0
        plot_right = axes[0].get_position().x1
        plot_center = (plot_left + plot_right) / 2
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   fontsize=7, framealpha=0.9,
                   bbox_to_anchor=(plot_center, 0.02))

    out = args.output or "plots/cot_only.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
