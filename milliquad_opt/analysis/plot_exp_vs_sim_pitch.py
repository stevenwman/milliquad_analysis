"""Side-by-side experimental vs simulation pitch amplitude plots for flat and step.

Left column: experimental per-trial data (from raw CSVs via plot_pitch_vs_freq).
Right column: simulation per-trial data (from validation CSVs).
Both: scatter dots + shaded std bands (no mean lines).

Usage:
    uv run python -m analysis.plot_exp_vs_sim_pitch \
        results/20260228T013353_rk4_flat \
        results/20260228T093833_rk4_step_cold
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 14
import matplotlib.pyplot as plt
import numpy as np

# Add experimental_data to path for import
_EXP_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "experimental_data")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from plot_pitch_vs_freq import extract_flat_pitch, extract_step_pitch  # noqa: E402

from analysis._common import detect_terrain  # noqa: E402
from analysis.plot_validation import (  # noqa: E402
    COLORS,
    LABELS,
    PLOT_ORDER,
    TERRAIN_TITLES,
    build_plot_data,
    load_validation_csv,
    plot_panel,
)

# Morph keys (experimental) -> scene keys (simulation)
_MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}


def _remap_exp_data(exp_data: dict) -> dict:
    return {_MORPH_TO_SCENE[m]: exp_data[m] for m in exp_data if m in _MORPH_TO_SCENE}


# Frequencies to exclude from comparison (same as velocity: f50 excluded from flat)
_EXCLUDE_FREQS: dict[str, list[float]] = {
    "flat": [50.0],
}


def _strip_freqs(data: dict, freqs_to_remove: list[float]):
    """Remove specified frequencies from plot data (both scatter and summary)."""
    for scene in data:
        d = data[scene]
        # Strip scatter points
        keep = [j for j, f in enumerate(d["freqs"]) if f not in freqs_to_remove]
        d["freqs"] = [d["freqs"][j] for j in keep]
        d["trials"] = [d["trials"][j] for j in keep]
        # Strip summary stats
        keep_m = [j for j, f in enumerate(d["mean_freqs"]) if f not in freqs_to_remove]
        d["mean_freqs"] = [d["mean_freqs"][j] for j in keep_m]
        d["means"] = [d["means"][j] for j in keep_m]
        d["stds"] = [d["stds"][j] for j in keep_m]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=pathlib.Path,
        help="Run dirs for flat and/or step (auto-detected from name)",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    exp_extractors = {"flat": extract_flat_pitch, "step": extract_step_pitch}

    entries: list[tuple[str, list[dict], pathlib.Path]] = []
    for run_dir in args.run_dirs:
        csv_path = run_dir / "validation_trials.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        terrain = detect_terrain(run_dir)
        if terrain not in exp_extractors:
            print(f"WARNING: {terrain} has no experimental pitch data, skipping")
            continue
        rows = load_validation_csv(csv_path)
        entries.append((terrain, rows, run_dir))

    if not entries:
        sys.exit("No flat/step validation CSVs found")

    n = len(entries)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n), squeeze=False)

    ylabel = "Pitch Amplitude RMS (\u00b0)"

    for i, (terrain, rows, _) in enumerate(entries):
        title = TERRAIN_TITLES.get(terrain, terrain.title())

        exclude = _EXCLUDE_FREQS.get(terrain, [])

        # Left: experimental
        exp_raw = exp_extractors[terrain]()
        exp_data = _remap_exp_data(exp_raw)
        if exclude:
            _strip_freqs(exp_data, exclude)
        plot_panel(axes[i, 0], exp_data, f"{title}: Experimental", ylabel)

        # Right: simulation
        sim_data = build_plot_data(rows, "pitch_rms", selected_only=True)
        if exclude:
            _strip_freqs(sim_data, exclude)
        plot_panel(axes[i, 1], sim_data, f"{title}: Simulation", ylabel)

        # Share y-axis range per row
        y_lo = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
        y_hi = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        axes[i, 0].set_ylim(y_lo, y_hi)
        axes[i, 1].set_ylim(y_lo, y_hi)

    # Single legend on top-left
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="upper left", fontsize=12, framealpha=0.9)
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg and ax is not axes[0, 0]:
            leg.remove()

    fig.tight_layout()
    out = args.output or "plots/exp_vs_sim_pitch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
