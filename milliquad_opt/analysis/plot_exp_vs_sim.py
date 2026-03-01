"""Side-by-side experimental vs simulation velocity plots for flat and step.

Left column: experimental per-trial data (extracted from raw CSVs).
Right column: simulation per-trial data (from validation CSVs).
Both: all trials as scatter, shaded std bands, mean line.

Usage:
    uv run python -m analysis.plot_exp_vs_sim \
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

from plot_velocity_vs_freq import extract_flat, extract_step  # noqa: E402

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

# Morph keys (experimental) → scene keys (simulation)
_MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}


def _remap_exp_data(exp_data: dict) -> dict:
    """Convert morph-keyed experimental data to scene-keyed format."""
    return {_MORPH_TO_SCENE[m]: exp_data[m] for m in exp_data if m in _MORPH_TO_SCENE}


# Known failure modes
# exp_only: robot physically breaks (sim doesn't reproduce)
# shared: robot can't move (sim also reproduces near-zero velocity)
_EXP_ONLY_FAILURES: dict[str, dict[str, list[float]]] = {
    "flat": {"scene_wheel": [50.0]},
}
_SHARED_FAILURES: dict[str, dict[str, list[float]]] = {
    "step": {"scene_wheel": [10.0, 20.0]},
}


def _get_exp_failures(terrain: str) -> dict[str, list[float]]:
    """All failure freqs for experimental panel (exp-only + shared)."""
    merged: dict[str, list[float]] = {}
    for src in (_EXP_ONLY_FAILURES, _SHARED_FAILURES):
        for scene, freqs in src.get(terrain, {}).items():
            merged.setdefault(scene, []).extend(freqs)
    return merged


def _get_sim_failures(terrain: str) -> dict[str, list[float]]:
    """Failure freqs for simulation panel (shared only)."""
    return _SHARED_FAILURES.get(terrain, {})


def _inject_failure_zeros(data: dict, failures: dict[str, list[float]]):
    """Set mean/std=0 at failure freqs, strip scatter, sort by freq."""
    for scene, fail_freqs in failures.items():
        if scene not in data:
            continue
        d = data[scene]
        for ff in fail_freqs:
            # Remove scatter points
            keep = [j for j, f in enumerate(d["freqs"]) if f != ff]
            d["freqs"] = [d["freqs"][j] for j in keep]
            d["trials"] = [d["trials"][j] for j in keep]
            # Set mean/std to zero (or inject if missing)
            if ff in d["mean_freqs"]:
                idx = d["mean_freqs"].index(ff)
                d["means"][idx] = 0.0
                d["stds"][idx] = 0.0
            else:
                d["mean_freqs"].append(ff)
                d["means"].append(0.0)
                d["stds"].append(0.0)
        # Sort by frequency so fill_between draws correctly
        order = sorted(range(len(d["mean_freqs"])), key=lambda k: d["mean_freqs"][k])
        d["mean_freqs"] = [d["mean_freqs"][k] for k in order]
        d["means"] = [d["means"][k] for k in order]
        d["stds"] = [d["stds"][k] for k in order]


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

    # Experimental extractors by terrain
    exp_extractors = {"flat": extract_flat, "step": extract_step}

    entries: list[tuple[str, list[dict], pathlib.Path]] = []
    for run_dir in args.run_dirs:
        csv_path = run_dir / "validation_trials.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        terrain = detect_terrain(run_dir)
        if terrain not in exp_extractors:
            print(f"WARNING: {terrain} has no experimental data, skipping")
            continue
        rows = load_validation_csv(csv_path)
        entries.append((terrain, rows, run_dir))

    if not entries:
        sys.exit("No flat/step validation CSVs found")

    n = len(entries)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n), squeeze=False)

    for i, (terrain, rows, _) in enumerate(entries):
        title = TERRAIN_TITLES.get(terrain, terrain.title())
        exp_failures = _get_exp_failures(terrain)
        sim_failures = _get_sim_failures(terrain)

        # Left: experimental
        exp_raw = exp_extractors[terrain]()
        exp_data = _remap_exp_data(exp_raw)

        # At failure freqs: strip scatter, set mean/std=0 so shading tapers to zero
        _inject_failure_zeros(exp_data, exp_failures)

        plot_panel(axes[i, 0], exp_data, f"{title}: Experimental", "Forward Velocity (mm/s)", exp_failures)

        # Right: simulation — inject shared failures as zeros, show X markers
        sim_data = build_plot_data(rows, "vx", selected_only=True)
        _inject_failure_zeros(sim_data, sim_failures)

        plot_panel(axes[i, 1], sim_data, f"{title}: Simulation", "Forward Velocity (mm/s)", sim_failures)

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
    out = args.output or "plots/exp_vs_sim_velocity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
