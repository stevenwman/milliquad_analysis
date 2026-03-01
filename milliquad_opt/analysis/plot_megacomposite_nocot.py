"""Megacomposite (no COT): 3 terrains × 4 columns (vel exp|sim, pitch exp|sim).

Layout (3×4):
              vel_exp    vel_sim    pitch_exp   pitch_sim
    flat        ✓          ✓          ✓           ✓
    step        ✓          ✓          ✓           ✓
    rough       ✓          ✓        blank         ✓

Rough panels use scatter_only mode (no shading, intra-morphology spreading).
Blank panels = no experimental data available yet; axes hidden.

Usage:
    uv run python -m analysis.plot_megacomposite_nocot \
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
matplotlib.rcParams["font.size"] = 12
import matplotlib.pyplot as plt

# Add experimental_data to path for import
_EXP_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "experimental_data")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from plot_velocity_vs_freq import extract_flat, extract_step_q60, extract_rough  # noqa: E402
from plot_pitch_vs_freq import extract_flat_pitch, extract_step_pitch_q60  # noqa: E402

from analysis._common import detect_terrain  # noqa: E402
from analysis.plot_validation import (  # noqa: E402
    COLORS,
    GATE_END,
    LABELS,
    PITCH_EXCLUDE,
    PLOT_ORDER,
    TERRAIN_TITLES,
    build_all_failed_freqs,
    build_plot_data,
    load_validation_csv,
    plot_panel,
    strip_freqs,
)

_MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}

# Row order
_TERRAIN_ORDER = ["flat", "step", "rough"]

# Experimental extractors (None = no data, leave blank)
_VEL_EXTRACTORS = {"flat": extract_flat, "step": extract_step_q60, "rough": extract_rough}
_PITCH_EXTRACTORS = {"flat": extract_flat_pitch, "step": extract_step_pitch_q60}


def _remap_exp_data(exp_data: dict) -> dict:
    return {_MORPH_TO_SCENE[m]: exp_data[m] for m in exp_data if m in _MORPH_TO_SCENE}


# Failure modes from experimental observations
_EXP_ONLY_FAILURES: dict[str, dict[str, list[float]]] = {
    "flat": {"scene_wheel": [50.0]},
}
_SHARED_FAILURES: dict[str, dict[str, list[float]]] = {
    "step": {"scene_wheel": [10.0, 20.0]},
}


def _get_exp_failures(terrain: str) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for src in (_EXP_ONLY_FAILURES, _SHARED_FAILURES):
        for scene, freqs in src.get(terrain, {}).items():
            merged.setdefault(scene, []).extend(freqs)
    return merged


def _get_sim_failures(terrain: str) -> dict[str, list[float]]:
    return _SHARED_FAILURES.get(terrain, {})


def _inject_na_zeros(data: dict, total_trials: int = 5):
    """Add zero-valued trials for n/a entries in experimental data.

    Assumes each (scene, freq) condition had `total_trials` attempts.
    Missing trials (n/a) are injected as zeros; scatter_only mode plots
    these as X markers at y=0.
    """
    for d in data.values():
        for freq in d["mean_freqs"]:
            n_success = sum(1 for f in d["freqs"] if f == freq)
            n_failed = total_trials - n_success
            if n_failed > 0:
                d["freqs"].extend([freq] * n_failed)
                d["trials"].extend([0.0] * n_failed)


def _strip_failure_freqs(data: dict, failures: dict[str, list[float]]):
    """Remove failure frequencies from data entirely.

    Strips both individual trials and mean/std entries at failure freqs.
    X markers are handled separately by plot_panel via the failures dict.
    """
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

    # Load and bucket by terrain
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
    n_cols = 4  # vel_exp, vel_sim, pitch_exp, pitch_sim
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), squeeze=False)

    for i, terrain in enumerate(present):
        rows, run_dir = terrain_data[terrain]
        title = TERRAIN_TITLES.get(terrain, terrain.title())
        ge = GATE_END.get(terrain)
        exp_failures = _get_exp_failures(terrain)
        sim_failures = _get_sim_failures(terrain)
        pitch_exclude = PITCH_EXCLUDE.get(terrain, [])
        all_failed = build_all_failed_freqs(rows, selected_only=True, gate_end=ge)
        so = terrain.startswith("rough")  # scatter_only for rough

        # Col 0: Exp velocity
        vel_ext = _VEL_EXTRACTORS.get(terrain)
        if vel_ext is not None:
            exp_vel = _remap_exp_data(vel_ext())
            _strip_failure_freqs(exp_vel, exp_failures)
            if so:
                _inject_na_zeros(exp_vel)
            plot_panel(axes[i, 0], exp_vel, f"{title}: Exp Velocity", "Forward Velocity (mm/s)", exp_failures, scatter_only=so)
        else:
            axes[i, 0].set_visible(False)

        # Col 1: Sim velocity
        sim_vel = build_plot_data(rows, "vx", selected_only=True, exclude_invalid=True, gate_end=ge)
        _strip_failure_freqs(sim_vel, sim_failures)
        plot_panel(axes[i, 1], sim_vel, f"{title}: Sim Velocity", "Forward Velocity (mm/s)", sim_failures, all_failed, scatter_only=so)

        # Share y-axis for velocity pair if both exist
        if vel_ext is not None:
            y_lo = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
            y_hi = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
            axes[i, 0].set_ylim(y_lo, y_hi)
            axes[i, 1].set_ylim(y_lo, y_hi)

        # Col 2: Exp pitch
        pitch_ext = _PITCH_EXTRACTORS.get(terrain)
        if pitch_ext is not None:
            exp_pitch = _remap_exp_data(pitch_ext())
            _strip_failure_freqs(exp_pitch, exp_failures)
            if pitch_exclude:
                strip_freqs(exp_pitch, pitch_exclude)
            plot_panel(axes[i, 2], exp_pitch, f"{title}: Exp Pitch", "Pitch Amplitude RMS (\u00b0)", exp_failures, scatter_only=so)
        else:
            axes[i, 2].set_visible(False)

        # Col 3: Sim pitch
        sim_pitch = build_plot_data(rows, "pitch_rms", selected_only=True, exclude_invalid=True, gate_end=ge)
        _strip_failure_freqs(sim_pitch, sim_failures)
        if pitch_exclude:
            strip_freqs(sim_pitch, pitch_exclude)
        plot_panel(axes[i, 3], sim_pitch, f"{title}: Sim Pitch", "Pitch Amplitude RMS (\u00b0)", sim_failures, all_failed=all_failed, scatter_only=so)

        # Share y-axis for pitch pair if both exist
        if pitch_ext is not None:
            y_lo = min(axes[i, 2].get_ylim()[0], axes[i, 3].get_ylim()[0])
            y_hi = max(axes[i, 2].get_ylim()[1], axes[i, 3].get_ylim()[1])
            axes[i, 2].set_ylim(y_lo, y_hi)
            axes[i, 3].set_ylim(y_lo, y_hi)

    # Single legend from first visible axes
    for ax in axes.flat:
        if ax.get_visible():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper left", fontsize=10, framealpha=0.9)
                break
    # Remove duplicate legends
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg and ax is not axes.flat[0]:
            leg.remove()

    fig.tight_layout()
    out = args.output or "plots/megacomposite_nocot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
