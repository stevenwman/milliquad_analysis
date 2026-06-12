"""COT-only column figure (65% gate): 3 terrains x 1 column (sim COT).

Copy of plot_cot_only.py -- identical except step terrain COT
is recomputed at 65% spatial gate (83.5mm) from NPZ data.
Flat/rough unchanged.

Usage:
    uv run python -m analysis.plot_cot_only_065 \
        results/20260228T013353_rk4_flat \
        results/20260228T230022_step_q60_rk-warm \
        results/20260228T202903_rough_spatial_rk4
"""

from __future__ import annotations

import argparse
import csv as _csv
import pathlib
import sys

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 6
import matplotlib.pyplot as plt
import numpy as np

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

# Step terrain: recompute COT at 65% spatial gate from NPZ
_STEP_START_X = 0.05
_STEP_END_X = 0.1015
_CUTOFF_065 = _STEP_START_X + 0.65 * (_STEP_END_X - _STEP_START_X)


def _recompute_step_cot_065(rows: list[dict], run_dir: pathlib.Path):
    """Override COT values using 65% spatial gate from NPZ data."""
    npz_files = sorted(run_dir.glob("*_validation_trajectories.npz"))
    if not npz_files:
        print("WARNING: no NPZ in run dir, keeping 90% COT values")
        return
    d = np.load(str(npz_files[-1]))

    # Check omega is present
    if not any(k.endswith("_omega") for k in d.files):
        print("WARNING: NPZ has no omega arrays (re-run validation), keeping 90% COT")
        return

    # Build trial -> scene mapping from CSV
    csv_path = run_dir / "validation_trials.csv"
    if not csv_path.exists():
        # Try timestamped
        csv_files = sorted(run_dir.glob("*_validation_trials.csv"))
        if csv_files:
            csv_path = csv_files[-1]
        else:
            print("WARNING: no validation CSV found, keeping 90% COT values")
            return

    trial_map: list[str] = []
    scene_map: list[str] = []
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            trial_map.append(f"{r['ref_id']}_t{r['trial']}")
            scene_map.append(r["scene"])

    # Mass cache
    import mujoco
    from config import MJCF_PATHS
    mass_cache: dict[str, float] = {}

    recomputed = 0
    for i, row in enumerate(rows):
        if i >= len(trial_map):
            break
        tkey = trial_map[i]
        scene = scene_map[i]

        try:
            tau_ext = d[f"{tkey}_tau_ext"]   # (T, 4, 3)
            omega = d[f"{tkey}_omega"]       # (T, 4, 3)
            pos_x = d[f"{tkey}_pos_x"]       # (T,)
            pos_y = d[f"{tkey}_pos_y"]       # (T,)
            time_arr = d[f"{tkey}_time"]     # (T,)
        except KeyError:
            continue

        enter_idx = int(np.searchsorted(pos_x, _STEP_START_X))
        gate_indices = np.where(pos_x >= _CUTOFF_065)[0]
        if len(gate_indices) == 0 or gate_indices[0] <= enter_idx + 1:
            row["cot"] = None  # can't measure at 65% — invalidate stale 90% value
            continue
        gate_idx = int(gate_indices[0])

        # Mass
        if scene not in mass_cache:
            mjcf = MJCF_PATHS.get(scene)
            if mjcf is None:
                continue
            model = mujoco.MjModel.from_xml_path(mjcf)
            mass_cache[scene] = float(np.sum(model.body_mass))
        mass = mass_cache[scene]

        # Power integration over 65% gate
        t_active = tau_ext[enter_idx:gate_idx + 1]   # (N, 4, 3)
        o_active = omega[enter_idx:gate_idx + 1]     # (N, 4, 3)
        time_active = time_arr[enter_idx:gate_idx + 1]

        n = len(t_active) - 1
        if n < 1:
            continue

        power = np.sum(t_active[:n] * o_active[:n], axis=(1, 2))  # (n,)
        dt = np.diff(time_active)  # (n,)
        energy = float(np.sum(power * dt))

        # 2D distance
        dx = float(pos_x[gate_idx] - pos_x[enter_idx])
        dy = float(pos_y[gate_idx] - pos_y[enter_idx])
        distance = np.sqrt(dx**2 + dy**2)

        mgd = mass * 9.81 * distance
        if mgd < 1e-12:
            continue

        row["cot"] = float(energy / mgd)
        recomputed += 1

    print(f"  Recomputed COT at 65% gate for {recomputed}/{len(rows)} step trials")


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
        if terrain == "step":
            _recompute_step_cot_065(rows, run_dir)
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

        # Uniform padding below 0 (5% of top) so 0-tick + X markers are visible
        top = ax.get_ylim()[1]
        ax.set_ylim(bottom=-0.05 * top)

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
        ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, labelsize=10)
        ax.tick_params(axis="x", which="both", labelsize=10)

        # Remove xlabel except bottom row
        if i < n_rows - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Drive frequency (Hz)", fontsize=9)

    # Title
    axes[0].set_title("Cost of Transport (65% gate)", fontsize=10, fontweight="bold", pad=8)

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
        plot_left = axes[0].get_position().x0
        plot_right = axes[0].get_position().x1
        plot_center = (plot_left + plot_right) / 2
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   fontsize=7, framealpha=0.9,
                   bbox_to_anchor=(plot_center, 0.04))

    out = args.output or "plots/cot_only_065.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
