"""Plot velocity and COT vs frequency from validation trial CSVs.

Produces publication-quality figures matching the style of:
  - experimental_data/plots/velocity_vs_freq_flat_clean.png
  - mujoco_refactor/results/cot_flat_vs_step.png

Usage:
    uv run python -m analysis.plot_validation results/20260228T013353_rk4_flat
    uv run python -m analysis.plot_validation \
        results/20260228T013353_rk4_flat \
        results/20260228T093833_rk4_step_cold \
        results/20260228T102010_rk4_rough
"""

from __future__ import annotations

import argparse
import csv
import importlib
import pathlib
import sys

import matplotlib
matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 14
import matplotlib.pyplot as plt
import numpy as np

from analysis._common import detect_terrain

# Style constants (matching mujoco_refactor/morphology_style.py)
COLORS = {
    "scene1": "#1E88E5",
    "scene2": "#FFC107",
    "scene4": "#007561",
    "scene_wheel": "#D81B60",
}
LABELS = {"scene1": "L1", "scene2": "L2", "scene4": "L4", "scene_wheel": "WR"}
PLOT_ORDER = ["scene1", "scene2", "scene4", "scene_wheel"]
TERRAIN_TITLES = {"flat": "Flat Terrain", "step": "Step Terrain", "rough": "Rough Terrain"}


def load_validation_csv(csv_path: pathlib.Path) -> list[dict]:
    """Load validation_trials.csv and return parsed rows."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "ref_id": row["ref_id"],
                "scene": row["scene"],
                "freq": float(row["ctrl_freq"]),
                "target": float(row["target_speed"]) if row["target_speed"] else None,
                "vx": float(row["vx"]) if row["vx"] else None,
                "cot": float(row["cot"]) if row["cot"] else None,
                "crash": row["crash"] == "True",
                "selected": row["selected"] == "True",
                "min_window_vx": float(row["min_window_vx"]) if row.get("min_window_vx") else 0.0,
                "pitch_rms": float(row["pitch_rms"]) if row.get("pitch_rms") else None,
                "stalled": row.get("stalled", "False") == "True",
            })
    return rows


def build_plot_data(rows: list[dict], metric: str,
                    min_vx: float | None = None,
                    selected_only: bool = False,
                    exclude_stalled: bool = False) -> dict:
    """Group trials by scene.

    Returns {scene: {freqs, trials, mean_freqs, means, stds}}.
    All trials shown as scatter; shading = std (matching experimental plots).
    metric: "vx" (converted to mm/s), "cot", or "pitch_rms" (degrees).
    min_vx: if set, exclude trials with vx below this (m/s).
    selected_only: if True, use only selected=True trials (top 3 by vel error).
    exclude_stalled: if True, exclude trials flagged as stalled (5-period window).
    """
    valid = [r for r in rows if not r["crash"]]
    if selected_only:
        valid = [r for r in valid if r.get("selected", False)]
    if exclude_stalled:
        valid = [r for r in valid if not r.get("stalled", False)]
    if min_vx is not None:
        valid = [r for r in valid if r["vx"] is not None and abs(r["vx"]) >= min_vx]

    data = {}
    for scene in PLOT_ORDER:
        scene_rows = [r for r in valid if r["scene"] == scene]
        freqs = sorted(set(r["freq"] for r in scene_rows))

        trial_freqs: list[float] = []
        trial_vals: list[float] = []
        mean_freqs: list[float] = []
        means: list[float] = []
        stds: list[float] = []

        for freq in freqs:
            freq_rows = [r for r in scene_rows if r["freq"] == freq]
            if metric == "vx":
                vals = [r["vx"] * 1000 for r in freq_rows if r["vx"] is not None]
            elif metric == "pitch_rms":
                vals = [r["pitch_rms"] for r in freq_rows if r["pitch_rms"] is not None]
            else:
                vals = [r["cot"] for r in freq_rows if r["cot"] is not None]

            if not vals:
                continue

            trial_freqs.extend([freq] * len(vals))
            trial_vals.extend(vals)
            mean_freqs.append(freq)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

        data[scene] = {
            "freqs": trial_freqs,
            "trials": trial_vals,
            "mean_freqs": mean_freqs,
            "means": means,
            "stds": stds,
        }
    return data


def get_failure_modes(terrain: str) -> dict[str, list[float]]:
    """Get failure mode refs (target=0) from terrain config."""
    config_mod = importlib.import_module(f"config_{terrain}")
    failures: dict[str, list[float]] = {}
    for r in config_mod.REFERENCE_DATA:
        if r["speed"] < 1e-9:
            failures.setdefault(r["scene"], []).append(r["ctrl_freq"])
    return failures


def plot_panel(
    ax,
    data: dict,
    title: str,
    ylabel: str,
    failures: dict[str, list[float]] | None = None,
):
    """Plot one terrain panel with shaded std bands and scatter dots."""
    n = len(PLOT_ORDER)
    dodge_width = 1.2  # total spread in Hz
    for idx, scene in enumerate(PLOT_ORDER):
        d = data[scene]
        if not d["mean_freqs"]:
            continue
        dx = (idx - (n - 1) / 2) * (dodge_width / (n - 1))
        freq_arr = np.array(d["mean_freqs"]) + dx
        mean = np.array(d["means"])
        std = np.array(d["stds"])

        ax.fill_between(
            freq_arr, mean - std, mean + std,
            color=COLORS[scene], alpha=0.2, label=LABELS[scene],
        )
        scatter_freqs = np.array(d["freqs"]) + dx
        ax.scatter(scatter_freqs, d["trials"], color=COLORS[scene], alpha=0.6, s=30, zorder=3)

    if failures:
        for scene, freqs in failures.items():
            if scene in COLORS:
                for freq in freqs:
                    ax.plot(
                        freq, 0, "x", color=COLORS[scene],
                        markersize=10, markeredgewidth=2.5, zorder=5,
                    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    all_freqs = set(f for d in data.values() for f in d["mean_freqs"])
    if failures:
        for freqs in failures.values():
            all_freqs.update(freqs)
    all_freqs_sorted = sorted(all_freqs)
    if all_freqs_sorted:
        ax.set_xticks([int(f) for f in all_freqs_sorted])
        ax.set_xlim(all_freqs_sorted[0] - 3, all_freqs_sorted[-1] + 3)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dirs", nargs="+", type=pathlib.Path,
        help="One or more run directories with validation_trials.csv",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=None,
        help="Output directory (default: first run_dir)",
    )
    parser.add_argument("--no-show", action="store_true", help="Don't call plt.show()")
    args = parser.parse_args()

    entries: list[tuple[str, list[dict], pathlib.Path]] = []
    for run_dir in args.run_dirs:
        csv_path = run_dir / "validation_trials.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        terrain = detect_terrain(run_dir)
        rows = load_validation_csv(csv_path)
        entries.append((terrain, rows, run_dir))

    if not entries:
        sys.exit("No validation CSVs found")

    output_dir = args.output_dir or pathlib.Path(".")

    # Generate separate figures per terrain (each run_dir has different params)
    for terrain, rows, run_dir in entries:
        title = TERRAIN_TITLES.get(terrain, terrain.replace("_", " ").title())
        failures = get_failure_modes(terrain)

        # Velocity
        fig_v, ax_v = plt.subplots(figsize=(7, 5))
        vx_data = build_plot_data(rows, "vx", selected_only=True)
        plot_panel(ax_v, vx_data, title, "Forward Velocity (mm/s)", failures)
        ax_v.legend(loc="upper left", fontsize=12, framealpha=0.9)
        fig_v.tight_layout()
        vel_path = output_dir / f"velocity_vs_freq_{terrain}.png"
        fig_v.savefig(vel_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {vel_path}")
        plt.close(fig_v)

        # COT — filter out stalled trials (5-period sustained velocity check)
        fig_c, ax_c = plt.subplots(figsize=(7, 5))
        cot_data = build_plot_data(rows, "cot", selected_only=True, exclude_stalled=True)
        plot_panel(ax_c, cot_data, title, "Cost of Transport")
        ax_c.legend(loc="upper left", fontsize=12, framealpha=0.9)
        fig_c.tight_layout()
        cot_path = output_dir / f"cot_vs_freq_{terrain}.png"
        fig_c.savefig(cot_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {cot_path}")
        plt.close(fig_c)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
