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
from matplotlib.ticker import MaxNLocator
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
                "target": float(row.get("target_speed", "")) if row.get("target_speed") else None,
                "vx": float(row["vx"]) if row.get("vx") else None,
                "cot": float(row["cot"]) if row.get("cot") else None,
                "crash": row.get("crash", "False") == "True",
                "selected": row.get("selected", "True") == "True",
                "min_window_vx": float(row["min_window_vx"]) if row.get("min_window_vx") else 0.0,
                "max_x": float(row["max_x"]) if row.get("max_x") else None,
                "pitch_rms": float(row["pitch_rms"]) if row.get("pitch_rms") else None,
                "stalled": row.get("stalled", "False") == "True",
            })
    return rows


def _is_valid_trial(r: dict, gate_end: float | None = None,
                    gate_exempt: frozenset[tuple[str, float]] | None = None) -> bool:
    """A trial is valid if it cleared the gate (rough/step) AND isn't inverted.

    gate_end: required x position for rough/step terrain (None for flat = always passes).
    gate_exempt: (scene, freq) pairs exempt from gate check. Falls back to module-level
        GATE_EXEMPT if not provided.
    """
    exempt = gate_exempt if gate_exempt is not None else GATE_EXEMPT
    if gate_end is not None and r["max_x"] is not None:
        if (r["scene"], r["freq"]) not in exempt and r["max_x"] < gate_end:
            return False
    inverted = r["pitch_rms"] is not None and r["pitch_rms"] > INVERTED_PITCH_THRESHOLD
    return not inverted


def build_plot_data(rows: list[dict], metric: str,
                    min_vx: float | None = None,
                    selected_only: bool = False,
                    exclude_stalled: bool = False,
                    exclude_invalid: bool = False,
                    gate_end: float | None = None,
                    gate_exempt: frozenset[tuple[str, float]] | None = None) -> dict:
    """Group trials by scene.

    Returns {scene: {freqs, trials, mean_freqs, means, stds}}.
    All trials shown as scatter; shading = std (matching experimental plots).
    metric: "vx" (converted to mm/s), "cot", or "pitch_rms" (degrees).
    min_vx: if set, exclude trials with vx below this (m/s).
    selected_only: if True, use only selected=True trials (top 3 by vel error).
    exclude_stalled: if True, exclude trials flagged as stalled (5-period window).
    exclude_invalid: if True, exclude trials that didn't clear gate or are inverted.
    gate_end: x position threshold for gate-clearing (rough/step). None for flat.
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
            
            freq_vals_valid = []
            
            for r in freq_rows:
                is_valid = _is_valid_trial(r, gate_end, gate_exempt)
                if exclude_invalid and not is_valid:
                    # Plot as failure (X marker at 0.0) but don't include in means
                    trial_freqs.append(freq)
                    trial_vals.append(0.0)
                    continue

                if metric == "vx":
                    v = r["vx"] * 1000 if r["vx"] is not None else None
                elif metric == "pitch_rms":
                    v = r["pitch_rms"]
                else:
                    v = r["cot"]

                if v is not None:
                    trial_freqs.append(freq)
                    trial_vals.append(v)
                    freq_vals_valid.append(v)

            if not freq_vals_valid and trial_freqs and trial_freqs[-1] == freq:
                # All trials failed, still need mean_freqs for x-axis ticking
                mean_freqs.append(freq)
                means.append(0.0)
                stds.append(0.0)
            elif freq_vals_valid:
                mean_freqs.append(freq)
                means.append(float(np.mean(freq_vals_valid)))
                stds.append(float(np.std(freq_vals_valid, ddof=1)) if len(freq_vals_valid) > 1 else 0.0)

        data[scene] = {
            "freqs": trial_freqs,
            "trials": trial_vals,
            "mean_freqs": mean_freqs,
            "means": means,
            "stds": stds,
        }
    return data


INVERTED_PITCH_THRESHOLD = 30.0  # degrees — pitch_rms above this = inverted
GATE_END = {"rough": 0.155, "step": 0.1015}  # m — robot must reach this x to count as valid

# scene1_f10 on rough: robot traverses terrain successfully but moves too slowly (1-leg @ 10Hz)
# to reach ROUGH_END_X. Exempt from gate check per visual inspection.
GATE_EXEMPT = {("scene1", 10.0)}

# Frequencies to exclude from pitch plots. Empty = show all.
# Inverted trials (pitch_rms > 30°) are already excluded by exclude_invalid.
PITCH_EXCLUDE: dict[str, list[float]] = {}


def strip_freqs(data: dict, freqs_to_remove: list[float]):
    """Remove specific frequencies from plot data (in-place)."""
    for scene in data:
        d = data[scene]
        keep = [j for j, f in enumerate(d["freqs"]) if f not in freqs_to_remove]
        d["freqs"] = [d["freqs"][j] for j in keep]
        d["trials"] = [d["trials"][j] for j in keep]
        keep_m = [j for j, f in enumerate(d["mean_freqs"]) if f not in freqs_to_remove]
        d["mean_freqs"] = [d["mean_freqs"][j] for j in keep_m]
        d["means"] = [d["means"][j] for j in keep_m]
        d["stds"] = [d["stds"][j] for j in keep_m]


def build_all_failed_freqs(rows: list[dict],
                           selected_only: bool = False,
                           gate_end: float | None = None,
                           gate_exempt: frozenset[tuple[str, float]] | None = None) -> dict[str, dict[float, int]]:
    """Find (scene, freq) combos where ALL trials are invalid (didn't clear gate or inverted).

    Returns {scene: {freq: count}} for use as X markers with counts on plots.
    """
    valid = [r for r in rows if not r["crash"]]
    if selected_only:
        valid = [r for r in valid if r.get("selected", False)]

    failed: dict[str, dict[float, int]] = {}
    for scene in PLOT_ORDER:
        scene_rows = [r for r in valid if r["scene"] == scene]
        freqs = sorted(set(r["freq"] for r in scene_rows))
        for freq in freqs:
            freq_rows = [r for r in scene_rows if r["freq"] == freq]
            if freq_rows and all(not _is_valid_trial(r, gate_end, gate_exempt) for r in freq_rows):
                failed.setdefault(scene, {})[freq] = len(freq_rows)
    return failed


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
    all_failed: dict[str, dict[float, int]] | None = None,
    scatter_only: bool = False,
    intra_spread: float | None = None,
    scatter_dodge_width: float | None = None,
    scatter_mean_line: bool = False,
):
    """Plot one terrain panel with shaded std bands and scatter dots.

    failures: (scene, freq) where target=0 (experimental failure mode). X at y=0.
    all_failed: (scene, freq) where ALL selected trials are invalid (stalled/inverted).
        X at y=0, colored by scene.
    scatter_only: if True, no shading — scatter dots only with intra-morphology
        horizontal spreading (sorted by value, left-to-right lowest-to-highest).
    """
    n = len(PLOT_ORDER)
    dodge_width = 3.5  # total spread in Hz (non-scatter mode)
    if scatter_dodge_width is None:
        scatter_dodge_width = 15.0  # wider spread for scatter_only (freq ticks 20 Hz apart)
    if intra_spread is None:
        intra_spread = 3.0  # Hz, spread within one morphology's slot (scatter_only)
    # morphology gap = 15/3 = 5 Hz; clearance = 5 - 3 = 2 Hz
    for idx, scene in enumerate(PLOT_ORDER):
        d = data[scene]
        if not d["mean_freqs"]:
            continue
        dw = scatter_dodge_width if scatter_only else dodge_width
        dx = (idx - (n - 1) / 2) * (dw / (n - 1))

        if scatter_only:
            # Group trials by freq, sort by value, spread within dodge slot
            # Trials with value <= 0 are plotted as X markers (n/a / failure)
            scatter_x: list[float] = []
            scatter_y: list[float] = []
            fail_counts: dict[float, int] = {}  # freq → count of failed trials
            unique_freqs = sorted(set(d["freqs"]))
            for freq in unique_freqs:
                vals = sorted(v for f, v in zip(d["freqs"], d["trials"]) if f == freq)
                n_fail = sum(1 for v in vals if v <= 0)
                valid_vals = [v for v in vals if v > 0]
                nt = len(valid_vals)
                if nt == 1:
                    offsets = [0.0]
                elif nt > 1:
                    offsets = np.linspace(-intra_spread / 2, intra_spread / 2, nt).tolist()
                else:
                    offsets = []
                for off, val in zip(offsets, valid_vals):
                    scatter_x.append(freq + dx + off)
                    scatter_y.append(val)
                if n_fail > 0:
                    fail_counts[freq] = n_fail
            if scatter_x:
                ax.scatter(scatter_x, scatter_y, color=COLORS[scene], alpha=0.6, s=30,
                           zorder=3, label=LABELS[scene])
            has_label = bool(scatter_x)
            for freq, n_fail in fail_counts.items():
                fx = freq + dx
                ax.plot(fx, 0, "x", color=COLORS[scene],
                        markersize=8, markeredgewidth=2, zorder=5,
                        label=LABELS[scene] if not has_label else None)
                has_label = True
                if n_fail > 1:
                    ax.annotate(str(n_fail), (fx, 0), textcoords="offset points",
                                xytext=(0, -6), ha="center", va="top",
                                fontsize=14, fontweight="bold", color=COLORS[scene])
            # Mean line through scatter dots
            if scatter_mean_line and d["mean_freqs"]:
                mf = np.array(d["mean_freqs"]) + dx
                mm = np.array(d["means"])
                valid_m = mm > 0
                if valid_m.any():
                    ax.plot(mf[valid_m], mm[valid_m], color=COLORS[scene],
                            linewidth=1.2, alpha=0.7, zorder=4)
        else:
            freq_arr = np.array(d["mean_freqs"]) + dx
            mean = np.array(d["means"])
            std = np.array(d["stds"])
            ax.fill_between(
                freq_arr, mean - std, mean + std,
                color=COLORS[scene], alpha=0.2, label=LABELS[scene],
            )
            freqs_arr = np.array(d["freqs"])
            trials_arr = np.array(d["trials"])
            valid_mask = trials_arr > 0
            if valid_mask.any():
                ax.scatter(freqs_arr[valid_mask] + dx, trials_arr[valid_mask],
                           color=COLORS[scene], alpha=0.6, s=30, zorder=3)
            # Collapse per-trial failures into one X + count per freq
            if (~valid_mask).any():
                fail_freqs = freqs_arr[~valid_mask]
                for uf in sorted(set(fail_freqs)):
                    nf = int(np.sum(fail_freqs == uf))
                    fx = uf + dx
                    ax.plot(fx, 0, "x", color=COLORS[scene],
                            markersize=8, markeredgewidth=2, zorder=5)
                    if nf > 1:
                        ax.annotate(str(nf), (fx, 0), textcoords="offset points",
                                    xytext=(0, -6), ha="center", va="top",
                                    fontsize=14, fontweight="bold", color=COLORS[scene])

    # X markers for target=0 failure modes (dodged by morphology)
    if failures:
        for scene, freqs in failures.items():
            if scene in COLORS and scene in PLOT_ORDER:
                idx = PLOT_ORDER.index(scene)
                dw = scatter_dodge_width if scatter_only else dodge_width
                fail_dx = (idx - (n - 1) / 2) * (dw / (n - 1))
                for freq in freqs:
                    ax.plot(
                        freq + fail_dx, 0, "x", color=COLORS[scene],
                        markersize=10, markeredgewidth=2.5, zorder=5,
                    )

    # X markers for all-invalid (scene, freq) combos (dodged by morphology)
    if all_failed:
        for scene, freq_counts in all_failed.items():
            if scene in COLORS and scene in PLOT_ORDER:
                idx = PLOT_ORDER.index(scene)
                dw = scatter_dodge_width if scatter_only else dodge_width
                fail_dx = (idx - (n - 1) / 2) * (dw / (n - 1))
                for freq, count in freq_counts.items():
                    fx = freq + fail_dx
                    ax.plot(
                        fx, 0, "x", color=COLORS[scene],
                        markersize=10, markeredgewidth=2.5, zorder=5,
                    )
                    if count > 1:
                        ax.annotate(str(count), (fx, 0), textcoords="offset points",
                                    xytext=(0, -6), ha="center", va="top",
                                    fontsize=14, fontweight="bold", color=COLORS[scene])

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.grid(axis="y", alpha=0.3)

    all_freqs = set(f for d in data.values() for f in d["mean_freqs"])
    if failures:
        for freqs in failures.values():
            all_freqs.update(freqs)
    if all_failed:
        for freqs in all_failed.values():
            all_freqs.update(freqs)
    all_freqs_sorted = sorted(all_freqs)
    if all_freqs_sorted:
        pad = max(scatter_dodge_width / 2 + intra_spread / 2 + 2, 3) if scatter_only else 3
        ax.set_xlim(all_freqs_sorted[0] - pad, all_freqs_sorted[-1] + pad)
        # Bracket ticks + grey gap bands (unified for all modes)
        if scatter_only:
            half_spread = scatter_dodge_width / 2 + intra_spread / 2
        else:
            half_spread = dodge_width / 2 + 0.75
        if half_spread < 0.1:
            # No dodge — plain frequency ticks, no grey bands
            ax.set_xticks(all_freqs_sorted)
            ax.set_xticklabels([str(int(f)) for f in all_freqs_sorted])
        else:
            edge_ticks = []
            for f in all_freqs_sorted:
                edge_ticks.extend([f - half_spread, f + half_spread])
            ax.set_xticks(edge_ticks)
            ax.set_xticklabels([""] * len(edge_ticks))
            ax.set_xticks(all_freqs_sorted, minor=True)
            ax.set_xticklabels([str(int(f)) for f in all_freqs_sorted], minor=True)
            ax.tick_params(which="minor", length=0)
            # Grey bands in gaps between bracket zones (white inside brackets)
            x_lo = all_freqs_sorted[0] - pad
            x_hi = all_freqs_sorted[-1] + pad
            ax.axvspan(x_lo, all_freqs_sorted[0] - half_spread, color="#f0f0f0", zorder=0)
            for j in range(len(all_freqs_sorted) - 1):
                ax.axvspan(all_freqs_sorted[j] + half_spread,
                           all_freqs_sorted[j + 1] - half_spread,
                           color="#f0f0f0", zorder=0)
            ax.axvspan(all_freqs_sorted[-1] + half_spread, x_hi, color="#f0f0f0", zorder=0)


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

    output_dir = args.output_dir or pathlib.Path("plots")

    # Generate separate figures per terrain (each run_dir has different params)
    for terrain, rows, run_dir in entries:
        title = TERRAIN_TITLES.get(terrain, terrain.replace("_", " ").title())
        ge = GATE_END.get(terrain)
        all_failed = build_all_failed_freqs(rows, selected_only=True, gate_end=ge)
        so = terrain.startswith("rough")  # scatter_only for rough

        # Velocity
        fig_v, ax_v = plt.subplots(figsize=(7, 5))
        vx_data = build_plot_data(rows, "vx", selected_only=True, exclude_invalid=True, gate_end=ge)
        plot_panel(ax_v, vx_data, title, "Forward Velocity (mm/s)", all_failed=all_failed, scatter_only=so)
        ax_v.legend(loc="upper left", fontsize=12, framealpha=0.9)
        fig_v.tight_layout()
        vel_path = output_dir / f"velocity_vs_freq_{terrain}.png"
        fig_v.savefig(vel_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {vel_path}")
        plt.close(fig_v)

        # COT
        fig_c, ax_c = plt.subplots(figsize=(7, 5))
        cot_data = build_plot_data(rows, "cot", selected_only=True, exclude_invalid=True, gate_end=ge)
        plot_panel(ax_c, cot_data, title, "Cost of Transport", all_failed=all_failed, scatter_only=so)
        ax_c.legend(loc="upper left", fontsize=12, framealpha=0.9)
        fig_c.tight_layout()
        cot_path = output_dir / f"cot_vs_freq_{terrain}.png"
        fig_c.savefig(cot_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {cot_path}")
        plt.close(fig_c)

        # Pitch RMS
        fig_p, ax_p = plt.subplots(figsize=(7, 5))
        pitch_data = build_plot_data(rows, "pitch_rms", selected_only=True, exclude_invalid=True, gate_end=ge)
        pitch_excl = PITCH_EXCLUDE.get(terrain, [])
        if pitch_excl:
            strip_freqs(pitch_data, pitch_excl)
        plot_panel(ax_p, pitch_data, title, "Pitch Amplitude RMS (\u00b0)", all_failed=all_failed, scatter_only=so)
        ax_p.legend(loc="upper left", fontsize=12, framealpha=0.9)
        fig_p.tight_layout()
        pitch_path = output_dir / f"pitch_vs_freq_{terrain}.png"
        fig_p.savefig(pitch_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {pitch_path}")
        plt.close(fig_p)

    # Composite 3×3 if multiple terrains provided
    if len(entries) > 1:
        metrics = [
            ("vx", "Forward Velocity (mm/s)"),
            ("cot", "Cost of Transport"),
            ("pitch_rms", "Pitch Amplitude RMS (\u00b0)"),
        ]
        n_rows = len(entries)
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows),
                                 squeeze=False)

        for i, (terrain, rows, _) in enumerate(entries):
            title_t = TERRAIN_TITLES.get(terrain, terrain.replace("_", " ").title())
            ge = GATE_END.get(terrain)
            all_failed = build_all_failed_freqs(rows, selected_only=True, gate_end=ge)
            so = terrain.startswith("rough")
            for j, (metric, ylabel) in enumerate(metrics):
                pdata = build_plot_data(rows, metric, selected_only=True, exclude_invalid=True, gate_end=ge)
                if metric == "pitch_rms":
                    pitch_excl = PITCH_EXCLUDE.get(terrain, [])
                    if pitch_excl:
                        strip_freqs(pdata, pitch_excl)
                panel_title = f"{title_t}: {ylabel}"
                plot_panel(axes[i, j], pdata, panel_title, ylabel, all_failed=all_failed, scatter_only=so)

        # Single legend from top-left
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[0, 0].legend(handles, labels, loc="upper left", fontsize=10, framealpha=0.9)
        for ax in axes.flat:
            leg = ax.get_legend()
            if leg and ax is not axes[0, 0]:
                leg.remove()

        fig.tight_layout()
        comp_path = output_dir / "sim_composite.png"
        fig.savefig(comp_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {comp_path}")
        plt.close(fig)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
