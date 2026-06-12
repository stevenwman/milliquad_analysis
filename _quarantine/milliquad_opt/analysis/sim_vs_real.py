"""Sim vs Real comparison metrics.

Loads experimental ground truth from raw CSVs (via extract_* functions)
and simulation results from validation_trials.csv. Computes per-condition
and summary % error.

Usage:
    uv run python -m analysis.sim_vs_real \
        results/20260228T013353_rk4_flat \
        results/20260228T230022_step_q60_rk-warm
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# ── Experimental data extractors ──
_EXP_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "experimental_data")
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from plot_velocity_vs_freq import extract_flat, extract_step_q60, extract_rough  # noqa: E402

from analysis._common import detect_terrain  # noqa: E402
from analysis.plot_validation import (  # noqa: E402
    GATE_END,
    build_plot_data,
    load_validation_csv,
)

_MORPH_TO_SCENE = {"leg": "scene1", "2leg": "scene2", "4leg": "scene4", "wheel": "scene_wheel"}
_SCENE_TO_MORPH = {v: k for k, v in _MORPH_TO_SCENE.items()}

_VEL_EXTRACTORS = {"flat": extract_flat, "step": extract_step_q60, "rough": extract_rough}


def _get_exp_means(terrain: str) -> dict[str, dict[float, tuple[float, float]]]:
    """Returns {scene: {freq: (mean_mm_s, std_mm_s)}} from raw experimental CSVs."""
    extractor = _VEL_EXTRACTORS.get(terrain)
    if extractor is None:
        return {}
    raw = extractor()
    result = {}
    for morph, scene in _MORPH_TO_SCENE.items():
        d = raw[morph]
        result[scene] = {}
        for f, m, s in zip(d["mean_freqs"], d["means"], d["stds"]):
            result[scene][float(f)] = (m, s)
    return result


def _get_sim_means(csv_path: pathlib.Path, terrain: str) -> dict[str, dict[float, tuple[float, float, int]]]:
    """Returns {scene: {freq: (mean_mm_s, std_mm_s, n_valid)}} from validation CSV.

    Uses same validity filtering as plots (gate + pitch check), non-crashed only.
    """
    rows = load_validation_csv(csv_path)
    gate_end = GATE_END.get(terrain)
    sim_data = build_plot_data(rows, "vx", exclude_invalid=True, gate_end=gate_end)

    result = {}
    for scene, d in sim_data.items():
        result[scene] = {}
        for f, m, s in zip(d["mean_freqs"], d["means"], d["stds"]):
            # Count trials at this freq
            n = sum(1 for ff in d["freqs"] if ff == f)
            result[scene][float(f)] = (m, s, n)
    return result


def compare_velocity(run_dirs: list[pathlib.Path]) -> None:
    """Print per-condition and summary velocity % error for each terrain."""
    for run_dir in run_dirs:
        csv_path = run_dir / "validation_trials.csv"
        if not csv_path.exists():
            print(f"  SKIP {run_dir.name}: no validation_trials.csv")
            continue

        terrain = detect_terrain(run_dir)
        exp = _get_exp_means(terrain)
        sim = _get_sim_means(csv_path, terrain)

        if not exp:
            print(f"  SKIP {run_dir.name}: no experimental extractor for '{terrain}'")
            continue

        print(f"\n{'='*72}")
        print(f"  {terrain.upper()} — {run_dir.name}")
        print(f"{'='*72}")
        print(f"  {'Scene':<14s} {'Freq':>4s}  {'Exp (mm/s)':>10s}  {'Sim (mm/s)':>10s}  {'Err%':>7s}  {'n':>2s}")
        print(f"  {'-'*14} {'-'*4}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*2}")

        errors = []
        abs_errors = []
        for scene in ("scene1", "scene2", "scene4", "scene_wheel"):
            if scene not in exp:
                continue
            for freq in sorted(exp[scene].keys()):
                exp_mean, exp_std = exp[scene][freq]
                if scene in sim and freq in sim[scene]:
                    sim_mean, sim_std, n = sim[scene][freq]
                    if abs(exp_mean) > 0.1:  # skip near-zero targets (failure modes)
                        pct_err = (sim_mean - exp_mean) / exp_mean * 100
                        errors.append(pct_err)
                        abs_errors.append(abs(pct_err))
                        print(f"  {scene:<14s} {freq:4.0f}  {exp_mean:10.2f}  {sim_mean:10.2f}  {pct_err:+6.1f}%  {n:2d}")
                    else:
                        print(f"  {scene:<14s} {freq:4.0f}  {exp_mean:10.2f}  {sim_mean:10.2f}  {'(fail)':>7s}  {n:2d}")
                else:
                    print(f"  {scene:<14s} {freq:4.0f}  {exp_mean:10.2f}  {'n/a':>10s}  {'':>7s}")

        if errors:
            print(f"\n  Summary ({len(errors)} conditions, excluding failure modes):")
            print(f"    Mean  error:  {np.mean(errors):+.1f}%")
            print(f"    Mean |error|: {np.mean(abs_errors):.1f}%")
            print(f"    Median |err|: {np.median(abs_errors):.1f}%")
            print(f"    Max  |error|: {np.max(abs_errors):.1f}%  (worst case)")
            print(f"    Std  error:   {np.std(errors):.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Sim vs Real comparison metrics")
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path,
                        help="Result directories to compare")
    args = parser.parse_args()

    compare_velocity(args.run_dirs)


if __name__ == "__main__":
    main()
