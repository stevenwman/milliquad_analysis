#!/usr/bin/env python3
"""Compare best params across per-morphology and per-frequency optimization runs.

Finds the latest solo_* results directories and shows params side-by-side,
flagging which params converge (shared physics) vs diverge (group-specific).
"""

import csv
import math
import pathlib
import sys

from config import space

PARAM_NAMES = [dim.name for dim in space] + ["solimp_dmax"]
MORPHOLOGIES = ["scene1", "scene2", "scene4", "scene_wheel"]
FREQUENCIES = ["f10", "f30", "f50"]

results_dir = pathlib.Path(__file__).parent / "results"


def find_latest(suffix: str) -> pathlib.Path | None:
    """Find the latest results dir matching *_{suffix}."""
    candidates = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.endswith(f"_{suffix}")
    )
    return candidates[-1] if candidates else None


def load_best_params(run_dir: pathlib.Path) -> dict[str, float] | None:
    """Load params from the last row of optimization_bests.csv."""
    csv_path = run_dir / "optimization_bests.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    last = rows[-1]
    params = {}
    for p in PARAM_NAMES:
        if p in last:
            params[p] = float(last[p])
    params["_cost"] = float(last.get("cost", 0))
    params["_n_eval"] = int(last.get("n_eval", 0))
    return params


def log_range_frac(dim, value: float) -> float:
    """Fraction through the search range (log-space for log-uniform)."""
    lo, hi = dim.low, dim.high
    if dim.prior == "log-uniform" and lo > 0 and hi > 0 and value > 0:
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (value - lo) / (hi - lo) if hi != lo else 0.5


def print_comparison(labels: list[str], runs: dict, params: dict, group_label: str):
    """Print a side-by-side comparison table."""
    cols = [l for l in labels if l in params]
    if len(cols) < 2:
        print(f"  Need at least 2 {group_label} runs to compare (found {len(cols)})\n")
        return

    # Header
    print(f"\n{'param':<24}", end="")
    for c in cols:
        print(f"  {c:>14}", end="")
    print(f"  {'spread':>8}  {'verdict'}")
    print("-" * (24 + 16 * len(cols) + 22))

    # Cost row
    print(f"{'COST':<24}", end="")
    for c in cols:
        print(f"  {params[c]['_cost']:>14.4f}", end="")
    print()

    # N_eval row
    print(f"{'n_eval':<24}", end="")
    for c in cols:
        print(f"  {params[c]['_n_eval']:>14.0f}", end="")
    print()
    print()

    # Param rows
    for pname in PARAM_NAMES:
        vals = [params[c].get(pname) for c in cols]
        if any(v is None for v in vals):
            continue

        dim = next((d for d in space if d.name == pname), None)
        if dim is not None:
            fracs = [log_range_frac(dim, v) for v in vals]
            spread = max(fracs) - min(fracs)
        else:
            spread = None

        print(f"  {pname:<22}", end="")
        for v in vals:
            print(f"  {v:>14.6g}", end="")

        if spread is not None:
            verdict = "CONVERGE" if spread < 0.15 else ("MIXED" if spread < 0.35 else "DIVERGE")
            print(f"  {spread:>7.1%}  {verdict}")
        else:
            print()

    print()
    for c in cols:
        print(f"  {c}: {runs[c].name}")
    print()


# --- Load per-morphology results ---
morph_runs = {}
morph_params = {}
for morph in MORPHOLOGIES:
    d = find_latest(f"solo_{morph}")
    if d is None:
        continue
    p = load_best_params(d)
    if p is None:
        continue
    morph_runs[morph] = d
    morph_params[morph] = p

# --- Load per-frequency results ---
freq_runs = {}
freq_params = {}
for freq in FREQUENCIES:
    d = find_latest(f"solo_{freq}")
    if d is None:
        continue
    p = load_best_params(d)
    if p is None:
        continue
    freq_runs[freq] = d
    freq_params[freq] = p

# --- Load combined result ---
combined_run = find_latest("combined")
combined_params = load_best_params(combined_run) if combined_run else None

# --- Print ---
print("=" * 80)
print("PER-MORPHOLOGY (all freqs for one morphology)")
print("  Divergence here → morphology-specific contact/coupling")
print("=" * 80)
print_comparison(MORPHOLOGIES, morph_runs, morph_params, "morphology")

print("=" * 80)
print("PER-FREQUENCY (all morphologies at one frequency)")
print("  Divergence here → model missing frequency-dependent physics")
print("=" * 80)
print_comparison(FREQUENCIES, freq_runs, freq_params, "frequency")

# --- Combined vs splits ---
if combined_params and (morph_params or freq_params):
    print("=" * 80)
    print("COMBINED vs PER-MORPHOLOGY")
    print("  Shows how much the global fit compromises vs per-morphology bests")
    print("=" * 80)
    all_labels = MORPHOLOGIES + ["COMBINED"]
    all_runs = dict(morph_runs)
    all_params = dict(morph_params)
    if combined_run:
        all_runs["COMBINED"] = combined_run
        all_params["COMBINED"] = combined_params
    print_comparison(all_labels, all_runs, all_params, "morph+combined")

    print("=" * 80)
    print("COMBINED vs PER-FREQUENCY")
    print("  Shows how much the global fit compromises vs per-frequency bests")
    print("=" * 80)
    all_labels = FREQUENCIES + ["COMBINED"]
    all_runs = dict(freq_runs)
    all_params = dict(freq_params)
    if combined_run:
        all_runs["COMBINED"] = combined_run
        all_params["COMBINED"] = combined_params
    print_comparison(all_labels, all_runs, all_params, "freq+combined")
elif combined_params:
    print("=" * 80)
    print("COMBINED (all morphologies + frequencies)")
    print("=" * 80)
    print(f"  Run: {combined_run.name}")
    print(f"  Cost: {combined_params['_cost']:.6f}")
    print(f"  Evals: {combined_params['_n_eval']}")
    print()
    for pname in PARAM_NAMES:
        if pname in combined_params:
            print(f"  {pname:<22}  {combined_params[pname]:>14.6g}")
    print()

print("Spread = range of positions through search space (log-space for log-uniform)")
print("CONVERGE (<15%): shared physics — safe to lock globally")
print("MIXED (15-35%): may benefit from per-group tuning")
print("DIVERGE (>35%): strong candidate for per-group fitting")
