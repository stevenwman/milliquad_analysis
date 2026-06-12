#!/usr/bin/env python3
"""Compare best parameters across multiple optimization runs.

No simulation — reads only CSV files.

Usage:
    uv run python -m analysis.compare_params results/20260228T*
    uv run python -m analysis.compare_params results/20260228T013353_rk4_flat results/20260228T102010_rk4_rough
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys

_PARENT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from config import space  # noqa: E402

PARAM_NAMES = [dim.name for dim in space] + ["solimp_dmax"]
_BOUNDS = {dim.name: (dim.low, dim.high) for dim in space}


def _short_label(dirname: str) -> str:
    """Extract short label from results dir name (e.g. '20260228T013353_rk4_flat' -> 'flat_W')."""
    parts = dirname.split("_")
    # Find terrain keyword
    terrain = ""
    for p in parts:
        if p in ("flat", "step", "rough"):
            terrain = p
            break
    cold = "cold" in dirname
    suffix = "C" if cold else "W"
    return f"{terrain}_{suffix}" if terrain else dirname[-10:]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path,
                        help="Results directories to compare")
    args = parser.parse_args()

    # Load best params from each run
    runs: list[tuple[str, dict[str, float], float]] = []
    for d in sorted(args.run_dirs):
        if not d.is_dir():
            print(f"Skipping {d} (not a directory)", file=sys.stderr)
            continue
        bests_csv = d / "optimization_bests.csv"
        if not bests_csv.exists():
            print(f"Skipping {d} (no optimization_bests.csv)", file=sys.stderr)
            continue
        rows = list(csv.DictReader(open(bests_csv)))
        if not rows:
            continue
        best = rows[-1]
        cost = float(best["cost"])
        params = {}
        for pname in PARAM_NAMES:
            if pname in best:
                raw = best[pname]
                if raw.startswith("np.float64("):
                    raw = raw[len("np.float64("):-1]
                params[pname] = float(raw)
        label = _short_label(d.name)
        runs.append((label, params, cost))

    if not runs:
        print("No valid runs found.")
        sys.exit(1)

    # Print header
    labels = [r[0] for r in runs]
    col_w = max(10, max(len(l) for l in labels) + 1)
    header = f"{'param':<28}" + "".join(f"{l:>{col_w}}" for l in labels) + f"{'range':>{col_w}}"
    print(header)
    print("-" * len(header))

    # Print costs first
    cost_line = f"{'COST':<28}" + "".join(f"{r[2]:>{col_w}.4f}" for r in runs)
    print(cost_line)
    print()

    # Print each parameter
    for pname in PARAM_NAMES:
        vals = [r[1].get(pname) for r in runs]
        if all(v is None for v in vals):
            continue

        # Format values
        parts = []
        numeric_vals = []
        for v in vals:
            if v is None:
                parts.append(f"{'—':>{col_w}}")
            else:
                numeric_vals.append(v)
                # Choose format based on magnitude
                if abs(v) < 1e-4:
                    parts.append(f"{v:>{col_w}.2e}")
                elif abs(v) < 0.01:
                    parts.append(f"{v:>{col_w}.5f}")
                elif abs(v) < 100:
                    parts.append(f"{v:>{col_w}.4f}")
                else:
                    parts.append(f"{v:>{col_w}.2f}")

        # Range (max - min)
        if len(numeric_vals) >= 2:
            lo, hi = min(numeric_vals), max(numeric_vals)
            # Use log-range for log-uniform params
            is_log = pname in _BOUNDS and next(
                (d for d in space if d.name == pname), None
            ) and next((d for d in space if d.name == pname)).prior == "log-uniform"
            if is_log and lo > 0:
                rng = math.log10(hi / lo)
                rng_str = f"{rng:>{col_w - 1}.1f}x"
            else:
                rng_str = f"{hi - lo:>{col_w}.4f}"
        else:
            rng_str = f"{'—':>{col_w}}"

        line = f"{pname:<28}" + "".join(parts) + rng_str
        print(line)


if __name__ == "__main__":
    main()
