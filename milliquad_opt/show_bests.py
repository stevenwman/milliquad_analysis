#!/usr/bin/env python3
"""Pretty-print optimization_bests.csv as readable tables.

Usage:
    uv run python show_bests.py                          # latest results dir
    uv run python show_bests.py results/20260227T.../     # specific dir
    uv run python show_bests.py path/to/optimization_bests.csv
"""

import argparse
import csv
import math
import pathlib
import sys

from config import space

_BOUNDS = {dim.name: (dim.low, dim.high) for dim in space}
_PARAM_NAMES = [dim.name for dim in space] + ["solimp_dmax"]


def _bound_flag(name: str, value: float) -> str:
    if name not in _BOUNDS:
        return ""
    lo, hi = _BOUNDS[name]
    dim = next(d for d in space if d.name == name)
    is_log = dim.prior == "log-uniform"
    if is_log and lo > 0 and hi > 0 and value > 0:
        log_lo, log_hi, log_v = math.log(lo), math.log(hi), math.log(value)
        frac = (log_v - log_lo) / (log_hi - log_lo)
    else:
        frac = (value - lo) / (hi - lo) if hi != lo else 0.5
    if frac <= 0.02:
        return " << LO"
    if frac >= 0.98:
        return " >> HI"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Show optimization bests")
    parser.add_argument("path", nargs="?", default=None, help="Path to results dir or CSV")
    parser.add_argument("--terrain", "-t", default=None,
                        help="Terrain config for targets (flat, step, rough, flat_no20)")
    args = parser.parse_args()

    if args.path:
        p = pathlib.Path(args.path)
        if p.is_dir():
            path = str(p / "optimization_bests.csv")
        else:
            path = str(p)
    else:
        results_dir = pathlib.Path("results")
        run_dirs = sorted(
            d for d in results_dir.iterdir()
            if d.is_dir() and (d / "optimization_bests.csv").exists()
        ) if results_dir.exists() else []
        if not run_dirs:
            print("No results directories found with optimization_bests.csv")
            sys.exit(0)
        path = str(run_dirs[-1] / "optimization_bests.csv")
        print(f"Using: {path}")

    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"No rows in {path}")
        sys.exit(0)

    # Detect ref IDs from vel_* columns
    ref_ids = [k[4:] for k in rows[0] if k.startswith("vel_")]

    # Determine terrain type for correct reference targets
    terrain = args.terrain
    if terrain is None:
        # Auto-detect from results dir name (e.g. "20260228T..._rk4_flat")
        dir_name = pathlib.Path(path).parent.name
        for t in ["flat_no20", "flat", "step", "rough"]:
            if t in dir_name:
                terrain = t
                break
    if terrain is None:
        terrain = "flat"  # fallback

    targets = {}
    from config import reference_rows as _rr
    try:
        mod = __import__(f"config_{terrain}")
        for row in _rr(mod.REFERENCE_DATA):
            targets[row["id"]] = row["speed"]
    except Exception as e:
        print(f"  WARNING: could not load config_{terrain}: {e}", file=sys.stderr)

    has_lateral = f"lateral_{ref_ids[0]}" in rows[0] if ref_ids else False
    has_yaw = f"yaw_{ref_ids[0]}" in rows[0] if ref_ids else False

    for r in rows:
        print("=" * 72)
        elapsed = r.get('elapsed_min', '')
        elapsed_str = f"  t={elapsed}min" if elapsed else ""
        print(f"  {r['timestamp']}  n={r['n_eval']}{elapsed_str}  id={r['id']}  cost={r['cost']}")

        if has_lateral and has_yaw:
            print(f"  {'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9} {'Δ%':>5} {'tumble':>7} {'lateral':>8} {'yaw':>5}")
            print(f"  {'-' * 70}")
        elif has_lateral:
            print(f"  {'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9} {'Δ%':>5} {'tumble':>7} {'lateral':>8}")
            print(f"  {'-' * 65}")
        else:
            print(f"  {'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9} {'Δ%':>5}")
            print(f"  {'-' * 50}")

        for rid in ref_ids:
            t = targets.get(rid, 0.0)
            s = float(r.get(f"vel_{rid}", 0))
            d = (s - t) * 100
            dpct = ((s - t) / t * 100) if t != 0 else 0.0
            if has_lateral:
                tmb = float(r.get(f"tumble_{rid}", 0))
                lat = float(r.get(f"lateral_{rid}", 0)) * 100
                if has_yaw:
                    yaw = float(r.get(f"yaw_{rid}", 0))
                    print(f"  {rid:<18} {t:>6.3f}  {s:>6.3f}  {d:>+7.1f}cs {dpct:>+4.0f}%  {tmb:>6.4f}  {lat:>6.1f}cm  {yaw:>4.0f}°")
                else:
                    print(f"  {rid:<18} {t:>6.3f}  {s:>6.3f}  {d:>+7.1f}cs {dpct:>+4.0f}%  {tmb:>6.4f}  {lat:>6.1f}cm")
            else:
                print(f"  {rid:<18} {t:>6.3f}  {s:>6.3f}  {d:>+7.1f}cs {dpct:>+4.0f}%")

        # Show params for last (current best) entry
        if r is rows[-1]:
            print(f"\n  {'param':<28} {'value':>14}  {'flag'}")
            print(f"  {'-' * 50}")
            for pname in _PARAM_NAMES:
                if pname in r:
                    raw = r[pname]
                    if raw.startswith("np.float64("):
                        raw = raw[len("np.float64("):-1]
                    val = float(raw)
                    flag = _bound_flag(pname, val)
                    print(f"  {pname:<28} {val:>14.6g}  {flag}")
        print()

    print(f"Total: {len(rows)} best(s) recorded")


if __name__ == "__main__":
    main()
