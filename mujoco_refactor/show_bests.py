#!/usr/bin/env python3
"""Pretty-print optimization_bests.csv as readable tables."""

import csv
import pathlib
import sys
from config import reference_rows

ref_rows = reference_rows()
ref_ids = [r["id"] for r in ref_rows]
targets = {r["id"]: r["speed"] for r in ref_rows}

# Default: latest results dir; override with explicit path as argv[1]
if len(sys.argv) > 1:
    path = sys.argv[1]
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

# Check if new columns exist (backwards-compat with old CSVs)
has_lateral = f"lateral_{ref_ids[0]}" in rows[0]

for r in rows:
    print("=" * 72)
    elapsed = r.get('elapsed_min', '')
    elapsed_str = f"  t={elapsed}min" if elapsed else ""
    print(f"  {r['timestamp']}  n={r['n_eval']}{elapsed_str}  id={r['id']}  cost={r['cost']}")
    has_yaw = f"yaw_{ref_ids[0]}" in r
    if has_lateral:
        if has_yaw:
            print(f"  {'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9} {'Δ%':>5} {'tumble':>7} {'lateral':>8} {'yaw':>5} {'pitch':>6}")
            print(f"  {'-' * 78}")
        else:
            print(f"  {'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9} {'Δ%':>5} {'tumble':>7} {'lateral':>8} {'pitch':>6}")
            print(f"  {'-' * 72}")
    else:
        print(f"  {'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9} {'Δ%':>5}")
        print(f"  {'-' * 50}")
    for rid in ref_ids:
        t = targets[rid]
        s = float(r.get(f"vel_{rid}", 0))
        d = (s - t) * 100
        dpct = ((s - t) / t * 100) if t != 0 else 0.0
        if has_lateral:
            tmb = float(r.get(f"tumble_{rid}", 0))
            lat = float(r.get(f"lateral_{rid}", 0)) * 100  # cm
            yaw = float(r.get(f"yaw_{rid}", 0)) if has_yaw else None
            pit = float(r.get(f"pitch_rms_{rid}", 0))
            if has_yaw:
                print(f"  {rid:<18} {t:>6.3f}  {s:>6.3f}  {d:>+7.1f}cs {dpct:>+4.0f}%  {tmb:>6.4f}  {lat:>6.1f}cm  {yaw:>4.0f}°  {pit:>4.1f}°")
            else:
                print(f"  {rid:<18} {t:>6.3f}  {s:>6.3f}  {d:>+7.1f}cs {dpct:>+4.0f}%  {tmb:>6.4f}  {lat:>6.1f}cm  {pit:>4.1f}°")
        else:
            print(f"  {rid:<18} {t:>6.3f}  {s:>6.3f}  {d:>+7.1f}cs {dpct:>+4.0f}%")
    print()

print(f"Total: {len(rows)} best(s) recorded")
