#!/usr/bin/env python3
"""Pretty-print optimization_bests.csv as readable tables."""

import csv
import sys
from config import reference_rows, BEST_CSV_PATH

ref_rows = reference_rows()
ref_ids = [r["id"] for r in ref_rows]
targets = {r["id"]: r["speed"] for r in ref_rows}

path = sys.argv[1] if len(sys.argv) > 1 else BEST_CSV_PATH
rows = list(csv.DictReader(open(path)))

if not rows:
    print(f"No rows in {path}")
    sys.exit(0)

for r in rows:
    print("=" * 72)
    print(f"  {r['timestamp']}  n={r['n_eval']}  id={r['id']}  cost={r['cost']}")
    print(f"  {'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9}")
    print(f"  {'-' * 44}")
    for rid in ref_ids:
        t = targets[rid]
        s = float(r.get(f"vel_{rid}", 0))
        d = (s - t) * 100
        print(f"  {rid:<18} {t:>6.3f}  {s:>6.3f}  {d:>+7.1f}cs")
    print()

print(f"Total: {len(rows)} best(s) recorded")
