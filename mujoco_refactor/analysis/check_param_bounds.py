"""Check where top solutions cluster for solref/solimp/friction params across all runs."""
import csv
import pathlib
import sys
import statistics

PARAMS = [
    "sliding_friction", "torsional_friction", "rolling_friction",
    "solref_timeconst", "solref_dampratio",
    "solimp_dmin", "solimp_dmax", "solimp_width", "solimp_midpoint", "solimp_power",
]

results_dir = pathlib.Path(__file__).parent.parent / "results"
dirs = sorted(results_dir.iterdir())

rows = []
for d in dirs:
    csv_path = d / "multi_optimization_results.csv"
    if not csv_path.exists():
        continue
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "cost" not in row or "solimp_dmin" not in row:
                continue
            cost = float(row["cost"])
            if cost >= 1e6:
                continue
            rows.append((cost, row))

rows.sort(key=lambda x: x[0])
print(f"Total non-failure rows: {len(rows)}")

for n, label in [(50, "Top 50"), (100, "Top 100"), (500, "Top 500")]:
    subset = rows[:n]
    print(f"\n{'='*80}")
    print(f"{label} solutions (cost range: {subset[0][0]:.4f} — {subset[-1][0]:.4f})")
    print(f"{'param':<22} {'min':>12} {'p5':>12} {'median':>12} {'p95':>12} {'max':>12}")
    print("-" * 82)
    for p in PARAMS:
        vals = sorted(float(r[p]) for _, r in subset)
        p5 = vals[max(0, len(vals)//20)]
        p95 = vals[min(len(vals)-1, len(vals)*19//20)]
        med = statistics.median(vals)
        print(f"{p:<22} {min(vals):>12.6g} {p5:>12.6g} {med:>12.6g} {p95:>12.6g} {max(vals):>12.6g}")
