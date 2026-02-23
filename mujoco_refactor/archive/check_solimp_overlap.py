"""Quick check: how often does solimp_dmin >= solimp_dmax in optimization history?"""
import csv
import pathlib
import sys

results_dir = pathlib.Path(__file__).parent / "results"
if len(sys.argv) > 1:
    dirs = [pathlib.Path(sys.argv[1])]
else:
    dirs = sorted(results_dir.iterdir())

total = 0
violations = 0
violation_costs = []
ok_costs = []

for d in dirs:
    csv_path = d / "multi_optimization_results.csv"
    if not csv_path.exists():
        continue
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "solimp_dmin" not in row or "solimp_dmax" not in row:
                continue
            dmin = float(row["solimp_dmin"])
            dmax = float(row["solimp_dmax"])
            cost = float(row["cost"])
            total += 1
            if dmin >= dmax:
                violations += 1
                violation_costs.append(cost)
            else:
                ok_costs.append(cost)

print(f"Total rows scanned: {total}")
print(f"Violations (dmin >= dmax): {violations} ({100*violations/total:.1f}%)" if total else "No data")
if violation_costs:
    violation_costs.sort()
    ok_costs.sort()
    print(f"\nViolation costs:  min={min(violation_costs):.4f}  median={violation_costs[len(violation_costs)//2]:.4f}  max={max(violation_costs):.4f}")
    print(f"OK costs:         min={min(ok_costs):.4f}  median={ok_costs[len(ok_costs)//2]:.4f}  max={max(ok_costs):.4f}")
    # How many violations are in the top 100 overall?
    all_costs = sorted([(c, "bad") for c in violation_costs] + [(c, "ok") for c in ok_costs])
    top100_bad = sum(1 for _, tag in all_costs[:100] if tag == "bad")
    print(f"\nViolations in top 100 overall: {top100_bad}")
else:
    print("\nNo violations found.")
