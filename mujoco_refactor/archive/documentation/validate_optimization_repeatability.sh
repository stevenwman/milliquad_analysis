#!/usr/bin/env bash
#
# Validate optimization repeatability by re-running the 3 most recent configs
# for 64 evaluations and comparing CSV entries.
#
# Expected: With same random seed and X0, CMA-ES should produce identical
# parameter proposals and costs (assuming simulation is deterministic).
#
# Usage:
#   ./validate_optimization_repeatability.sh
#

set -e
cd "$(dirname "$0")"

echo "========================================================================"
echo "OPTIMIZATION REPEATABILITY VALIDATION"
echo "========================================================================"
echo
echo "This will re-run 3 recent optimization configs for 64 evals each"
echo "and compare multi_optimization_results.csv entries to verify repeatability."
echo
echo "Target runs:"
echo "  1. 20260225T122342_flat_10_30_50        (16-dim flat, f10/f30/f50)"
echo "  2. 20260225T003517_flat_16dim_corrected_warm  (16-dim flat, all freqs)"
echo "  3. 20260224T175707_step_16dim_v1       (16-dim step)"
echo
echo "Each run will execute 64 evals (8 batches × 8 points/batch)"
echo "Estimated time: ~10-15 minutes total"
echo
read -p "Proceed? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ============================================================================
# 1. Flat terrain, f10/f30/f50 only (16-dim)
# ============================================================================

echo
echo "========================================================================"
echo "[1/3] Running: flat_10_30_50 (16-dim, --freqs 10 30 50)"
echo "========================================================================"

python optimizer_new.py \
    --freqs 10 30 50 \
    --n-calls 64 \
    --suffix validate_10_30_50

VALIDATE_DIR_1=$(ls -1dt results/*/config.py | head -1 | xargs dirname)
echo "Results saved to: $VALIDATE_DIR_1"

# ============================================================================
# 2. Flat terrain, all freqs (16-dim, corrected warm-start)
# ============================================================================

echo
echo "========================================================================"
echo "[2/3] Running: flat_16dim_corrected_warm (16-dim, all freqs)"
echo "========================================================================"

python optimizer_new.py \
    --n-calls 64 \
    --suffix validate_16dim_flat

VALIDATE_DIR_2=$(ls -1dt results/*/config.py | head -1 | xargs dirname)
echo "Results saved to: $VALIDATE_DIR_2"

# ============================================================================
# 3. Step terrain (16-dim)
# ============================================================================

echo
echo "========================================================================"
echo "[3/3] Running: step_16dim (16-dim step terrain)"
echo "========================================================================"

python optimizer_step.py \
    --n-calls 64 \
    --suffix validate_step

VALIDATE_DIR_3=$(ls -1dt results/*/config.py | head -1 | xargs dirname)
echo "Results saved to: $VALIDATE_DIR_3"

# ============================================================================
# Comparison analysis
# ============================================================================

echo
echo "========================================================================"
echo "COMPARISON ANALYSIS"
echo "========================================================================"

cat > /tmp/compare_csvs.py <<'PYEOF'
#!/usr/bin/env python3
"""Compare multi_optimization_results.csv entries for repeatability."""

import sys
import csv
import numpy as np
from pathlib import Path

def compare_runs(orig_dir: str, valid_dir: str, label: str):
    """Compare first 64 rows of original vs validation run."""

    orig_csv = Path(orig_dir) / "multi_optimization_results.csv"
    valid_csv = Path(valid_dir) / "multi_optimization_results.csv"

    if not orig_csv.exists() or not valid_csv.exists():
        print(f"✗ {label}: CSV files not found")
        return False

    # Read first 64 data rows (skip header)
    with open(orig_csv) as f:
        orig_rows = list(csv.DictReader(f))[:64]

    with open(valid_csv) as f:
        valid_rows = list(csv.DictReader(f))[:64]

    if len(orig_rows) < 64:
        print(f"⚠ {label}: Original run has only {len(orig_rows)} rows (expected ≥64)")
        n_compare = len(orig_rows)
    else:
        n_compare = 64

    if len(valid_rows) < n_compare:
        print(f"✗ {label}: Validation run has only {len(valid_rows)} rows (expected {n_compare})")
        return False

    # Compare costs
    orig_costs = np.array([float(r["cost"]) for r in orig_rows[:n_compare]])
    valid_costs = np.array([float(r["cost"]) for r in valid_rows[:n_compare]])

    cost_diff = np.abs(orig_costs - valid_costs)
    max_diff = np.max(cost_diff)
    mean_diff = np.mean(cost_diff)

    # Check parameter values (first 5 params as spot check)
    param_names = [k for k in orig_rows[0].keys() if k not in ["id", "cost", "elapsed_min"]][:5]
    param_diffs = []

    for pname in param_names:
        orig_vals = np.array([float(r[pname]) for r in orig_rows[:n_compare]])
        valid_vals = np.array([float(r[pname]) for r in valid_rows[:n_compare]])
        rel_diff = np.abs(orig_vals - valid_vals) / (np.abs(orig_vals) + 1e-12)
        param_diffs.append(np.max(rel_diff))

    max_param_diff = max(param_diffs) if param_diffs else 0.0

    # Determine repeatability
    cost_repeatable = max_diff < 1e-6
    param_repeatable = max_param_diff < 1e-6

    status = "✓" if (cost_repeatable and param_repeatable) else "✗"

    print(f"{status} {label}:")
    print(f"    Rows compared: {n_compare}")
    print(f"    Cost max diff: {max_diff:.2e} (mean: {mean_diff:.2e})")
    print(f"    Param max diff: {max_param_diff:.2e} (spot check: {', '.join(param_names)})")

    if not cost_repeatable or not param_repeatable:
        print(f"    ⚠ NOT REPEATABLE - check random seed / X0 / simulation determinism")
        return False

    return True

if __name__ == "__main__":
    runs = [
        ("results/20260225T122342_flat_10_30_50", sys.argv[1], "Flat f10/f30/f50"),
        ("results/20260225T003517_flat_16dim_corrected_warm", sys.argv[2], "Flat 16-dim all freqs"),
        ("results/20260224T175707_step_16dim_v1", sys.argv[3], "Step 16-dim"),
    ]

    print()
    all_repeatable = True
    for orig, valid, label in runs:
        repeatable = compare_runs(orig, valid, label)
        all_repeatable = all_repeatable and repeatable

    print()
    if all_repeatable:
        print("✓ ALL RUNS REPEATABLE")
    else:
        print("✗ SOME RUNS NOT REPEATABLE - investigate confounding factors")
        print()
        print("Common causes:")
        print("  - X0 (warm-start) differs between runs")
        print("  - Random seed (OPTIMIZER_RANDOM_STATE) changed")
        print("  - Simulation jitter seeds changed")
        print("  - config.py modified between original and validation run")

    sys.exit(0 if all_repeatable else 1)
PYEOF

python /tmp/compare_csvs.py "$VALIDATE_DIR_1" "$VALIDATE_DIR_2" "$VALIDATE_DIR_3"

echo
echo "========================================================================"
echo "X0 CONFOUNDING CHECK"
echo "========================================================================"
echo
echo "Original run X0 configurations:"
echo

echo "[1] Flat f10/f30/f50:"
grep -A 15 "^CMAES_X0:" results/20260225T122342_flat_10_30_50/config.py | head -16

echo
echo "[2] Flat 16-dim all freqs:"
grep -A 15 "^CMAES_X0:" results/20260225T003517_flat_16dim_corrected_warm/config.py | head -16

echo
echo "[3] Step 16-dim:"
grep -A 15 "^CMAES_X0:" results/20260224T175707_step_16dim_v1/config_step_new.py 2>/dev/null | head -16 \
    || echo "(config_step_new.py not found in results dir)"

echo
echo "Validation run X0 configurations:"
echo
echo "[1] Flat f10/f30/f50:"
grep -A 15 "^CMAES_X0:" "$VALIDATE_DIR_1/config.py" | head -16

echo
echo "[2] Flat 16-dim all freqs:"
grep -A 15 "^CMAES_X0:" "$VALIDATE_DIR_2/config.py" | head -16

echo
echo "[3] Step 16-dim:"
grep -A 15 "^CMAES_X0:" "$VALIDATE_DIR_3/config_step.py" 2>/dev/null | head -16 \
    || grep -A 15 "^CMAES_X0:" "$VALIDATE_DIR_3/config.py" | head -16

echo
echo "========================================================================"
echo "VALIDATION COMPLETE"
echo "========================================================================"
echo
echo "Results directories:"
echo "  Original runs:"
echo "    1. results/20260225T122342_flat_10_30_50"
echo "    2. results/20260225T003517_flat_16dim_corrected_warm"
echo "    3. results/20260224T175707_step_16dim_v1"
echo
echo "  Validation runs:"
echo "    1. $VALIDATE_DIR_1"
echo "    2. $VALIDATE_DIR_2"
echo "    3. $VALIDATE_DIR_3"
echo
