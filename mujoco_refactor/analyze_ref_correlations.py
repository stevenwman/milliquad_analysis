#!/usr/bin/env python3
"""Analyze correlations between reference conditions to identify redundant ones."""

import csv
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Load optimization results to analyze
RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/20260222T181114_with_20hz_no-deadzone"

csv_path = Path(RUN_DIR) / "multi_optimization_results.csv"
if not csv_path.exists():
    print(f"Error: {csv_path} not found")
    sys.exit(1)

print("=" * 80)
print("REFERENCE CONDITION CORRELATION ANALYSIS")
print("=" * 80)
print(f"Source: {RUN_DIR}")
print()

# Load CSV
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} evaluations")

# Find all per-reference cost columns
# They should be named like: cost_scene1_f10, cost_scene2_f20, etc.
cost_cols = [col for col in df.columns if col.startswith('cost_') and '_f' in col]

if not cost_cols:
    # Try alternative naming: might be individual reference IDs as columns
    # Check for columns with numeric values that look like costs
    potential_cost_cols = [col for col in df.columns if col not in ['n_eval', 'timestamp', 'elapsed_min', 'hash', 'cost']]

    if potential_cost_cols:
        # Check first few rows to see if they're numeric costs
        sample = df[potential_cost_cols].head()
        cost_cols = [col for col in potential_cost_cols if pd.api.types.is_numeric_dtype(df[col])]

if not cost_cols:
    print("Error: Could not find per-reference cost columns in CSV")
    print(f"Available columns: {list(df.columns)[:20]}")
    sys.exit(1)

print(f"Found {len(cost_cols)} per-reference cost columns:")
for col in sorted(cost_cols):
    print(f"  - {col}")
print()

# Extract per-reference costs
costs_df = df[cost_cols].copy()

# Remove rows with NaN or inf (failed evaluations)
costs_df = costs_df.replace([np.inf, -np.inf], np.nan).dropna()

# Also remove rows where ANY cost > 1000 (simulation failures)
failed_mask = (costs_df > 1000).any(axis=1)
costs_df = costs_df[~failed_mask]

print(f"Using {len(costs_df)} valid evaluations (after removing NaN/inf/failures)")
print()

# Compute correlation matrix
corr_matrix = costs_df.corr()

print("=" * 80)
print("CORRELATION MATRIX")
print("=" * 80)
print(corr_matrix.round(3))
print()

# Find highly correlated pairs (correlation > 0.85)
HIGH_CORR_THRESHOLD = 0.85
high_corr_pairs = []

for i, col1 in enumerate(cost_cols):
    for j, col2 in enumerate(cost_cols):
        if i < j:  # Only upper triangle (avoid duplicates)
            corr = corr_matrix.loc[col1, col2]
            if corr > HIGH_CORR_THRESHOLD:
                high_corr_pairs.append((col1, col2, corr))

# Sort by correlation
high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

print("=" * 80)
print(f"HIGHLY CORRELATED PAIRS (correlation > {HIGH_CORR_THRESHOLD})")
print("=" * 80)

if high_corr_pairs:
    for col1, col2, corr in high_corr_pairs:
        print(f"{corr:.3f}  {col1:<30} <-> {col2}")
    print()
    print("Recommendation: Consider dropping one condition from each highly correlated pair")
else:
    print("No highly correlated pairs found - all conditions provide unique information")

print()

# Compute average correlation for each reference (how redundant is it?)
avg_corr = corr_matrix.mean().sort_values(ascending=False)

print("=" * 80)
print("AVERAGE CORRELATION PER REFERENCE (redundancy score)")
print("=" * 80)
print("Higher = more redundant with other conditions")
print()

for ref, corr in avg_corr.items():
    redundancy = "HIGH" if corr > 0.7 else ("MEDIUM" if corr > 0.5 else "LOW")
    print(f"{ref:<35} {corr:.3f}  [{redundancy}]")

print()
print("=" * 80)
print("SUGGESTED MINIMAL REFERENCE SET")
print("=" * 80)

# Greedy selection: pick references that maximize coverage
# Start with the least redundant reference, then add references that are least correlated with selected set
selected_refs = []
remaining_refs = list(cost_cols)

# Start with least redundant
first_ref = avg_corr.idxmin()
selected_refs.append(first_ref)
remaining_refs.remove(first_ref)

# Iteratively add references with lowest average correlation to selected set
while remaining_refs and len(selected_refs) < len(cost_cols):
    # Compute average correlation of each remaining ref with selected refs
    avg_corr_with_selected = {}
    for ref in remaining_refs:
        avg_corr_with_selected[ref] = corr_matrix.loc[ref, selected_refs].mean()

    # Pick reference with lowest correlation to selected set
    next_ref = min(avg_corr_with_selected, key=avg_corr_with_selected.get)

    # Stop if correlation is too high (diminishing returns)
    if avg_corr_with_selected[next_ref] > 0.9:
        break

    selected_refs.append(next_ref)
    remaining_refs.remove(next_ref)

print(f"Minimal set ({len(selected_refs)} references, down from {len(cost_cols)}):")
for i, ref in enumerate(selected_refs, 1):
    print(f"  {i}. {ref}")

print()
print(f"Computational savings: {(1 - len(selected_refs)/len(cost_cols))*100:.1f}%")
print()

# Dropped references
dropped_refs = [ref for ref in cost_cols if ref not in selected_refs]
if dropped_refs:
    print(f"Dropped references ({len(dropped_refs)}):")
    for ref in dropped_refs:
        # Show which reference it's most correlated with
        max_corr_ref = corr_matrix.loc[ref, selected_refs].idxmax()
        max_corr = corr_matrix.loc[ref, max_corr_ref]
        print(f"  - {ref:<35} (corr={max_corr:.3f} with {max_corr_ref})")
