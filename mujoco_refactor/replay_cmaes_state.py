"""Reconstruct CMA-ES state from a completed run's CSV history.

Reads multi_optimization_results.csv, replays the ask/tell sequence
through a fresh CMA-ES instance (same seed/config), and dumps
cmaes_state.pkl into the run directory for use with --resume-from.

Usage:
    uv run python replay_cmaes_state.py results/20260218T223844
"""

import csv
import pathlib
import pickle
import sys

import numpy as np

# Use the saved config from the run directory (validates consistency)
from config import (
    BATCH_SIZE,
    CMAES_SIGMA0,
    CMAES_X0,
    OPTIMIZER_RANDOM_STATE,
    space,
)
from optimizer import _cmaes_space_info, _cmaes_to_real


def main():
    if len(sys.argv) < 2:
        print("Usage: replay_cmaes_state.py <results_dir>")
        sys.exit(1)

    run_dir = pathlib.Path(sys.argv[1])
    csv_path = run_dir / "multi_optimization_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # Read all evaluated points and costs from the CSV
    param_names = [d.name for d in space]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} evaluations from {csv_path}")

    if len(rows) % BATCH_SIZE != 0:
        print(f"WARNING: {len(rows)} rows is not a multiple of BATCH_SIZE={BATCH_SIZE}")
        print(f"  Truncating to {(len(rows) // BATCH_SIZE) * BATCH_SIZE} rows")

    n_batches = len(rows) // BATCH_SIZE

    # Create a fresh CMA-ES with the same initial conditions
    import cma

    x0, lower, upper, is_log = _cmaes_space_info()
    opts = {
        "bounds": [lower, upper],
        "seed": OPTIMIZER_RANDOM_STATE,
        "popsize": BATCH_SIZE,
        "verbose": -1,
        "tolfun": 1e-8,
        "tolx": 1e-10,
    }
    es = cma.CMAEvolutionStrategy(x0, CMAES_SIGMA0, opts)

    # Replay: for each batch, call ask() (to advance RNG), then tell() with historical costs
    for batch_idx in range(n_batches):
        batch_rows = rows[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        # ask() to keep RNG in sync — we discard the points (they match the CSV)
        asked_points = es.ask()

        # Extract costs from CSV rows
        costs = [float(r["cost"]) for r in batch_rows]

        # Extract params from CSV and convert to internal space for tell()
        internal_points = []
        for r in batch_rows:
            internal = []
            for dim, log_flag in zip(space, is_log):
                val = float(r[dim.name])
                internal.append(np.log10(val) if log_flag else val)
            internal_points.append(internal)

        es.tell(internal_points, costs)

        if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
            n_done = (batch_idx + 1) * BATCH_SIZE
            print(f"  Replayed {n_done}/{n_batches * BATCH_SIZE} evals, sigma={es.sigma:.4g}")

    n_done = n_batches * BATCH_SIZE
    space_bounds = [(d.name, d.low, d.high, d.prior) for d in space]
    state_path = run_dir / "cmaes_state.pkl"
    with open(state_path, "wb") as f:
        pickle.dump({"es": es, "n_done": n_done, "space_bounds": space_bounds}, f)

    print(f"\nSaved CMA-ES state to {state_path}")
    print(f"  n_done={n_done}, sigma={es.sigma:.4g}")
    print(f"\nResume with:")
    print(f"  uv run python optimizer.py --resume-from {run_dir}")


if __name__ == "__main__":
    main()
