"""
Replay an optimization run in the MuJoCo passive viewer.

Reads a row from the results CSV by run ID (or picks the best),
reconstructs sim params, and launches the viewer for each scene.

Usage:
    uv run python replay.py                  # replay best (lowest cost)
    uv run python replay.py f37ba9db         # replay specific run ID
    uv run python replay.py f37ba9db scene4  # replay one scene only
"""

import csv
import sys

from config import (
    CSV_PATH,
    DEFAULT_CTRL_FREQ,
    MJCF_PATHS,
    SIM_DURATION,
    reference_rows,
    sim_params_from_point,
    space,
)
import importlib

# ---- Simulation module selector ----
# Switch between vectorized (4.68x faster) and original (bit-exact) simulation.
# Hot-swap: change to "simulation" to use original implementation.
SIM_MODULE = "simulation_fast"
_sim = importlib.import_module(SIM_MODULE)


def _load_csv(csv_path: str = CSV_PATH) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _find_row(rows: list[dict], run_id: str | None) -> dict:
    if run_id is None:
        best = min(rows, key=lambda r: float(r["cost"]))
        print(f"No ID specified — using best row: id={best['id']} cost={best['cost']}")
        return best
    for r in rows:
        if r["id"] == run_id:
            return r
    # Allow prefix match
    matches = [r for r in rows if r["id"].startswith(run_id)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(m["id"] for m in matches)
        print(f"Ambiguous prefix '{run_id}' matches: {ids}")
        sys.exit(1)
    print(f"Run ID '{run_id}' not found in {CSV_PATH}")
    sys.exit(1)


def _row_to_point(row: dict) -> list[float]:
    """Extract optimizer point (list in space order) from a CSV row."""
    return [float(row[dim.name]) for dim in space]


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    scene_filter = sys.argv[2] if len(sys.argv) > 2 else None

    rows = _load_csv()
    if not rows:
        print(f"CSV {CSV_PATH} is empty.")
        sys.exit(1)

    row = _find_row(rows, run_id)
    point = _row_to_point(row)
    sim_params = sim_params_from_point(point)

    print(f"Replaying id={row['id']} (cost={row['cost']})")
    print(f"Params: { {dim.name: float(row[dim.name]) for dim in space} }")

    ref_rows = reference_rows()
    if scene_filter:
        ref_rows = [r for r in ref_rows if r["scene"] == scene_filter]
        if not ref_rows:
            print(f"No reference rows match scene '{scene_filter}'. Available: {list(MJCF_PATHS.keys())}")
            sys.exit(1)

    for ref in ref_rows:
        scene = ref["scene"]
        mjcf_path = MJCF_PATHS[scene]
        freq = ref.get("ctrl_freq", DEFAULT_CTRL_FREQ)

        print(f"\n--- Launching viewer: {scene} (freq={freq} Hz, target={ref['speed']} m/s) ---")
        print("  Press SPACE to unpause.")

        sp = dict(sim_params)
        sp["drive_freq"] = freq
        _sim.run_simulation(
            sp,
            mjcf_path=mjcf_path,
            sim_duration=SIM_DURATION,
            visualize=True,
        )


if __name__ == "__main__":
    main()
