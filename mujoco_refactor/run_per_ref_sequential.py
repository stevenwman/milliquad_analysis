#!/usr/bin/env python3
"""Run one CMA-ES optimization per reference, sequentially.

Each run gets all CPU cores. Sequential order ensures no resource contention
and maximum convergence quality per run.

Usage:
    cd mujoco_refactor
    uv run python run_per_ref_sequential.py
    uv run python run_per_ref_sequential.py --n-calls 4800
    uv run python run_per_ref_sequential.py --mode step --n-calls 2400
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["step", "multi"], default="multi",
                    help="step: 12 step refs; multi: 19 flat+step refs (default: multi)")
parser.add_argument("--n-calls", type=int, default=2400,
                    help="CMA-ES evaluations per ref (default: 2400)")
parser.add_argument("--sweep-name", type=str, default=None,
                    help="Sweep folder name under results/ (default: per_ref_sweep_TIMESTAMP)")
args = parser.parse_args()

if args.mode == "step":
    from config_step_new import reference_rows
    optimizer = "optimizer_step.py"
    extra = []
else:
    from config_multi_terrain import reference_rows
    optimizer = "optimizer_multi_terrain.py"
    extra = []

refs = reference_rows()

_ts = datetime.now().strftime('%Y%m%dT%H%M%S')
sweep_name = f"{_ts}_{args.sweep_name}" if args.sweep_name else f"{_ts}_per_ref_{args.mode}"
sweep_dir = Path("results") / sweep_name
sweep_dir.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = sweep_dir / "summary.csv"
SUMMARY_FIELDS = ["ref_id", "terrain", "target_speed", "best_cost", "best_vel", "delta_pct", "elapsed_min", "status"]

with open(SUMMARY_CSV, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()

print("=" * 70)
print(f"Sequential per-ref optimizer  [mode={args.mode}]")
print(f"  {len(refs)} refs × {args.n_calls} evals each")
print(f"  Optimizer: {optimizer}")
print(f"  Sweep dir: {sweep_dir}/")
print("=" * 70)
print()


def _read_best(run_dir: Path, ref_id: str) -> dict | None:
    """Read the last (best) row from optimization_bests.csv."""
    bests_path = run_dir / "optimization_bests.csv"
    if not bests_path.exists():
        return None
    try:
        with open(bests_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return rows[-1]
    except Exception:
        return None


def _print_summary_table(summary_rows: list[dict]) -> None:
    """Print the accumulated summary table."""
    header = f"{'ref_id':<26} {'terrain':<6} {'target':>7} {'best_vel':>9} {'Δ%':>6} {'cost':>8} {'min':>6}  status"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        delta_str = f"{r['delta_pct']:>+.0f}%" if r["delta_pct"] is not None else "  n/a"
        vel_str = f"{r['best_vel']:.3f}" if r["best_vel"] is not None else "  n/a"
        cost_str = f"{r['best_cost']:.4f}" if r["best_cost"] is not None else "  n/a"
        print(
            f"{r['ref_id']:<26} {r['terrain']:<6} {r['target_speed']:>7.4f}"
            f" {vel_str:>9} {delta_str:>6} {cost_str:>8} {r['elapsed_min']:>6.1f}  {r['status']}"
        )
    print()


t_total_start = time.time()
summary_rows = []

for i, row in enumerate(refs):
    ref_id = row["id"]
    terrain = row.get("terrain", "step")
    target = row["speed"]
    print(f"[{i+1}/{len(refs)}] Starting {ref_id}  ({terrain}, target={target:.4f} m/s)")

    run_dir = sweep_dir / ref_id
    cmd = [sys.executable, optimizer,
           "--scenes", row["scene"],
           "--freqs", str(row["ctrl_freq"]),
           "--n-calls", str(args.n_calls),
           "--run-dir", str(run_dir),
           ] + extra
    if args.mode == "multi":
        cmd += ["--terrain", terrain]

    t0 = time.time()
    env = {**os.environ, "MUJOCO_GL": "egl"}
    rc = subprocess.run(cmd, env=env).returncode
    elapsed = (time.time() - t0) / 60.0
    total_elapsed = (time.time() - t_total_start) / 60.0

    # Read best result from completed run
    best_row = _read_best(run_dir, ref_id) if rc == 0 else None
    best_cost = float(best_row["cost"]) if best_row else None
    best_vel = float(best_row.get(f"vel_{ref_id}", 0)) if best_row else None
    delta_pct = ((best_vel - target) / target * 100) if (best_vel is not None and target != 0) else None

    status = "OK" if rc == 0 else f"FAIL(rc={rc})"

    # Append to summary CSV
    summary_row = {
        "ref_id": ref_id,
        "terrain": terrain,
        "target_speed": target,
        "best_cost": best_cost,
        "best_vel": best_vel,
        "delta_pct": delta_pct,
        "elapsed_min": elapsed,
        "status": status,
    }
    summary_rows.append(summary_row)
    with open(SUMMARY_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow(summary_row)

    # Print running summary table
    print(f"  → {status}  took {elapsed:.1f}min  total elapsed {total_elapsed:.1f}min")
    remaining = len(refs) - (i + 1)
    if remaining > 0:
        eta = total_elapsed / (i + 1) * remaining
        print(f"  → ~{eta:.0f}min remaining ({remaining} refs left)")
    print()
    _print_summary_table(summary_rows)

total_min = (time.time() - t_total_start) / 60.0
ok = sum(1 for r in summary_rows if r["status"] == "OK")
print("=" * 70)
print(f"DONE  {ok}/{len(refs)} succeeded  in {total_min:.1f} min")
print(f"Summary saved to: {SUMMARY_CSV}")
failed = [r["ref_id"] for r in summary_rows if r["status"] != "OK"]
if failed:
    print(f"Failed: {failed}")
print("=" * 70)
