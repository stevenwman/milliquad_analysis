#!/usr/bin/env python3
"""Cross-terrain generalization test.

Takes best params from multiple runs, evaluates each on all 3 terrains.
Produces the cross-terrain error matrix (like PARAM_TRIFECTA_ANALYSIS.md).

Usage:
    uv run python -m analysis.cross_terrain_eval \
        results/20260228T013353_rk4_flat \
        results/20260228T093833_rk4_step_cold \
        results/20260228T102010_rk4_rough
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

import mujoco
import numpy as np

from analysis._common import (
    load_best_point,
    detect_terrain,
    extract_velocity,
    SETTLE_TIME,
)
from config import sim_params_from_point, reference_rows


def _eval_on_terrain(
    sim_params_base: dict,
    eval_terrain: str,
    sim_module,
    run_label: str = "",
) -> dict[str, float | str]:
    """Evaluate params on a single terrain. Returns dict of per-ref velocity errors."""
    config_mod = importlib.import_module(f"config_{eval_terrain}")
    MJCF_PATHS = config_mod.MJCF_PATHS
    SIM_DURATION = config_mod.SIM_DURATION
    ref_rows = list(reference_rows(config_mod.REFERENCE_DATA))
    active_refs = [r for r in ref_rows if r["speed"] > 1e-9 and r["scene"] in MJCF_PATHS]

    step_start_x = getattr(config_mod, "STEP_START_X", None)
    step_end_x = getattr(config_mod, "STEP_END_X", None)

    extra_kwargs = {}
    if eval_terrain.startswith("rough"):
        extra_kwargs["spawn_offset"] = (
            config_mod.SPAWN_X, 0.0, config_mod.SPAWN_Z_RAISE
        )

    prefix = f"  [{run_label} → {eval_terrain}]" if run_label else f"  [{eval_terrain}]"
    errors = []
    n_crash = 0

    for i, ref_row in enumerate(active_refs):
        scene = ref_row["scene"]
        target = ref_row["speed"]
        rid = ref_row.get("id", f"{scene}_f{ref_row.get('ctrl_freq', '?')}")
        freq = ref_row.get("ctrl_freq", 10.0)

        print(f"{prefix} ({i+1}/{len(active_refs)}) {rid}  f={freq:.0f}Hz  target={target*100:.1f}cm/s ...",
              end="", flush=True, file=sys.stderr)

        sp = dict(sim_params_base)
        sp["drive_freq"] = freq

        try:
            traj = sim_module.run_simulation(
                sp,
                mjcf_path=MJCF_PATHS[scene],
                sim_duration=SIM_DURATION,
                visualize=False,
                progress=False,
                ignore_stuck_detection=True,
                **extra_kwargs,
            )
        except Exception as e:
            n_crash += 1
            print(f" CRASH ({e.__class__.__name__})", file=sys.stderr)
            continue

        if traj is None:
            n_crash += 1
            print(" CRASH (None)", file=sys.stderr)
            continue

        vx = extract_velocity(traj, SETTLE_TIME, step_start_x, step_end_x)
        err = abs(vx - target) / target * 100
        errors.append(err)
        print(f" vx={vx*100:.1f}cm/s  err={err:.1f}%", file=sys.stderr)

    n_refs = len(active_refs)

    if n_crash > 0 and len(errors) < n_refs // 2:
        return {"mean_err": float("inf"), "label": "CRASH", "n_ok": len(errors), "n_refs": n_refs}

    if not errors:
        return {"mean_err": float("inf"), "label": "FAIL", "n_ok": 0, "n_refs": n_refs}

    return {
        "mean_err": float(np.mean(errors)),
        "label": f"{np.mean(errors):.1f}%",
        "n_ok": len(errors),
        "n_refs": n_refs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument("--eval-terrains", nargs="+", default=["flat", "step", "rough"])
    args = parser.parse_args()

    import simulation as sim_module

    # Collect runs
    runs = []
    for d in sorted(args.run_dirs):
        if not d.is_dir():
            continue
        terrain = detect_terrain(d)
        point = load_best_point(d)
        sim_params = sim_params_from_point(point)
        # Short label
        cold = "cold" in d.name
        label = f"{terrain}_{'C' if cold else 'W'}"
        runs.append({"dir": d, "terrain": terrain, "label": label, "sim_params": sim_params})

    if not runs:
        print("No valid runs found.")
        sys.exit(1)

    eval_terrains = args.eval_terrains

    # Header
    label_w = max(12, max(len(r["label"]) for r in runs) + 1)
    col_w = 12
    header = f"{'Params from':<{label_w}}" + "".join(f"{t:>{col_w}}" for t in eval_terrains)
    print("\nCross-Terrain Generalization (mean velocity error %)\n")
    print(header)
    print("-" * len(header))

    # Evaluate each run on each terrain
    for run in runs:
        parts = []
        for et in eval_terrains:
            result = _eval_on_terrain(run["sim_params"], et, sim_module, run_label=run["label"])
            native = (et == run["terrain"])
            cell = result["label"]
            if native:
                cell += "*"
            parts.append(f"{cell:>{col_w}}")
            print(f"  → {run['label']} on {et}: {result['label']} ({result['n_ok']}/{result['n_refs']} refs)\n",
                  file=sys.stderr)
        print(f"{run['label']:<{label_w}}" + "".join(parts))

    print(f"\n* = native terrain (optimized for this)")


if __name__ == "__main__":
    main()
