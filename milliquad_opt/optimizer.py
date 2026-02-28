"""
Unified CMA-ES optimizer for flat, step, and rough terrain.

Usage:
    uv run python optimizer.py --terrain flat --suffix rk4_flat
    uv run python optimizer.py --terrain step --suffix rk4_step
    uv run python optimizer.py --terrain rough --suffix rk4_rough
"""

import csv
import importlib
import multiprocessing
import os
import pathlib
import pickle
import shutil
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np

from config import (
    COST_FAILURE,
    DEFAULT_CTRL_FREQ,
    SIMULATION_TIMEOUT,
    point_to_params,
    sim_params_from_point,
    space,
    reference_rows,
    reference_ids,
    csv_fieldnames,
)


class OptResult(NamedTuple):
    fun: float
    x: list[float]


# ---------------------------------------------------------------------------
# Globals set at startup from terrain config
# ---------------------------------------------------------------------------
_REF_ROWS: list[dict] = []
_REF_INDEX_BY_ID: dict[str, int] = {}

# These are set from the terrain config module in main
MJCF_PATHS: dict[str, str] = {}
_calculate_cost = None  # terrain-specific cost function
_TERRAIN: str = ""

# Terrain config values (set in main)
SIM_DURATION: float = 3.0
N_CALLS: int = 4800
BATCH_SIZE: int = 16
CMAES_X0: dict[str, float] | None = None
CMAES_SIGMA0: float = 0.3
OPTIMIZER_RANDOM_STATE: int = 69420
VELOCITY_VARIANCE_WEIGHT: float = 2.0
JITTER_AGGREGATION: str = "median"
CSV_PATH: str = ""
BEST_CSV_PATH: str = ""

# Flat/step jitter params
INIT_YAW_JITTER_DEG: float = 0.0
INIT_JITTER_TRIALS: int = 3
INIT_JITTER_SEED: int = 12345

# Rough jitter params
Y_JITTER: float = 0.0
Y_JITTER_SEED: int = 77777
SPAWN_X: float = 0.0
SPAWN_Z_RAISE: float = 0.01

# Extra CSV columns (e.g. "progress" for step terrain)
_EXTRA_CSV_COLS: list[str] = []


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

def _evaluate_one_scene(args):
    """Run one trial for one (point, reference row, trial_index).

    All needed state is passed via the task tuple to survive multiprocessing spawn.
    """
    (point_index, point, ref_row, trial_index, show_progress,
     global_point_index, mjcf_path, task_cfg) = args

    _sim = importlib.import_module("simulation")
    from config import sim_params_from_point as _spp

    terrain = task_cfg["terrain"]
    sim_duration = task_cfg["sim_duration"]
    n_refs = task_cfg["n_refs"]
    ref_idx = task_cfg["ref_idx"]
    n_trials = task_cfg["n_trials"]

    # Load terrain-specific cost function in worker
    cost_mod = importlib.import_module(f"config_{terrain}")
    calc_cost = cost_mod.calculate_cost

    sim_params = _spp(point)
    scene_name = ref_row["scene"]
    target_velocity = ref_row["speed"]
    speed_std = ref_row.get("speed_std", 0.0)
    weight = ref_row.get("weight", 1.0)
    sim_params["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)

    t0 = time.perf_counter()

    if terrain.startswith("rough"):
        y_jitter_seed = task_cfg["y_jitter_seed"]
        y_jitter = task_cfg["y_jitter"]
        spawn_x = task_cfg["spawn_x"]
        spawn_z = task_cfg["spawn_z"]
        seed = y_jitter_seed + (global_point_index * n_refs + ref_idx) * n_trials + trial_index
        rng = np.random.default_rng(seed)
        y_offset = rng.uniform(-y_jitter, y_jitter)
        spawn_offset = (spawn_x, y_offset, spawn_z)
        try:
            trajectory = _sim.run_simulation(
                sim_params,
                mjcf_path=mjcf_path,
                sim_duration=sim_duration,
                visualize=False,
                progress=show_progress,
                wall_timeout=SIMULATION_TIMEOUT,
                spawn_offset=spawn_offset,
                ignore_stuck_detection=True,
            )
        except Exception as e:
            print(f"  [WARN] Sim crashed ({ref_row['id']} trial {trial_index}): {e}", flush=True)
            trajectory = None
    else:
        jitter_seed = task_cfg["jitter_seed"]
        yaw_jitter_deg = task_cfg["yaw_jitter_deg"]
        seed = jitter_seed + (global_point_index * n_refs + ref_idx) * n_trials + trial_index
        trajectory = _sim.run_simulation(
            sim_params,
            mjcf_path=mjcf_path,
            sim_duration=sim_duration,
            visualize=False,
            progress=show_progress,
            wall_timeout=SIMULATION_TIMEOUT,
            init_yaw_jitter_deg=yaw_jitter_deg,
            rng_seed=seed,
        )

    if trajectory is None:
        cost_data = {
            "total_cost": COST_FAILURE,
            "avg_forward_velocity": 0.0,
            "tumble_penalty": 0.0,
            "lateral_displacement": 0.0,
            "yaw_deviation_deg": 0.0,
            "progress_penalty": 0.0,
        }
    else:
        cost_data = calc_cost(
            trajectory,
            target_velocity,
            speed_std=speed_std,
            verbose=False,
        )

    wall_time = time.perf_counter() - t0

    return (
        point_index,
        ref_row["id"],
        scene_name,
        cost_data["total_cost"],
        cost_data["avg_forward_velocity"],
        cost_data.get("tumble_penalty", 0.0),
        cost_data.get("lateral_displacement", 0.0),
        cost_data.get("yaw_deviation_deg", 0.0),
        cost_data.get("progress_penalty", 0.0),
        weight,
        wall_time,
    )


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def _aggregate_scene_results(points: list, scene_results: list) -> list[dict]:
    """Turn per-trial results into full result dicts (one per point)."""
    by_point = defaultdict(
        lambda: {
            "ref_trials_costs": defaultdict(list),
            "ref_trials_velocities": defaultdict(list),
            "ref_trials_tumble": defaultdict(list),
            "ref_trials_lateral": defaultdict(list),
            "ref_trials_yaw": defaultdict(list),
            "ref_trials_progress": defaultdict(list),
            "ref_weights": {},
            "ref_scene": {},
            "scene_costs": defaultdict(float),
            "scene_vel_num": defaultdict(float),
            "scene_weight": defaultdict(float),
            "scene_wall_times": [],
            "has_failure": False,
        }
    )

    for result_tuple in scene_results:
        (point_index, ref_id, scene_name, cost, velocity, tumble,
         lateral, yaw_deg, progress, weight, wall_time) = result_tuple
        d = by_point[point_index]
        d["ref_trials_costs"][ref_id].append(cost)
        d["ref_trials_velocities"][ref_id].append(velocity)
        d["ref_trials_tumble"][ref_id].append(tumble)
        d["ref_trials_lateral"][ref_id].append(lateral)
        d["ref_trials_yaw"][ref_id].append(yaw_deg)
        d["ref_trials_progress"][ref_id].append(progress)
        if cost >= COST_FAILURE:
            d["has_failure"] = True
        d["ref_weights"][ref_id] = weight
        d["ref_scene"][ref_id] = scene_name
        d["scene_wall_times"].append(wall_time)

    results = []
    for point_index in sorted(by_point):
        d = by_point[point_index]
        params = point_to_params(points[point_index])
        ref_costs = {}
        ref_avg_velocities = {}
        ref_tumble = {}
        ref_lateral = {}
        ref_yaw = {}
        ref_progress = {}
        ref_best_trial = {}

        for ref_id, trials in d["ref_trials_costs"].items():
            scene = d["ref_scene"][ref_id]
            weight = d["ref_weights"][ref_id]

            if JITTER_AGGREGATION == "best":
                best_idx = int(np.argmin(trials))
                agg_cost = float(trials[best_idx])
                agg_vel = float(d["ref_trials_velocities"][ref_id][best_idx])
                agg_tumble = float(d["ref_trials_tumble"][ref_id][best_idx])
                agg_lateral = float(d["ref_trials_lateral"][ref_id][best_idx])
                agg_yaw = float(d["ref_trials_yaw"][ref_id][best_idx])
                agg_progress = float(d["ref_trials_progress"][ref_id][best_idx])
                ref_best_trial[ref_id] = best_idx
            else:
                agg_cost = float(np.median(trials))
                agg_vel = float(np.median(d["ref_trials_velocities"][ref_id]))
                agg_tumble = float(np.median(d["ref_trials_tumble"][ref_id]))
                agg_lateral = float(np.median(d["ref_trials_lateral"][ref_id]))
                agg_yaw = float(np.median(d["ref_trials_yaw"][ref_id]))
                agg_progress = float(np.median(d["ref_trials_progress"][ref_id]))
                ref_best_trial[ref_id] = -1

            ref_costs[ref_id] = agg_cost
            ref_avg_velocities[ref_id] = agg_vel
            ref_tumble[ref_id] = agg_tumble
            ref_lateral[ref_id] = agg_lateral
            ref_yaw[ref_id] = agg_yaw
            ref_progress[ref_id] = agg_progress

            d["scene_costs"][scene] += weight * agg_cost
            d["scene_vel_num"][scene] += weight * agg_vel
            d["scene_weight"][scene] += weight

        scene_avg_velocities = {
            s: (d["scene_vel_num"][s] / d["scene_weight"][s] if d["scene_weight"][s] > 0 else 0.0)
            for s in MJCF_PATHS
        }
        scene_costs = dict(d["scene_costs"])

        if d["has_failure"]:
            total_cost = COST_FAILURE
        else:
            total_cost = sum(scene_costs.values())
            if len(ref_avg_velocities) > 1:
                targets_by_id = {row["id"]: row["speed"] for row in _REF_ROWS}
                rel_errors = [
                    (ref_avg_velocities[rid] - targets_by_id[rid]) / targets_by_id[rid]
                    for rid in ref_avg_velocities
                    if targets_by_id.get(rid, 0) > 0
                ]
                if len(rel_errors) > 1:
                    total_cost += VELOCITY_VARIANCE_WEIGHT * float(np.var(rel_errors))

        point_wall = max(d["scene_wall_times"]) if d["scene_wall_times"] else 0.0
        results.append({
            "id": str(uuid.uuid4().hex)[:8],
            "cost": total_cost,
            "params": params,
            "scene_costs": scene_costs,
            "scene_avg_velocities": scene_avg_velocities,
            "ref_costs": ref_costs,
            "ref_avg_velocities": ref_avg_velocities,
            "ref_tumble": ref_tumble,
            "ref_lateral": ref_lateral,
            "ref_yaw": ref_yaw,
            "ref_progress": ref_progress,
            "ref_best_trial": ref_best_trial,
            "ref_weights": d["ref_weights"],
            "ref_scene": d["ref_scene"],
            "wall_time": point_wall,
        })
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _get_fieldnames():
    return csv_fieldnames(_REF_ROWS, MJCF_PATHS, extra_per_ref=_EXTRA_CSV_COLS)


def _append_result_to_csv(res: dict, elapsed_min: float) -> None:
    row = {"id": res["id"], "cost": res["cost"], "elapsed_min": f"{elapsed_min:.1f}"}
    for scene in MJCF_PATHS:
        row[f"velocity_{scene}"] = res["scene_avg_velocities"].get(scene, 0)
        row[f"cost_{scene}"] = res["scene_costs"].get(scene, 0)
    for rid in [r["id"] for r in _REF_ROWS]:
        row[f"velocity_{rid}"] = res["ref_avg_velocities"].get(rid, 0)
        row[f"cost_{rid}"] = res["ref_costs"].get(rid, 0)
        row[f"lateral_{rid}"] = res.get("ref_lateral", {}).get(rid, 0)
        row[f"tumble_{rid}"] = res.get("ref_tumble", {}).get(rid, 0)
        row[f"yaw_{rid}"] = res.get("ref_yaw", {}).get(rid, 0)
        for extra in _EXTRA_CSV_COLS:
            row[f"{extra}_{rid}"] = res.get(f"ref_{extra}", {}).get(rid, 0)
    row.update(res["params"])
    p = res["params"]
    row["solimp_dmax"] = p["solimp_dmin"] + p["solimp_delta_d"] * (0.9999 - p["solimp_dmin"])
    try:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_get_fieldnames())
            writer.writerow(row)
    except Exception as e:
        print(f"  [Warning] Could not append to CSV: {e}")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_point_results(results: list[dict], n_this: int) -> None:
    for i, r in enumerate(results):
        wt = r.get("wall_time", 0)
        print(f"    [{i+1}/{n_this}] id={r['id']}  cost={r['cost']:.4f}  time={wt:.1f}s")
        _print_ref_table(r, _REF_ROWS, indent=6)


def _print_ref_table(r: dict, ref_rows: list[dict], indent: int = 4) -> None:
    rv = r.get("ref_avg_velocities", {})
    rt = r.get("ref_tumble", {})
    rl = r.get("ref_lateral", {})
    ry = r.get("ref_yaw", {})
    pad = " " * indent
    print(f"{pad}{'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9} {'Δ%':>5} {'tumble':>7} {'lateral':>8} {'yaw':>5}")
    print(f"{pad}{'-'*70}")
    for row in ref_rows:
        rid = row["id"]
        target = row["speed"]
        sim_v = rv.get(rid, 0.0)
        delta = (sim_v - target) * 100
        delta_pct = ((sim_v - target) / target * 100) if target != 0 else 0.0
        tmb = rt.get(rid, 0.0)
        lat = rl.get(rid, 0.0) * 100
        yaw = ry.get(rid, 0.0)
        print(f"{pad}{rid:<18} {target:>6.3f}  {sim_v:>6.3f}  {delta:>+7.1f}cs {delta_pct:>+4.0f}%  {tmb:>6.4f}  {lat:>6.1f}cm  {yaw:>4.0f}°")


_best_cost_so_far: float = float("inf")


def _best_csv_fieldnames() -> list[str]:
    ref_ids = [row["id"] for row in _REF_ROWS]
    cols = (
        ["timestamp", "elapsed_min", "n_eval", "id", "cost"]
        + [f"vel_{rid}" for rid in ref_ids]
        + [f"lateral_{rid}" for rid in ref_ids]
        + [f"tumble_{rid}" for rid in ref_ids]
        + [f"yaw_{rid}" for rid in ref_ids]
    )
    for extra in _EXTRA_CSV_COLS:
        cols += [f"{extra}_{rid}" for rid in ref_ids]
    cols += [dim.name for dim in space] + ["solimp_dmax"]
    return cols


def _append_best_csv(best: dict, n_done: int, elapsed_min: float) -> None:
    ref_ids = [row["id"] for row in _REF_ROWS]
    rv = best.get("ref_avg_velocities", {})
    rl = best.get("ref_lateral", {})
    rt = best.get("ref_tumble", {})
    ry = best.get("ref_yaw", {})

    with open(BEST_CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_best_csv_fieldnames())
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_min": f"{elapsed_min:.1f}",
            "n_eval": n_done,
            "id": best["id"],
            "cost": f"{best['cost']:.6f}",
        }
        for rid in ref_ids:
            row[f"vel_{rid}"] = f"{rv.get(rid, 0.0):.4f}"
            row[f"lateral_{rid}"] = f"{rl.get(rid, 0.0):.6f}"
            row[f"tumble_{rid}"] = f"{rt.get(rid, 0.0):.6f}"
            row[f"yaw_{rid}"] = f"{ry.get(rid, 0.0):.1f}"
        for extra in _EXTRA_CSV_COLS:
            ref_extra = best.get(f"ref_{extra}", {})
            for rid in ref_ids:
                row[f"{extra}_{rid}"] = f"{ref_extra.get(rid, 0.0):.4f}"
        for dim in space:
            row[dim.name] = float(best['params'][dim.name])
        bp = best["params"]
        row["solimp_dmax"] = float(bp['solimp_dmin'] + bp['solimp_delta_d'] * (0.9999 - bp['solimp_dmin']))
        w.writerow(row)


def _print_best_so_far(all_results: list[dict], n_done: int, elapsed_min: float) -> None:
    global _best_cost_so_far
    best = min(all_results, key=lambda r: r["cost"])
    is_new_best = best["cost"] < _best_cost_so_far
    marker = " ★ NEW BEST" if is_new_best else ""
    print(f"  Best so far (n={n_done}): cost={best['cost']:.6f}  id={best['id']}{marker}")
    _print_ref_table(best, _REF_ROWS, indent=4)

    if is_new_best:
        _best_cost_so_far = best["cost"]
        _append_best_csv(best, n_done, elapsed_min)


# ---------------------------------------------------------------------------
# CMA-ES space mapping
# ---------------------------------------------------------------------------

def _cmaes_space_info():
    x0, lower, upper, is_log = [], [], [], []
    for dim in space:
        lo, hi = dim.low, dim.high
        if dim.prior == "log-uniform":
            is_log.append(True)
            lower.append(np.log10(lo))
            upper.append(np.log10(hi))
            if CMAES_X0 is not None:
                x0.append(np.log10(CMAES_X0[dim.name]))
            else:
                x0.append(0.5 * (np.log10(lo) + np.log10(hi)))
        else:
            is_log.append(False)
            lower.append(lo)
            upper.append(hi)
            if CMAES_X0 is not None:
                x0.append(CMAES_X0[dim.name])
            else:
                x0.append(0.5 * (lo + hi))
    return x0, lower, upper, is_log


def _cmaes_to_real(x_internal, is_log):
    return [10.0 ** v if log else v for v, log in zip(x_internal, is_log)]


def _create_cmaes_optimizer(es_override=None):
    import cma

    _, lower, upper, is_log = _cmaes_space_info()

    if es_override is not None:
        es = es_override
    else:
        x0, _, _, _ = _cmaes_space_info()
        opts = {
            "bounds": [lower, upper],
            "seed": OPTIMIZER_RANDOM_STATE,
            "popsize": BATCH_SIZE,
            "verbose": -1,
            "tolfun": 1e-8,
            "tolx": 1e-10,
        }
        es = cma.CMAEvolutionStrategy(x0, CMAES_SIGMA0, opts)

    def ask(n_points):
        internal_points = es.ask()
        return [_cmaes_to_real(p, is_log) for p in internal_points]

    def tell(points, costs):
        internal_points = []
        for pt in points:
            internal = [np.log10(v) if log else v for v, log in zip(pt, is_log)]
            internal_points.append(internal)
        es.tell(internal_points, costs)

    return ask, tell, es


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def _run_batch_optimization(all_results: list[dict], pool: multiprocessing.Pool,
                            es_resume=None) -> OptResult:
    ask, tell, es = _create_cmaes_optimizer(es_override=es_resume)
    if es_resume is not None:
        print(f"  Backend: CMA-ES RESUMED (sigma={es.sigma:.4g}, popsize={BATCH_SIZE})")
    else:
        warm = "warm-start" if CMAES_X0 is not None else "cold-start"
        print(f"  Backend: CMA-ES (sigma0={CMAES_SIGMA0}, popsize={BATCH_SIZE}, {warm})")

    n_done = 0
    batch_num = 0
    t_run_start = time.perf_counter()

    while n_done < N_CALLS:
        n_this = min(BATCH_SIZE, N_CALLS - n_done)
        batch_num += 1
        n_trials = max(1, INIT_JITTER_TRIALS)
        t_batch_start = time.perf_counter()
        print(f"\n--- Batch {batch_num}: asking for {n_this} points ({n_done + 1}–{n_done + n_this} / {N_CALLS}), {n_this * len(_REF_ROWS) * n_trials} tasks ---")

        t_ask = time.perf_counter()
        points = ask(n_this)
        t_ask = time.perf_counter() - t_ask

        tasks = []
        for i, point in enumerate(points):
            for ref_idx, ref_row in enumerate(_REF_ROWS):
                mjcf_path = MJCF_PATHS[ref_row["scene"]]
                task_cfg = {
                    "terrain": _TERRAIN,
                    "sim_duration": SIM_DURATION,
                    "n_refs": len(_REF_ROWS),
                    "ref_idx": ref_idx,
                    "n_trials": n_trials,
                }
                if _TERRAIN.startswith("rough"):
                    task_cfg["y_jitter_seed"] = Y_JITTER_SEED
                    task_cfg["y_jitter"] = Y_JITTER
                    task_cfg["spawn_x"] = SPAWN_X
                    task_cfg["spawn_z"] = SPAWN_Z_RAISE
                else:
                    task_cfg["jitter_seed"] = INIT_JITTER_SEED
                    task_cfg["yaw_jitter_deg"] = INIT_YAW_JITTER_DEG
                for trial_idx in range(n_trials):
                    tasks.append((
                        i, point, ref_row, trial_idx, False,
                        n_done + i, mjcf_path, task_cfg,
                    ))

        t_sim = time.perf_counter()
        scene_results = list(pool.imap_unordered(_evaluate_one_scene, tasks, chunksize=1))
        t_sim = time.perf_counter() - t_sim

        t_agg = time.perf_counter()
        results = _aggregate_scene_results(points, scene_results)
        costs = [r["cost"] for r in results]
        t_agg = time.perf_counter() - t_agg

        t_tell = time.perf_counter()
        tell(points, costs)
        t_tell = time.perf_counter() - t_tell

        t_csv = time.perf_counter()
        elapsed_min = (time.perf_counter() - t_run_start) / 60.0
        for r in results:
            all_results.append(r)
            _append_result_to_csv(r, elapsed_min)
        t_csv = time.perf_counter() - t_csv

        n_done += n_this

        _print_point_results(results, n_this)

        batch_wall = time.perf_counter() - t_batch_start
        elapsed = time.perf_counter() - t_run_start
        elapsed_min = elapsed / 60.0
        print(f"  Batch wall: {batch_wall:.1f}s | Elapsed: {elapsed_min:.1f}min | Costs: min={min(costs):.4f}, max={max(costs):.4f}")
        print(f"  Profile: ask={t_ask:.3f}s sim={t_sim:.2f}s agg={t_agg:.3f}s tell={t_tell:.3f}s csv={t_csv:.3f}s")

        _print_best_so_far(all_results, n_done, elapsed_min)

        # Save CMA-ES state every batch
        state_path = pathlib.Path(BEST_CSV_PATH).parent / "cmaes_state.pkl"
        space_bounds = [(d.name, d.low, d.high, d.prior) for d in space]
        with open(state_path, "wb") as f:
            pickle.dump({"es": es, "n_done": n_done, "space_bounds": space_bounds}, f)

    best = min(all_results, key=lambda r: r["cost"])
    return OptResult(
        fun=best["cost"],
        x=[best["params"][dim.name] for dim in space],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified CMA-ES parameter optimization")
    parser.add_argument("--terrain", required=True,
                        help="Terrain type to optimize for (must have a matching config_{terrain}.py)")
    parser.add_argument("--suffix", "-s", type=str, default="", help="Suffix for results folder")
    parser.add_argument("--scenes", nargs="+", default=None, help="Filter to these scenes")
    parser.add_argument("--freqs", nargs="+", type=float, default=None, help="Filter to these freqs")
    parser.add_argument("--n-calls", type=int, default=None, help="Override N_CALLS")
    parser.add_argument("--warm-start-from", type=str, default=None,
                        help="Results dir to warm-start from (reads optimization_bests.csv)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Results dir with cmaes_state.pkl to resume from")
    args = parser.parse_args()

    # Load terrain-specific config
    _TERRAIN = args.terrain
    config_mod = importlib.import_module(f"config_{args.terrain}")

    MJCF_PATHS = config_mod.MJCF_PATHS
    _calculate_cost = config_mod.calculate_cost
    SIM_DURATION = config_mod.SIM_DURATION
    N_CALLS = config_mod.N_CALLS
    BATCH_SIZE = config_mod.BATCH_SIZE
    CMAES_X0 = config_mod.CMAES_X0
    CMAES_SIGMA0 = config_mod.CMAES_SIGMA0
    OPTIMIZER_RANDOM_STATE = config_mod.OPTIMIZER_RANDOM_STATE
    VELOCITY_VARIANCE_WEIGHT = config_mod.VELOCITY_VARIANCE_WEIGHT
    JITTER_AGGREGATION = config_mod.JITTER_AGGREGATION

    if _TERRAIN.startswith("rough"):
        INIT_JITTER_TRIALS = config_mod.INIT_JITTER_TRIALS
        Y_JITTER = config_mod.Y_JITTER
        Y_JITTER_SEED = config_mod.Y_JITTER_SEED
        SPAWN_X = config_mod.SPAWN_X
        SPAWN_Z_RAISE = config_mod.SPAWN_Z_RAISE
    else:
        INIT_YAW_JITTER_DEG = config_mod.INIT_YAW_JITTER_DEG
        INIT_JITTER_TRIALS = config_mod.INIT_JITTER_TRIALS
        INIT_JITTER_SEED = config_mod.INIT_JITTER_SEED

    if _TERRAIN.startswith("step"):
        _EXTRA_CSV_COLS = ["progress", "best_trial"]
    elif JITTER_AGGREGATION == "best":
        _EXTRA_CSV_COLS = ["best_trial"]

    # Build reference rows
    _REF_ROWS = reference_rows(config_mod.REFERENCE_DATA)
    _REF_INDEX_BY_ID = {row["id"]: i for i, row in enumerate(_REF_ROWS)}

    # Resume / warm-start
    es_resume = None
    if args.resume_from and args.warm_start_from:
        print("ERROR: --resume-from and --warm-start-from are mutually exclusive")
        sys.exit(1)
    if args.resume_from:
        resume_path = pathlib.Path(args.resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "cmaes_state.pkl"
        if not resume_path.exists():
            print(f"ERROR: resume state not found: {resume_path}")
            sys.exit(1)
        with open(resume_path, "rb") as f:
            state = pickle.load(f)
        es_resume = state["es"]
        print(f"Resuming from {resume_path} (sigma={es_resume.sigma:.4g}, prev evals={state['n_done']})")

    if args.warm_start_from:
        ws_path = pathlib.Path(args.warm_start_from)
        if ws_path.is_dir():
            ws_path = ws_path / "optimization_bests.csv"
        if not ws_path.exists():
            print(f"ERROR: warm-start file not found: {ws_path}")
            sys.exit(1)
        with open(ws_path) as f:
            ws_rows = list(csv.DictReader(f))
        if not ws_rows:
            print(f"ERROR: warm-start file is empty: {ws_path}")
            sys.exit(1)
        ws_last = ws_rows[-1]
        CMAES_X0 = {dim.name: float(ws_last[dim.name]) for dim in space}
        print(f"Warm-starting from {ws_path} (cost={ws_last['cost']})")

    # Filter references
    if args.scenes:
        _REF_ROWS = [r for r in _REF_ROWS if r["scene"] in args.scenes]
    if args.freqs:
        _REF_ROWS = [r for r in _REF_ROWS if r["ctrl_freq"] in args.freqs]
    if args.scenes or args.freqs:
        _REF_INDEX_BY_ID = {row["id"]: i for i, row in enumerate(_REF_ROWS)}
        if not _REF_ROWS:
            print(f"ERROR: no refs match scenes={args.scenes} freqs={args.freqs}")
            sys.exit(1)
        print(f"Filtered to {len(_REF_ROWS)} refs: {[r['id'] for r in _REF_ROWS]}")

    if args.n_calls is not None:
        N_CALLS = args.n_calls

    print(f"\nOptimizing {_TERRAIN} terrain for {N_CALLS} evals in batches of {BATCH_SIZE}")
    print(f"Aggregation: {JITTER_AGGREGATION}, Trials: {INIT_JITTER_TRIALS}")
    print("Reference targets:")
    for row in _REF_ROWS:
        print(f"  - {row['id']}: speed={row['speed']:.4f} m/s | weight={row['weight']}")

    # Create run directory
    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.suffix:
        run_tag += f"_{args.suffix}"
    run_dir = pathlib.Path("results") / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pathlib.Path(__file__).parent / "config.py", run_dir / "config.py")
    shutil.copy2(pathlib.Path(__file__).parent / f"config_{args.terrain}.py", run_dir / f"config_{args.terrain}.py")
    print(f"  Run directory: {run_dir}/")

    CSV_PATH = str(run_dir / "multi_optimization_results.csv")
    BEST_CSV_PATH = str(run_dir / "optimization_bests.csv")

    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_get_fieldnames()).writeheader()
    with open(BEST_CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_best_csv_fieldnames()).writeheader()

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    all_results = []
    pool = None

    try:
        n_trials = max(1, INIT_JITTER_TRIALS)
        tasks_per_batch = BATCH_SIZE * len(_REF_ROWS) * n_trials
        pool_size = max(1, min(os.cpu_count() or 16, tasks_per_batch))
        print(f"Worker pool size: {pool_size} (tasks per batch: {tasks_per_batch})")
        pool = multiprocessing.Pool(processes=pool_size)
        result = _run_batch_optimization(all_results, pool, es_resume=es_resume)
    finally:
        if pool:
            print("\n--- Finalizing: Terminating worker pool. ---")
            pool.terminate()
            pool.join()
            pool = None

    print(f"\n--- Optimization Finished ---")
    print(f"Lowest Cost Found: {result.fun:.6f}")
    print("Best Parameters:")
    best_params = {dim.name: value for dim, value in zip(space, result.x)}
    for name, value in best_params.items():
        print(f"  {name}: {value:.6f}")
    dmax = best_params["solimp_dmin"] + best_params["solimp_delta_d"] * (0.9999 - best_params["solimp_dmin"])
    print(f"  solimp_dmax: {dmax:.6f}  (derived)")

    # Record best rollout videos
    print("\n--- Recording Best Rollout ---")
    os.environ.setdefault("MUJOCO_GL", "egl")
    sim_module = importlib.import_module("simulation")
    sorted_results = sorted(all_results, key=lambda r: r["cost"])

    for i in range(min(1, len(sorted_results))):
        result_data = sorted_results[i]
        rank = i + 1
        print(f"\n#{rank}: Cost={result_data['cost']:.6f}")

        sim_params = sim_params_from_point(
            [result_data["params"][dim.name] for dim in space]
        )
        for ref_row in _REF_ROWS:
            scene = ref_row["scene"]
            mjcf_path = MJCF_PATHS[scene]
            ref_id = ref_row["id"]
            video_path = run_dir / f"rank_{rank:02d}_{ref_id}.mp4"
            print(f"  Recording {ref_id} → {video_path}...")
            sim_params_scene = dict(sim_params)
            sim_params_scene["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)

            extra_kwargs = {}
            if _TERRAIN.startswith("rough"):
                extra_kwargs["spawn_offset"] = (SPAWN_X, 0.0, SPAWN_Z_RAISE)

            sim_module.run_simulation(
                sim_params_scene,
                mjcf_path=mjcf_path,
                sim_duration=SIM_DURATION + 2.0,
                record_path=str(video_path),
                **extra_kwargs,
            )

    print(f"\n  Results saved to {run_dir}/")
