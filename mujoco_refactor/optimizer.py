"""
Batch Bayesian optimization for multi-scene simulation parameter tuning.

Uses scikit-optimize with parallel batch evaluation across multiple robot
configurations (2-leg and 4-leg scenes).

Usage:
    cd mujoco_refactor
    uv run python optimizer.py
"""

import csv
import multiprocessing
import time
import uuid
from collections import defaultdict
from typing import Any, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from skopt import Optimizer

from config import (
    BASE_ESTIMATOR,
    BATCH_SIZE,
    COST_FAILURE,
    CSV_PATH,
    MJCF_PATHS,
    N_CALLS,
    NUM_SCENES,
    POOL_SIZE,
    PROFILE_BATCH,
    SEED_FROM_OLD_CSV,
    SEED_POINT,
    SETTLE_TIME,
    SIM_DURATION,
    TARGET_VELOCITIES,
    TUMBLE_COST_WEIGHT,
    TUMBLE_PENALTY_SCALE,
    TUMBLE_THRESHOLD,
    VELOCITY_COST_WEIGHT,
    csv_fieldnames,
    point_to_params,
    sim_params_from_point,
    space,
    VERBOSE_BATCH,
)


class OptResult(NamedTuple):
    """Result of optimization run, matching skopt's result interface."""
    fun: float  # best cost found
    x: list[float]  # best point (in space order)


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

# Unit vector for uprightness check (world Z-axis)
_UP_VECTOR = np.array([0.0, 0.0, 1.0])


def calculate_cost(
    trajectory: list[dict],
    target_velocity: float,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Calculate cost from simulation trajectory.

    Penalizes deviation from target velocity and tumbling instability.
    Returns dict with total_cost, avg_forward_velocity, tumble_penalty.
    """
    if not trajectory:
        return {"total_cost": COST_FAILURE, "avg_forward_velocity": 0, "tumble_penalty": 0}

    # Forward velocity cost (over active time, after settle)
    final_state = trajectory[-1]
    start_state = trajectory[0]
    for state in trajectory:
        if state["time"] >= SETTLE_TIME:
            start_state = state
            break

    active_duration = final_state["time"] - start_state["time"]
    avg_forward_velocity = 0.0
    if active_duration > 1e-6:
        forward_displacement = final_state["pos"][0] - start_state["pos"][0]
        avg_forward_velocity = forward_displacement / active_duration

    velocity_error = (avg_forward_velocity - target_velocity) ** 2

    # Stability cost (tumbling penalty)
    tumble_penalty = 0.0
    for state in trajectory:
        quat = state["quat"]
        body_z_axis = R.from_quat(quat, scalar_first=True).apply(_UP_VECTOR)
        uprightness = np.dot(body_z_axis, _UP_VECTOR)
        if uprightness < TUMBLE_THRESHOLD:
            tumble_penalty += (1 - uprightness) * TUMBLE_PENALTY_SCALE

    total_cost = VELOCITY_COST_WEIGHT * velocity_error + TUMBLE_COST_WEIGHT * tumble_penalty

    if verbose:
        print(
            f"    Avg Vel: {avg_forward_velocity:.3f} m/s | "
            f"Vel Err: {velocity_error:.4f} | "
            f"Tumble Pen: {tumble_penalty:.4f} | "
            f"Total Cost: {total_cost:.4f}"
        )

    return {
        "total_cost": total_cost,
        "avg_forward_velocity": avg_forward_velocity,
        "tumble_penalty": tumble_penalty,
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

def _evaluate_one_scene(args):
    """
    Run one scene for one point. Used by 1-scene-per-process pool.

    args: (point_index, point, scene_name, mjcf_path)
    Returns: (point_index, scene_name, cost, velocity, tumble, wall_time)
    """
    point_index, point, scene_name, mjcf_path = args

    # Lazy import in subprocess (separate memory space)
    import simulation as _sim
    from config import sim_params_from_point as _sim_params_from_point

    sim_params = _sim_params_from_point(point)
    target_velocity = TARGET_VELOCITIES[scene_name]
    t0 = time.perf_counter()
    trajectory = _sim.run_simulation(
        sim_params, mjcf_path=mjcf_path, sim_duration=SIM_DURATION, visualize=False
    )
    wall_time = time.perf_counter() - t0

    if trajectory is None:
        cost_data = {"total_cost": COST_FAILURE, "avg_forward_velocity": 0.0, "tumble_penalty": 0.0}
    else:
        cost_data = calculate_cost(trajectory, target_velocity, verbose=False)

    return (
        point_index,
        scene_name,
        cost_data["total_cost"],
        cost_data["avg_forward_velocity"],
        cost_data.get("tumble_penalty", 0.0),
        wall_time,
    )


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def _aggregate_scene_results(points: list, scene_results: list) -> list[dict]:
    """Turn list of per-scene results into list of full result dicts (one per point)."""
    by_point = defaultdict(
        lambda: {"scene_costs": {}, "scene_avg_velocities": {}, "scene_tumble": {}, "scene_wall_times": []}
    )
    for point_index, scene_name, cost, velocity, tumble, wall_time in scene_results:
        by_point[point_index]["scene_costs"][scene_name] = cost
        by_point[point_index]["scene_avg_velocities"][scene_name] = velocity
        by_point[point_index]["scene_tumble"][scene_name] = tumble
        by_point[point_index]["scene_wall_times"].append(wall_time)

    results = []
    for point_index in sorted(by_point):
        d = by_point[point_index]
        params = point_to_params(points[point_index])
        total_cost = sum(d["scene_costs"].values())
        point_wall = max(d["scene_wall_times"]) if d["scene_wall_times"] else 0.0
        results.append({
            "id": str(uuid.uuid4().hex)[:8],
            "cost": total_cost,
            "params": params,
            "scene_costs": d["scene_costs"],
            "scene_avg_velocities": d["scene_avg_velocities"],
            "scene_tumble": d["scene_tumble"],
            "wall_time": point_wall,
        })
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _append_result_to_csv(res: dict[str, Any]) -> None:
    """Append one result row to the CSV."""
    row = {"id": res["id"], "cost": res["cost"]}
    for scene in MJCF_PATHS:
        row[f"velocity_{scene}"] = res["scene_avg_velocities"].get(scene, 0)
        row[f"cost_{scene}"] = res["scene_costs"].get(scene, 0)
    row.update(res["params"])
    try:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
            writer.writerow(row)
    except Exception as e:
        print(f"  [Warning] Could not append to CSV: {e}")


# ---------------------------------------------------------------------------
# Batch summary printing (extracted from main loop — fix #10)
# ---------------------------------------------------------------------------

def _print_point_results(results: list[dict], n_this: int) -> None:
    """Print per-point cost/residual/tumble for one batch."""
    for i, r in enumerate(results):
        sav = r["scene_avg_velocities"]
        st = r.get("scene_tumble", {})
        residuals_cm_s = " | ".join(
            f"{s}: {(sav.get(s, 0) - TARGET_VELOCITIES[s]) * 100:+.1f} cm/s"
            for s in MJCF_PATHS
        )
        tumbles = " | ".join(f"{s}: {st.get(s, 0.0):.4f}" for s in MJCF_PATHS)
        wt = r.get("wall_time", 0)
        print(f"    [{i+1}/{n_this}] cost={r['cost']:.4f} | residual: {residuals_cm_s} | tumble: {tumbles} | time: {wt:.1f}s")


def _print_best_so_far(all_results: list[dict], n_done: int) -> None:
    """Print best result across all completed iterations."""
    best = min(all_results, key=lambda r: r["cost"])
    sav = best["scene_avg_velocities"]
    st = best.get("scene_tumble", {})
    res_str = " | ".join(
        f"{s}: {(sav.get(s, 0) - TARGET_VELOCITIES[s]) * 100:+.1f} cm/s"
        for s in MJCF_PATHS
    )
    tum_str = " | ".join(f"{s}: {st.get(s, 0.0):.4f}" for s in MJCF_PATHS)
    print(f"  Best so far (n={n_done}): cost={best['cost']:.6f} | residual: {res_str} | tumble: {tum_str} | id={best['id']}")


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def _run_batch_optimization(all_results: list[dict], pool: multiprocessing.Pool) -> OptResult:
    """
    Batch Bayesian optimization: propose BATCH_SIZE points, evaluate all scenes
    in parallel, tell optimizer the costs, repeat.

    Args:
        all_results: Mutable list to accumulate results (replaces global).
        pool: Pre-created multiprocessing pool.
    """
    optimizer = Optimizer(
        dimensions=space,
        base_estimator=BASE_ESTIMATOR,
        n_initial_points=20,
        random_state=42,
    )
    n_done = 0

    # Optional seed from older optimization
    if SEED_FROM_OLD_CSV:
        print("\n--- Seed from old CSV (optimization_results.csv abe1b74c, fudges=1, damping=7e-10) ---")
        scene_results = [
            _evaluate_one_scene((0, SEED_POINT, sn, path))
            for sn, path in MJCF_PATHS.items()
        ]
        seed_results = _aggregate_scene_results([SEED_POINT], scene_results)
        seed_result = seed_results[0]
        optimizer.tell([SEED_POINT], [seed_result["cost"]])
        all_results.append(seed_result)
        _append_result_to_csv(seed_result)
        n_done = 1
        sav = seed_result["scene_avg_velocities"]
        st = seed_result.get("scene_tumble", {})
        res_str = " | ".join(
            f"{s}: {(sav.get(s, 0) - TARGET_VELOCITIES[s]) * 100:+.1f} cm/s"
            for s in MJCF_PATHS
        )
        tum_str = " | ".join(f"{s}: {st.get(s, 0.0):.4f}" for s in MJCF_PATHS)
        print(f"  Seed cost={seed_result['cost']:.6f} | residual: {res_str} | tumble: {tum_str} | id={seed_result['id']}\n")

    batch_num = 0
    while n_done < N_CALLS:
        n_this = min(BATCH_SIZE, N_CALLS - n_done)
        batch_num += 1
        print(f"\n--- Batch {batch_num}: asking for {n_this} points ({n_done + 1}\u2013{n_done + n_this} / {N_CALLS}), {n_this * NUM_SCENES} scene tasks ---")

        t_ask = time.perf_counter()
        points = optimizer.ask(n_points=n_this)
        if n_this == 1:
            points = [points]
        t_ask = time.perf_counter() - t_ask

        # One scene per process: (point_index, point, scene_name, mjcf_path)
        tasks = [
            (i, point, scene_name, mjcf_path)
            for i, point in enumerate(points)
            for scene_name, mjcf_path in MJCF_PATHS.items()
        ]
        t_sim = time.perf_counter()
        scene_results = pool.map(_evaluate_one_scene, tasks)
        t_sim = time.perf_counter() - t_sim

        t_agg = time.perf_counter()
        results = _aggregate_scene_results(points, scene_results)
        costs = [r["cost"] for r in results]
        t_agg = time.perf_counter() - t_agg

        t_tell = time.perf_counter()
        optimizer.tell(points, costs)
        t_tell = time.perf_counter() - t_tell

        t_csv = time.perf_counter()
        for r in results:
            all_results.append(r)
            _append_result_to_csv(r)
        t_csv = time.perf_counter() - t_csv

        n_done += n_this

        t_verbose = 0.0
        if VERBOSE_BATCH:
            t_verbose = time.perf_counter()
            _print_point_results(results, n_this)
            t_verbose = time.perf_counter() - t_verbose

        batch_wall = t_ask + t_sim + t_agg + t_tell + t_csv + t_verbose
        print(f"  Batch wall: {batch_wall:.1f}s | Costs: min={min(costs):.4f}, max={max(costs):.4f}")

        if PROFILE_BATCH:
            parts = f"ask={t_ask:.3f}s sim={t_sim:.2f}s agg={t_agg:.3f}s tell={t_tell:.3f}s csv={t_csv:.3f}s"
            if VERBOSE_BATCH:
                parts += f" verbose={t_verbose:.3f}s"
            print(f"  Profile: {parts}")

        _print_best_so_far(all_results, n_done)

    best = min(all_results, key=lambda r: r["cost"])
    return OptResult(
        fun=best["cost"],
        x=[best["params"][dim.name] for dim in space],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running Bayesian optimization for {N_CALLS} evaluations in batches of {BATCH_SIZE}...")
    print(f"Target velocities: {TARGET_VELOCITIES}")

    # Write CSV header once
    try:
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fieldnames()).writeheader()
    except Exception as e:
        print(f"  [Warning] Could not create CSV: {e}")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Local state — no globals (fix #4)
    all_results = []
    pool = None

    try:
        pool = multiprocessing.Pool(processes=POOL_SIZE)
        result = _run_batch_optimization(all_results, pool)
    finally:
        if pool:
            print("\n--- Finalizing: Terminating worker pool. ---")
            pool.terminate()
            pool.join()
            pool = None

    if all_results:
        print(f"\n--- Results appended to {CSV_PATH} after each batch (rows in evaluation order) ---")

    print("\n--- Optimization Finished ---")
    best_cost = result.fun
    print(f"Lowest Cost Found: {best_cost:.6f}")
    print("Best Parameters:")
    for dim, value in zip(space, result.x):
        print(f"  {dim.name}: {value:.6f}")

    # Record top 5 rollouts
    print("\n--- Recording Top 5 Best Rollouts ---")
    import simulation as sim_module

    sorted_results = sorted(all_results, key=lambda r: r["cost"])

    for i in range(min(5, len(sorted_results))):
        result_data = sorted_results[i]
        rank = i + 1

        print(f"\n#{rank}: Cost={result_data['cost']:.6f}")
        for scene, velocity in result_data["scene_avg_velocities"].items():
            print(f"  - {scene}: Avg Velocity={velocity:.4f} m/s (Cost: {result_data['scene_costs'].get(scene, 'N/A'):.4f})")

        sim_params = sim_params_from_point(
            [result_data["params"][dim.name] for dim in space]
        )

        video_dir = "top_5_rollouts_multi"
        for scene, mjcf_path in MJCF_PATHS.items():
            video_path = f"{video_dir}/rank_{rank}_{scene}_id_{result_data['id']}_cost_{result_data['cost']:.4f}.mp4"
            print(f"  Recording video for {scene} to {video_path}...")
            sim_module.run_simulation(
                sim_params,
                mjcf_path=mjcf_path,
                sim_duration=10.0,
                record_path=video_path,
            )
