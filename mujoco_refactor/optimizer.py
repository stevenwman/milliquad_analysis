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
import os
import pathlib
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from skopt import Optimizer

from config import (
    BASE_ESTIMATOR,
    BATCH_SIZE,
    BEST_CSV_PATH,
    COST_FAILURE,
    CSV_PATH,
    INIT_JITTER_SEED,
    INIT_JITTER_TRIALS,
    INIT_YAW_JITTER_DEG,
    MJCF_PATHS,
    N_CALLS,
    PROFILE_BATCH,
    SIMULATION_TIMEOUT,
    SETTLE_TIME,
    SIM_DURATION,
    DEFAULT_CTRL_FREQ,
    reference_rows,
    reference_ids,
    PITCH_RMS_TARGET_DEG,
    PITCH_RMS_WEIGHT,
    TUMBLE_COST_WEIGHT,
    TUMBLE_PENALTY_SCALE,
    TUMBLE_THRESHOLD,
    VELOCITY_COST_WEIGHT,
    VELOCITY_VARIANCE_WEIGHT,
    LATERAL_COST_WEIGHT,
    csv_fieldnames,
    point_to_params,
    sim_params_from_point,
    space,
    VERBOSE_BATCH,
)

# ---- Simulation module selector ----
# Switch between vectorized (4.68x faster) and original (bit-exact) simulation.
# Hot-swap: change to "simulation" to use original implementation.
SIM_MODULE = "simulation_fast"


class OptResult(NamedTuple):
    """Result of optimization run, matching skopt's result interface."""
    fun: float  # best cost found
    x: list[float]  # best point (in space order)


_REF_ROWS = reference_rows()
if not _REF_ROWS:
    raise ValueError("No reference rows defined. Populate REFERENCE_DATA in config.py.")
_SCENE_TARGETS = {}
for _row in _REF_ROWS:
    _scene = _row["scene"]
    _w = float(_row.get("weight", 1.0))
    _SCENE_TARGETS.setdefault(_scene, {"num": 0.0, "den": 0.0})
    _SCENE_TARGETS[_scene]["num"] += _w * float(_row["speed"])
    _SCENE_TARGETS[_scene]["den"] += _w
_SCENE_TARGETS = {
    s: (v["num"] / v["den"] if v["den"] > 0 else 0.0)
    for s, v in _SCENE_TARGETS.items()
}

# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

# Robot body-z points DOWN in normal operation (INITIAL_QUATERNION = 180° about y).
# Uprightness = dot(body_z, nominal_down); +1 = normal, < threshold = flipped.
_BODY_Z_LOCAL = np.array([0.0, 0.0, 1.0])
_NOMINAL_BODY_Z_WORLD = np.array([0.0, 0.0, -1.0])


def calculate_cost(
    trajectory: list[dict],
    target_velocity: float,
    pitch_target_deg: float | None = None,
    pitch_weight: float | None = None,
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

    # Lateral displacement penalty (y-axis drift)
    lateral_displacement = 0.0
    if active_duration > 1e-6:
        lateral_displacement = abs(final_state["pos"][1] - start_state["pos"][1])
    lateral_error = lateral_displacement ** 2

    # Stability cost (tumbling penalty, normalized to per-step average)
    # Checks alignment of body-z with its nominal world direction (down).
    # uprightness ≈ +1 during normal walking, drops toward -1 when flipped.
    tumble_penalty = 0.0
    for state in trajectory:
        quat = state["quat"]
        body_z_axis = R.from_quat(quat, scalar_first=True).apply(_BODY_Z_LOCAL)
        uprightness = np.dot(body_z_axis, _NOMINAL_BODY_Z_WORLD)
        if uprightness < TUMBLE_THRESHOLD:
            tumble_penalty += (1 - uprightness) * TUMBLE_PENALTY_SCALE
    tumble_penalty /= max(len(trajectory), 1)

    # Pitch RMS amplitude (optional, detrended, in degrees)
    pitch_vals = []
    for state in trajectory:
        if state["time"] < SETTLE_TIME:
            continue
        quat = state["quat"]
        pitch = R.from_quat(quat, scalar_first=True).as_euler("xyz", degrees=False)[1]
        pitch_vals.append(pitch)
    pitch_rms_deg = 0.0
    if pitch_vals:
        pitch_vals = np.asarray(pitch_vals)
        pitch_vals = pitch_vals - np.mean(pitch_vals)
        pitch_rms_deg = np.sqrt(np.mean(pitch_vals ** 2)) * (180.0 / np.pi)

    if pitch_target_deg is None:
        pitch_target_deg = PITCH_RMS_TARGET_DEG
    if pitch_weight is None:
        pitch_weight = PITCH_RMS_WEIGHT
    pitch_error = (pitch_rms_deg - pitch_target_deg) ** 2

    total_cost = (
        VELOCITY_COST_WEIGHT * velocity_error
        + TUMBLE_COST_WEIGHT * tumble_penalty
        + LATERAL_COST_WEIGHT * lateral_error
        + pitch_weight * pitch_error
    )

    if verbose:
        print(
            f"    Avg Vel: {avg_forward_velocity:.3f} m/s | "
            f"Vel Err: {velocity_error:.4f} | "
            f"Lateral: {lateral_displacement:.4f} m | "
            f"Tumble Pen: {tumble_penalty:.4f} | "
            f"Pitch RMS: {pitch_rms_deg:.2f} deg | "
            f"Total Cost: {total_cost:.4f}"
        )

    return {
        "total_cost": total_cost,
        "avg_forward_velocity": avg_forward_velocity,
        "lateral_displacement": lateral_displacement,
        "tumble_penalty": tumble_penalty,
        "pitch_rms_deg": pitch_rms_deg,
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

def _evaluate_one_scene(args):
    """
    Run one trial for one (point, reference row). Used by process pool.

    args: (point_index, point, ref_row, trial_index, show_progress)
    Returns: (point_index, ref_id, scene_name, cost, velocity, tumble, pitch_rms, lateral, weight, wall_time)
    """
    point_index, point, ref_row, trial_index, show_progress = args

    # Lazy import in subprocess (separate memory space)
    import importlib
    _sim = importlib.import_module(SIM_MODULE)
    from config import sim_params_from_point as _sim_params_from_point

    sim_params = _sim_params_from_point(point)
    scene_name = ref_row["scene"]
    mjcf_path = MJCF_PATHS[scene_name]
    target_velocity = ref_row["speed"]
    pitch_target_deg = ref_row.get("pitch_amp_deg", PITCH_RMS_TARGET_DEG)
    pitch_weight = ref_row.get("pitch_weight", PITCH_RMS_WEIGHT)
    weight = ref_row.get("weight", 1.0)
    sim_params["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)

    # Deterministic jitter seeds (stable across processes).
    ref_idx = sum(ord(c) for c in ref_row["id"]) % 1000
    t0 = time.perf_counter()
    seed = INIT_JITTER_SEED + 100000 * point_index + 1000 * ref_idx + trial_index
    trajectory = _sim.run_simulation(
        sim_params,
        mjcf_path=mjcf_path,
        sim_duration=SIM_DURATION,
        visualize=False,
        progress=show_progress,
        wall_timeout=SIMULATION_TIMEOUT,
        init_yaw_jitter_deg=INIT_YAW_JITTER_DEG,
        rng_seed=seed,
    )
    if trajectory is None:
        cost_data = {
            "total_cost": COST_FAILURE,
            "avg_forward_velocity": 0.0,
            "tumble_penalty": 0.0,
            "pitch_rms_deg": 0.0,
            "lateral_displacement": 0.0,
        }
    else:
        cost_data = calculate_cost(
            trajectory,
            target_velocity,
            pitch_target_deg=pitch_target_deg,
            pitch_weight=pitch_weight,
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
        cost_data.get("pitch_rms_deg", 0.0),
        cost_data.get("lateral_displacement", 0.0),
        weight,
        wall_time,
    )


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def _aggregate_scene_results(points: list, scene_results: list) -> list[dict]:
    """Turn list of per-trial results into full result dicts (one per point)."""
    by_point = defaultdict(
        lambda: {
            "ref_trials_costs": defaultdict(list),
            "ref_trials_velocities": defaultdict(list),
            "ref_trials_tumble": defaultdict(list),
            "ref_trials_pitch_rms": defaultdict(list),
            "ref_trials_lateral": defaultdict(list),
            "ref_weights": {},
            "ref_scene": {},
            "scene_costs": defaultdict(float),
            "scene_vel_num": defaultdict(float),
            "scene_tumble_num": defaultdict(float),
            "scene_lateral_num": defaultdict(float),
            "scene_weight": defaultdict(float),
            "scene_wall_times": [],
            "has_failure": False,
        }
    )
    for (
        point_index,
        ref_id,
        scene_name,
        cost,
        velocity,
        tumble,
        pitch_rms,
        lateral,
        weight,
        wall_time,
    ) in scene_results:
        d = by_point[point_index]
        d["ref_trials_costs"][ref_id].append(cost)
        d["ref_trials_velocities"][ref_id].append(velocity)
        d["ref_trials_tumble"][ref_id].append(tumble)
        d["ref_trials_pitch_rms"][ref_id].append(pitch_rms)
        d["ref_trials_lateral"][ref_id].append(lateral)
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
        ref_pitch_rms = {}
        ref_lateral = {}

        for ref_id, trials in d["ref_trials_costs"].items():
            scene = d["ref_scene"][ref_id]
            weight = d["ref_weights"][ref_id]
            mean_cost = float(np.mean(trials))
            mean_vel = float(np.mean(d["ref_trials_velocities"][ref_id]))
            mean_tumble = float(np.mean(d["ref_trials_tumble"][ref_id]))
            mean_pitch = float(np.mean(d["ref_trials_pitch_rms"][ref_id]))
            mean_lateral = float(np.mean(d["ref_trials_lateral"][ref_id]))

            ref_costs[ref_id] = mean_cost
            ref_avg_velocities[ref_id] = mean_vel
            ref_tumble[ref_id] = mean_tumble
            ref_pitch_rms[ref_id] = mean_pitch
            ref_lateral[ref_id] = mean_lateral

            d["scene_costs"][scene] += weight * mean_cost
            d["scene_vel_num"][scene] += weight * mean_vel
            d["scene_tumble_num"][scene] += weight * mean_tumble
            d["scene_lateral_num"][scene] += weight * mean_lateral
            d["scene_weight"][scene] += weight

        scene_avg_velocities = {
            s: (d["scene_vel_num"][s] / d["scene_weight"][s] if d["scene_weight"][s] > 0 else 0.0)
            for s in MJCF_PATHS
        }
        scene_tumble = {
            s: (d["scene_tumble_num"][s] / d["scene_weight"][s] if d["scene_weight"][s] > 0 else 0.0)
            for s in MJCF_PATHS
        }
        scene_lateral = {
            s: (d["scene_lateral_num"][s] / d["scene_weight"][s] if d["scene_weight"][s] > 0 else 0.0)
            for s in MJCF_PATHS
        }
        scene_costs = dict(d["scene_costs"])
        if d["has_failure"]:
            total_cost = COST_FAILURE
        else:
            total_cost = sum(scene_costs.values())
            # Variance penalty: penalize uneven relative velocity errors across refs
            if len(ref_avg_velocities) > 1:
                ref_rows_local = reference_rows()
                targets_by_id = {row["id"]: row["speed"] for row in ref_rows_local}
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
            "scene_tumble": scene_tumble,
            "scene_lateral": scene_lateral,
            "ref_costs": ref_costs,
            "ref_avg_velocities": ref_avg_velocities,
            "ref_tumble": ref_tumble,
            "ref_pitch_rms": ref_pitch_rms,
            "ref_lateral": ref_lateral,
            "ref_weights": d["ref_weights"],
            "ref_scene": d["ref_scene"],
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
    for rid in reference_ids():
        row[f"velocity_{rid}"] = res["ref_avg_velocities"].get(rid, 0)
        row[f"cost_{rid}"] = res["ref_costs"].get(rid, 0)
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
    """Print per-point results as a compact table with one row per reference."""
    ref_rows = reference_rows()
    for i, r in enumerate(results):
        wt = r.get("wall_time", 0)
        print(f"    [{i+1}/{n_this}] id={r['id']}  cost={r['cost']:.4f}  time={wt:.1f}s")
        _print_ref_table(r, ref_rows, indent=6)


def _print_ref_table(r: dict, ref_rows: list[dict], indent: int = 4) -> None:
    """Print a compact table of per-reference-row results."""
    rv = r.get("ref_avg_velocities", {})
    rt = r.get("ref_tumble", {})
    rp = r.get("ref_pitch_rms", {})
    rl = r.get("ref_lateral", {})
    pad = " " * indent
    # Header
    print(f"{pad}{'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9} {'tumble':>7} {'lateral':>8} {'pitch':>6}")
    print(f"{pad}{'-'*66}")
    for row in ref_rows:
        rid = row["id"]
        target = row["speed"]
        sim_v = rv.get(rid, 0.0)
        delta = (sim_v - target) * 100  # cm/s
        tmb = rt.get(rid, 0.0)
        lat = rl.get(rid, 0.0) * 100  # cm
        pit = rp.get(rid, 0.0)
        print(f"{pad}{rid:<18} {target:>6.3f}  {sim_v:>6.3f}  {delta:>+7.1f}cs  {tmb:>6.4f}  {lat:>6.1f}cm  {pit:>4.1f}°")


_best_cost_so_far: float = float("inf")


def _append_best_csv(best: dict, n_done: int) -> None:
    """Append a new-best row to the running bests CSV."""
    from datetime import datetime
    ref_rows = reference_rows()
    ref_ids = [row["id"] for row in ref_rows]
    rv = best.get("ref_avg_velocities", {})

    fieldnames = (
        ["timestamp", "n_eval", "id", "cost"]
        + [f"vel_{rid}" for rid in ref_ids]
        + [dim.name for dim in space]
    )
    with open(BEST_CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_eval": n_done,
            "id": best["id"],
            "cost": f"{best['cost']:.6f}",
        }
        for rid in ref_ids:
            row[f"vel_{rid}"] = f"{rv.get(rid, 0.0):.4f}"
        for dim in space:
            row[dim.name] = f"{best['params'][dim.name]:.8g}"
        w.writerow(row)


def _print_best_so_far(all_results: list[dict], n_done: int) -> None:
    """Print best result and append to bests CSV if improved."""
    global _best_cost_so_far
    best = min(all_results, key=lambda r: r["cost"])
    ref_rows = reference_rows()
    is_new_best = best["cost"] < _best_cost_so_far
    marker = " ★ NEW BEST" if is_new_best else ""
    print(f"  Best so far (n={n_done}): cost={best['cost']:.6f}  id={best['id']}{marker}")
    _print_ref_table(best, ref_rows, indent=4)

    if is_new_best:
        _best_cost_so_far = best["cost"]
        _append_best_csv(best, n_done)


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def _run_batch_optimization(all_results: list[dict], pool: multiprocessing.Pool) -> OptResult:
    """
    Batch Bayesian optimization: propose BATCH_SIZE points, evaluate all references
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

    batch_num = 0
    while n_done < N_CALLS:
        n_this = min(BATCH_SIZE, N_CALLS - n_done)
        batch_num += 1
        n_trials = max(1, INIT_JITTER_TRIALS)
        t_batch_start = time.perf_counter()
        print(f"\n--- Batch {batch_num}: asking for {n_this} points ({n_done + 1}\u2013{n_done + n_this} / {N_CALLS}), {n_this * len(_REF_ROWS) * n_trials} tasks ---")

        t_ask = time.perf_counter()
        points = optimizer.ask(n_points=n_this)
        if n_this == 1:
            points = [points]
        t_ask = time.perf_counter() - t_ask

        # One reference row per process: (point_index, point, ref_row)
        tasks = [
            (
                i,
                point,
                ref_row,
                trial_idx,
                (i == 0 and ref_idx == 0 and trial_idx == 0),
            )
            for i, point in enumerate(points)
            for ref_idx, ref_row in enumerate(_REF_ROWS)
            for trial_idx in range(n_trials)
        ]
        t_sim = time.perf_counter()
        # Queue-style scheduling to keep workers busy for heterogeneous task runtimes.
        scene_results = list(pool.imap_unordered(_evaluate_one_scene, tasks, chunksize=1))
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

        batch_wall_actual = time.perf_counter() - t_batch_start
        batch_wall_sum = t_ask + t_sim + t_agg + t_tell + t_csv + t_verbose
        overhead = batch_wall_actual - batch_wall_sum
        print(f"  Batch wall: {batch_wall_actual:.1f}s (profiled: {batch_wall_sum:.1f}s, overhead: {overhead:.2f}s) | Costs: min={min(costs):.4f}, max={max(costs):.4f}")

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
    print("Reference targets:")
    for row in _REF_ROWS:
        print(
            f"  - {row['id']}: scene={row['scene']} | "
            f"ctrl_freq={row['ctrl_freq']} Hz | "
            f"speed={row['speed']} m/s | "
            f"weight={row['weight']} | "
            f"pitch_weight={row.get('pitch_weight', PITCH_RMS_WEIGHT)}"
        )

    # Write CSV headers once (overwrite from previous runs)
    try:
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fieldnames()).writeheader()
    except Exception as e:
        print(f"  [Warning] Could not create CSV: {e}")

    ref_ids = [row["id"] for row in reference_rows()]
    best_fieldnames = (
        ["timestamp", "n_eval", "id", "cost"]
        + [f"vel_{rid}" for rid in ref_ids]
        + [dim.name for dim in space]
    )
    try:
        with open(BEST_CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=best_fieldnames).writeheader()
    except Exception as e:
        print(f"  [Warning] Could not create bests CSV: {e}")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Local state — no globals (fix #4)
    all_results = []
    pool = None

    try:
        n_trials = max(1, INIT_JITTER_TRIALS)
        tasks_per_batch = BATCH_SIZE * len(_REF_ROWS) * n_trials
        pool_size = max(1, min(os.cpu_count() or 16, tasks_per_batch))
        print(f"Worker pool size: {pool_size} (tasks per batch: {tasks_per_batch})")
        pool = multiprocessing.Pool(processes=pool_size)
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

    # Record top 3 rollouts
    print("\n--- Recording Top 3 Best Rollouts ---")
    import importlib
    sim_module = importlib.import_module(SIM_MODULE)

    sorted_results = sorted(all_results, key=lambda r: r["cost"])
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_root = pathlib.Path("top_3_rollouts_multi") / run_stamp
    replay_root.mkdir(parents=True, exist_ok=True)
    print(f"Replay output root: {replay_root}")

    for i in range(min(3, len(sorted_results))):
        result_data = sorted_results[i]
        rank = i + 1

        print(f"\n#{rank}: Cost={result_data['cost']:.6f}")
        for scene, velocity in result_data["scene_avg_velocities"].items():
            print(f"  - {scene}: Avg Velocity={velocity:.4f} m/s (Cost: {result_data['scene_costs'].get(scene, 'N/A'):.4f})")

        sim_params = sim_params_from_point(
            [result_data["params"][dim.name] for dim in space]
        )

        run_dir = replay_root / f"rank_{rank:02d}_id_{result_data['id']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for ref_row in _REF_ROWS:
            scene = ref_row["scene"]
            mjcf_path = MJCF_PATHS[scene]
            ref_id = ref_row["id"]
            video_path = run_dir / f"{ref_id}.mp4"
            print(f"  Recording video for {ref_id} to {video_path}...")
            sim_params_scene = dict(sim_params)
            sim_params_scene["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)
            sim_module.run_simulation(
                sim_params_scene,
                mjcf_path=mjcf_path,
                sim_duration=10.0,
                record_path=str(video_path),
            )
