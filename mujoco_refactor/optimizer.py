"""
Batch optimization for multi-scene simulation parameter tuning.

Supports two backends (set OPTIMIZER_BACKEND in config.py):
  - "skopt": Bayesian optimization via scikit-optimize (GP/RF surrogate)
  - "cmaes": CMA Evolution Strategy via pycma

Usage:
    cd mujoco_refactor
    uv run python optimizer.py
"""

import csv
import multiprocessing
import os
import pathlib
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from config import (
    ACQ_FUNC,
    ACQ_FUNC_KWARGS,
    BASE_ESTIMATOR,
    BATCH_SIZE,
    BEST_CSV_PATH,
    CMAES_SIGMA0,
    CMAES_X0,
    COST_FAILURE,
    CSV_PATH,
    INIT_JITTER_SEED,
    INIT_JITTER_TRIALS,
    INIT_YAW_JITTER_DEG,
    MJCF_PATHS,
    N_CALLS,
    N_INITIAL_POINTS,
    OPTIMIZER_BACKEND,
    OPTIMIZER_NOISE,
    OPTIMIZER_RANDOM_STATE,
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
    YAW_THRESHOLD_DEG,
    YAW_COST_WEIGHT,
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
# Collision-free ref index for deterministic jitter seeds.
_REF_INDEX_BY_ID: dict[str, int] = {row["id"]: i for i, row in enumerate(_REF_ROWS)}

# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

# Robot body-z points DOWN in normal operation (INITIAL_QUATERNION = 180° about y).
# Uprightness = dot(body_z, nominal_down); +1 = normal, < threshold = flipped.
_BODY_Z_LOCAL = np.array([0.0, 0.0, 1.0])
_NOMINAL_BODY_Z_WORLD = np.array([0.0, 0.0, -1.0])
_BODY_X_LOCAL = np.array([1.0, 0.0, 0.0])


def calculate_cost(
    trajectory: list[dict],
    target_velocity: float,
    speed_std: float = 0.0,
    pitch_target_deg: float | None = None,
    pitch_weight: float | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Calculate cost from simulation trajectory.

    Penalizes deviation from target velocity and tumbling instability.
    If speed_std > 0, velocity errors within 1-sigma of target cost zero (dead zone).
    Returns dict with total_cost, avg_forward_velocity, tumble_penalty.
    """
    if not trajectory:
        return {"total_cost": COST_FAILURE, "avg_forward_velocity": 0, "tumble_penalty": 0,
                "lateral_displacement": 0, "yaw_deviation_deg": 0, "pitch_rms_deg": 0}

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

    # Dead-zone velocity error: no penalty within 1-sigma of target
    vel_deviation = avg_forward_velocity - target_velocity
    if speed_std > 0.0 and abs(vel_deviation) <= speed_std:
        velocity_error = 0.0
    elif speed_std > 0.0:
        # Penalize only the excess beyond the dead zone
        excess = abs(vel_deviation) - speed_std
        velocity_error = (excess / target_velocity) ** 2
    else:
        velocity_error = (vel_deviation / target_velocity) ** 2

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

    # Yaw spin-out check: compare final heading to initial heading.
    # Only penalizes large deviations (> YAW_THRESHOLD_DEG); normal yaw oscillation is fine.
    start_body_x = R.from_quat(start_state["quat"], scalar_first=True).apply(_BODY_X_LOCAL)
    end_body_x = R.from_quat(final_state["quat"], scalar_first=True).apply(_BODY_X_LOCAL)
    start_heading = start_body_x[:2]
    end_heading = end_body_x[:2]
    start_norm = np.linalg.norm(start_heading)
    end_norm = np.linalg.norm(end_heading)
    yaw_deviation_deg = 0.0
    yaw_penalty = 0.0
    if start_norm > 1e-6 and end_norm > 1e-6:
        cos_yaw = np.clip(np.dot(start_heading / start_norm, end_heading / end_norm), -1.0, 1.0)
        yaw_deviation_deg = np.degrees(np.arccos(cos_yaw))
        if yaw_deviation_deg > YAW_THRESHOLD_DEG:
            excess = yaw_deviation_deg - YAW_THRESHOLD_DEG
            yaw_penalty = (excess / 90.0) ** 2

    total_cost = (
        VELOCITY_COST_WEIGHT * velocity_error
        + TUMBLE_COST_WEIGHT * tumble_penalty
        + LATERAL_COST_WEIGHT * lateral_error
        + YAW_COST_WEIGHT * yaw_penalty
        + pitch_weight * pitch_error
    )

    if verbose:
        print(
            f"    Avg Vel: {avg_forward_velocity:.3f} m/s | "
            f"Vel Err: {velocity_error:.4f} | "
            f"Lateral: {lateral_displacement:.4f} m | "
            f"Tumble Pen: {tumble_penalty:.4f} | "
            f"Yaw: {yaw_deviation_deg:.1f}° | "
            f"Pitch RMS: {pitch_rms_deg:.2f} deg | "
            f"Total Cost: {total_cost:.4f}"
        )

    return {
        "total_cost": total_cost,
        "avg_forward_velocity": avg_forward_velocity,
        "lateral_displacement": lateral_displacement,
        "tumble_penalty": tumble_penalty,
        "yaw_deviation_deg": yaw_deviation_deg,
        "pitch_rms_deg": pitch_rms_deg,
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

def _evaluate_one_scene(args):
    """
    Run one trial for one (point, reference row). Used by process pool.

    args: (point_index, point, ref_row, trial_index, show_progress, global_point_index)
    Returns: (point_index, ref_id, scene_name, cost, velocity, tumble, pitch_rms, lateral, weight, wall_time)
    """
    point_index, point, ref_row, trial_index, show_progress, global_point_index = args

    # Lazy import in subprocess (separate memory space)
    import importlib
    _sim = importlib.import_module(SIM_MODULE)
    from config import sim_params_from_point as _sim_params_from_point

    sim_params = _sim_params_from_point(point)
    scene_name = ref_row["scene"]
    mjcf_path = MJCF_PATHS[scene_name]
    target_velocity = ref_row["speed"]
    speed_std = ref_row.get("speed_std", 0.0)
    pitch_target_deg = ref_row.get("pitch_amp_deg", PITCH_RMS_TARGET_DEG)
    pitch_weight = ref_row.get("pitch_weight", PITCH_RMS_WEIGHT)
    weight = ref_row.get("weight", 1.0)
    sim_params["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)

    # Deterministic jitter seeds (collision-free across refs/trials/points).
    n_refs = len(_REF_ROWS)
    n_trials = max(1, INIT_JITTER_TRIALS)
    ref_idx = _REF_INDEX_BY_ID[ref_row["id"]]
    t0 = time.perf_counter()
    seed = INIT_JITTER_SEED + (global_point_index * n_refs + ref_idx) * n_trials + trial_index
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
            "yaw_deviation_deg": 0.0,
        }
    else:
        cost_data = calculate_cost(
            trajectory,
            target_velocity,
            speed_std=speed_std,
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
        cost_data.get("yaw_deviation_deg", 0.0),
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
            "ref_trials_yaw": defaultdict(list),
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
        yaw_deg,
        weight,
        wall_time,
    ) in scene_results:
        d = by_point[point_index]
        d["ref_trials_costs"][ref_id].append(cost)
        d["ref_trials_velocities"][ref_id].append(velocity)
        d["ref_trials_tumble"][ref_id].append(tumble)
        d["ref_trials_pitch_rms"][ref_id].append(pitch_rms)
        d["ref_trials_lateral"][ref_id].append(lateral)
        d["ref_trials_yaw"][ref_id].append(yaw_deg)
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
        ref_yaw = {}

        for ref_id, trials in d["ref_trials_costs"].items():
            scene = d["ref_scene"][ref_id]
            weight = d["ref_weights"][ref_id]
            mean_cost = float(np.median(trials))
            mean_vel = float(np.median(d["ref_trials_velocities"][ref_id]))
            mean_tumble = float(np.median(d["ref_trials_tumble"][ref_id]))
            mean_pitch = float(np.median(d["ref_trials_pitch_rms"][ref_id]))
            mean_lateral = float(np.median(d["ref_trials_lateral"][ref_id]))
            mean_yaw = float(np.median(d["ref_trials_yaw"][ref_id]))

            ref_costs[ref_id] = mean_cost
            ref_avg_velocities[ref_id] = mean_vel
            ref_tumble[ref_id] = mean_tumble
            ref_pitch_rms[ref_id] = mean_pitch
            ref_lateral[ref_id] = mean_lateral
            ref_yaw[ref_id] = mean_yaw

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
            "ref_yaw": ref_yaw,
            "ref_weights": d["ref_weights"],
            "ref_scene": d["ref_scene"],
            "wall_time": point_wall,
        })
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _append_result_to_csv(res: dict[str, Any], elapsed_min: float) -> None:
    """Append one result row to the CSV."""
    row = {"id": res["id"], "cost": res["cost"], "elapsed_min": f"{elapsed_min:.1f}"}
    for scene in MJCF_PATHS:
        row[f"velocity_{scene}"] = res["scene_avg_velocities"].get(scene, 0)
        row[f"cost_{scene}"] = res["scene_costs"].get(scene, 0)
    for rid in [row["id"] for row in _REF_ROWS]:
        row[f"velocity_{rid}"] = res["ref_avg_velocities"].get(rid, 0)
        row[f"cost_{rid}"] = res["ref_costs"].get(rid, 0)
        row[f"lateral_{rid}"] = res.get("ref_lateral", {}).get(rid, 0)
        row[f"tumble_{rid}"] = res.get("ref_tumble", {}).get(rid, 0)
        row[f"yaw_{rid}"] = res.get("ref_yaw", {}).get(rid, 0)
        row[f"pitch_rms_{rid}"] = res.get("ref_pitch_rms", {}).get(rid, 0)
    row.update(res["params"])
    p = res["params"]
    row["solimp_dmax"] = p["solimp_dmin"] + p["solimp_delta_d"] * (0.9999 - p["solimp_dmin"])
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
    ref_rows = _REF_ROWS  # use filtered set when --scenes is active
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
    ry = r.get("ref_yaw", {})
    pad = " " * indent
    # Header
    print(f"{pad}{'ref_id':<18} {'target':>7} {'sim':>7} {'Δvel':>9} {'Δ%':>5} {'tumble':>7} {'lateral':>8} {'yaw':>5} {'pitch':>6}")
    print(f"{pad}{'-'*78}")
    for row in ref_rows:
        rid = row["id"]
        target = row["speed"]
        sim_v = rv.get(rid, 0.0)
        delta = (sim_v - target) * 100  # cm/s
        delta_pct = ((sim_v - target) / target * 100) if target != 0 else 0.0
        tmb = rt.get(rid, 0.0)
        lat = rl.get(rid, 0.0) * 100  # cm
        yaw = ry.get(rid, 0.0)
        pit = rp.get(rid, 0.0)
        print(f"{pad}{rid:<18} {target:>6.3f}  {sim_v:>6.3f}  {delta:>+7.1f}cs {delta_pct:>+4.0f}%  {tmb:>6.4f}  {lat:>6.1f}cm  {yaw:>4.0f}°  {pit:>4.1f}°")


_best_cost_so_far: float = float("inf")


def _best_csv_fieldnames() -> list[str]:
    """Column names for the bests CSV (shared by header writer and append)."""
    ref_ids = [row["id"] for row in _REF_ROWS]
    return (
        ["timestamp", "elapsed_min", "n_eval", "id", "cost"]
        + [f"vel_{rid}" for rid in ref_ids]
        + [f"lateral_{rid}" for rid in ref_ids]
        + [f"tumble_{rid}" for rid in ref_ids]
        + [f"yaw_{rid}" for rid in ref_ids]
        + [f"pitch_rms_{rid}" for rid in ref_ids]
        + [dim.name for dim in space]
        + ["solimp_dmax"]  # derived from delta_d
    )


def _append_best_csv(best: dict, n_done: int, elapsed_min: float) -> None:
    """Append a new-best row to the running bests CSV."""
    from datetime import datetime
    ref_rows = _REF_ROWS
    ref_ids = [row["id"] for row in ref_rows]
    rv = best.get("ref_avg_velocities", {})
    rl = best.get("ref_lateral", {})
    rt = best.get("ref_tumble", {})
    ry = best.get("ref_yaw", {})
    rp = best.get("ref_pitch_rms", {})

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
            row[f"pitch_rms_{rid}"] = f"{rp.get(rid, 0.0):.2f}"
        for dim in space:
            row[dim.name] = repr(best['params'][dim.name])
        bp = best["params"]
        row["solimp_dmax"] = repr(bp['solimp_dmin'] + bp['solimp_delta_d'] * (0.9999 - bp['solimp_dmin']))
        w.writerow(row)


def _print_best_so_far(all_results: list[dict], n_done: int, elapsed_min: float) -> None:
    """Print best result and append to bests CSV if improved."""
    global _best_cost_so_far
    best = min(all_results, key=lambda r: r["cost"])
    ref_rows = _REF_ROWS
    is_new_best = best["cost"] < _best_cost_so_far
    marker = " ★ NEW BEST" if is_new_best else ""
    print(f"  Best so far (n={n_done}): cost={best['cost']:.6f}  id={best['id']}{marker}")
    _print_ref_table(best, ref_rows, indent=4)

    if is_new_best:
        _best_cost_so_far = best["cost"]
        _append_best_csv(best, n_done, elapsed_min)


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CMA-ES space mapping helpers
# ---------------------------------------------------------------------------

def _cmaes_space_info():
    """Build CMA-ES bounds, initial point, and log-space flags from skopt space.

    If CMAES_X0 is set, uses those values as the starting point (warm start).
    Otherwise, starts from the midpoint of each dimension (cold start).
    Log-uniform dimensions are optimized in log10-space for CMA-ES.
    Returns (x0, lower, upper, is_log) all as lists of length len(space).
    """
    x0 = []
    lower = []
    upper = []
    is_log = []
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
    """Map CMA-ES internal point back to real (original) space."""
    point = []
    for val, log_flag in zip(x_internal, is_log):
        if log_flag:
            point.append(10.0 ** val)
        else:
            point.append(val)
    return point


# ---------------------------------------------------------------------------
# Optimizer factories (ask/tell wrappers)
# ---------------------------------------------------------------------------

def _create_skopt_optimizer():
    """Create a scikit-optimize Bayesian optimizer."""
    from skopt import Optimizer

    estimator = BASE_ESTIMATOR
    if BASE_ESTIMATOR == "gp" and OPTIMIZER_NOISE is not None:
        from skopt.learning import GaussianProcessRegressor
        from skopt.learning.gaussian_process.kernels import Matern
        estimator = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            noise=OPTIMIZER_NOISE,
            n_restarts_optimizer=2,
            normalize_y=True,
        )
    opt_kwargs = dict(
        dimensions=space,
        base_estimator=estimator,
        n_initial_points=N_INITIAL_POINTS,
        random_state=OPTIMIZER_RANDOM_STATE,
        acq_func=ACQ_FUNC,
    )
    if ACQ_FUNC_KWARGS:
        opt_kwargs["acq_func_kwargs"] = ACQ_FUNC_KWARGS

    optimizer = Optimizer(**opt_kwargs)

    def ask(n_points):
        pts = optimizer.ask(n_points=n_points)
        if n_points == 1:
            pts = [pts]
        return pts

    def tell(points, costs):
        optimizer.tell(points, costs)

    return ask, tell


def _create_cmaes_optimizer(es_override=None):
    """Create a CMA-ES optimizer (pycma) with log-space mapping.

    If es_override is provided (a pre-existing CMAEvolutionStrategy),
    it is used directly instead of creating a new one (for --resume-from).
    Returns (ask, tell, es) where es is the raw pycma object.
    """
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
            "verbose": -1,  # suppress pycma's own output
            "tolfun": 1e-8,
            "tolx": 1e-10,
        }
        es = cma.CMAEvolutionStrategy(x0, CMAES_SIGMA0, opts)

    def ask(n_points):
        """Ask returns popsize points (n_points is ignored — CMA-ES has fixed population)."""
        internal_points = es.ask()
        return [_cmaes_to_real(p, is_log) for p in internal_points]

    def tell(points, costs):
        """Tell maps points back to internal space before telling CMA-ES."""
        internal_points = []
        for pt in points:
            internal = []
            for val, log_flag in zip(pt, is_log):
                internal.append(np.log10(val) if log_flag else val)
            internal_points.append(internal)
        es.tell(internal_points, costs)

    return ask, tell, es


def _run_batch_optimization(all_results: list[dict], pool: multiprocessing.Pool,
                            es_resume=None) -> OptResult:
    """
    Batch optimization loop: propose points, evaluate all references
    in parallel, tell optimizer the costs, repeat.

    Supports both skopt (Bayesian) and CMA-ES backends via ask/tell interface.

    Args:
        all_results: Mutable list to accumulate results (replaces global).
        pool: Pre-created multiprocessing pool.
        es_resume: Optional pre-loaded CMA-ES state to resume from.
    """
    es = None  # only set for CMA-ES backend
    if OPTIMIZER_BACKEND == "cmaes":
        ask, tell, es = _create_cmaes_optimizer(es_override=es_resume)
        if es_resume is not None:
            print(f"  Backend: CMA-ES RESUMED (sigma={es.sigma:.4g}, popsize={BATCH_SIZE})")
        else:
            warm = "warm-start" if CMAES_X0 is not None else "cold-start"
            print(f"  Backend: CMA-ES (sigma0={CMAES_SIGMA0}, popsize={BATCH_SIZE}, {warm})")
    else:
        ask, tell = _create_skopt_optimizer()
        es = None
        print(f"  Backend: skopt ({BASE_ESTIMATOR}, acq={ACQ_FUNC})")

    n_done = 0
    batch_num = 0
    t_run_start = time.perf_counter()

    while n_done < N_CALLS:
        n_this = min(BATCH_SIZE, N_CALLS - n_done)
        batch_num += 1
        n_trials = max(1, INIT_JITTER_TRIALS)
        t_batch_start = time.perf_counter()
        print(f"\n--- Batch {batch_num}: asking for {n_this} points ({n_done + 1}\u2013{n_done + n_this} / {N_CALLS}), {n_this * len(_REF_ROWS) * n_trials} tasks ---")

        t_ask = time.perf_counter()
        points = ask(n_this)
        t_ask = time.perf_counter() - t_ask

        # One reference row per process: (point_index, point, ref_row)
        tasks = [
            (
                i,
                point,
                ref_row,
                trial_idx,
                False,  # suppress per-sim progress output
                n_done + i,  # global point index for unique jitter seeds
            )
            for i, point in enumerate(points)
            for ref_idx, ref_row in enumerate(_REF_ROWS)
            for trial_idx in range(n_trials)
        ]
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

        t_verbose = 0.0
        if VERBOSE_BATCH:
            t_verbose = time.perf_counter()
            _print_point_results(results, n_this)
            t_verbose = time.perf_counter() - t_verbose

        batch_wall_actual = time.perf_counter() - t_batch_start
        elapsed = time.perf_counter() - t_run_start
        elapsed_min = elapsed / 60.0
        batch_wall_sum = t_ask + t_sim + t_agg + t_tell + t_csv + t_verbose
        overhead = batch_wall_actual - batch_wall_sum
        print(f"  Batch wall: {batch_wall_actual:.1f}s | Elapsed: {elapsed_min:.1f}min | Costs: min={min(costs):.4f}, max={max(costs):.4f}")

        if PROFILE_BATCH:
            parts = f"ask={t_ask:.3f}s sim={t_sim:.2f}s agg={t_agg:.3f}s tell={t_tell:.3f}s csv={t_csv:.3f}s"
            if VERBOSE_BATCH:
                parts += f" verbose={t_verbose:.3f}s"
            print(f"  Profile: {parts}")

        _print_best_so_far(all_results, n_done, elapsed_min)

        # Save CMA-ES state every batch for resumption.
        # State (sigma, covariance, paths) evolves every generation,
        # not just at new bests — so save after every batch.
        if es is not None:
            import pickle
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
    import shutil

    parser = argparse.ArgumentParser(description="CMA-ES parameter optimization")
    parser.add_argument("--suffix", "-s", type=str, default="", help="Suffix appended to results folder name")
    parser.add_argument("--scenes", nargs="+", default=None, help="Only optimize for these scene keys (e.g. scene1 scene4)")
    parser.add_argument("--freqs", nargs="+", type=float, default=None, help="Only optimize for these ctrl freqs (e.g. 10 30)")
    parser.add_argument("--n-calls", type=int, default=None, help="Override N_CALLS from config")
    parser.add_argument("--warm-start-from", type=str, default=None,
                        help="Results dir (or path to optimization_bests.csv) to warm-start from. "
                             "Reads best params from last row of optimization_bests.csv.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Results dir containing cmaes_state.pkl to resume from. "
                             "Restores full CMA-ES state (sigma, covariance, paths).")
    args = parser.parse_args()

    # Resume: load full CMA-ES state (mutually exclusive with warm-start)
    es_resume = None
    if args.resume_from and args.warm_start_from:
        print("ERROR: --resume-from and --warm-start-from are mutually exclusive")
        sys.exit(1)
    if args.resume_from:
        import pickle
        resume_path = pathlib.Path(args.resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "cmaes_state.pkl"
        if not resume_path.exists():
            print(f"ERROR: resume state not found: {resume_path}")
            sys.exit(1)
        with open(resume_path, "rb") as f:
            state = pickle.load(f)
        es_resume = state["es"]
        prev_n_done = state["n_done"]
        # Validate space bounds match the pickled state
        saved_bounds = state.get("space_bounds")
        if saved_bounds is not None:
            current_bounds = [(d.name, d.low, d.high, d.prior) for d in space]
            if saved_bounds != current_bounds:
                resume_dir = resume_path.parent if resume_path.name == "cmaes_state.pkl" else resume_path
                print("ERROR: current config.py space does not match the resumed run.")
                print(f"  Copy the saved config: cp {resume_dir}/config.py mujoco_refactor/config.py")
                sys.exit(1)
        print(f"Resuming from {resume_path} (sigma={es_resume.sigma:.4g}, prev evals={prev_n_done})")

    # Warm-start: load best params from a previous run
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

    # Filter reference rows by scene and/or frequency
    if args.scenes:
        _REF_ROWS = [r for r in _REF_ROWS if r["scene"] in args.scenes]
    if args.freqs:
        _REF_ROWS = [r for r in _REF_ROWS if r["ctrl_freq"] in args.freqs]
    if args.scenes or args.freqs:
        _REF_INDEX_BY_ID = {row["id"]: i for i, row in enumerate(_REF_ROWS)}
        if not _REF_ROWS:
            print(f"ERROR: no reference rows match scenes={args.scenes} freqs={args.freqs}")
            sys.exit(1)
        print(f"Filtered to {len(_REF_ROWS)} refs: {[r['id'] for r in _REF_ROWS]}")

    # Override eval budget if requested
    if args.n_calls is not None:
        N_CALLS = args.n_calls

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

    # Create run directory and save config snapshot
    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.suffix:
        run_tag += f"_{args.suffix}"
    run_dir_results = pathlib.Path("results") / run_tag
    run_dir_results.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pathlib.Path(__file__).parent / "config.py", run_dir_results / "config.py")
    print(f"  Run directory: {run_dir_results}/")

    # Write CSVs directly into run directory
    global CSV_PATH, BEST_CSV_PATH
    CSV_PATH = str(run_dir_results / "multi_optimization_results.csv")
    BEST_CSV_PATH = str(run_dir_results / "optimization_bests.csv")

    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fieldnames()).writeheader()
    with open(BEST_CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_best_csv_fieldnames()).writeheader()

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
        result = _run_batch_optimization(all_results, pool, es_resume=es_resume)
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
    best_params = {dim.name: value for dim, value in zip(space, result.x)}
    for name, value in best_params.items():
        print(f"  {name}: {value:.6f}")
    dmax = best_params["solimp_dmin"] + best_params["solimp_delta_d"] * (0.9999 - best_params["solimp_dmin"])
    print(f"  solimp_dmax: {dmax:.6f}  (derived)")

    # Record top 1 rollout into the run results directory
    print("\n--- Recording Best Rollout ---")
    import importlib
    sim_module = importlib.import_module(SIM_MODULE)

    sorted_results = sorted(all_results, key=lambda r: r["cost"])

    for i in range(min(1, len(sorted_results))):
        result_data = sorted_results[i]
        rank = i + 1

        print(f"\n#{rank}: Cost={result_data['cost']:.6f}")
        for scene, velocity in result_data["scene_avg_velocities"].items():
            scene_cost = result_data["scene_costs"].get(scene)
            cost_str = f"{scene_cost:.4f}" if scene_cost is not None else "N/A"
            print(f"  - {scene}: Avg Velocity={velocity:.4f} m/s (Cost: {cost_str})")

        sim_params = sim_params_from_point(
            [result_data["params"][dim.name] for dim in space]
        )

        for ref_row in _REF_ROWS:
            scene = ref_row["scene"]
            mjcf_path = MJCF_PATHS[scene]
            ref_id = ref_row["id"]
            video_path = run_dir_results / f"rank_{rank:02d}_{ref_id}.mp4"
            print(f"  Recording video for {ref_id} to {video_path}...")
            sim_params_scene = dict(sim_params)
            sim_params_scene["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)
            sim_module.run_simulation(
                sim_params_scene,
                mjcf_path=mjcf_path,
                sim_duration=5.0,
                record_path=str(video_path),
            )

    print(f"\n  Results saved to {run_dir_results}/")
