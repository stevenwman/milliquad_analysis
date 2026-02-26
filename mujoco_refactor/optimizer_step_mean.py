"""
Step terrain optimizer: fits contact/friction/magnetic params to experimental
step-terrain velocities using CMA-ES.

Same 13-param search space as the flat optimizer. Key differences:
  - Step-aware cost: velocity measured only in the step region (x >= flat_lead)
  - Reference data: 10 step terrain conditions (mid-300 vx)
  - Pre-builds step terrain XMLs at startup
  - No pitch RMS term

Usage:
    cd mujoco_refactor
    uv run python optimizer_step.py --suffix step_v1
    uv run python optimizer_step.py --scenes scene4 --n-calls 16 --suffix step_smoke
"""

import csv
import multiprocessing
import os
import pathlib
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from config_step import (
    BATCH_SIZE,
    BEST_CSV_PATH,
    CMAES_SIGMA0,
    CMAES_X0,
    COST_FAILURE,
    CSV_PATH,
    DEFAULT_CTRL_FREQ,
    FAILURE_MODE_VEL_SCALE,
    INIT_JITTER_SEED,
    INIT_JITTER_TRIALS,
    INIT_YAW_JITTER_DEG,
    LATERAL_COST_WEIGHT,
    MJCF_PATHS,
    N_CALLS,
    OPTIMIZER_RANDOM_STATE,
    PROFILE_BATCH,
    SETTLE_TIME,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    STEP_PRESET,
    STEP_START_X,
    STEP_END_X,
    PROGRESS_COST_WEIGHT,
    TUMBLE_COST_WEIGHT,
    TUMBLE_PENALTY_SCALE,
    TUMBLE_THRESHOLD,
    VELOCITY_COST_WEIGHT,
    VELOCITY_DEADZONE,
    VELOCITY_VARIANCE_WEIGHT,
    VERBOSE_BATCH,
    YAW_COST_WEIGHT,
    YAW_THRESHOLD_DEG,
    csv_fieldnames,
    point_to_params,
    reference_ids,
    reference_rows,
    sim_params_from_point,
    space,
)

# ---- Simulation module selector ----
SIM_MODULE = "simulation_fast_new"


class OptResult(NamedTuple):
    """Result of optimization run, matching skopt's result interface."""
    fun: float
    x: list[float]


_REF_ROWS = reference_rows()
if not _REF_ROWS:
    raise ValueError("No reference rows defined. Populate REFERENCE_DATA in config_step.py.")
_REF_INDEX_BY_ID: dict[str, int] = {row["id"]: i for i, row in enumerate(_REF_ROWS)}

# Module-level dict — populated in __main__ before Pool creation.
# Workers receive paths via task tuple (spawn re-imports see empty dict).
MJCF_STEP_PATHS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Step XML generation (copied from terrain_test.py)
# ---------------------------------------------------------------------------

def _inject_steps(xml_path: str, preset: dict, out_xml: str) -> str:
    """Add step box geoms to the MJCF and write to out_xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    flat_lead = preset["flat_lead"]
    step_height = preset["step_height"]
    step_length = preset["step_length"]
    step_count = preset["step_count"]
    final_step_length = preset["final_step_length"]
    step_width = preset["step_width"]

    for i in range(step_count):
        is_final = (i == step_count - 1)
        length = final_step_length if is_final else step_length

        if is_final:
            pos_x = flat_lead + (step_count - 1) * step_length + length / 2.0
        else:
            pos_x = flat_lead + i * step_length + length / 2.0
        pos_z = (i + 1) * step_height - step_height / 2.0

        geom = ET.SubElement(worldbody, "geom")
        geom.set("name", f"step_{i}")
        geom.set("type", "box")
        geom.set("size", f"{length/2.0} {step_width/2.0} {step_height/2.0}")
        geom.set("pos", f"{pos_x} 0.0 {pos_z}")
        geom.set("rgba", "0.5 0.5 0.5 1")

    tree.write(out_xml)
    return out_xml


# ---------------------------------------------------------------------------
# Step-aware cost function
# ---------------------------------------------------------------------------

_BODY_Z_LOCAL = np.array([0.0, 0.0, 1.0])
_NOMINAL_BODY_Z_WORLD = np.array([0.0, 0.0, -1.0])
_BODY_X_LOCAL = np.array([1.0, 0.0, 0.0])


def calculate_cost(
    trajectory: list[dict],
    target_velocity: float,
    speed_std: float = 0.0,
    step_start_x: float = 0.0,
    step_end_x: float = 0.0,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Step-aware cost function.

    Measures forward velocity only after the robot enters the step field
    (pos[0] >= step_start_x). No settle time — step position is the only gate.
    Penalizes deviation from target velocity, tumbling, lateral drift, and
    yaw spin-out. No pitch RMS.
    """
    fail = {
        "total_cost": COST_FAILURE, "avg_forward_velocity": 0,
        "tumble_penalty": 0, "lateral_displacement": 0, "yaw_deviation_deg": 0,
        "progress_penalty": 1.0,
    }
    if not trajectory:
        return fail

    # Find when robot enters step field (x >= step_start_x)
    enter_state = None
    for state in trajectory:
        if state["pos"][0] >= step_start_x:
            enter_state = state
            break

    if enter_state is None:
        return fail

    final_state = trajectory[-1]
    active_duration = final_state["time"] - enter_state["time"]
    avg_forward_velocity = 0.0
    if active_duration > 1e-6:
        forward_displacement = final_state["pos"][0] - enter_state["pos"][0]
        avg_forward_velocity = forward_displacement / active_duration

    # Velocity error
    vel_deviation = avg_forward_velocity - target_velocity
    if target_velocity > 1e-6:
        # Normal case: relative squared error
        if VELOCITY_DEADZONE and speed_std > 0.0 and abs(vel_deviation) <= speed_std:
            velocity_error = 0.0
        elif VELOCITY_DEADZONE and speed_std > 0.0:
            excess = abs(vel_deviation) - speed_std
            velocity_error = (excess / target_velocity) ** 2
        else:
            velocity_error = (vel_deviation / target_velocity) ** 2
    else:
        # Failure mode (target=0): penalize any movement, normalized so that
        # moving at FAILURE_MODE_VEL_SCALE produces cost=1.0
        velocity_error = (avg_forward_velocity / FAILURE_MODE_VEL_SCALE) ** 2

    # Lateral displacement (from step entry to end)
    lateral_displacement = abs(final_state["pos"][1] - enter_state["pos"][1])
    lateral_error = lateral_displacement ** 2

    # Tumble penalty (over entire trajectory, normalized per-step)
    tumble_penalty = 0.0
    for state in trajectory:
        quat = state["quat"]
        body_z_axis = R.from_quat(quat, scalar_first=True).apply(_BODY_Z_LOCAL)
        uprightness = np.dot(body_z_axis, _NOMINAL_BODY_Z_WORLD)
        if uprightness < TUMBLE_THRESHOLD:
            tumble_penalty += (1 - uprightness) * TUMBLE_PENALTY_SCALE
    tumble_penalty /= max(len(trajectory), 1)

    # Yaw spin-out (from step entry to end)
    start_body_x = R.from_quat(enter_state["quat"], scalar_first=True).apply(_BODY_X_LOCAL)
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

    # Progress penalty: penalize incomplete step traversal
    progress_penalty = 0.0
    if target_velocity > 1e-6 and step_end_x > step_start_x:
        final_x = final_state["pos"][0]
        step_distance = step_end_x - step_start_x
        progress_fraction = np.clip((final_x - step_start_x) / step_distance, 0.0, 1.0)
        progress_penalty = (1.0 - progress_fraction) ** 2

    total_cost = (
        VELOCITY_COST_WEIGHT * velocity_error
        + TUMBLE_COST_WEIGHT * tumble_penalty
        + LATERAL_COST_WEIGHT * lateral_error
        + YAW_COST_WEIGHT * yaw_penalty
        + PROGRESS_COST_WEIGHT * progress_penalty
    )

    if verbose:
        print(
            f"    Avg Vel: {avg_forward_velocity:.3f} m/s | "
            f"Vel Err: {velocity_error:.4f} | "
            f"Lateral: {lateral_displacement:.4f} m | "
            f"Tumble Pen: {tumble_penalty:.4f} | "
            f"Yaw: {yaw_deviation_deg:.1f}° | "
            f"Progress: {progress_penalty:.4f} | "
            f"Total Cost: {total_cost:.4f}"
        )

    return {
        "total_cost": total_cost,
        "avg_forward_velocity": avg_forward_velocity,
        "lateral_displacement": lateral_displacement,
        "tumble_penalty": tumble_penalty,
        "yaw_deviation_deg": yaw_deviation_deg,
        "progress_penalty": progress_penalty,
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

def _evaluate_one_scene(args):
    """
    Run one trial for one (point, reference row).

    args: (point_index, point, ref_row, trial_index, show_progress,
           global_point_index, mjcf_path)
    """
    point_index, point, ref_row, trial_index, show_progress, global_point_index, mjcf_path = args

    import importlib
    _sim = importlib.import_module(SIM_MODULE)
    from config_step import sim_params_from_point as _sim_params_from_point

    sim_params = _sim_params_from_point(point)
    scene_name = ref_row["scene"]
    target_velocity = ref_row["speed"]
    speed_std = ref_row.get("speed_std", 0.0)
    weight = ref_row.get("weight", 1.0)
    sim_params["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)

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
            "lateral_displacement": 0.0,
            "yaw_deviation_deg": 0.0,
            "progress_penalty": 1.0,
        }
    else:
        cost_data = calculate_cost(
            trajectory,
            target_velocity,
            speed_std=speed_std,
            step_start_x=STEP_START_X,
            step_end_x=STEP_END_X,
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
    """Turn list of per-trial results into full result dicts (one per point)."""
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
            "scene_tumble_num": defaultdict(float),
            "scene_lateral_num": defaultdict(float),
            "scene_weight": defaultdict(float),
            "scene_wall_times": [],
            "has_failure": False,
        }
    )
    for (
        point_index, ref_id, scene_name, cost, velocity,
        tumble, lateral, yaw_deg, progress, weight, wall_time,
    ) in scene_results:
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
            # Mean aggregation across jitter trials
            mean_cost = float(np.mean(trials))
            mean_vel = float(np.mean(d["ref_trials_velocities"][ref_id]))
            mean_tumble = float(np.mean(d["ref_trials_tumble"][ref_id]))
            mean_lateral = float(np.mean(d["ref_trials_lateral"][ref_id]))
            mean_yaw = float(np.mean(d["ref_trials_yaw"][ref_id]))
            mean_progress = float(np.mean(d["ref_trials_progress"][ref_id]))

            ref_costs[ref_id] = mean_cost
            ref_avg_velocities[ref_id] = mean_vel
            ref_tumble[ref_id] = mean_tumble
            ref_lateral[ref_id] = mean_lateral
            ref_yaw[ref_id] = mean_yaw
            ref_progress[ref_id] = mean_progress
            ref_best_trial[ref_id] = -1  # N/A for mean aggregation

            d["scene_costs"][scene] += weight * best_cost
            d["scene_vel_num"][scene] += weight * best_vel
            d["scene_tumble_num"][scene] += weight * best_tumble
            d["scene_lateral_num"][scene] += weight * best_lateral
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
        row[f"progress_{rid}"] = res.get("ref_progress", {}).get(rid, 0)
        row[f"best_trial_{rid}"] = res.get("ref_best_trial", {}).get(rid, "")
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
# Batch summary printing
# ---------------------------------------------------------------------------

def _print_point_results(results: list[dict], n_this: int) -> None:
    """Print per-point results as a compact table with one row per reference."""
    ref_rows = _REF_ROWS
    for i, r in enumerate(results):
        wt = r.get("wall_time", 0)
        print(f"    [{i+1}/{n_this}] id={r['id']}  cost={r['cost']:.4f}  time={wt:.1f}s")
        _print_ref_table(r, ref_rows, indent=6)


def _print_ref_table(r: dict, ref_rows: list[dict], indent: int = 4) -> None:
    """Print a compact table of per-reference-row results."""
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
    """Column names for the bests CSV."""
    ref_ids = [row["id"] for row in _REF_ROWS]
    return (
        ["timestamp", "elapsed_min", "n_eval", "id", "cost"]
        + [f"vel_{rid}" for rid in ref_ids]
        + [f"lateral_{rid}" for rid in ref_ids]
        + [f"tumble_{rid}" for rid in ref_ids]
        + [f"yaw_{rid}" for rid in ref_ids]
        + [f"best_trial_{rid}" for rid in ref_ids]
        + [dim.name for dim in space]
        + ["solimp_dmax"]
    )


def _append_best_csv(best: dict, n_done: int, elapsed_min: float) -> None:
    """Append a new-best row to the running bests CSV."""
    ref_ids = [row["id"] for row in _REF_ROWS]
    rv = best.get("ref_avg_velocities", {})
    rl = best.get("ref_lateral", {})
    rt = best.get("ref_tumble", {})
    ry = best.get("ref_yaw", {})
    rb = best.get("ref_best_trial", {})

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
            row[f"best_trial_{rid}"] = rb.get(rid, "")
        for dim in space:
            row[dim.name] = float(best['params'][dim.name])
        bp = best["params"]
        row["solimp_dmax"] = float(bp['solimp_dmin'] + bp['solimp_delta_d'] * (0.9999 - bp['solimp_dmin']))
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
# CMA-ES space mapping
# ---------------------------------------------------------------------------

def _cmaes_space_info():
    """Build CMA-ES bounds, initial point, and log-space flags from skopt space."""
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
    """Map CMA-ES internal point back to real (original) space."""
    return [10.0 ** val if log_flag else val for val, log_flag in zip(x_internal, is_log)]


def _create_cmaes_optimizer(es_override=None):
    """Create a CMA-ES optimizer (pycma) with log-space mapping."""
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
            internal = [np.log10(val) if log_flag else val for val, log_flag in zip(pt, is_log)]
            internal_points.append(internal)
        es.tell(internal_points, costs)

    return ask, tell, es


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def _run_batch_optimization(all_results: list[dict], pool: multiprocessing.Pool,
                            es_resume=None) -> OptResult:
    """Batch optimization loop: propose, evaluate, tell, repeat."""
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
        print(f"\n--- Batch {batch_num}: asking for {n_this} points ({n_done + 1}\u2013{n_done + n_this} / {N_CALLS}), {n_this * len(_REF_ROWS) * n_trials} tasks ---")

        t_ask = time.perf_counter()
        points = ask(n_this)
        t_ask = time.perf_counter() - t_ask

        tasks = [
            (
                i, point, ref_row, trial_idx, False, n_done + i,
                MJCF_STEP_PATHS[ref_row["scene"]],
            )
            for i, point in enumerate(points)
            for ref_row in _REF_ROWS
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
        print(f"  Batch wall: {batch_wall_actual:.1f}s | Elapsed: {elapsed_min:.1f}min | Costs: min={min(costs):.4f}, max={max(costs):.4f}")

        if PROFILE_BATCH:
            parts = f"ask={t_ask:.3f}s sim={t_sim:.2f}s agg={t_agg:.3f}s tell={t_tell:.3f}s csv={t_csv:.3f}s"
            if VERBOSE_BATCH:
                parts += f" verbose={t_verbose:.3f}s"
            print(f"  Profile: {parts}")

        _print_best_so_far(all_results, n_done, elapsed_min)

        # Save CMA-ES state every batch for resumption
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

    parser = argparse.ArgumentParser(description="Step terrain CMA-ES parameter optimization")
    parser.add_argument("--suffix", "-s", type=str, default="", help="Suffix appended to results folder name")
    parser.add_argument("--scenes", nargs="+", default=None, help="Only optimize for these scene keys")
    parser.add_argument("--freqs", nargs="+", type=float, default=None, help="Only optimize for these ctrl freqs")
    parser.add_argument("--n-calls", type=int, default=None, help="Override N_CALLS")
    parser.add_argument("--warm-start-from", type=str, default=None,
                        help="Results dir (or optimization_bests.csv) to warm-start from")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Results dir containing cmaes_state.pkl to resume from")
    args = parser.parse_args()

    # Resume: load full CMA-ES state
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
        saved_bounds = state.get("space_bounds")
        if saved_bounds is not None:
            current_bounds = [(d.name, d.low, d.high, d.prior) for d in space]
            if saved_bounds != current_bounds:
                print("ERROR: current config space does not match the resumed run.")
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

    # Filter reference rows
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

    if args.n_calls is not None:
        N_CALLS = args.n_calls

    # --- Build step terrain XMLs ---
    h_mm = STEP_PRESET["step_height"] * 1000
    l_mm = STEP_PRESET["step_length"] * 1000
    n_steps = STEP_PRESET["step_count"]
    lead_mm = STEP_PRESET["flat_lead"] * 1000
    step_tag = f"step_{n_steps}x{h_mm:.0f}mm_{l_mm:.1f}L_{lead_mm:.0f}lead"

    print(f"Building step terrain XMLs ({step_tag}) ...")
    for scene, base_xml in MJCF_PATHS.items():
        src_dir = pathlib.Path(base_xml).parent
        stem = pathlib.Path(base_xml).stem
        out_xml = str(src_dir / f"{stem}_{step_tag}.xml")
        _inject_steps(base_xml, STEP_PRESET, out_xml)
        MJCF_STEP_PATHS[scene] = out_xml
        print(f"  {scene}: {out_xml}")

    print(f"\nRunning step terrain optimization for {N_CALLS} evaluations in batches of {BATCH_SIZE}...")
    print(f"Step-aware cost: velocity measured for x >= {STEP_START_X*1000:.0f}mm (step field)")
    print("Reference targets:")
    for row in _REF_ROWS:
        print(
            f"  - {row['id']}: scene={row['scene']} | "
            f"ctrl_freq={row['ctrl_freq']} Hz | "
            f"speed={row['speed']} m/s ({row['speed']*1000:.1f} mm/s) | "
            f"weight={row['weight']}"
        )

    # Create run directory and save config snapshots
    run_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.suffix:
        run_tag += f"_{args.suffix}"
    else:
        run_tag += "_step"
    run_dir_results = pathlib.Path("results") / run_tag
    run_dir_results.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pathlib.Path(__file__).parent / "config_step.py", run_dir_results / "config_step.py")
    shutil.copy2(pathlib.Path(__file__).parent / "config_new.py", run_dir_results / "config_new.py")
    print(f"  Run directory: {run_dir_results}/")

    # Write CSVs into run directory
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
        print(f"\n--- Results appended to {CSV_PATH} after each batch ---")

    print("\n--- Optimization Finished ---")
    best_cost = result.fun
    print(f"Lowest Cost Found: {best_cost:.6f}")
    print("Best Parameters:")
    best_params = {dim.name: value for dim, value in zip(space, result.x)}
    for name, value in best_params.items():
        print(f"  {name}: {value:.6f}")
    dmax = best_params["solimp_dmin"] + best_params["solimp_delta_d"] * (0.9999 - best_params["solimp_dmin"])
    print(f"  solimp_dmax: {dmax:.6f}  (derived)")

    # Record top 1 rollout
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
            mjcf_path = MJCF_STEP_PATHS[scene]
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
