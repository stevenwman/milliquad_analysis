"""
Multi-terrain optimizer: fits contact/friction/magnetic params to BOTH flat and
step terrain experimental velocities using CMA-ES.

Key features:
  - 19 reference conditions (11 flat + 8 step)
  - Terrain-specific aggregation: MEDIAN for flat, BEST for step
  - Hierarchical cost: within-terrain → across-terrain weighted sum
  - scene_wheel f20 = failure mode constraint (target velocity = 0.0)

Usage:
    cd mujoco_refactor
    uv run python optimizer_multi_terrain.py --suffix multi_v1
    uv run python optimizer_multi_terrain.py --n-calls 100 --suffix smoke
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

from config_multi_terrain import (
    BATCH_SIZE,
    BEST_CSV_PATH,
    CMAES_SIGMA0,
    CMAES_X0,
    COST_FAILURE,
    CSV_PATH,
    DEFAULT_CTRL_FREQ,
    FLAT_TERRAIN_WEIGHT,
    INIT_JITTER_SEED,
    INIT_JITTER_TRIALS,
    INIT_YAW_JITTER_DEG,
    FLAT_LATERAL_COST_WEIGHT,
    FLAT_TUMBLE_COST_WEIGHT,
    FLAT_VELOCITY_COST_WEIGHT,
    FLAT_YAW_COST_WEIGHT,
    STEP_LATERAL_COST_WEIGHT,
    STEP_TUMBLE_COST_WEIGHT,
    STEP_VELOCITY_COST_WEIGHT,
    STEP_YAW_COST_WEIGHT,
    MJCF_PATHS,
    N_CALLS,
    OPTIMIZER_RANDOM_STATE,
    PROFILE_BATCH,
    SETTLE_TIME,
    SIM_DURATION,
    SIMULATION_TIMEOUT,
    STEP_PRESET,
    STEP_START_X,
    STEP_TERRAIN_WEIGHT,
    TUMBLE_PENALTY_SCALE,
    TUMBLE_THRESHOLD,
    VELOCITY_DEADZONE,
    VELOCITY_VARIANCE_WEIGHT,
    VERBOSE_BATCH,
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
    raise ValueError("No reference rows defined. Populate REFERENCE_DATA in config_multi_terrain.py.")
_REF_INDEX_BY_ID: dict[str, int] = {row["id"]: i for i, row in enumerate(_REF_ROWS)}

# Module-level dicts — populated in __main__ before Pool creation.
# Workers receive paths via task tuple (spawn re-imports see empty dicts).
MJCF_FLAT_PATHS: dict[str, str] = {}  # scene -> flat XML path
MJCF_STEP_PATHS: dict[str, str] = {}  # scene -> step XML path


# ---------------------------------------------------------------------------
# Step XML generation
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
# Terrain-specific cost functions
# ---------------------------------------------------------------------------

_BODY_Z_LOCAL = np.array([0.0, 0.0, 1.0])
_NOMINAL_BODY_Z_WORLD = np.array([0.0, 0.0, -1.0])
_BODY_X_LOCAL = np.array([1.0, 0.0, 0.0])


def calculate_cost_flat(
    trajectory: list[dict],
    target_velocity: float,
    speed_std: float = 0.0,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Flat terrain cost function (after settle time).

    Similar to the original 13-dim optimizer cost function.
    """
    if not trajectory:
        return {
            "total_cost": COST_FAILURE,
            "avg_forward_velocity": 0.0,
            "tumble_penalty": 0.0,
            "lateral_displacement": 0.0,
            "yaw_deviation_deg": 0.0,
        }

    # Find post-settle states
    active_states = [s for s in trajectory if s["time"] >= SETTLE_TIME]
    if not active_states:
        return {
            "total_cost": COST_FAILURE,
            "avg_forward_velocity": 0.0,
            "tumble_penalty": 0.0,
            "lateral_displacement": 0.0,
            "yaw_deviation_deg": 0.0,
        }

    # Velocity (from settle to end)
    start_state = active_states[0]
    final_state = active_states[-1]
    active_duration = final_state["time"] - start_state["time"]
    if active_duration < 1e-6:
        return {
            "total_cost": COST_FAILURE,
            "avg_forward_velocity": 0.0,
            "tumble_penalty": 0.0,
            "lateral_displacement": 0.0,
            "yaw_deviation_deg": 0.0,
        }

    forward_displacement = final_state["pos"][0] - start_state["pos"][0]
    avg_velocity = forward_displacement / active_duration

    # Velocity cost
    if target_velocity > 1e-6:
        # Normal case: relative error
        if VELOCITY_DEADZONE:
            err = abs(avg_velocity - target_velocity)
            if err <= speed_std:
                velocity_cost = 0.0
            else:
                velocity_cost = ((err - speed_std) / target_velocity) ** 2
        else:
            velocity_cost = ((avg_velocity - target_velocity) / target_velocity) ** 2
    else:
        # Failure mode constraint (target ≈ 0): absolute velocity penalty
        velocity_cost = avg_velocity ** 2

    # Lateral displacement
    lateral_displacement = abs(final_state["pos"][1])
    lateral_cost = lateral_displacement

    # Tumble (pitch+roll RMS)
    tumbles = []
    for s in active_states:
        quat = s["quat"]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        body_z = rot.apply(_BODY_Z_LOCAL)
        tumble = np.arccos(np.clip(np.dot(body_z, _NOMINAL_BODY_Z_WORLD), -1, 1))
        tumbles.append(tumble)
    tumble_rms = np.sqrt(np.mean(np.array(tumbles) ** 2))
    tumble_penalty = max(0.0, tumble_rms - TUMBLE_THRESHOLD) * TUMBLE_PENALTY_SCALE

    # Yaw deviation
    body_x = rot.apply(_BODY_X_LOCAL)
    yaw_rad = np.arctan2(body_x[1], body_x[0])
    yaw_deg = np.rad2deg(abs(yaw_rad))
    yaw_cost = max(0.0, yaw_deg - YAW_THRESHOLD_DEG) / 180.0

    total_cost = (
        FLAT_VELOCITY_COST_WEIGHT * velocity_cost
        + FLAT_LATERAL_COST_WEIGHT * lateral_cost
        + FLAT_TUMBLE_COST_WEIGHT * tumble_penalty
        + FLAT_YAW_COST_WEIGHT * yaw_cost
    )

    return {
        "total_cost": total_cost,
        "avg_forward_velocity": avg_velocity,
        "tumble_penalty": tumble_penalty,
        "lateral_displacement": lateral_displacement,
        "yaw_deviation_deg": yaw_deg,
    }


def calculate_cost_step(
    trajectory: list[dict],
    target_velocity: float,
    speed_std: float = 0.0,
    step_start_x: float = 0.0,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Step terrain cost function (step-aware velocity measurement).

    Velocity measured only after robot enters step field (x >= step_start_x).
    Similar to optimizer_step.py cost function.
    """
    if not trajectory:
        return {
            "total_cost": COST_FAILURE,
            "avg_forward_velocity": 0.0,
            "tumble_penalty": 0.0,
            "lateral_displacement": 0.0,
            "yaw_deviation_deg": 0.0,
        }

    # Find first state after step start
    start_state = None
    for state in trajectory:
        if state["pos"][0] >= step_start_x:
            start_state = state
            break

    if start_state is None:
        return {
            "total_cost": COST_FAILURE,
            "avg_forward_velocity": 0.0,
            "tumble_penalty": 0.0,
            "lateral_displacement": 0.0,
            "yaw_deviation_deg": 0.0,
        }

    final_state = trajectory[-1]
    active_duration = final_state["time"] - start_state["time"]

    if active_duration < 1e-6:
        return {
            "total_cost": COST_FAILURE,
            "avg_forward_velocity": 0.0,
            "tumble_penalty": 0.0,
            "lateral_displacement": 0.0,
            "yaw_deviation_deg": 0.0,
        }

    forward_displacement = final_state["pos"][0] - start_state["pos"][0]
    avg_velocity = forward_displacement / active_duration

    # Velocity cost (no deadzone for steps)
    if target_velocity > 1e-6:
        # Normal case: relative error
        velocity_cost = ((avg_velocity - target_velocity) / target_velocity) ** 2
    else:
        # Failure mode constraint (target ≈ 0): absolute velocity penalty
        velocity_cost = avg_velocity ** 2

    # Lateral displacement
    lateral_displacement = abs(final_state["pos"][1])
    lateral_cost = lateral_displacement

    # Tumble (only in step region)
    tumbles = []
    for s in trajectory:
        if s["pos"][0] >= step_start_x:
            quat = s["quat"]
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            body_z = rot.apply(_BODY_Z_LOCAL)
            tumble = np.arccos(np.clip(np.dot(body_z, _NOMINAL_BODY_Z_WORLD), -1, 1))
            tumbles.append(tumble)
    tumble_rms = np.sqrt(np.mean(np.array(tumbles) ** 2)) if tumbles else 0.0
    tumble_penalty = max(0.0, tumble_rms - TUMBLE_THRESHOLD) * TUMBLE_PENALTY_SCALE

    # Yaw deviation
    quat = final_state["quat"]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    body_x = rot.apply(_BODY_X_LOCAL)
    yaw_rad = np.arctan2(body_x[1], body_x[0])
    yaw_deg = np.rad2deg(abs(yaw_rad))
    yaw_cost = max(0.0, yaw_deg - YAW_THRESHOLD_DEG) / 180.0

    total_cost = (
        STEP_VELOCITY_COST_WEIGHT * velocity_cost
        + STEP_LATERAL_COST_WEIGHT * lateral_cost
        + STEP_TUMBLE_COST_WEIGHT * tumble_penalty
        + STEP_YAW_COST_WEIGHT * yaw_cost
    )

    return {
        "total_cost": total_cost,
        "avg_forward_velocity": avg_velocity,
        "tumble_penalty": tumble_penalty,
        "lateral_displacement": lateral_displacement,
        "yaw_deviation_deg": yaw_deg,
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
    from config_multi_terrain import sim_params_from_point as _sim_params_from_point

    sim_params = _sim_params_from_point(point)
    scene_name = ref_row["scene"]
    terrain_type = ref_row["terrain"]
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
        }
    else:
        if terrain_type == "flat":
            cost_data = calculate_cost_flat(
                trajectory,
                target_velocity,
                speed_std=speed_std,
                verbose=False,
            )
        elif terrain_type == "step":
            cost_data = calculate_cost_step(
                trajectory,
                target_velocity,
                speed_std=speed_std,
                step_start_x=STEP_START_X,
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown terrain type: {terrain_type}")

    wall_time = time.perf_counter() - t0

    return (
        point_index,
        ref_row["id"],
        scene_name,
        terrain_type,
        cost_data["total_cost"],
        cost_data["avg_forward_velocity"],
        cost_data.get("tumble_penalty", 0.0),
        cost_data.get("lateral_displacement", 0.0),
        cost_data.get("yaw_deviation_deg", 0.0),
        weight,
        wall_time,
    )


# ---------------------------------------------------------------------------
# Result aggregation (terrain-specific)
# ---------------------------------------------------------------------------

def _aggregate_scene_results(points: list, scene_results: list) -> list[dict]:
    """
    Turn list of per-trial results into full result dicts (one per point).

    Implements terrain-specific aggregation:
      - Flat terrain: MEDIAN of jitter trials
      - Step terrain: BEST (argmin cost) of jitter trials
    """
    by_point = defaultdict(
        lambda: {
            "ref_trials_costs": defaultdict(list),
            "ref_trials_velocities": defaultdict(list),
            "ref_trials_tumble": defaultdict(list),
            "ref_trials_lateral": defaultdict(list),
            "ref_trials_yaw": defaultdict(list),
            "ref_weights": {},
            "ref_scene": {},
            "ref_terrain": {},
            "scene_costs": defaultdict(lambda: defaultdict(float)),  # [scene][terrain] -> cost
            "scene_vel_num": defaultdict(lambda: defaultdict(float)),
            "scene_tumble_num": defaultdict(lambda: defaultdict(float)),
            "scene_lateral_num": defaultdict(lambda: defaultdict(float)),
            "scene_weight": defaultdict(lambda: defaultdict(float)),
            "terrain_costs": defaultdict(float),  # [terrain] -> cost
            "terrain_weight": defaultdict(float),
            "scene_wall_times": [],
            "has_failure": False,
        }
    )
    for (
        point_index, ref_id, scene_name, terrain_type, cost, velocity,
        tumble, lateral, yaw_deg, weight, wall_time,
    ) in scene_results:
        d = by_point[point_index]
        d["ref_trials_costs"][ref_id].append(cost)
        d["ref_trials_velocities"][ref_id].append(velocity)
        d["ref_trials_tumble"][ref_id].append(tumble)
        d["ref_trials_lateral"][ref_id].append(lateral)
        d["ref_trials_yaw"][ref_id].append(yaw_deg)
        if cost >= COST_FAILURE:
            d["has_failure"] = True
        d["ref_weights"][ref_id] = weight
        d["ref_scene"][ref_id] = scene_name
        d["ref_terrain"][ref_id] = terrain_type
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
        ref_best_trial = {}

        for ref_id, trials in d["ref_trials_costs"].items():
            scene = d["ref_scene"][ref_id]
            terrain = d["ref_terrain"][ref_id]
            weight = d["ref_weights"][ref_id]

            # Terrain-specific aggregation
            if terrain == "flat":
                # MEDIAN aggregation for flat terrain
                median_idx = int(np.argsort(trials)[len(trials) // 2])
                agg_cost = float(trials[median_idx])
                agg_vel = float(d["ref_trials_velocities"][ref_id][median_idx])
                agg_tumble = float(d["ref_trials_tumble"][ref_id][median_idx])
                agg_lateral = float(d["ref_trials_lateral"][ref_id][median_idx])
                agg_yaw = float(d["ref_trials_yaw"][ref_id][median_idx])
                best_trial_idx = median_idx
            elif terrain == "step":
                # BEST (argmin) aggregation for step terrain
                best_idx = int(np.argmin(trials))
                agg_cost = float(trials[best_idx])
                agg_vel = float(d["ref_trials_velocities"][ref_id][best_idx])
                agg_tumble = float(d["ref_trials_tumble"][ref_id][best_idx])
                agg_lateral = float(d["ref_trials_lateral"][ref_id][best_idx])
                agg_yaw = float(d["ref_trials_yaw"][ref_id][best_idx])
                best_trial_idx = best_idx
            else:
                raise ValueError(f"Unknown terrain: {terrain}")

            ref_costs[ref_id] = agg_cost
            ref_avg_velocities[ref_id] = agg_vel
            ref_tumble[ref_id] = agg_tumble
            ref_lateral[ref_id] = agg_lateral
            ref_yaw[ref_id] = agg_yaw
            ref_best_trial[ref_id] = best_trial_idx

            # Accumulate per-scene-terrain and per-terrain costs
            d["scene_costs"][scene][terrain] += weight * agg_cost
            d["scene_vel_num"][scene][terrain] += weight * agg_vel
            d["scene_tumble_num"][scene][terrain] += weight * agg_tumble
            d["scene_lateral_num"][scene][terrain] += weight * agg_lateral
            d["scene_weight"][scene][terrain] += weight
            d["terrain_costs"][terrain] += weight * agg_cost
            d["terrain_weight"][terrain] += weight

        # Compute scene-level and terrain-level averages
        scene_avg_velocities = {}
        scene_avg_costs = {}
        for scene in d["scene_costs"]:
            for terrain in d["scene_costs"][scene]:
                key = f"{scene}_{terrain}"
                w = d["scene_weight"][scene][terrain]
                scene_avg_velocities[key] = d["scene_vel_num"][scene][terrain] / w if w > 0 else 0.0
                scene_avg_costs[key] = d["scene_costs"][scene][terrain] / w if w > 0 else 0.0

        terrain_avg_velocities = {}
        terrain_avg_costs = {}
        for terrain in d["terrain_costs"]:
            w = d["terrain_weight"][terrain]
            terrain_avg_velocities[terrain] = sum(
                d["scene_vel_num"][s][terrain] for s in d["scene_vel_num"]
            ) / w if w > 0 else 0.0
            terrain_avg_costs[terrain] = d["terrain_costs"][terrain] / w if w > 0 else 0.0

        # Add velocity variance penalty within each terrain
        # (penalizes inconsistent performance across references)
        ref_rows_lookup = {row["id"]: row for row in reference_rows()}
        terrain_variance_penalties = {}

        for terrain in ["flat", "step"]:
            # Get all references for this terrain
            terrain_refs = [rid for rid in ref_avg_velocities if ref_rows_lookup[rid]["terrain"] == terrain]

            if len(terrain_refs) > 1:
                # Compute relative velocity errors for this terrain
                rel_errors = []
                for rid in terrain_refs:
                    target_vel = ref_rows_lookup[rid]["speed"]
                    if target_vel > 1e-6:  # Skip failure mode constraints (target=0)
                        sim_vel = ref_avg_velocities[rid]
                        rel_error = (sim_vel - target_vel) / target_vel
                        rel_errors.append(rel_error)

                # Add variance penalty if we have multiple valid errors
                if len(rel_errors) > 1:
                    variance_penalty = VELOCITY_VARIANCE_WEIGHT * float(np.var(rel_errors))
                    terrain_variance_penalties[terrain] = variance_penalty
                    terrain_avg_costs[terrain] += variance_penalty
                else:
                    terrain_variance_penalties[terrain] = 0.0
            else:
                terrain_variance_penalties[terrain] = 0.0

        # Hierarchical cost: weighted sum across terrains (includes variance)
        total_cost = (
            FLAT_TERRAIN_WEIGHT * terrain_avg_costs.get("flat", 0.0)
            + STEP_TERRAIN_WEIGHT * terrain_avg_costs.get("step", 0.0)
        )

        avg_wall_time = float(np.mean(d["scene_wall_times"])) if d["scene_wall_times"] else 0.0
        result = {
            "id": str(uuid.uuid4().hex[:8]),
            "cost": total_cost,
            "params": params,
            "ref_costs": ref_costs,
            "ref_avg_velocities": ref_avg_velocities,
            "ref_tumble": ref_tumble,
            "ref_lateral": ref_lateral,
            "ref_yaw": ref_yaw,
            "ref_best_trial": ref_best_trial,
            "scene_avg_velocities": scene_avg_velocities,
            "scene_avg_costs": scene_avg_costs,
            "terrain_avg_velocities": terrain_avg_velocities,
            "terrain_avg_costs": terrain_avg_costs,
            "wall_time": avg_wall_time,
            "has_failure": d["has_failure"],
        }
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def _append_result_to_csv(r: dict, elapsed_min: float) -> None:
    """Append one optimization result to the main CSV."""
    try:
        needs_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fieldnames())
            if needs_header:
                w.writeheader()
            params = r["params"]
            ref_costs = r.get("ref_costs", {})
            ref_velocities = r.get("ref_avg_velocities", {})
            ref_lateral = r.get("ref_lateral", {})
            ref_tumble = r.get("ref_tumble", {})
            ref_yaw = r.get("ref_yaw", {})
            ref_best_trial = r.get("ref_best_trial", {})
            scene_velocities = r.get("scene_avg_velocities", {})
            scene_costs = r.get("scene_avg_costs", {})
            terrain_velocities = r.get("terrain_avg_velocities", {})
            terrain_costs = r.get("terrain_avg_costs", {})
            row = {
                "id": r["id"],
                "cost": f"{r['cost']:.6f}",
                "elapsed_min": f"{elapsed_min:.1f}",
            }
            for t in ["flat", "step"]:
                row[f"velocity_{t}"] = f"{terrain_velocities.get(t, 0.0):.4f}"
                row[f"cost_{t}"] = f"{terrain_costs.get(t, 0.0):.6f}"
            for key in scene_velocities:
                row[f"velocity_{key}"] = f"{scene_velocities[key]:.4f}"
            for key in scene_costs:
                row[f"cost_{key}"] = f"{scene_costs[key]:.6f}"
            for rid in reference_ids():
                row[f"velocity_{rid}"] = f"{ref_velocities.get(rid, 0.0):.4f}"
                row[f"cost_{rid}"] = f"{ref_costs.get(rid, 0.0):.6f}"
                row[f"lateral_{rid}"] = f"{ref_lateral.get(rid, 0.0):.6f}"
                row[f"tumble_{rid}"] = f"{ref_tumble.get(rid, 0.0):.6f}"
                row[f"yaw_{rid}"] = f"{ref_yaw.get(rid, 0.0):.1f}"
                row[f"best_trial_{rid}"] = ref_best_trial.get(rid, "")
            for dim in space:
                row[dim.name] = float(params[dim.name])
            row["solimp_dmax"] = float(params['solimp_dmin'] + params['solimp_delta_d'] * (0.9999 - params['solimp_dmin']))
            w.writerow(row)
    except Exception as e:
        print(f"  [Warning] Could not append to CSV: {e}")


# ---------------------------------------------------------------------------
# Batch summary printing
# ---------------------------------------------------------------------------

def _print_point_results(results: list[dict], n_this: int) -> None:
    """Print per-point results as a compact table."""
    ref_rows = _REF_ROWS
    for i, r in enumerate(results):
        wt = r.get("wall_time", 0)
        print(f"    [{i+1}/{n_this}] id={r['id']}  cost={r['cost']:.4f}  time={wt:.1f}s")

        # Print terrain-level summary
        terrain_vel = r.get("terrain_avg_velocities", {})
        terrain_cost = r.get("terrain_avg_costs", {})
        print(f"      TERRAIN LEVEL:")
        print(f"        FLAT: vel={terrain_vel.get('flat', 0)*100:>5.1f} cm/s  cost={terrain_cost.get('flat', 0):>7.4f}")
        print(f"        STEP: vel={terrain_vel.get('step', 0)*100:>5.1f} cm/s  cost={terrain_cost.get('step', 0):>7.4f}")

        # Print scene-level breakdown
        scene_vel = r.get("scene_avg_velocities", {})
        scene_cost = r.get("scene_avg_costs", {})
        print(f"      SCENE LEVEL:")
        for scene in ["scene1", "scene2", "scene4", "scene_wheel"]:
            flat_key = f"{scene}_flat"
            step_key = f"{scene}_step"
            flat_vel = scene_vel.get(flat_key, 0) * 100
            flat_cost = scene_cost.get(flat_key, 0)
            step_vel = scene_vel.get(step_key, 0) * 100
            step_cost = scene_cost.get(step_key, 0)
            print(f"        {scene:12}  flat: vel={flat_vel:>5.1f}cm/s cost={flat_cost:>7.4f}   step: vel={step_vel:>5.1f}cm/s cost={step_cost:>7.4f}")

        print(f"      INDIVIDUAL REFS:")
        _print_ref_table(r, ref_rows, indent=8)


def _print_ref_table(r: dict, ref_rows: list[dict], indent: int = 4) -> None:
    """Print a compact table of per-reference-row results."""
    rv = r.get("ref_avg_velocities", {})
    rt = r.get("ref_tumble", {})
    rl = r.get("ref_lateral", {})
    ry = r.get("ref_yaw", {})
    pad = " " * indent
    print(f"{pad}{'ref_id':<22} {'target':>7} {'sim':>7} {'Δvel':>9} {'Δ%':>5} {'tumble':>7} {'lateral':>8} {'yaw':>5}")
    print(f"{pad}{'-'*80}")
    for row in ref_rows:
        rid = row["id"]
        target = row["speed"]
        sim_v = rv.get(rid, 0.0)
        delta = (sim_v - target) * 100
        delta_pct = ((sim_v - target) / target * 100) if target != 0 else 0.0
        tmb = rt.get(rid, 0.0)
        lat = rl.get(rid, 0.0) * 100
        yaw = ry.get(rid, 0.0)
        print(f"{pad}{rid:<22} {target:>6.3f}  {sim_v:>6.3f}  {delta:>+7.1f}cs {delta_pct:>+4.0f}%  {tmb:>6.4f}  {lat:>6.1f}cm  {yaw:>4.0f}°")


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

    # Print terrain-level costs
    terrain_cost = best.get("terrain_avg_costs", {})
    print(f"    FLAT: {terrain_cost.get('flat', 0):.4f}  |  STEP: {terrain_cost.get('step', 0):.4f}")

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
                            run_dir: pathlib.Path, es_resume=None) -> OptResult:
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
        print(f"\n--- Batch {batch_num}: asking for {n_this} points ({n_done + 1}–{n_done + n_this} / {N_CALLS}), {n_this * len(_REF_ROWS) * n_trials} tasks ---")

        t_ask = time.perf_counter()
        points = ask(n_this)
        t_ask = time.perf_counter() - t_ask

        # Build tasks with terrain-specific MJCF paths
        tasks = []
        for i, point in enumerate(points):
            for ref_row in _REF_ROWS:
                scene = ref_row["scene"]
                terrain = ref_row["terrain"]
                if terrain == "flat":
                    mjcf_path = MJCF_FLAT_PATHS[scene]
                elif terrain == "step":
                    mjcf_path = MJCF_STEP_PATHS[scene]
                else:
                    raise ValueError(f"Unknown terrain: {terrain}")

                for trial_idx in range(n_trials):
                    tasks.append((
                        i, point, ref_row, trial_idx, False, n_done + i, mjcf_path
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
        import pickle
        state_path = run_dir / "cmaes_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump({
                "es": es,
                "n_done": n_done,
                "space_bounds": [(d.name, d.low, d.high, d.prior) for d in space],
            }, f)

    best = min(all_results, key=lambda r: r["cost"])
    return OptResult(fun=best["cost"], x=list(best["params"].values()))


if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Multi-terrain CMA-ES parameter optimization")
    parser.add_argument("--suffix", "-s", type=str, default="", help="Suffix appended to results folder name")
    parser.add_argument("--scenes", nargs="+", default=None, help="Only optimize for these scene keys")
    parser.add_argument("--freqs", nargs="+", type=float, default=None, help="Only optimize for these ctrl freqs")
    parser.add_argument("--n-calls", type=int, default=None, help="Override N_CALLS")
    parser.add_argument("--warm-start-from", type=str, default=None,
                        help="Results dir (or optimization_bests.csv) to warm-start from")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Results dir containing cmaes_state.pkl to resume from")
    parser.add_argument("--terrain", nargs="+", default=None,
                        help="Only optimize for these terrain types (flat, step)")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Override worker pool size (default: os.cpu_count())")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Override results directory (skips auto timestamp naming)")
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
    if args.terrain:
        _REF_ROWS = [r for r in _REF_ROWS if r["terrain"] in args.terrain]
    if args.scenes or args.freqs or args.terrain:
        _REF_INDEX_BY_ID = {row["id"]: i for i, row in enumerate(_REF_ROWS)}
        if not _REF_ROWS:
            print(f"ERROR: no reference rows match scenes={args.scenes} freqs={args.freqs} terrain={args.terrain}")
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

    print(f"Building terrain XMLs ({step_tag}) ...")
    for scene, base_xml in MJCF_PATHS.items():
        # Flat terrain: use original XML
        MJCF_FLAT_PATHS[scene] = base_xml

        # Step terrain: generate step XML
        src_dir = pathlib.Path(base_xml).parent
        stem = pathlib.Path(base_xml).stem
        out_xml = str(src_dir / f"{stem}_{step_tag}.xml")
        _inject_steps(base_xml, STEP_PRESET, out_xml)
        MJCF_STEP_PATHS[scene] = out_xml

    print(f"\nRunning multi-terrain optimization for {N_CALLS} evaluations in batches of {BATCH_SIZE}...")
    print(f"Cost hierarchy: flat_weight={FLAT_TERRAIN_WEIGHT}, step_weight={STEP_TERRAIN_WEIGHT}")
    print(f"Flat aggregation: MEDIAN | Step aggregation: BEST (argmin)")
    print("Reference targets:")
    for row in _REF_ROWS:
        print(
            f"  - {row['id']}: scene={row['scene']} | terrain={row['terrain']} | "
            f"ctrl_freq={row['ctrl_freq']} Hz | "
            f"speed={row['speed']} m/s ({row['speed']*1000:.1f} mm/s) | "
            f"weight={row['weight']}"
        )

    # Create run directory and save config snapshots
    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        suffix = f"_{args.suffix}" if args.suffix else ""
        run_dir = pathlib.Path("results") / f"{timestamp}_multi_terrain{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults directory: {run_dir}")

    # Save config snapshot
    shutil.copy2("config_multi_terrain.py", run_dir / "config_multi_terrain.py")
    CSV_PATH = str(run_dir / CSV_PATH)
    BEST_CSV_PATH = str(run_dir / BEST_CSV_PATH)

    # Initialize best CSV header
    with open(BEST_CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_best_csv_fieldnames())
        w.writeheader()

    # Run optimization with multiprocessing
    with multiprocessing.Pool(processes=args.pool_size or os.cpu_count()) as pool:
        all_results = []
        result = _run_batch_optimization(all_results, pool, run_dir, es_resume=es_resume)

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best cost: {result.fun:.6f}")
    print(f"Results saved to: {run_dir}")
    print(f"  - {CSV_PATH}")
    print(f"  - {BEST_CSV_PATH}")

    # Record best rollout for each ref
    print("\n--- Recording Best Rollout ---")
    import importlib as _importlib
    _sim_rec = _importlib.import_module(SIM_MODULE)
    best_result = min(all_results, key=lambda r: r["cost"])
    best_sim_params = sim_params_from_point(
        [best_result["params"][dim.name] for dim in space]
    )
    for ref_row in _REF_ROWS:
            scene = ref_row["scene"]
            terrain = ref_row["terrain"]
            ref_id = ref_row["id"]
            mjcf_path = MJCF_FLAT_PATHS[scene] if terrain == "flat" else MJCF_STEP_PATHS[scene]
            video_path = run_dir / f"best_{ref_id}.mp4"
            sim_params_rec = dict(best_sim_params)
            sim_params_rec["drive_freq"] = ref_row.get("ctrl_freq", DEFAULT_CTRL_FREQ)
            print(f"  Recording {ref_id} → {video_path.name}")
            _sim_rec.run_simulation(
                sim_params_rec,
                mjcf_path=mjcf_path,
                sim_duration=SIM_DURATION,
                record_path=str(video_path),
            )
