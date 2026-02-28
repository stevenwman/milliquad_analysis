"""Rough terrain optimization config.

Defines REFERENCE_DATA (7 rough conditions, no wheel), terrain geometry,
time-gated cost function with Y-jitter, and CMAES_X0.
"""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from config import (
    PACKAGE_DIR,
    SETTLE_TIME,
    COST_FAILURE,
    _make_ref_id,
)

# ---------------------------------------------------------------------------
# Scene XMLs (rough terrain, RK4 baked in)
# No wheel — only 40% success rate on rough terrain
# ---------------------------------------------------------------------------
MJCF_PATHS: dict[str, str] = {
    "scene1": str(PACKAGE_DIR / "robots" / "quad" / "scene_1_rough.xml"),
    "scene2": str(PACKAGE_DIR / "robots" / "quad" / "scene_2_rough.xml"),
    "scene4": str(PACKAGE_DIR / "robots" / "quad" / "scene_4_rough.xml"),
}

# ---------------------------------------------------------------------------
# Terrain geometry (fixed layout — seed=42 heightmap)
# ---------------------------------------------------------------------------
TERRAIN_NX = 10
TERRAIN_NY = 6
TERRAIN_SL = 0.005
N_TILES = 3
FLAT_LEAD = 0.005

# Spawn offset: center of terrain in X, 10mm above ground
_TOTAL_NX = TERRAIN_NX * N_TILES
_X_HALF = _TOTAL_NX * TERRAIN_SL / 2.0
SPAWN_X = FLAT_LEAD + _X_HALF
SPAWN_Z_RAISE = 0.01

# ---------------------------------------------------------------------------
# Simulation / optimization parameters
# ---------------------------------------------------------------------------
SIM_DURATION = 2.0
N_CALLS = 4800
BATCH_SIZE = 16
INIT_JITTER_TRIALS = 3
Y_JITTER = 0.003           # ±3mm Y offset
Y_JITTER_SEED = 77777
VELOCITY_DEADZONE = False
JITTER_AGGREGATION = "median"

# Cost weights
VELOCITY_COST_WEIGHT = 5.0
TUMBLE_COST_WEIGHT = 2.0
LATERAL_COST_WEIGHT = 5.0
VELOCITY_VARIANCE_WEIGHT = 2.0
YAW_COST_WEIGHT = 1.0
YAW_THRESHOLD_DEG = 60.0
TUMBLE_THRESHOLD = 0.17     # cos(80°) — penalize tilt past 80°
TUMBLE_PENALTY_SCALE = 0.1

COST_FAILURE_VAL = 1e6

# CMA-ES
CMAES_SIGMA0 = 0.3
OPTIMIZER_RANDOM_STATE = 69420

# ---------------------------------------------------------------------------
# Warm-start: best from zzz_rough_v2 (Euler, cost=0.3949)
# ---------------------------------------------------------------------------
CMAES_X0: dict[str, float] | None = {
    "sliding_friction": 0.625649065196989,
    "torsional_friction": 0.0001577896719328836,
    "rolling_friction": 1.6999818728634566e-06,
    "solref_timeconst": 0.0008451887608978624,
    "solref_dampratio": 4.077074329871974,
    "solimp_dmin": 0.3285187962568,
    "solimp_delta_d": 0.602195907788976,
    "solimp_width": 2.101968017904397e-05,
    "solimp_midpoint": 0.8771918576448682,
    "solimp_power": 5.413792421292177,
    "magnetic_moment_fudge": 0.8957245963324068,
    "magnetic_field_fudge": 1.165693360752599,
    "dof_damping": 9.535192105821565e-10,
    "noslip_iterations": 0.37419076188152406,
    "noslip_tolerance": 1.1735746000014794e-06,
    "margin": 4.577884774903134e-06,
}

# ---------------------------------------------------------------------------
# Reference data (7 rough conditions, >=60% success rate)
# ---------------------------------------------------------------------------
REFERENCE_DATA: list[dict[str, Any]] = [
    # Single leg (scene1) — f10: 80%, f30: 80%
    {"scene": "scene1", "ctrl_freq": 10.0, "speed": 0.04292, "speed_std": 0.00101, "weight": 1.0},
    {"scene": "scene1", "ctrl_freq": 30.0, "speed": 0.08162, "speed_std": 0.00974, "weight": 1.0},
    # Double leg (scene2) — f10: 100%, f30: 80%, f50: 60%
    {"scene": "scene2", "ctrl_freq": 10.0, "speed": 0.06559, "speed_std": 0.00548, "weight": 1.0},
    {"scene": "scene2", "ctrl_freq": 30.0, "speed": 0.12888, "speed_std": 0.00027, "weight": 1.0},
    {"scene": "scene2", "ctrl_freq": 50.0, "speed": 0.10624, "speed_std": 0.03240, "weight": 1.0},
    # Quad leg (scene4) — f10: 100%, f30: 80%
    {"scene": "scene4", "ctrl_freq": 10.0, "speed": 0.08565, "speed_std": 0.00695, "weight": 1.0},
    {"scene": "scene4", "ctrl_freq": 30.0, "speed": 0.14602, "speed_std": 0.03684, "weight": 1.0},
]

# ---------------------------------------------------------------------------
# Cost function (time-gated, after SETTLE_TIME — same as flat)
# ---------------------------------------------------------------------------

_BODY_Z_LOCAL = np.array([0.0, 0.0, 1.0])
_NOMINAL_BODY_Z_WORLD = np.array([0.0, 0.0, -1.0])
_BODY_X_LOCAL = np.array([1.0, 0.0, 0.0])


def calculate_cost(
    trajectory: list[dict],
    target_velocity: float,
    speed_std: float = 0.0,
    verbose: bool = True,
) -> dict[str, float]:
    """Rough cost: time-gated (after SETTLE_TIME), no spatial gating."""
    fail = {
        "total_cost": COST_FAILURE_VAL, "avg_forward_velocity": 0,
        "tumble_penalty": 0, "lateral_displacement": 0, "yaw_deviation_deg": 0,
    }
    if not trajectory:
        return fail

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

    vel_deviation = avg_forward_velocity - target_velocity
    if VELOCITY_DEADZONE and speed_std > 0.0 and abs(vel_deviation) <= speed_std:
        velocity_error = 0.0
    elif VELOCITY_DEADZONE and speed_std > 0.0:
        excess = abs(vel_deviation) - speed_std
        velocity_error = (excess / target_velocity) ** 2
    else:
        velocity_error = (vel_deviation / target_velocity) ** 2

    lateral_displacement = 0.0
    if active_duration > 1e-6:
        lateral_displacement = abs(final_state["pos"][1] - start_state["pos"][1])
    lateral_error = lateral_displacement ** 2

    tumble_penalty = 0.0
    for state in trajectory:
        quat = state["quat"]
        body_z_axis = R.from_quat(quat, scalar_first=True).apply(_BODY_Z_LOCAL)
        uprightness = np.dot(body_z_axis, _NOMINAL_BODY_Z_WORLD)
        if uprightness < TUMBLE_THRESHOLD:
            tumble_penalty += (1 - uprightness) * TUMBLE_PENALTY_SCALE
    tumble_penalty /= max(len(trajectory), 1)

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
    )

    if verbose:
        print(
            f"    Avg Vel: {avg_forward_velocity:.3f} m/s | "
            f"Vel Err: {velocity_error:.4f} | "
            f"Lateral: {lateral_displacement:.4f} m | "
            f"Tumble: {tumble_penalty:.4f} | "
            f"Yaw: {yaw_deviation_deg:.1f}° | "
            f"Total: {total_cost:.4f}"
        )

    return {
        "total_cost": total_cost,
        "avg_forward_velocity": avg_forward_velocity,
        "lateral_displacement": lateral_displacement,
        "tumble_penalty": tumble_penalty,
        "yaw_deviation_deg": yaw_deviation_deg,
    }
