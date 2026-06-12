"""Rough terrain config with spatially-gated cost function.

Thin overlay on config_rough.py — imports everything, overrides only
calculate_cost to gate velocity/lateral/yaw to the hfield bounds
instead of using SETTLE_TIME.

Usage:
    uv run python optimizer.py --suffix rough_spatial_warm_rk4 \
        --warm-start-from results/20260228T102010_rk4_rough
    # (optimizer.py dispatches to config_{terrain}.calculate_cost)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from config import COST_FAILURE

# Re-export everything from config_rough so optimizer.py sees all the same
# constants (MJCF_PATHS, REFERENCE_DATA, SIM_DURATION, etc.)
from config_rough import *  # noqa: F401,F403
from config_rough import (
    FLAT_LEAD,
    _X_HALF,
    VELOCITY_DEADZONE,
    VELOCITY_COST_WEIGHT,
    TUMBLE_COST_WEIGHT,
    LATERAL_COST_WEIGHT,
    YAW_COST_WEIGHT,
    YAW_THRESHOLD_DEG,
    TUMBLE_THRESHOLD,
    TUMBLE_PENALTY_SCALE,
)

# ---------------------------------------------------------------------------
# Spatial gating bounds (hfield edges in world X)
# ---------------------------------------------------------------------------
ROUGH_START_X = FLAT_LEAD              # 0.005m — left edge of hfield
ROUGH_END_X = FLAT_LEAD + 2 * _X_HALF  # 0.155m — right edge of hfield

# ---------------------------------------------------------------------------
# Cost function (spatially-gated to hfield bounds)
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
    """Rough cost: spatially-gated between ROUGH_START_X and ROUGH_END_X."""
    fail = {
        "total_cost": COST_FAILURE, "avg_forward_velocity": 0,
        "tumble_penalty": 0, "lateral_displacement": 0, "yaw_deviation_deg": 0,
    }
    if not trajectory:
        return fail

    # Find spatial window: first timestep at hfield start, first at hfield end
    enter_idx = None
    exit_idx = None
    for i, state in enumerate(trajectory):
        if enter_idx is None and state["pos"][0] >= ROUGH_START_X:
            enter_idx = i
        if state["pos"][0] >= ROUGH_END_X:
            exit_idx = i
            break
    if enter_idx is None:
        return fail
    # Robot never reached end of hfield — use last timestep
    if exit_idx is None:
        exit_idx = len(trajectory) - 1
    if exit_idx <= enter_idx:
        return fail

    start_state = trajectory[enter_idx]
    end_state = trajectory[exit_idx]

    active_duration = end_state["time"] - start_state["time"]
    avg_forward_velocity = 0.0
    if active_duration > 1e-6:
        forward_displacement = end_state["pos"][0] - start_state["pos"][0]
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
        lateral_displacement = abs(end_state["pos"][1] - start_state["pos"][1])
    lateral_error = lateral_displacement ** 2

    # Tumble: evaluate over full trajectory (tumble anywhere is bad)
    tumble_penalty = 0.0
    for state in trajectory:
        quat = state["quat"]
        body_z_axis = R.from_quat(quat, scalar_first=True).apply(_BODY_Z_LOCAL)
        uprightness = np.dot(body_z_axis, _NOMINAL_BODY_Z_WORLD)
        if uprightness < TUMBLE_THRESHOLD:
            tumble_penalty += (1 - uprightness) * TUMBLE_PENALTY_SCALE
    tumble_penalty /= max(len(trajectory), 1)

    start_body_x = R.from_quat(start_state["quat"], scalar_first=True).apply(_BODY_X_LOCAL)
    end_body_x = R.from_quat(end_state["quat"], scalar_first=True).apply(_BODY_X_LOCAL)
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
