"""Flat terrain optimization config with per-condition time gating.

Copy of config_flat.py. Only change: each REFERENCE_DATA entry carries
``trial_duration`` (mean experimental recording length in seconds).
The cost function truncates the sim trajectory to SETTLE_TIME + trial_duration
so velocity is measured over the same window as the experiment.

Run with:  uv run python optimizer.py --terrain flat_tg --suffix flat_tg
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
# Scene XMLs (flat terrain, RK4 baked in)
# ---------------------------------------------------------------------------
MJCF_PATHS: dict[str, str] = {
    "scene1":      str(PACKAGE_DIR / "robots" / "quad"  / "scene_1_flat.xml"),
    "scene2":      str(PACKAGE_DIR / "robots" / "quad"  / "scene_2_flat.xml"),
    "scene4":      str(PACKAGE_DIR / "robots" / "quad"  / "scene_4_flat.xml"),
    "scene_wheel": str(PACKAGE_DIR / "robots" / "wheel" / "scene_wheel_flat.xml"),
}

# ---------------------------------------------------------------------------
# Simulation / optimization parameters
# ---------------------------------------------------------------------------
SIM_DURATION = 3.0
N_CALLS = 4800
BATCH_SIZE = 16
INIT_YAW_JITTER_DEG = 2
INIT_JITTER_TRIALS = 3
INIT_JITTER_SEED = 12345
VELOCITY_DEADZONE = False
JITTER_AGGREGATION = "median"

# Cost weights
VELOCITY_COST_WEIGHT = 5.0
TUMBLE_COST_WEIGHT = 1.0
LATERAL_COST_WEIGHT = 5.0
VELOCITY_VARIANCE_WEIGHT = 2.0
YAW_COST_WEIGHT = 1.0
YAW_THRESHOLD_DEG = 60.0
TUMBLE_THRESHOLD = 0.0
TUMBLE_PENALTY_SCALE = 0.1

# CMA-ES
CMAES_SIGMA0 = 0.3
OPTIMIZER_RANDOM_STATE = 69420

# ---------------------------------------------------------------------------
# Warm-start: best from 20260225T122342_flat_10_30_50 (Euler, cost=0.1276)
# ---------------------------------------------------------------------------
CMAES_X0: dict[str, float] | None = {
    "sliding_friction": 0.4067415162382437,
    "torsional_friction": 0.0001598832061548777,
    "rolling_friction": 4.591348631378625e-06,
    "solref_timeconst": 0.002046553443113131,
    "solref_dampratio": 3.8186561171444096,
    "solimp_dmin": 0.43524706654498085,
    "solimp_delta_d": 0.988954449244754,
    "solimp_width": 4.6887126279039634e-05,
    "solimp_midpoint": 0.6473380150564967,
    "solimp_power": 5.037567575958023,
    "magnetic_moment_fudge": 0.6545005423370444,
    "magnetic_field_fudge": 1.1389204964634818,
    "dof_damping": 5.324867233622363e-10,
    "noslip_iterations": 0.16236228774673808,
    "noslip_tolerance": 1.0341954878538795e-06,
    "margin": 0.0006920906564202648,
}

# ---------------------------------------------------------------------------
# Reference data (15 flat conditions + 1 failure @ weight=0)
#
# trial_duration = mean experimental recording length (seconds), computed
# from CSVs in experimental_data/csv/flat/.  Used by calculate_cost to
# truncate the sim measurement window.
# ---------------------------------------------------------------------------
REFERENCE_DATA: list[dict[str, Any]] = [
    # Single leg (scene1)
    {"scene": "scene1", "ctrl_freq": 10.0, "speed": 0.0512, "speed_std": 0.0024, "weight": 1.0, "trial_duration": 2.625},
    {"scene": "scene1", "ctrl_freq": 20.0, "speed": 0.1264, "speed_std": 0.0047, "weight": 1.0, "trial_duration": 1.093},
    {"scene": "scene1", "ctrl_freq": 30.0, "speed": 0.1187, "speed_std": 0.0127, "weight": 1.0, "trial_duration": 1.197},
    {"scene": "scene1", "ctrl_freq": 50.0, "speed": 0.1483, "speed_std": 0.0131, "weight": 1.0, "trial_duration": 1.023},
    # Double leg (scene2)
    {"scene": "scene2", "ctrl_freq": 10.0, "speed": 0.0832, "speed_std": 0.0014, "weight": 1.0, "trial_duration": 1.567},
    {"scene": "scene2", "ctrl_freq": 20.0, "speed": 0.1131, "speed_std": 0.0420, "weight": 1.0, "trial_duration": 1.021},
    {"scene": "scene2", "ctrl_freq": 30.0, "speed": 0.1796, "speed_std": 0.0179, "weight": 1.0, "trial_duration": 0.827},
    {"scene": "scene2", "ctrl_freq": 50.0, "speed": 0.2633, "speed_std": 0.0257, "weight": 1.0, "trial_duration": 0.663},
    # Quad leg (scene4)
    {"scene": "scene4", "ctrl_freq": 10.0, "speed": 0.1121, "speed_std": 0.0060, "weight": 1.0, "trial_duration": 1.245},
    {"scene": "scene4", "ctrl_freq": 20.0, "speed": 0.1841, "speed_std": 0.0156, "weight": 1.0, "trial_duration": 0.712},
    {"scene": "scene4", "ctrl_freq": 30.0, "speed": 0.2747, "speed_std": 0.0207, "weight": 1.0, "trial_duration": 0.589},
    {"scene": "scene4", "ctrl_freq": 50.0, "speed": 0.3274, "speed_std": 0.0556, "weight": 1.0, "trial_duration": 0.547},
    # Wheel
    {"scene": "scene_wheel", "ctrl_freq": 10.0, "speed": 0.1432, "speed_std": 0.0013, "weight": 1.0, "trial_duration": 0.965},
    {"scene": "scene_wheel", "ctrl_freq": 20.0, "speed": 0.3058, "speed_std": 0.0068, "weight": 1.0, "trial_duration": 0.478},
    {"scene": "scene_wheel", "ctrl_freq": 30.0, "speed": 0.4493, "speed_std": 0.0183, "weight": 1.0, "trial_duration": 0.384},
    # WR f50: exp robot self-destructs at 50Hz; sim succeeds (~720 mm/s). Validation only.
    {"scene": "scene_wheel", "ctrl_freq": 50.0, "speed": 0.0, "speed_std": 0.0, "weight": 0.0},
]

# ---------------------------------------------------------------------------
# Trial-duration lookup (keyed by (speed, speed_std) — unique per condition)
# ---------------------------------------------------------------------------
_TRIAL_DURATION_MAP: dict[tuple[float, float], float] = {}
for _row in REFERENCE_DATA:
    if "trial_duration" in _row:
        _TRIAL_DURATION_MAP[(_row["speed"], _row.get("speed_std", 0.0))] = _row["trial_duration"]

# ---------------------------------------------------------------------------
# Cost function (time-gated per condition)
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
    """Flat cost with per-condition time gating.

    Velocity (and lateral/yaw) are measured from SETTLE_TIME to
    SETTLE_TIME + trial_duration, where trial_duration is the mean
    experimental recording length for this condition.  Falls back to
    full trajectory if no duration is registered (e.g. weight=0 refs).
    """
    fail = {
        "total_cost": COST_FAILURE, "avg_forward_velocity": 0,
        "tumble_penalty": 0, "lateral_displacement": 0, "yaw_deviation_deg": 0,
    }
    if not trajectory:
        return fail

    # --- determine measurement window ---
    trial_duration = _TRIAL_DURATION_MAP.get((target_velocity, speed_std))

    if trial_duration is not None:
        end_time = SETTLE_TIME + trial_duration
    else:
        end_time = trajectory[-1]["time"]

    # Find start state (first state at or after SETTLE_TIME)
    start_state = trajectory[0]
    start_idx = 0
    for i, state in enumerate(trajectory):
        if state["time"] >= SETTLE_TIME:
            start_state = state
            start_idx = i
            break

    # Find end state (last state at or before end_time)
    final_state = start_state
    end_idx = start_idx
    for i in range(start_idx, len(trajectory)):
        if trajectory[i]["time"] <= end_time:
            final_state = trajectory[i]
            end_idx = i
        else:
            break

    active_duration = final_state["time"] - start_state["time"]
    avg_forward_velocity = 0.0
    if active_duration > 1e-6:
        forward_displacement = final_state["pos"][0] - start_state["pos"][0]
        avg_forward_velocity = forward_displacement / active_duration

    vel_deviation = avg_forward_velocity - target_velocity
    if target_velocity == 0.0:
        velocity_error = 0.0
    elif VELOCITY_DEADZONE and speed_std > 0.0 and abs(vel_deviation) <= speed_std:
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

    # Tumble: only within measurement window
    tumble_penalty = 0.0
    window = trajectory[start_idx:end_idx + 1]
    for state in window:
        quat = state["quat"]
        body_z_axis = R.from_quat(quat, scalar_first=True).apply(_BODY_Z_LOCAL)
        uprightness = np.dot(body_z_axis, _NOMINAL_BODY_Z_WORLD)
        if uprightness < TUMBLE_THRESHOLD:
            tumble_penalty += (1 - uprightness) * TUMBLE_PENALTY_SCALE
    tumble_penalty /= max(len(window), 1)

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
        window_str = f" [{trial_duration:.2f}s]" if trial_duration is not None else ""
        print(
            f"    Avg Vel: {avg_forward_velocity:.3f} m/s{window_str} | "
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
