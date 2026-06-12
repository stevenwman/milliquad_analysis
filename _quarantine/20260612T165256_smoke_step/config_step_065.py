"""Step terrain optimization config (65% spatial gate).

Copy of config_step.py with active_cutoff changed from 90% to 65% of step region.
This avoids measuring velocity/pitch near the cliff edge where artifacts occur.

Defines REFERENCE_DATA (12 step conditions including wheel failure modes),
step geometry, spatial-gated cost function, and CMAES_X0.
"""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from config import (
    PACKAGE_DIR,
    COST_FAILURE,
    _make_ref_id,
)

# ---------------------------------------------------------------------------
# Scene XMLs (step terrain, RK4 baked in)
# ---------------------------------------------------------------------------
MJCF_PATHS: dict[str, str] = {
    "scene1":      str(PACKAGE_DIR / "robots" / "quad"  / "scene_1_step.xml"),
    "scene2":      str(PACKAGE_DIR / "robots" / "quad"  / "scene_2_step.xml"),
    "scene4":      str(PACKAGE_DIR / "robots" / "quad"  / "scene_4_step.xml"),
    "scene_wheel": str(PACKAGE_DIR / "robots" / "wheel" / "scene_wheel_step.xml"),
}

# ---------------------------------------------------------------------------
# Step terrain geometry
# ---------------------------------------------------------------------------
STEP_PRESET: dict[str, float | int] = {
    "step_height": 0.001,
    "step_length": 0.0045,
    "step_count": 8,
    "final_step_length": 0.02,
    "step_width": 0.1,
    "flat_lead": 0.05,
}

STEP_START_X: float = STEP_PRESET["flat_lead"]
STEP_END_X: float = (
    STEP_PRESET["flat_lead"]
    + (STEP_PRESET["step_count"] - 1) * STEP_PRESET["step_length"]
    + STEP_PRESET["final_step_length"]
)

# ---------------------------------------------------------------------------
# Simulation / optimization parameters
# ---------------------------------------------------------------------------
SIM_DURATION = 5.0
N_CALLS = 4800
BATCH_SIZE = 16
INIT_YAW_JITTER_DEG = 2
INIT_JITTER_TRIALS = 3
INIT_JITTER_SEED = 12345
VELOCITY_DEADZONE = False
JITTER_AGGREGATION = "best"  # argmin cost

# Cost weights
VELOCITY_COST_WEIGHT = 5.0
TUMBLE_COST_WEIGHT = 1.0
LATERAL_COST_WEIGHT = 1.0
VELOCITY_VARIANCE_WEIGHT = 2.0
YAW_COST_WEIGHT = 0.0       # disabled — cliff-fall artifact
YAW_THRESHOLD_DEG = 60.0
TUMBLE_THRESHOLD = 0.0
TUMBLE_PENALTY_SCALE = 0.1
PROGRESS_COST_WEIGHT = 2.0

# Failure mode velocity scale
FAILURE_MODE_VEL_SCALE = 0.05

# CMA-ES
CMAES_SIGMA0 = 0.5
OPTIMIZER_RANDOM_STATE = 69420

# ---------------------------------------------------------------------------
# Warm-start: best from 20260225T225248_step_argmin_progress (Euler, cost=0.2096)
# ---------------------------------------------------------------------------
CMAES_X0: dict[str, float] | None = {
    "sliding_friction": 0.4986415629153183,
    "torsional_friction": 0.007850943496611652,
    "rolling_friction": 0.00020110398060597134,
    "solref_timeconst": 0.0025844846675452367,
    "solref_dampratio": 1.5006241725620073,
    "solimp_dmin": 0.21292074405301045,
    "solimp_delta_d": 0.8609783847610987,
    "solimp_width": 0.0002017720085791609,
    "solimp_midpoint": 0.8300673256628996,
    "solimp_power": 4.408972484420905,
    "magnetic_moment_fudge": 0.9213609195063706,
    "magnetic_field_fudge": 0.7333437950639141,
    "dof_damping": 1.1663628231467944e-09,
    "noslip_iterations": 31.371769945968545,
    "noslip_tolerance": 0.0006255666106575557,
    "margin": 5.330483403847906e-05,
}

# ---------------------------------------------------------------------------
# Reference data (12 step conditions)
# ---------------------------------------------------------------------------
REFERENCE_DATA: list[dict[str, Any]] = [
    # Single leg (scene1)
    {"scene": "scene1",      "ctrl_freq": 10.0, "speed": 0.0199, "speed_std": 0.0018, "weight": 1.0},
    {"scene": "scene1",      "ctrl_freq": 20.0, "speed": 0.0473, "speed_std": 0.0106, "weight": 1.0},
    {"scene": "scene1",      "ctrl_freq": 30.0, "speed": 0.0331, "speed_std": 0.0066, "weight": 1.0},
    # Double leg (scene2)
    {"scene": "scene2",      "ctrl_freq": 10.0, "speed": 0.0542, "speed_std": 0.0105, "weight": 1.0},
    {"scene": "scene2",      "ctrl_freq": 20.0, "speed": 0.0894, "speed_std": 0.0275, "weight": 1.0},
    {"scene": "scene2",      "ctrl_freq": 30.0, "speed": 0.1335, "speed_std": 0.0129, "weight": 1.0},
    # Quad leg (scene4)
    {"scene": "scene4",      "ctrl_freq": 10.0, "speed": 0.0716, "speed_std": 0.0074, "weight": 1.0},
    {"scene": "scene4",      "ctrl_freq": 20.0, "speed": 0.1038, "speed_std": 0.0120, "weight": 1.0},
    {"scene": "scene4",      "ctrl_freq": 30.0, "speed": 0.0898, "speed_std": 0.0202, "weight": 1.0},
    # Wheel — f10/f20 are failure modes (does not move), f30 moves
    {"scene": "scene_wheel", "ctrl_freq": 10.0, "speed": 0.0000, "speed_std": 0.0,    "weight": 1.0},
    {"scene": "scene_wheel", "ctrl_freq": 20.0, "speed": 0.0000, "speed_std": 0.0,    "weight": 1.0},
    {"scene": "scene_wheel", "ctrl_freq": 30.0, "speed": 0.0938, "speed_std": 0.0097, "weight": 1.0},
]

# ---------------------------------------------------------------------------
GATE_FRACTION = 0.65

# Cost function (spatial-gated, step-aware) — 65% gate
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
    """Step-aware cost: spatial-gated (pos[0] >= STEP_START_X), with progress penalty.

    Uses 65% gate (vs 90% in config_step.py) to avoid cliff-edge artifacts.
    """
    fail = {
        "total_cost": COST_FAILURE, "avg_forward_velocity": 0,
        "tumble_penalty": 0, "lateral_displacement": 0, "yaw_deviation_deg": 0,
        "progress_penalty": 1.0,
    }
    if not trajectory:
        return fail

    step_start_x = STEP_START_X
    step_end_x = STEP_END_X

    # Find when robot enters step field
    enter_state = None
    for state in trajectory:
        if state["pos"][0] >= step_start_x:
            enter_state = state
            break

    if enter_state is None:
        return fail

    # Exit state: last state within 65% of step region (avoids cliff-edge artifacts)
    final_state = trajectory[-1]
    active_cutoff = step_start_x + 0.65 * (step_end_x - step_start_x)
    exit_state = enter_state
    for state in trajectory:
        if state["pos"][0] > active_cutoff:
            break
        exit_state = state

    active_duration = exit_state["time"] - enter_state["time"]
    avg_forward_velocity = 0.0
    if active_duration > 1e-6:
        forward_displacement = exit_state["pos"][0] - enter_state["pos"][0]
        avg_forward_velocity = forward_displacement / active_duration

    # Velocity error
    vel_deviation = avg_forward_velocity - target_velocity
    if target_velocity > 1e-6:
        if VELOCITY_DEADZONE and speed_std > 0.0 and abs(vel_deviation) <= speed_std:
            velocity_error = 0.0
        elif VELOCITY_DEADZONE and speed_std > 0.0:
            excess = abs(vel_deviation) - speed_std
            velocity_error = (excess / target_velocity) ** 2
        else:
            velocity_error = (vel_deviation / target_velocity) ** 2
    else:
        # Failure mode (target=0): penalize any movement
        velocity_error = (avg_forward_velocity / FAILURE_MODE_VEL_SCALE) ** 2

    # Lateral displacement (within active step region)
    lateral_displacement = abs(exit_state["pos"][1] - enter_state["pos"][1])
    lateral_error = lateral_displacement ** 2

    # Tumble penalty (within 65% of step region)
    tumble_penalty = 0.0
    tumble_count = 0
    for state in trajectory:
        if state["pos"][0] > active_cutoff:
            break
        quat = state["quat"]
        body_z_axis = R.from_quat(quat, scalar_first=True).apply(_BODY_Z_LOCAL)
        uprightness = np.dot(body_z_axis, _NOMINAL_BODY_Z_WORLD)
        if uprightness < TUMBLE_THRESHOLD:
            tumble_penalty += (1 - uprightness) * TUMBLE_PENALTY_SCALE
        tumble_count += 1
    tumble_penalty /= max(tumble_count, 1)

    # Yaw spin-out (within active step region)
    start_body_x = R.from_quat(enter_state["quat"], scalar_first=True).apply(_BODY_X_LOCAL)
    end_body_x = R.from_quat(exit_state["quat"], scalar_first=True).apply(_BODY_X_LOCAL)
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
    if target_velocity > 1e-6:
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
            f"Tumble: {tumble_penalty:.4f} | "
            f"Yaw: {yaw_deviation_deg:.1f}° | "
            f"Progress: {progress_penalty:.4f} | "
            f"Total: {total_cost:.4f}"
        )

    return {
        "total_cost": total_cost,
        "avg_forward_velocity": avg_forward_velocity,
        "lateral_displacement": lateral_displacement,
        "tumble_penalty": tumble_penalty,
        "yaw_deviation_deg": yaw_deviation_deg,
        "progress_penalty": progress_penalty,
    }
