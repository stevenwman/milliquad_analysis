"""Step terrain optimization config — 16-dim extended parameter space.

Thin overlay on config_new.py. Imports the 16-dim search space and param
conversion, defines step-specific reference data including wheel f10/f20
as failure mode constraints (target=0).

The 3 new dims vs config.py:
  - noslip_iterations: MuJoCo no-slip solver iterations (0=default)
  - noslip_tolerance:  convergence threshold for no-slip solver
  - margin:            contact detection distance (o_margin)
"""

from typing import Any

from config_new import (
    # Simulation constants
    SETTLE_TIME,
    SIM_TIMESTEP,
    INITIAL_Z_HEIGHT,
    INITIAL_QUATERNION,
    INITIAL_LEG_ANGLES,
    LEG_BODY_OFFSET,
    MAGNETIC_MOMENT,
    MAGNETIC_FIELD_MAGNITUDE,
    MU0_OVER_4PI,
    R_EPS,
    SIMULATION_TIMEOUT,
    STUCK_CHECK_INTERVAL,
    STUCK_THRESHOLD,
    # Scene XMLs (flat — used as base for step XML generation)
    PACKAGE_DIR,
    MJCF_PATHS,
    # 16-dim search space and param conversion
    space,
    point_to_params,
    sim_params_from_point,
    _make_ref_id,
    # Video / viewer constants
    VIDEO_FRAMERATE,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    CAMERA_DISTANCE_RECORD,
    CAMERA_DISTANCE_VIEWER,
)

# ---------------------------------------------------------------------------
# Step-specific overrides
# ---------------------------------------------------------------------------
SIM_DURATION = 5.0          # robot must traverse steps
N_CALLS = 4800
BATCH_SIZE = 16
INIT_YAW_JITTER_DEG = 2
INIT_JITTER_TRIALS = 2
INIT_JITTER_SEED = 12345
VELOCITY_DEADZONE = False
DEFAULT_CTRL_FREQ = 30.0

# Cost weights
VELOCITY_COST_WEIGHT = 5.0
TUMBLE_COST_WEIGHT = 1.0
LATERAL_COST_WEIGHT = 1.0
VELOCITY_VARIANCE_WEIGHT = 2.0
YAW_COST_WEIGHT = 0.0
YAW_THRESHOLD_DEG = 60.0
TUMBLE_THRESHOLD = 0.0
TUMBLE_PENALTY_SCALE = 0.1
COST_FAILURE = 1e6

# Failure mode velocity scale: normalizes avg_velocity² penalty so that a
# wheel moving at FAILURE_MODE_VEL_SCALE produces cost=1.0 — same order of
# magnitude as a normal reference with 100% velocity error.
FAILURE_MODE_VEL_SCALE = 0.05  # m/s

# CMA-ES
CMAES_SIGMA0 = 0.5          # larger than flat's 0.3 — step terrain needs more exploration
OPTIMIZER_RANDOM_STATE = 69420
VERBOSE_BATCH = True
PROFILE_BATCH = True

# CSV filenames (written into run_dir at runtime)
CSV_PATH = "multi_optimization_results.csv"
BEST_CSV_PATH = "optimization_bests.csv"

# ---------------------------------------------------------------------------
# Warm-start: 13-dim step params + safe defaults for the 3 new dims
# Source: config_step.py CMAES_X0 (flat-optimized, warm-started into step)
# ---------------------------------------------------------------------------
CMAES_X0: dict[str, float] | None = None  # Cold start from search space midpoints

# ---------------------------------------------------------------------------
# Step terrain reference data
# Source: experimental_data/csv/steps/, forward velocity (vx), q75-300 window
# scene_wheel f10/f20: failure modes — robot does not move on steps
# scene_wheel f30:     target = 0.0938 m/s (does move)
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
# Step terrain geometry
# ---------------------------------------------------------------------------
STEP_PRESET: dict[str, float | int] = {
    "step_height":       0.001,   # 1mm
    "step_length":       0.0045,  # 4.5mm
    "step_count":        8,
    "final_step_length": 0.02,    # 20mm extended platform
    "step_width":        0.1,     # 100mm
    "flat_lead":         0.05,    # 50mm flat ground before first step
}

STEP_START_X: float = STEP_PRESET["flat_lead"]
STEP_END_X: float = (
    STEP_PRESET["flat_lead"]
    + (STEP_PRESET["step_count"] - 1) * STEP_PRESET["step_length"]
    + STEP_PRESET["final_step_length"]
)  # 0.1015m — trailing edge of final step platform

# Progress penalty: penalizes incomplete step traversal
PROGRESS_COST_WEIGHT = 2.0


# ---------------------------------------------------------------------------
# Helper functions (mirrors config_step.py structure)
# ---------------------------------------------------------------------------

def reference_rows() -> list[dict[str, Any]]:
    """Return reference rows for optimization from REFERENCE_DATA."""
    rows = []
    seen_ids = set()
    for row in REFERENCE_DATA:
        scene = row["scene"]
        ctrl_freq = float(row.get("ctrl_freq", DEFAULT_CTRL_FREQ))
        speed = float(row["speed"])
        weight = float(row.get("weight", 1.0))
        ref_id = str(row.get("id", _make_ref_id(scene, ctrl_freq)))
        if ref_id in seen_ids:
            raise ValueError(f"Duplicate reference id '{ref_id}' in REFERENCE_DATA")
        seen_ids.add(ref_id)
        speed_std = row.get("speed_std", 0.0)
        rows.append({
            "id": ref_id,
            "scene": scene,
            "ctrl_freq": ctrl_freq,
            "speed": speed,
            "speed_std": float(speed_std),
            "weight": weight,
        })
    return rows


def reference_ids() -> list[str]:
    return [row["id"] for row in reference_rows()]


def csv_fieldnames() -> list[str]:
    param_names = [dim.name for dim in space] + ["solimp_dmax"]
    rids = reference_ids()
    scene_names = list(MJCF_PATHS.keys())
    return (
        ["id", "cost", "elapsed_min"]
        + [f"velocity_{s}" for s in scene_names]
        + [f"cost_{s}" for s in scene_names]
        + [f"velocity_{rid}" for rid in rids]
        + [f"cost_{rid}" for rid in rids]
        + [f"lateral_{rid}" for rid in rids]
        + [f"tumble_{rid}" for rid in rids]
        + [f"yaw_{rid}" for rid in rids]
        + [f"progress_{rid}" for rid in rids]
        + [f"best_trial_{rid}" for rid in rids]
        + param_names
    )
