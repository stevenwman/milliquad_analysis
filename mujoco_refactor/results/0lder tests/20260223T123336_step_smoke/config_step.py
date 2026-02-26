"""Step terrain optimization configuration.

Thin overlay on config.py — imports shared search space and param conversion,
defines step-specific reference data and constants.

See STEP_OPTIMIZER_PLAN.md for design rationale.
"""

from typing import Any

from config import (
    # Re-export simulation constants
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
    # Search space (same 13 dims) and param conversion
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
SIM_DURATION = 5.0          # longer than flat's 3.0s — robot must traverse steps
N_CALLS = 2400
BATCH_SIZE = 8
INIT_YAW_JITTER_DEG = 2
INIT_JITTER_TRIALS = 3
INIT_JITTER_SEED = 12345
VELOCITY_DEADZONE = False
DEFAULT_CTRL_FREQ = 30.0

# Cost weights
VELOCITY_COST_WEIGHT = 5.0
TUMBLE_COST_WEIGHT = 1.0
LATERAL_COST_WEIGHT = 5.0
VELOCITY_VARIANCE_WEIGHT = 2.0
YAW_COST_WEIGHT = 1.0
YAW_THRESHOLD_DEG = 60.0
TUMBLE_THRESHOLD = 0.0
TUMBLE_PENALTY_SCALE = 0.1
COST_FAILURE = 1e6

# CMA-ES
CMAES_SIGMA0 = 0.3
OPTIMIZER_RANDOM_STATE = 69420
VERBOSE_BATCH = True
PROFILE_BATCH = True

# CSV filenames (written into run_dir at runtime)
CSV_PATH = "multi_optimization_results.csv"
BEST_CSV_PATH = "optimization_bests.csv"

# ---------------------------------------------------------------------------
# Warm-start from best of 20260222T181114_with_20hz_no-deadzone (cost=0.380)
# ---------------------------------------------------------------------------
CMAES_X0: dict[str, float] | None = {
    "sliding_friction": 0.48734565718766704,
    "torsional_friction": 0.00025167486271239974,
    "rolling_friction": 4.115853923336379e-06,
    "solref_timeconst": 0.002316749205053682,
    "solref_dampratio": 3.3733148987037813,
    "solimp_dmin": 0.45155836837876284,
    "solimp_delta_d": 0.6302833354616117,
    "solimp_width": 2.005399484434065e-05,
    "solimp_midpoint": 0.2865197391827468,
    "solimp_power": 5.231485448700575,
    "magnetic_moment_fudge": 0.6532045074731974,
    "magnetic_field_fudge": 1.0437234064669991,
    "dof_damping": 4.989444645973366e-10,
}

# ---------------------------------------------------------------------------
# Step terrain reference data
# Source: experimental_data/csv/steps/, forward velocity (vx), mid-300 window
# See STEP_TERRAIN_VALIDATION.md for method comparison
# ---------------------------------------------------------------------------
REFERENCE_DATA: list[dict[str, Any]] = [
    # Single leg (scene1)
    {"scene": "scene1",      "ctrl_freq": 10.0, "speed": 0.0156, "speed_std": 0.0014, "weight": 1.0},
    {"scene": "scene1",      "ctrl_freq": 20.0, "speed": 0.0421, "speed_std": 0.0049, "weight": 1.0},
    {"scene": "scene1",      "ctrl_freq": 30.0, "speed": 0.0241, "speed_std": 0.0248, "weight": 1.0},
    # Double leg (scene2)
    {"scene": "scene2",      "ctrl_freq": 10.0, "speed": 0.0319, "speed_std": 0.0137, "weight": 1.0},
    {"scene": "scene2",      "ctrl_freq": 20.0, "speed": 0.0914, "speed_std": 0.0325, "weight": 1.0},
    {"scene": "scene2",      "ctrl_freq": 30.0, "speed": 0.1351, "speed_std": 0.0220, "weight": 1.0},
    # Quad leg (scene4)
    {"scene": "scene4",      "ctrl_freq": 10.0, "speed": 0.0800, "speed_std": 0.0061, "weight": 1.0},
    {"scene": "scene4",      "ctrl_freq": 20.0, "speed": 0.0996, "speed_std": 0.0165, "weight": 1.0},
    {"scene": "scene4",      "ctrl_freq": 30.0, "speed": 0.0783, "speed_std": 0.0342, "weight": 1.0},
    # Wheel (scene_wheel) — 30Hz only
    {"scene": "scene_wheel", "ctrl_freq": 30.0, "speed": 0.0896, "speed_std": 0.0189, "weight": 1.0},
]

# ---------------------------------------------------------------------------
# Step terrain geometry — uniform for all morphologies
# ---------------------------------------------------------------------------
STEP_PRESET: dict[str, float | int] = {
    "step_height": 0.001,        # 1mm
    "step_length": 0.0045,       # 4.5mm
    "step_count": 8,
    "final_step_length": 0.02,   # 20mm extended platform
    "step_width": 0.1,           # 100mm
    "flat_lead": 0.05,           # 50mm flat ground before first step
}

# Derived: x-position where the step field begins (used by cost function)
STEP_START_X: float = STEP_PRESET["flat_lead"]


# ---------------------------------------------------------------------------
# Helper functions (same structure as config.py)
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
    """Reference IDs for CSV columns."""
    return [row["id"] for row in reference_rows()]


def csv_fieldnames() -> list[str]:
    """Column names for the results CSV."""
    param_names = [dim.name for dim in space] + ["solimp_dmax"]
    rids = reference_ids()
    ref_cost_names = [f"cost_{rid}" for rid in rids]
    ref_vel_names = [f"velocity_{rid}" for rid in rids]
    ref_lateral_names = [f"lateral_{rid}" for rid in rids]
    ref_tumble_names = [f"tumble_{rid}" for rid in rids]
    ref_yaw_names = [f"yaw_{rid}" for rid in rids]
    # Scene-level aggregates
    scene_names = list(MJCF_PATHS.keys())
    scene_cost_names = [f"cost_{s}" for s in scene_names]
    scene_vel_names = [f"velocity_{s}" for s in scene_names]
    return (
        ["id", "cost", "elapsed_min"]
        + scene_vel_names + scene_cost_names
        + ref_vel_names + ref_cost_names
        + ref_lateral_names + ref_tumble_names + ref_yaw_names
        + param_names
    )
