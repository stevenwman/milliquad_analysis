"""Rough terrain optimization config — 16-dim extended parameter space.

Thin overlay on config_new.py. Imports the 16-dim search space and param
conversion, defines rough-terrain-specific reference data from experimental
random terrain trials (≥60% success rate only).

Terrain generation:
    python terrain_mesh.py --nX 10 --nY 6 --sL 0.005 \
      --height-mean 0.002 --height-std 0.001 --z-safe 0.00025 --seed 42
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
    # Scene XMLs (flat — used as base for rough XML generation)
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
# Rough-terrain-specific overrides
# ---------------------------------------------------------------------------
SIM_DURATION = 2.0
N_CALLS = 4800
BATCH_SIZE = 16
INIT_JITTER_TRIALS = 3
Y_JITTER = 0.003           # ±3mm Y offset
Y_JITTER_SEED = 77777      # base seed for deterministic Y-jitter RNG
VELOCITY_DEADZONE = False
DEFAULT_CTRL_FREQ = 30.0

# Cost weights (same as flat optimizer)
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
# Warm-start: from flat_10_30_50 best params
# Source: results/20260225T122342_flat_10_30_50/optimization_bests.csv
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
    "noslip_iterations": 0,
    "noslip_tolerance": 1e-6,
    "margin": 0.0,
}

# ---------------------------------------------------------------------------
# Rough terrain reference data
# Source: experimental_data/csv/random_terrain_raw.csv
# Only configs with ≥60% success rate (recomputed from raw trial counts)
# ---------------------------------------------------------------------------
REFERENCE_DATA: list[dict[str, Any]] = [
    # Single leg (scene1) — f10: 80%, f30: 80%
    {"scene": "scene1",      "ctrl_freq": 10.0, "speed": 0.04292, "speed_std": 0.00101, "weight": 1.0},
    {"scene": "scene1",      "ctrl_freq": 30.0, "speed": 0.08162, "speed_std": 0.00974, "weight": 1.0},
    # Double leg (scene2) — f10: 100%, f30: 80%, f50: 60%
    {"scene": "scene2",      "ctrl_freq": 10.0, "speed": 0.06559, "speed_std": 0.00548, "weight": 1.0},
    {"scene": "scene2",      "ctrl_freq": 30.0, "speed": 0.12888, "speed_std": 0.00027, "weight": 1.0},
    {"scene": "scene2",      "ctrl_freq": 50.0, "speed": 0.10624, "speed_std": 0.03240, "weight": 1.0},
    # Quad leg (scene4) — f10: 100%, f30: 80%
    {"scene": "scene4",      "ctrl_freq": 10.0, "speed": 0.08565, "speed_std": 0.00695, "weight": 1.0},
    {"scene": "scene4",      "ctrl_freq": 30.0, "speed": 0.14602, "speed_std": 0.03684, "weight": 1.0},
    # Wheel — f30: 60% (3/5 raw trials successful)
    {"scene": "scene_wheel", "ctrl_freq": 30.0, "speed": 0.15412, "speed_std": 0.02769, "weight": 1.0},
]

# ---------------------------------------------------------------------------
# Terrain geometry (fixed layout — seed=42 heightmap from eval_rough_terrain.py)
# ---------------------------------------------------------------------------
TERRAIN_NX = 10
TERRAIN_NY = 6
TERRAIN_SL = 0.005          # 5mm tile side
TERRAIN_HEIGHT_MEAN = 0.002  # 2mm
TERRAIN_HEIGHT_STD = 0.001   # 1mm
TERRAIN_Z_SAFE = 0.00025
TERRAIN_SEED = 42
N_TILES = 3                 # 3 copies tiled along +X
FLAT_LEAD = 0.005           # 5mm flat lead (short — most sim on terrain)
PIXELS_PER_SQUARE = 8


# ---------------------------------------------------------------------------
# Helper functions
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
        + [f"best_trial_{rid}" for rid in rids]
        + param_names
    )
