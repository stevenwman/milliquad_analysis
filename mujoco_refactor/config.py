"""
Shared constants, search space definition, and parameter conversion.

Single source of truth for all values shared between simulation and optimization.
"""

import pathlib
from typing import Any

import numpy as np
from skopt.space import Real

# Directory containing this file — used to resolve MJCF paths regardless of CWD
PACKAGE_DIR = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
SETTLE_TIME = 0.1  # seconds before driving starts
STUCK_CHECK_INTERVAL = 5.0  # seconds between stuck checks
STUCK_THRESHOLD = 0.005  # minimum movement (meters) to avoid "stuck"
SIM_TIMESTEP = 1.0 / 2000.0  # MuJoCo timestep (2 kHz)

# Initial robot pose
INITIAL_Z_HEIGHT = 0.002  # meters above ground
INITIAL_QUATERNION = (0, 0, 1, 0)  # 180° rotation about y-axis (w, x, y, z)
INITIAL_LEG_ANGLES = np.pi  # all legs start at π radians
INIT_YAW_JITTER_DEG = 2  # max +/- yaw jitter (deg) applied at init; 0 = off
INIT_JITTER_TRIALS = 2  # number of jittered trials per point (>=1)
INIT_JITTER_SEED = 12345  # base seed for deterministic jitter

# Body indexing: leg bodies are offset from leg index (0-3) by this amount
# Body 0 = world, Body 1 = main chassis, Bodies 2-5 = legs FR/FL/BR/BL
LEG_BODY_OFFSET = 2

# Video recording
VIDEO_FRAMERATE = 60.0  # frames per second for video recording
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
CAMERA_DISTANCE_RECORD = 0.2  # camera distance when recording video
CAMERA_DISTANCE_VIEWER = 0.1  # camera distance in interactive viewer

# ---------------------------------------------------------------------------
# Physics / magnetic constants
# ---------------------------------------------------------------------------
MU0_OVER_4PI = 1e-7  # μ₀/(4π) in SI (N/A²)
R_EPS = 1e-6  # minimum r in dipole field to avoid 1/r³ blow-up (meters)
MAGNETIC_MOMENT = 1.13e-3
MAGNETIC_FIELD_MAGNITUDE = 2e-3

# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
MJCF_PATHS: dict[str, str] = {
    "scene1": str(PACKAGE_DIR / "multi_milli_quad" / "scene_1.xml"),
    "scene2": str(PACKAGE_DIR / "multi_milli_quad" / "scene_2.xml"),
    "scene4": str(PACKAGE_DIR / "multi_milli_quad" / "scene_4.xml"),
    "scene_wheel": str(PACKAGE_DIR / "wheel_milli_quad" / "scene_wheel.xml"),
}
DEFAULT_CTRL_FREQ = 30.0  # Hz when no per-row control frequency is provided

# Reference dataset for optimization.
# Fields:
#   id (optional): stable unique key for CSV columns and replay filenames
#   scene (required): key in MJCF_PATHS
#   ctrl_freq (optional): drive frequency in Hz (default DEFAULT_CTRL_FREQ)
#   speed (required): target speed in m/s
#   pitch_amp_deg (optional): target detrended RMS pitch amplitude in deg
#   pitch_weight (optional): per-row weight for pitch amplitude error term
#   weight (optional): per-row multiplier on total row cost when aggregating
REFERENCE_DATA: list[dict[str, Any]] = [
    # Single leg (scene1)
    {"scene": "scene1", "ctrl_freq": 10.0, "speed": 0.0512, "weight": 1.0},
    {"scene": "scene1", "ctrl_freq": 30.0, "speed": 0.1187, "weight": 1.0},
    {"scene": "scene1", "ctrl_freq": 50.0, "speed": 0.1483, "weight": 1.0},
    # Double leg (scene2)
    {"scene": "scene2", "ctrl_freq": 10.0, "speed": 0.0832, "weight": 1.0},
    {"scene": "scene2", "ctrl_freq": 30.0, "speed": 0.1796, "weight": 1.0},
    {"scene": "scene2", "ctrl_freq": 50.0, "speed": 0.2633, "weight": 1.0},
    # Quad leg (scene4)
    {"scene": "scene4", "ctrl_freq": 10.0, "speed": 0.1121, "weight": 1.0},
    {"scene": "scene4", "ctrl_freq": 30.0, "speed": 0.2747, "weight": 1.0},
    {"scene": "scene4", "ctrl_freq": 50.0, "speed": 0.3274, "weight": 1.0},
    # Wheel (scene_wheel)
    {"scene": "scene_wheel", "ctrl_freq": 10.0, "speed": 0.1432, "weight": 1.0},
    {"scene": "scene_wheel", "ctrl_freq": 30.0, "speed": 0.4493, "weight": 1.0},
]

# ---------------------------------------------------------------------------
# Optimization hyper-parameters
# ---------------------------------------------------------------------------
N_CALLS = 200  # total optimization iterations
SIM_DURATION = 3.0  # seconds per simulation run
SIMULATION_TIMEOUT = 35  # wall-clock seconds per worker
ROLLOUTS_PER_SCENE = 1  # sims per scene per iteration (>1 only for noisy sims)
BATCH_SIZE = 8  # points proposed per optimizer step
VERBOSE_BATCH = True
PROFILE_BATCH = True
# "rf" = random forest (fast ask/tell); "gp" = Gaussian process (slow at high n)
BASE_ESTIMATOR = "rf"

# ---------------------------------------------------------------------------
# Cost-function constants
# ---------------------------------------------------------------------------
TUMBLE_THRESHOLD = 0.0  # cos(angle) below this → "tumbling" (90° = past horizontal)
TUMBLE_PENALTY_SCALE = 0.1  # per-frame penalty when uprightness < threshold
COST_FAILURE = 1e6  # cost for failed / empty trajectory
VELOCITY_COST_WEIGHT = 5.0
TUMBLE_COST_WEIGHT = 1.0
LATERAL_COST_WEIGHT = 1.0  # penalizes lateral (y) displacement squared
VELOCITY_VARIANCE_WEIGHT = 2.0  # penalizes uneven velocity errors across references
PITCH_RMS_TARGET_DEG = 0.0  # target RMS pitch (deg); set when you have reference
PITCH_RMS_WEIGHT = 0.0  # set >0 to include RMS pitch in objective

# ---------------------------------------------------------------------------
# Search space (13 dimensions)
# ---------------------------------------------------------------------------
space: list[Real] = [
    Real(1e-5, 0.8, "log-uniform", name="sliding_friction"),
    Real(1e-5, 0.1, "log-uniform", name="torsional_friction"),
    Real(1e-5, 0.1, "log-uniform", name="rolling_friction"),
    Real(0.001, 0.1, "uniform", name="solref_timeconst"),
    Real(0.1, 2.0, "uniform", name="solref_dampratio"),
    Real(0.8, 0.99, "uniform", name="solimp_dmin"),
    Real(0.95, 0.999, "uniform", name="solimp_dmax"),
    Real(1e-4, 1e-2, "log-uniform", name="solimp_width"),
    Real(0.1, 0.9, "uniform", name="solimp_midpoint"),
    Real(1.0, 6.0, "uniform", name="solimp_power"),
    Real(0.5, 1.5, "uniform", name="magnetic_moment_fudge"),
    Real(0.5, 1.5, "uniform", name="magnetic_field_fudge"),
    Real(7e-11, 7e-9, "log-uniform", name="dof_damping"),
]

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
CSV_PATH = "multi_optimization_results.csv"
BEST_CSV_PATH = "optimization_bests.csv"


def _make_ref_id(scene: str, ctrl_freq: float) -> str:
    """Build a stable reference ID for CSV columns."""
    freq_str = f"{ctrl_freq:g}".replace(".", "p")
    return f"{scene}_f{freq_str}"


def reference_rows() -> list[dict[str, Any]]:
    """Return reference rows for optimization from REFERENCE_DATA."""
    rows = []
    seen_ids = set()
    for row in REFERENCE_DATA:
        scene = row["scene"]
        ctrl_freq = float(row.get("ctrl_freq", DEFAULT_CTRL_FREQ))
        speed = float(row["speed"])
        weight = float(row.get("weight", 1.0))
        pitch_amp_deg = row.get("pitch_amp_deg", None)
        pitch_weight = float(row.get("pitch_weight", PITCH_RMS_WEIGHT))
        ref_id = str(row.get("id", _make_ref_id(scene, ctrl_freq)))
        if ref_id in seen_ids:
            raise ValueError(f"Duplicate reference id '{ref_id}' in REFERENCE_DATA")
        seen_ids.add(ref_id)
        rows.append({
            "id": ref_id,
            "scene": scene,
            "ctrl_freq": ctrl_freq,
            "speed": speed,
            "pitch_amp_deg": pitch_amp_deg,
            "pitch_weight": pitch_weight,
            "weight": weight,
        })
    return rows


def reference_ids() -> list[str]:
    """Reference IDs for CSV columns."""
    return [row["id"] for row in reference_rows()]


def csv_fieldnames() -> list[str]:
    """Column names for the results CSV."""
    param_names = [dim.name for dim in space]
    scene_cost_names = [f"cost_{scene}" for scene in MJCF_PATHS]
    scene_vel_names = [f"velocity_{scene}" for scene in MJCF_PATHS]
    ref_cost_names = [f"cost_{rid}" for rid in reference_ids()]
    ref_vel_names = [f"velocity_{rid}" for rid in reference_ids()]
    return ["id", "cost"] + scene_vel_names + scene_cost_names + ref_vel_names + ref_cost_names + param_names


# ---------------------------------------------------------------------------
# Parameter conversion (single source of truth — fixes issue #3)
# ---------------------------------------------------------------------------

def point_to_params(point: list[float]) -> dict[str, float]:
    """Convert an optimizer point (list in space order) to a named dict."""
    return {dim.name: point[i] for i, dim in enumerate(space)}


def sim_params_from_point(point: list[float]) -> dict[str, Any]:
    """Build the sim_params dict consumed by simulation.run_simulation().

    This is the *only* place that maps optimizer space → simulation parameters.
    """
    params = point_to_params(point)
    m_mag = MAGNETIC_MOMENT * params["magnetic_moment_fudge"]
    kp_mag = m_mag * MAGNETIC_FIELD_MAGNITUDE * params["magnetic_field_fudge"]
    return {
        "ground_friction": [
            params["sliding_friction"],
            params["torsional_friction"],
            params["rolling_friction"],
        ],
        "solref": [params["solref_timeconst"], params["solref_dampratio"]],
        "solimp": [
            params["solimp_dmin"],
            params["solimp_dmax"],
            params["solimp_width"],
            params["solimp_midpoint"],
            params["solimp_power"],
        ],
        "dof_damping": params["dof_damping"],
        "kp_mag": kp_mag,
        "mag_params": {"m_mag": m_mag},
    }
