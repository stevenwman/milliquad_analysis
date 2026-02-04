"""
Shared constants, search space definition, and parameter conversion.

Single source of truth for all values shared between simulation and optimization.
"""

import os
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
VIDEO_FRAMERATE = 60.0  # frames per second for video recording

# ---------------------------------------------------------------------------
# Physics / magnetic constants
# ---------------------------------------------------------------------------
MU0_OVER_4PI = 1e-7  # μ₀/(4π) in SI (N/A²)
R_EPS = 1e-6  # minimum r in dipole field to avoid 1/r³ blow-up (meters)
MAGNETIC_MOMENT = 1.13e-3
MAGNETIC_FIELD_MAGNITUDE = 2e-3

# ---------------------------------------------------------------------------
# Contact solver defaults (solimp midpoint & power — previously magic numbers)
# ---------------------------------------------------------------------------
SOLIMP_MIDPOINT = 0.5
SOLIMP_POWER = 1.0

# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
TARGET_VELOCITIES: dict[str, float] = {
    "scene4": 0.21,  # 21 cm/s for 4-legged robot
    "scene2": 0.14,  # 14 cm/s for 2-legged robot
}
MJCF_PATHS: dict[str, str] = {
    "scene4": str(PACKAGE_DIR / "mulit_milli_quad" / "scene_4.xml"),
    "scene2": str(PACKAGE_DIR / "mulit_milli_quad" / "scene_2.xml"),
}

# ---------------------------------------------------------------------------
# Optimization hyper-parameters
# ---------------------------------------------------------------------------
N_CALLS = 200  # total optimization iterations
SIM_DURATION = 5.0  # seconds per simulation run
SIMULATION_TIMEOUT = 20  # wall-clock seconds per worker
ROLLOUTS_PER_SCENE = 1  # sims per scene per iteration (>1 only for noisy sims)
BATCH_SIZE = 8  # points proposed per batch (8×2 scenes = 16 tasks)
NUM_SCENES = len(MJCF_PATHS)
POOL_SIZE = min(os.cpu_count() or 16, BATCH_SIZE * NUM_SCENES)
VERBOSE_BATCH = True
PROFILE_BATCH = True
# "rf" = random forest (fast ask/tell); "gp" = Gaussian process (slow at high n)
BASE_ESTIMATOR = "rf"

# ---------------------------------------------------------------------------
# Seed from older single-scene optimization
# ---------------------------------------------------------------------------
SEED_FROM_OLD_CSV = False
# Order: sliding, torsional, rolling, solref_tc, solref_dr, solimp_dmin,
#        solimp_dmax, solimp_width, moment_fudge, field_fudge, dof_damping
SEED_POINT: list[float] = [
    0.00014225746640521907, 0.0021388784110800154, 5.292387847485097e-05,
    0.001, 0.7414912155887285, 0.9084351427617432, 0.9734506827063522,
    0.0037927813470769885,
    1.0, 1.0, 7e-10,
]

# ---------------------------------------------------------------------------
# Cost-function constants
# ---------------------------------------------------------------------------
TUMBLE_THRESHOLD = 0.3  # cos(angle) below this → "tumbling" (~72.5° from vertical)
TUMBLE_PENALTY_SCALE = 0.1  # per-frame penalty when uprightness < threshold
COST_FAILURE = 1e6  # cost for failed / empty trajectory
VELOCITY_COST_WEIGHT = 1.0
TUMBLE_COST_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# Search space (11 dimensions)
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
    Real(0.5, 1.5, "uniform", name="magnetic_moment_fudge"),
    Real(0.5, 1.5, "uniform", name="magnetic_field_fudge"),
    Real(7e-11, 7e-9, "log-uniform", name="dof_damping"),
]

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
CSV_PATH = "multi_optimization_results.csv"


def csv_fieldnames() -> list[str]:
    """Column names for the results CSV."""
    param_names = [dim.name for dim in space]
    scene_cost_names = [f"cost_{scene}" for scene in MJCF_PATHS]
    scene_vel_names = [f"velocity_{scene}" for scene in MJCF_PATHS]
    return ["id", "cost"] + scene_vel_names + scene_cost_names + param_names


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
            SOLIMP_MIDPOINT,
            SOLIMP_POWER,
        ],
        "dof_damping": params["dof_damping"],
        "kp_mag": kp_mag,
        "mag_params": {"m_mag": m_mag},
    }
