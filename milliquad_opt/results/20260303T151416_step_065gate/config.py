"""Shared constants, 16-dim search space, and parameter conversion.

Single source of truth for simulation constants and the optimizer ↔ simulation
interface.  Terrain-specific configs (config_flat.py, config_step.py,
config_rough.py) import from here and add their own REFERENCE_DATA, MJCF_PATHS,
cost weights, and CMAES_X0.
"""

import pathlib
from typing import Any

import numpy as np
from skopt.space import Real

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
SETTLE_TIME = 0.1           # seconds before driving starts
STUCK_CHECK_INTERVAL = 5.0  # seconds between stuck checks
STUCK_THRESHOLD = 0.005     # minimum movement (meters) to avoid "stuck"
SIM_TIMESTEP = 1.0 / 2000.0 # MuJoCo timestep (2 kHz)
SIMULATION_TIMEOUT = 35     # wall-clock seconds per worker

# Initial robot pose
INITIAL_Z_HEIGHT = 0.002
INITIAL_QUATERNION = (0, 0, 1, 0)  # 180° about Y (w, x, y, z)
INITIAL_LEG_ANGLES = np.pi
LEG_BODY_OFFSET = 2  # body 0=world, 1=chassis, 2-5=legs

# Video
VIDEO_FRAMERATE = 60.0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
CAMERA_DISTANCE_RECORD = 0.2
CAMERA_DISTANCE_VIEWER = 0.1

# Cost
COST_FAILURE = 1e6

# ---------------------------------------------------------------------------
# Physics / magnetic constants
# ---------------------------------------------------------------------------
MU0_OVER_4PI = 1e-7
R_EPS = 1e-6
MAGNETIC_MOMENT = 1.13e-3
MAGNETIC_FIELD_MAGNITUDE = 2e-3

# ---------------------------------------------------------------------------
# Scene configuration (flat — used as base for terrain overlays)
# ---------------------------------------------------------------------------
MJCF_PATHS: dict[str, str] = {
    "scene1":      str(PACKAGE_DIR / "robots" / "quad"  / "scene_1_flat.xml"),
    "scene2":      str(PACKAGE_DIR / "robots" / "quad"  / "scene_2_flat.xml"),
    "scene4":      str(PACKAGE_DIR / "robots" / "quad"  / "scene_4_flat.xml"),
    "scene_wheel": str(PACKAGE_DIR / "robots" / "wheel" / "scene_wheel_flat.xml"),
}
DEFAULT_CTRL_FREQ = 30.0

# ---------------------------------------------------------------------------
# 16-dim search space
# ---------------------------------------------------------------------------
space: list[Real] = [
    # Original 13 dimensions
    Real(0.01, 2.0, "log-uniform", name="sliding_friction"),
    Real(1e-6, 10.0, "log-uniform", name="torsional_friction"),
    Real(1e-6, 1e-3, "log-uniform", name="rolling_friction"),
    Real(1e-5, 1.0, "log-uniform", name="solref_timeconst"),
    Real(1.0, 10.0, "log-uniform", name="solref_dampratio"),
    Real(0.001, 0.999, "uniform", name="solimp_dmin"),
    Real(0.01, 0.99, "uniform", name="solimp_delta_d"),
    Real(1e-7, 1, "log-uniform", name="solimp_width"),
    Real(0.01, 0.99, "uniform", name="solimp_midpoint"),
    Real(2.0, 7.0, "uniform", name="solimp_power"),
    Real(0.5, 1.5, "uniform", name="magnetic_moment_fudge"),
    Real(0.5, 1.5, "uniform", name="magnetic_field_fudge"),
    Real(1e-10, 1e-8, "log-uniform", name="dof_damping"),
    # 3 solver parameters
    Real(0, 60, "uniform", name="noslip_iterations"),
    Real(1e-6, 1e-3, "log-uniform", name="noslip_tolerance"),
    Real(0.0, 0.005, "uniform", name="margin"),
]

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
CSV_PATH = "multi_optimization_results.csv"
BEST_CSV_PATH = "optimization_bests.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ref_id(scene: str, ctrl_freq: float) -> str:
    freq_str = f"{ctrl_freq:g}".replace(".", "p")
    return f"{scene}_f{freq_str}"


def reference_rows(ref_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build structured reference rows from a REFERENCE_DATA list."""
    rows = []
    seen_ids = set()
    for row in ref_data:
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


def reference_ids(ref_data: list[dict[str, Any]]) -> list[str]:
    return [row["id"] for row in reference_rows(ref_data)]


def csv_fieldnames(ref_data: list[dict[str, Any]], mjcf_paths: dict[str, str],
                   extra_per_ref: list[str] | None = None) -> list[str]:
    """Column names for results CSV.

    extra_per_ref: additional per-ref columns (e.g. ["progress", "best_trial"]).
    """
    param_names = [dim.name for dim in space] + ["solimp_dmax"]
    rids = reference_ids(ref_data)
    scene_names = list(mjcf_paths.keys())
    cols = (
        ["id", "cost", "elapsed_min"]
        + [f"velocity_{s}" for s in scene_names]
        + [f"cost_{s}" for s in scene_names]
        + [f"velocity_{rid}" for rid in rids]
        + [f"cost_{rid}" for rid in rids]
        + [f"lateral_{rid}" for rid in rids]
        + [f"tumble_{rid}" for rid in rids]
        + [f"yaw_{rid}" for rid in rids]
    )
    for extra in (extra_per_ref or []):
        cols += [f"{extra}_{rid}" for rid in rids]
    cols += param_names
    return cols


# ---------------------------------------------------------------------------
# Parameter conversion
# ---------------------------------------------------------------------------

def point_to_params(point: list[float] | dict[str, float]) -> dict[str, float]:
    """Convert an optimizer point (list in space order, or dict) to a named dict."""
    if isinstance(point, dict):
        return {dim.name: point[dim.name] for dim in space}
    return {dim.name: point[i] for i, dim in enumerate(space)}


def sim_params_from_point(point: list[float]) -> dict[str, Any]:
    """Build the sim_params dict consumed by simulation.run_simulation().

    Maps 16-dim optimizer space → simulation parameters.
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
            params["solimp_dmin"] + params["solimp_delta_d"] * (0.9999 - params["solimp_dmin"]),
            params["solimp_width"],
            params["solimp_midpoint"],
            params["solimp_power"],
        ],
        "dof_damping": params["dof_damping"],
        "kp_mag": kp_mag,
        "mag_params": {"m_mag": m_mag},
        "noslip_iterations": int(round(params["noslip_iterations"])),
        "noslip_tolerance": params["noslip_tolerance"],
        "margin": params["margin"],
    }
