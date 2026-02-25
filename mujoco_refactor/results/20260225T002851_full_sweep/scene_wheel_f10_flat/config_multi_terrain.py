"""Multi-terrain optimization configuration (flat + step).

Combines flat and step terrain reference data with reduced frequency set
based on correlation analysis. Includes scene_wheel f20 as failure mode
constraint (target velocity = 0) from experimental observations.

Reference set design (19 total):
- Flat terrain (11 refs): scene1/2/4 × f10/f30/f50 + scene_wheel f10/f30
- Step terrain (8 refs): scene1/2/4 × f10/f30 + scene_wheel f20/f30

See analyze_ref_correlations.py for correlation analysis justifying f20 dropout.
"""

from typing import Any

from config_new import (
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
    # Scene XMLs
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
# Multi-terrain overrides
# ---------------------------------------------------------------------------
SIM_DURATION = 3.0  # Same as flat terrain (faster evaluation)
N_CALLS = 4800      # More budget for multi-terrain (2× single-terrain)
BATCH_SIZE = 16
INIT_YAW_JITTER_DEG = 3
INIT_JITTER_TRIALS = 3
INIT_JITTER_SEED = 12345
VELOCITY_DEADZONE = False
DEFAULT_CTRL_FREQ = 30.0

# Terrain-level weights (hierarchical cost)
FLAT_TERRAIN_WEIGHT = 1.0   # Weight for flat terrain aggregate cost
STEP_TERRAIN_WEIGHT = 1.0   # Weight for step terrain aggregate cost

# Component weights (terrain-specific)
# Flat terrain: higher lateral penalty, yaw enabled
FLAT_VELOCITY_COST_WEIGHT = 5.0
FLAT_TUMBLE_COST_WEIGHT = 3.0
FLAT_LATERAL_COST_WEIGHT = 5.0
FLAT_YAW_COST_WEIGHT = 1.0

# Step terrain: lower lateral penalty, yaw disabled (cliff-fall artifact)
STEP_VELOCITY_COST_WEIGHT = 5.0
STEP_TUMBLE_COST_WEIGHT = 1.0
STEP_LATERAL_COST_WEIGHT = 1.0
STEP_YAW_COST_WEIGHT = 0.0

# Shared constants
VELOCITY_VARIANCE_WEIGHT = 2.0  # Not used currently
YAW_THRESHOLD_DEG = 60.0
TUMBLE_THRESHOLD = 0.0
TUMBLE_PENALTY_SCALE = 0.1
COST_FAILURE = 1e6

# CMA-ES
CMAES_SIGMA0 = 0.5
OPTIMIZER_RANDOM_STATE = 69420
VERBOSE_BATCH = False
PROFILE_BATCH = True

# CSV filenames (written into run_dir at runtime)
CSV_PATH = "multi_optimization_results.csv"
BEST_CSV_PATH = "optimization_bests.csv"

# ---------------------------------------------------------------------------
# Warm-start from best flat-terrain params (20260222T181114_with_20hz_no-deadzone)
# These params achieve cost=0.380 on flat but fail badly on steps (~93% error).
# Multi-terrain optimization should find params that balance both terrains.
# ---------------------------------------------------------------------------
CMAES_X0: dict[str, float] | None = None  # Cold start from search space midpoints

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
# Multi-terrain reference data
#
# Each reference includes a 'terrain' field:
#   - 'flat': regular MJCF, velocity measured after settle time
#   - 'step': step-suffixed MJCF, velocity measured after pos[0] >= STEP_START_X
#
# SPECIAL CASE: scene_wheel f20
#   Experimental data shows this configuration FAILS in real life (robot doesn't move).
#   Target velocity set to 0.0 m/s to constrain optimizer away from this failure mode.
#   This is NOT redundant with f10/f30 correlations — it captures a distinct failure mode.
#
# Aggregation method (within each terrain):
#   - Flat: MEDIAN of jitter trials (tolerates 1 outlier per 5 trials)
#   - Step: BEST (argmin cost) of jitter trials (robust to cliff-fall artifacts)
# ---------------------------------------------------------------------------
REFERENCE_DATA: list[dict[str, Any]] = [
    # ========================================================================
    # FLAT TERRAIN (11 references)
    # Source: experimental_data/csv/flat/, mean velocity over steady-state window
    # ========================================================================

    # Single leg (scene1) — flat
    {"scene": "scene1", "ctrl_freq": 10.0, "speed": 0.0512, "speed_std": 0.0024, "weight": 1.0, "terrain": "flat"},
    {"scene": "scene1", "ctrl_freq": 30.0, "speed": 0.1187, "speed_std": 0.0127, "weight": 1.0, "terrain": "flat"},
    {"scene": "scene1", "ctrl_freq": 50.0, "speed": 0.1483, "speed_std": 0.0131, "weight": 1.0, "terrain": "flat"},

    # Double leg (scene2) — flat
    {"scene": "scene2", "ctrl_freq": 10.0, "speed": 0.0832, "speed_std": 0.0014, "weight": 1.0, "terrain": "flat"},
    {"scene": "scene2", "ctrl_freq": 30.0, "speed": 0.1796, "speed_std": 0.0179, "weight": 1.0, "terrain": "flat"},
    {"scene": "scene2", "ctrl_freq": 50.0, "speed": 0.2633, "speed_std": 0.0257, "weight": 1.0, "terrain": "flat"},

    # Quad leg (scene4) — flat
    {"scene": "scene4", "ctrl_freq": 10.0, "speed": 0.1121, "speed_std": 0.0060, "weight": 1.0, "terrain": "flat"},
    {"scene": "scene4", "ctrl_freq": 30.0, "speed": 0.2747, "speed_std": 0.0207, "weight": 1.0, "terrain": "flat"},
    {"scene": "scene4", "ctrl_freq": 50.0, "speed": 0.3274, "speed_std": 0.0556, "weight": 1.0, "terrain": "flat"},

    # Wheel (scene_wheel) — flat
    {"scene": "scene_wheel", "ctrl_freq": 10.0, "speed": 0.1423, "speed_std": 0.0012, "weight": 1.0, "terrain": "flat"},
    {"scene": "scene_wheel", "ctrl_freq": 30.0, "speed": 0.4578, "speed_std": 0.0089, "weight": 1.0, "terrain": "flat"},

    # ========================================================================
    # STEP TERRAIN (8 references)
    # Source: experimental_data/csv/steps/, forward velocity (vx), q75-300 window
    # (75% index +/- 150 timesteps, clamped to bounds)
    # ========================================================================

    # Single leg (scene1) — step
    {"scene": "scene1", "ctrl_freq": 10.0, "speed": 0.0199, "speed_std": 0.0018, "weight": 1.0, "terrain": "step"},
    {"scene": "scene1", "ctrl_freq": 30.0, "speed": 0.0331, "speed_std": 0.0066, "weight": 1.0, "terrain": "step"},

    # Double leg (scene2) — step
    {"scene": "scene2", "ctrl_freq": 10.0, "speed": 0.0542, "speed_std": 0.0105, "weight": 1.0, "terrain": "step"},
    {"scene": "scene2", "ctrl_freq": 30.0, "speed": 0.1335, "speed_std": 0.0129, "weight": 1.0, "terrain": "step"},

    # Quad leg (scene4) — step
    {"scene": "scene4", "ctrl_freq": 10.0, "speed": 0.0716, "speed_std": 0.0074, "weight": 1.0, "terrain": "step"},
    {"scene": "scene4", "ctrl_freq": 30.0, "speed": 0.0898, "speed_std": 0.0202, "weight": 1.0, "terrain": "step"},

    # Wheel (scene_wheel) — step
    # f20: FAILURE MODE — robot doesn't move at 20Hz on steps, target=0
    {"scene": "scene_wheel", "ctrl_freq": 20.0, "speed": 0.0000, "speed_std": 0.0050, "weight": 2.0, "terrain": "step"},
    {"scene": "scene_wheel", "ctrl_freq": 30.0, "speed": 0.0938, "speed_std": 0.0097, "weight": 1.0, "terrain": "step"},
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def reference_rows() -> list[dict[str, Any]]:
    """Return reference rows for optimization from REFERENCE_DATA.

    Adds terrain suffix to reference IDs to distinguish flat vs step versions
    of the same scene/frequency combination.
    """
    rows = []
    seen_ids = set()
    for row in REFERENCE_DATA:
        scene = row["scene"]
        ctrl_freq = float(row.get("ctrl_freq", DEFAULT_CTRL_FREQ))
        speed = float(row["speed"])
        weight = float(row.get("weight", 1.0))
        terrain = row["terrain"]  # 'flat' or 'step'

        # Generate unique ID with terrain suffix
        base_id = _make_ref_id(scene, ctrl_freq)
        ref_id = f"{base_id}_{terrain}"

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
            "terrain": terrain,
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
    ref_best_trial_names = [f"best_trial_{rid}" for rid in rids]

    # Scene-level aggregates (per terrain)
    scene_names = list(MJCF_PATHS.keys())
    terrains = ["flat", "step"]
    scene_cost_names = [f"cost_{s}_{t}" for t in terrains for s in scene_names]
    scene_vel_names = [f"velocity_{s}_{t}" for t in terrains for s in scene_names]

    # Terrain-level aggregates
    terrain_cost_names = [f"cost_{t}" for t in terrains]
    terrain_vel_names = [f"velocity_{t}" for t in terrains]

    return (
        ["id", "cost", "elapsed_min"]
        + terrain_vel_names + terrain_cost_names
        + scene_vel_names + scene_cost_names
        + ref_vel_names + ref_cost_names
        + ref_lateral_names + ref_tumble_names + ref_yaw_names
        + ref_best_trial_names
        + param_names
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_reference_data():
    """Sanity checks on REFERENCE_DATA structure."""
    flat_refs = [r for r in REFERENCE_DATA if r["terrain"] == "flat"]
    step_refs = [r for r in REFERENCE_DATA if r["terrain"] == "step"]

    assert len(flat_refs) == 11, f"Expected 11 flat refs, got {len(flat_refs)}"
    assert len(step_refs) == 8, f"Expected 8 step refs, got {len(step_refs)}"
    assert len(REFERENCE_DATA) == 19, f"Expected 19 total refs, got {len(REFERENCE_DATA)}"

    # Check that all references have required fields
    for row in REFERENCE_DATA:
        assert "scene" in row, "Missing 'scene' field"
        assert "ctrl_freq" in row, "Missing 'ctrl_freq' field"
        assert "speed" in row, "Missing 'speed' field"
        assert "terrain" in row, "Missing 'terrain' field"
        assert row["terrain"] in ["flat", "step"], f"Invalid terrain: {row['terrain']}"

    # Check scene_wheel f20 failure mode (step only, target=0)
    wheel_f20_refs = [r for r in REFERENCE_DATA if r["scene"] == "scene_wheel" and r["ctrl_freq"] == 20.0]
    assert len(wheel_f20_refs) == 1, f"Expected 1 scene_wheel f20 ref (step only), got {len(wheel_f20_refs)}"
    for r in wheel_f20_refs:
        assert r["speed"] == 0.0, f"scene_wheel f20 should have speed=0.0 (failure mode), got {r['speed']}"
        assert r["weight"] == 2.0, f"scene_wheel f20 should have weight=2.0 (emphasize constraint), got {r['weight']}"
        assert r["terrain"] == "step", f"scene_wheel f20 should be step terrain only, got {r['terrain']}"

    print("✓ REFERENCE_DATA validation passed")
    print(f"  - {len(flat_refs)} flat terrain references")
    print(f"  - {len(step_refs)} step terrain references")
    print(f"  - 19 total references × {INIT_JITTER_TRIALS} jitter = {19 * INIT_JITTER_TRIALS} sims/eval")


if __name__ == "__main__":
    validate_reference_data()
    print("\nReference IDs:")
    for rid in reference_ids():
        print(f"  - {rid}")
