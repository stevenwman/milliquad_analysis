"""Trial validation and gating for L2/L4 analysis scripts.

Mirrors the filtering logic from plot_validation.py (GATE_END, GATE_EXEMPT,
INVERTED_PITCH_THRESHOLD) but works on raw NPZ arrays instead of CSV rows.

All l2_l4 scripts should use these for consistent trial filtering and
measurement windowing.
"""

from __future__ import annotations

import re

import numpy as np

# ---------------------------------------------------------------------------
# NPZ key parsing
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(r"^(.+?)_f(\d+(?:p\d+)?)_t(\d+)_(.+)$")

SCENE_LABELS = {
    "scene1": "1-spoke",
    "scene2": "2-spoke",
    "scene4": "4-spoke",
    "scene_wheel": "wheel",
}


def parse_key(key: str):
    """Parse NPZ key → (scene, freq, trial_idx, field) or None."""
    m = _KEY_RE.match(key)
    if not m:
        return None
    scene = m.group(1)
    freq = float(m.group(2).replace("p", "."))
    trial = int(m.group(3))
    field = m.group(4)
    return scene, freq, trial, field


# ---------------------------------------------------------------------------
# Constants (matching plot_validation.py)
# ---------------------------------------------------------------------------

SETTLE_TIME = 0.1  # seconds

# Robot must reach this x (m) to count as completed terrain traversal
GATE_END = {"rough": 0.155, "step": 0.1015}

# Exempt from gate check (slow but valid — confirmed visually)
GATE_EXEMPT = {("scene1", 10.0)}

# Pitch std above this = inverted robot, skip trial
INVERTED_PITCH_THRESHOLD = 30.0  # degrees

# Terrain spatial bounds for measurement windowing
# Rough: [FLAT_LEAD, FLAT_LEAD + 2*_X_HALF] = [0.005, 0.155]
ROUGH_START_X = 0.005
ROUGH_END_X = 0.155

# Step: [STEP_START_X, STEP_END_X] = [0.05, 0.1015]
STEP_START_X = 0.05
STEP_END_X = 0.1015


# ---------------------------------------------------------------------------
# Trial validation
# ---------------------------------------------------------------------------

def is_valid_trial(
    pos_x: np.ndarray,
    pitch: np.ndarray,
    terrain: str,
    scene: str | None = None,
    freq: float | None = None,
) -> bool:
    """Check if a trial completed the terrain and isn't inverted.

    Mirrors _is_valid_trial() from plot_validation.py but works on arrays.
    """
    # Gate check: did robot reach end of terrain?
    gate = GATE_END.get(terrain)
    if gate is not None:
        exempt = scene is not None and freq is not None and (scene, freq) in GATE_EXEMPT
        if not exempt and float(pos_x.max()) < gate:
            return False

    # Inversion check
    if float(np.std(pitch)) > INVERTED_PITCH_THRESHOLD:
        return False

    return True


# ---------------------------------------------------------------------------
# Measurement window (active mask)
# ---------------------------------------------------------------------------

def active_mask(
    time: np.ndarray,
    pos_x: np.ndarray,
    terrain: str,
) -> np.ndarray:
    """Boolean mask for the measurement window within the terrain.

    Flat:  time >= SETTLE_TIME (no terrain boundary)
    Rough: time >= SETTLE_TIME AND pos_x within 90% of terrain
    Step:  pos_x within [STEP_START_X, 90% of STEP_END_X]
           (no settle_time — step region is naturally after flat lead-in)
    """
    if terrain == "step":
        cutoff = STEP_START_X + 0.9 * (STEP_END_X - STEP_START_X)
        return (pos_x >= STEP_START_X) & (pos_x < cutoff)
    elif terrain == "rough":
        cutoff = ROUGH_START_X + 0.9 * (ROUGH_END_X - ROUGH_START_X)
        return (time >= SETTLE_TIME) & (pos_x >= ROUGH_START_X) & (pos_x < cutoff)
    else:
        # Flat
        return time >= SETTLE_TIME


# ---------------------------------------------------------------------------
# NPZ discovery
# ---------------------------------------------------------------------------

def find_npz(run_dir) -> "pathlib.Path":
    """Find the latest datetime-prefixed validation_trajectories.npz."""
    import pathlib
    import sys
    run_dir = pathlib.Path(run_dir)
    candidates = sorted(run_dir.glob("*_validation_trajectories.npz"))
    if candidates:
        return candidates[-1]
    fallback = run_dir / "validation_trajectories.npz"
    if fallback.exists():
        return fallback
    sys.exit(f"No validation_trajectories.npz in {run_dir}")


def detect_terrain(run_dir) -> str:
    """Auto-detect terrain type from dir name."""
    import pathlib
    name = pathlib.Path(run_dir).name
    for t in ["flat", "step", "rough"]:
        if t in name:
            return t
    return "flat"
