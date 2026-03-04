"""Shared utilities for the analysis toolkit.

Provides: param loading, terrain detection, pitch/COT/velocity computation.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

# Parent dir on sys.path so we can import config, simulation, etc.
_PARENT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from config import space, sim_params_from_point, reference_rows, SETTLE_TIME  # noqa: E402

PARAM_NAMES = [dim.name for dim in space]

# Leg body indices in leg_xpos array: [FR=0, FL=1, BR=2, BL=3]
FL_IDX = 1
BL_IDX = 3


# ---------------------------------------------------------------------------
# Param loading
# ---------------------------------------------------------------------------

def load_best_point(run_dir: pathlib.Path) -> list[float]:
    """Load best parameter point from a completed run (full precision from multi CSV)."""
    bests_csv = run_dir / "optimization_bests.csv"
    rows = list(csv.DictReader(open(bests_csv)))
    if not rows:
        sys.exit(f"ERROR: no rows in {bests_csv}")
    best = rows[-1]
    multi_csv = run_dir / "multi_optimization_results.csv"
    if multi_csv.exists():
        best_id = best["id"]
        for row in csv.DictReader(open(multi_csv)):
            if row["id"] == best_id:
                return [float(row[name]) for name in PARAM_NAMES]
    return [float(best[name]) for name in PARAM_NAMES]


def detect_terrain(run_dir: pathlib.Path) -> str:
    """Auto-detect base terrain type from results dir name.

    Maps 'rough_cold' -> 'rough', 'step_cold' -> 'step', etc.
    """
    name = run_dir.name
    for t in ["flat_no20", "flat", "step", "rough"]:
        if t in name:
            return t
    return "flat"


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------

def compute_pitch_series(traj: list[dict]) -> np.ndarray:
    """Per-timestep pitch (degrees) from FL/BL leg body positions.

    Yaw-invariant: uses horizontal distance sqrt(dx² + dy²) instead of
    world-frame dx, so 180° yaw rotations don't register as pitch.
    Result bounded to [-90, +90]°.
    """
    fl_pos = np.array([s["leg_xpos"][FL_IDX] for s in traj])
    bl_pos = np.array([s["leg_xpos"][BL_IDX] for s in traj])

    dx = fl_pos[:, 0] - bl_pos[:, 0]
    dy = fl_pos[:, 1] - bl_pos[:, 1]
    dz = fl_pos[:, 2] - bl_pos[:, 2]
    d_horiz = np.sqrt(dx**2 + dy**2)
    return np.degrees(np.arctan2(dz, d_horiz))


def compute_pitch_rms(
    traj: list[dict],
    settle_time: float = SETTLE_TIME,
    step_start_x: float | None = None,
    step_end_x: float | None = None,
) -> float:
    """Pitch amplitude RMS (degrees) from FL/BL leg body positions.

    Yaw-invariant. For flat/rough: time-gated after settle_time.
    For step: spatially-gated between step_start_x and 90% of step_end_x.
    """
    theta = compute_pitch_series(traj)

    if step_start_x is not None and step_end_x is not None:
        cutoff_x = step_start_x + 0.9 * (step_end_x - step_start_x)
        pos_x = np.array([s["pos"][0] for s in traj])
        enter_idx = np.searchsorted(pos_x, step_start_x)
        exit_indices = np.where(pos_x >= cutoff_x)[0]
        exit_idx = exit_indices[0] if len(exit_indices) else len(traj) - 1
        if exit_idx <= enter_idx or (exit_idx - enter_idx) < 10:
            return 0.0
        theta_active = theta[enter_idx:exit_idx + 1]
    else:
        t = np.array([s["time"] for s in traj])
        mask = t >= settle_time
        if mask.sum() < 10:
            return 0.0
        theta_active = theta[mask]

    theta_active = theta_active - theta_active[0]  # detrend
    return float(np.std(theta_active))


# ---------------------------------------------------------------------------
# COT (Cost of Transport)
# ---------------------------------------------------------------------------

def compute_cot(
    traj: list[dict],
    robot_mass: float,
    settle_time: float = SETTLE_TIME,
    g: float = 9.81,
    step_start_x: float | None = None,
    step_end_x: float | None = None,
) -> float | None:
    """Cost of transport using naive P = tau_ext . omega (correct under RK4).

    For flat/rough: time-gated after settle_time.
    For step: spatially-gated between step_start_x and 90% of step_end_x
    (excludes flat lead-in and cliff-fall).

    Returns COT = W_ext / (m * g * d), or None if trajectory is too short.
    """
    if step_start_x is not None and step_end_x is not None:
        # Spatial gating — same window as extract_velocity for step terrain
        cutoff_x = step_start_x + 0.9 * (step_end_x - step_start_x)
        enter_idx = None
        exit_idx = None
        for i, s in enumerate(traj):
            if enter_idx is None and s["pos"][0] >= step_start_x:
                enter_idx = i
            if s["pos"][0] >= cutoff_x:
                exit_idx = i
                break
        if enter_idx is None:
            return None
        if exit_idx is None:
            exit_idx = len(traj) - 1
        if exit_idx <= enter_idx:
            return None
        active = traj[enter_idx:exit_idx + 1]
    else:
        # Time gating
        start_idx = 0
        for i, s in enumerate(traj):
            if s["time"] >= settle_time:
                start_idx = i
                break
        active = traj[start_idx:]

    if len(active) < 2:
        return None
    if "tau_ext" not in active[0] or "omega" not in active[0]:
        return None

    n = len(active) - 1
    dt = np.empty(n)
    power = np.empty(n)

    for i in range(n):
        s = active[i]
        dt[i] = active[i + 1]["time"] - s["time"]
        power[i] = np.sum(s["tau_ext"] * s["omega"])

    energy = np.sum(power * dt)
    distance = np.linalg.norm(active[-1]["pos"][:2] - active[0]["pos"][:2])
    mgd = robot_mass * g * distance
    if mgd < 1e-12:
        return None
    return float(energy / mgd)


# ---------------------------------------------------------------------------
# Velocity extraction
# ---------------------------------------------------------------------------

def min_window_velocity(
    traj: list[dict],
    ctrl_freq: float,
    settle_time: float = SETTLE_TIME,
    n_periods: int = 5,
    step_start_x: float | None = None,
    step_end_x: float | None = None,
) -> float:
    """Minimum mean |vx| over any sliding window of n_periods actuation cycles.

    Returns the min window mean (m/s). Caller decides stall threshold.
    Returns 0.0 for degenerate trajectories.
    """
    # Select measurement window (same gating as extract_velocity / compute_cot)
    if step_start_x is not None and step_end_x is not None:
        cutoff_x = step_start_x + 0.9 * (step_end_x - step_start_x)
        enter_idx = None
        exit_idx = None
        for i, s in enumerate(traj):
            if enter_idx is None and s["pos"][0] >= step_start_x:
                enter_idx = i
            if s["pos"][0] >= cutoff_x:
                exit_idx = i
                break
        if enter_idx is None:
            return 0.0
        if exit_idx is None:
            exit_idx = len(traj) - 1
        active = traj[enter_idx:exit_idx + 1]
    else:
        start_idx = 0
        for i, s in enumerate(traj):
            if s["time"] >= settle_time:
                start_idx = i
                break
        active = traj[start_idx:]

    if len(active) < 2:
        return 0.0

    # Compute per-timestep forward velocity
    pos_x = np.array([s["pos"][0] for s in active])
    times = np.array([s["time"] for s in active])
    dt = times[1] - times[0]
    if dt < 1e-10:
        return 0.0
    vx = np.diff(pos_x) / np.diff(times)

    # Sliding window size
    period = 1.0 / ctrl_freq
    window_sec = n_periods * period
    window_steps = max(1, int(window_sec / dt))

    if len(vx) < window_steps:
        return float(np.mean(np.abs(vx)))

    # Sliding window mean of |vx|
    cumsum = np.cumsum(np.abs(vx))
    cumsum = np.insert(cumsum, 0, 0.0)
    window_means = (cumsum[window_steps:] - cumsum[:-window_steps]) / window_steps
    return float(np.min(window_means))


def extract_velocity(
    traj: list[dict],
    settle_time: float = SETTLE_TIME,
    step_start_x: float | None = None,
    step_end_x: float | None = None,
    gate_fraction: float = 0.9,
) -> float | None:
    """Extract forward velocity from trajectory.

    For flat/rough: time-gated after settle_time.
    For step: spatial-gated between step_start_x and gate_fraction of step region.

    Returns None if the trial doesn't reach the spatial gate (step terrain).
    """
    if step_start_x is not None and step_end_x is not None:
        # Spatial gating for step terrain
        cutoff_x = step_start_x + gate_fraction * (step_end_x - step_start_x)
        enter_idx = None
        exit_idx = None
        for i, s in enumerate(traj):
            if enter_idx is None and s["pos"][0] >= step_start_x:
                enter_idx = i
            if s["pos"][0] >= cutoff_x:
                exit_idx = i
                break
        if enter_idx is None:
            return None
        if exit_idx is None:
            return None  # didn't reach gate
        if exit_idx <= enter_idx:
            return None
        dt = traj[exit_idx]["time"] - traj[enter_idx]["time"]
        dx = traj[exit_idx]["pos"][0] - traj[enter_idx]["pos"][0]
        return dx / dt if dt > 1e-6 else 0.0
    else:
        # Time gating for flat/rough
        settle_idx = 0
        for i, s in enumerate(traj):
            if s["time"] >= settle_time:
                settle_idx = i
                break
        dt = traj[-1]["time"] - traj[settle_idx]["time"]
        dx = traj[-1]["pos"][0] - traj[settle_idx]["pos"][0]
        return dx / dt if dt > 1e-6 else 0.0
