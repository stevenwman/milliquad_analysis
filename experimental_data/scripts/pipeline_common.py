from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MM_SCALE = 1000.0


@dataclass(frozen=True)
class TrialData:
    t: np.ndarray
    vx: np.ndarray
    vy: np.ndarray | None
    speed: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    omega: np.ndarray


def base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def load_trial_csv(csv_name: str) -> np.ndarray:
    root = base_dir()
    candidates = [
        root / "csv" / "flat" / csv_name,
        root / "csv" / "steps" / csv_name,
        root / "csv" / "summary" / csv_name,
        root / csv_name,
    ]
    for path in candidates:
        if path.exists():
            return np.genfromtxt(path, delimiter=",", skip_header=2)
    raise FileNotFoundError(f"CSV not found in expected folders: {csv_name}")


def clean_for_interp(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]
    if t.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    unique_t, idx = np.unique(t, return_index=True)
    return unique_t, x[idx]
