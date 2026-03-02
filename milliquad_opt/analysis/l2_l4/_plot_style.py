"""Shared plot styling for L2/L4 hypothesis figures.

Colors/labels match mujoco_refactor/morphology_style.py and plot_validation.py.
"""

from __future__ import annotations

import matplotlib

matplotlib.rcParams["font.family"] = "TeX Gyre Pagella"
matplotlib.rcParams["font.size"] = 10

MORPH_COLORS: dict[str, str] = {
    "scene1": "#1E88E5",
    "scene2": "#FFC107",
    "scene4": "#007561",
    "scene_wheel": "#D81B60",
}

MORPH_LABELS: dict[str, str] = {
    "scene1": "L1",
    "scene2": "L2",
    "scene4": "L4",
    "scene_wheel": "WR",
}

MORPH_ORDER: list[str] = ["scene1", "scene2", "scene4", "scene_wheel"]

TERRAIN_TITLES: dict[str, str] = {
    "flat": "Flat",
    "step": "Step",
    "rough": "Rough",
}
