"""Shared morphology color/label constants for all plotting scripts.

Single source of truth — import from here, don't duplicate.
"""

# Paper color scheme
MORPH_COLORS: dict[str, str] = {
    "scene1":      "#1E88E5",  # Blue
    "scene2":      "#FFC107",  # Yellow
    "scene4":      "#007561",  # Green
    "scene_wheel": "#D81B60",  # Red
}

# Short labels for paper figures
MORPH_LABELS: dict[str, str] = {
    "scene1":      "L1",
    "scene2":      "L2",
    "scene4":      "L4",
    "scene_wheel": "WR",
}

# Canonical plot order
MORPH_ORDER: list[str] = ["scene1", "scene2", "scene4", "scene_wheel"]
