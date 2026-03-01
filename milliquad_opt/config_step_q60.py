"""Step terrain optimization config — q60 windowed targets.

Same as config_step.py but REFERENCE_DATA uses q60 windowing
(30% window centered at 60% of recording, indices 45%–75%).
See SIM_EXP_COMPARE_SHENANIGANS.md for details.
"""

from config_step import *  # noqa: F401,F403

from typing import Any

# Override REFERENCE_DATA with q60-derived targets
REFERENCE_DATA: list[dict[str, Any]] = [
    # Single leg (scene1)
    {"scene": "scene1",      "ctrl_freq": 10.0, "speed": 0.0169, "speed_std": 0.0020, "weight": 1.0},
    {"scene": "scene1",      "ctrl_freq": 20.0, "speed": 0.0462, "speed_std": 0.0018, "weight": 1.0},
    {"scene": "scene1",      "ctrl_freq": 30.0, "speed": 0.0279, "speed_std": 0.0079, "weight": 1.0},
    # Double leg (scene2)
    {"scene": "scene2",      "ctrl_freq": 10.0, "speed": 0.0415, "speed_std": 0.0217, "weight": 1.0},
    {"scene": "scene2",      "ctrl_freq": 20.0, "speed": 0.0835, "speed_std": 0.0371, "weight": 1.0},
    {"scene": "scene2",      "ctrl_freq": 30.0, "speed": 0.1098, "speed_std": 0.0245, "weight": 1.0},
    # Quad leg (scene4)
    {"scene": "scene4",      "ctrl_freq": 10.0, "speed": 0.0769, "speed_std": 0.0070, "weight": 1.0},
    {"scene": "scene4",      "ctrl_freq": 20.0, "speed": 0.0963, "speed_std": 0.0154, "weight": 1.0},
    {"scene": "scene4",      "ctrl_freq": 30.0, "speed": 0.0764, "speed_std": 0.0248, "weight": 1.0},
    # Wheel — f10/f20 are failure modes (does not move), f30 moves
    {"scene": "scene_wheel", "ctrl_freq": 10.0, "speed": 0.0000, "speed_std": 0.0,    "weight": 1.0},
    {"scene": "scene_wheel", "ctrl_freq": 20.0, "speed": 0.0000, "speed_std": 0.0,    "weight": 1.0},
    {"scene": "scene_wheel", "ctrl_freq": 30.0, "speed": 0.0972, "speed_std": 0.0046, "weight": 1.0},
]
