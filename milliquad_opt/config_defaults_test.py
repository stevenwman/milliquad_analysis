"""Minimal config for validating MuJoCo default params on scene2 f20 only."""

from config import space, sim_params_from_point, MJCF_PATHS  # noqa: F401

SIM_DURATION = 3.0
INIT_YAW_JITTER_DEG = 2

REFERENCE_DATA = [
    {"scene": "scene2", "ctrl_freq": 20.0, "speed": 0.1131, "speed_std": 0.0420, "weight": 1.0},
]
