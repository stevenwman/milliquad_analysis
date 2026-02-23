"""Terrain presets for terrain_test.py.

Separate from optimizer config.py so terrain validation settings
don't pollute the optimization search space / cost function config.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Bookkeeping constants
# ---------------------------------------------------------------------------
B_FIELD_MT = 2.0  # mT — external magnetic field amplitude (all experiments)

# ---------------------------------------------------------------------------
# Jitter defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_JITTER_TRIALS = 1
DEFAULT_JITTER_DEG = 2.0
JITTER_BASE_SEED = 99999

# ---------------------------------------------------------------------------
# Named terrain presets
# ---------------------------------------------------------------------------
# Each preset is a plain dict.  Required key: "type" (flat | step | rough).
# Optional key: "ctrl_freq" (Hz, default 30.0).
# All other keys are terrain-type-specific geometry params.
#
# Users: copy an existing preset, tweak values, give it a new name.
# CLI can also override individual keys, e.g. --step-height 0.003.

TERRAIN_PRESETS: dict[str, dict] = {
    # --- Flat (baseline) ---
    "flat": {
        "type": "flat",
        "ctrl_freq": 30.0,
    },

    # --- Step presets ---
    "step_default": {
        "type": "step",
        "ctrl_freq": 30.0,
        "step_height": 0.001,        # 1 mm
        "step_length": 0.0045,       # 4.5 mm
        "step_count": 8,
        "final_step_length": 0.02,   # 20 mm extended platform after last rise
        "step_width": 0.1,           # 100 mm
        "flat_lead": 0.0075,         # 7.5 mm flat ground before first step
    },
    "step_tall": {
        "type": "step",
        "ctrl_freq": 30.0,
        "step_height": 0.003,        # 3 mm
        "step_length": 0.0045,
        "step_count": 5,
        "final_step_length": 0.02,
        "step_width": 0.1,
        "flat_lead": 0.0075,
    },
    "step_many": {
        "type": "step",
        "ctrl_freq": 30.0,
        "step_height": 0.001,        # 1 mm
        "step_length": 0.0045,
        "step_count": 10,
        "final_step_length": 0.02,
        "step_width": 0.1,
        "flat_lead": 0.0075,
    },

    # --- Rough presets ---
    "rough_mild": {
        "type": "rough",
        "ctrl_freq": 30.0,
        "height_mean": 0.002,        # 2 mm mean tile height
        "height_std": 0.0005,        # 0.5 mm std
        "tile_size": 0.005,          # 5 mm square tiles
        "grid_nx": 20,
        "grid_ny": 10,
        "z_safe": 0.001,             # 1 mm min material thickness
        "seed": 42,
        "flat_lead": 0.05,           # 50 mm flat ground before terrain
    },
    "rough_harsh": {
        "type": "rough",
        "ctrl_freq": 30.0,
        "height_mean": 0.002,
        "height_std": 0.001,         # 1 mm std
        "tile_size": 0.005,
        "grid_nx": 20,
        "grid_ny": 10,
        "z_safe": 0.001,
        "seed": 42,
        "flat_lead": 0.05,
    },
}
