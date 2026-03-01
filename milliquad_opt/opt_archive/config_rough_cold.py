"""Rough terrain cold-start config.

Inherits everything from config_rough but removes warm-start (CMAES_X0=None)
and widens sigma for broader exploration.
"""

from config_rough import *  # noqa: F401,F403

CMAES_X0 = None
CMAES_SIGMA0 = 0.5
