"""Step terrain config — cold start (no warm-start point).

Inherits everything from config_step; overrides CMAES_X0 and sigma.
"""

from config_step import *  # noqa: F401, F403

# Cold start: let CMA-ES sample from the center of the space
CMAES_X0: dict | None = None
CMAES_SIGMA0 = 0.3
