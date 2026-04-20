"""Mutable per-iteration state for the IPM optimization loop.

Groups all scalar counters, tracking variables, and strategy-specific
state that change across iterations into a single object.  Replaces
~60 lines of scattered locals in ipm_driver.optimize() with one
IpmState instance that can be passed to helper methods.

Instance-wide configuration (max_iters, tolerances, etc.) stays in
the options dict.  Algorithmic state that persists across calls
(self._qf_refs, self._filter_theta_0, etc.) stays on self.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IpmState:
    """Per-iteration mutable state for the IPM loop.

    Categorized by concern:
      - Step tracking: results of the previous line search
      - Convergence tracking: counters for acceptable/precision floor
      - Quality-function mu bounds and mode
      - Rejection tracking: consecutive failures, zero-step recovery
      - Filter monotone fallback state (classical barrier path)
    """

    # Step tracking (updated after each accepted step)
    line_iters: int = 0
    alpha_x_prev: float = 0.0
    alpha_z_prev: float = 0.0
    x_index_prev: int = -1
    z_index_prev: int = -1

    # Convergence tracking
    prev_res_norm: float = float("inf")
    acceptable_counter: int = 0
    precision_floor_count: int = 0

    # Quality-function bounds and mode (set on first iteration)
    qf_mu_min: float = 0.0
    qf_mu_max: float = -1.0
    qf_free_mode: bool = True
    qf_monotone_mu: Optional[float] = None

    # Rejection tracking
    consecutive_rejections: int = 0
    zero_step_count: int = 0

    # Classical-barrier filter monotone fallback
    filter_monotone_mode: bool = False
    filter_monotone_mu: Optional[float] = None

    # Filter reset heuristic
    count_successive_filter_rejections: int = 0
    filter_reset_count: int = 0

    # Barrier parameter at the start of the most recent residual eval
    res_norm_mu: float = 0.0
