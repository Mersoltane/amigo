"""Amigo interior-point optimizer package.

Module layout (each file = one concept):

Orchestration:
  ipm_driver.py                  Main Optimizer class (pure brain)

Setup:
  problem_setup.py               __init__ helpers + loop-config helpers
  iterate_initialization.py      Pre-loop primal-dual iterate setup
  ipm_state.py                   IpmState dataclass (per-iteration state)

Convergence and options:
  convergence_check.py           Convergence criteria
  default_options.py             Default option values

Barrier parameter:
  barrier_update.py              Per-iteration orchestration (classical + QF)
  barrier_heuristic.py           LOQO-style data-driven update
  barrier_quality_function.py    QF oracle + golden-section search
  barrier_adaptive_mu.py         Globalization (progress/remember/safeguard)

Newton direction:
  newton_direction.py            KKT assembly + factorize + back-solve
  inertia_correction.py          Algorithm IC (Wachter & Biegler 2006)
  (iterative refinement lives in solvers/linear_solver.py - pure lin-alg)

Line search:
  merit_line_search.py           Backtracking Armijo with SOC
  filter_line_search.py          Filter LS + phi_mu + theta + watchdog
  filter_acceptance.py           2D filter data structure

Scaling and bound safeguards:
  optimality_scaling.py          KKT error scaling (s_d, s_c)
  bound_safeguards.py            Slack flooring + adaptive tau

Multipliers:
  multiplier_initialization.py   Least-squares / affine / zero start

Restoration and diagnostics:
  feasibility_restoration.py     Restoration phase
  iteration_logger.py            Iter-data assembly + progress table
  newton_diagnostics.py          Debug diagnostics (check_update_step)
  post_optimization.py           Output + post-opt derivatives

Solvers:
  solvers/                       Linear solver implementations
"""

from .ipm_driver import Optimizer
from .inertia_correction import InertiaCorrector
from .filter_acceptance import Filter
from .default_options import get_default_options

from .solvers import (
    DirectCudaSolver,
    LNKSInexactSolver,
    MumpsSolver,
    PardisoSolver,
    DirectPetscSolver,
    DirectScipySolver,
)
