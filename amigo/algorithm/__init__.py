"""Amigo interior-point optimizer package.

Module map, grouped by concern:

  ipm_driver.py                  Optimizer class, main loop
  ipm_state.py                   Per-iteration state
  evaluator.py                   Gradient/residual/Hessian evaluation cache
  newton_direction.py            KKT solve and step assembly
  convergence_check.py           Convergence, divergence, and stall tests
  iteration_logger.py            Iter-data assembly + progress table
  default_options.py             Default option values
  homotopy.py                    Path following over solves in one parameter
  regime.py                      Easy, hard, jam, or stuck labels per iteration
  step_contract.py               The dual step contract observer

  initialization/                Starting point construction
    iterate_initialization.py    Bounds push, slacks, bound duals
    multiplier_initialization.py Least-squares / affine / zero duals
    nlp_scaling.py               Gradient-based NLP scaling
    presolve.py                  Starting point via restoration

  barrier_strategy/              Barrier-parameter strategies
    base.py                      BarrierStrategy ABC, coupled tau
    monotone.py                  Monotone decrease with progress gate
    quality_function.py          Quality function + adaptive-mu globalization

  globalization/                 Step acceptance and rescue
    filter_line_search.py        Filter line search with second-order correction
    filter_acceptance.py         Filter and its acceptance rules
    funnel_line_search.py        Funnel line search
    merit_line_search.py         Augmented Lagrangian merit line search
    feasibility_restoration.py   Restoration phase on the l1 violation

  solvers/                       Linear solvers + inertia correction
    inertia_correction.py        Primal and dual regularization
"""

from .ipm_driver import Optimizer, WarmStart
from .default_options import get_default_options
from .homotopy import Homotopy, HomotopyHistory, HomotopyOptions, HomotopyStep

from .solvers import (
    AmigoSolver,
    # DirectCudaSolver,
    # LNKSInexactSolver,
    MumpsSolver,
    # PardisoSolver,
    # DirectPetscSolver,
    # DirectScipySolver,
)
