"""
Interior point optimizer
------------------------

  default_options.py             Every option with its default
  homotopy.py                    Path following over solves in one parameter
  simple_solver.py               Newton solver for square systems

  interior_point/                The interior point loop
    driver.py                    Optimizer class and main loop
    state.py                     Per iteration state
    evaluator.py                 Gradient, residual, and Hessian evaluation cache
    newton_direction.py          KKT solve and step assembly
    convergence_check.py         Convergence, divergence, and stall tests
    step_contract.py             The dual step contract observer

  monitoring/                    What a solve reports
    iteration_logger.py          Iteration data and the progress table
    regime.py                    Easy, hard, jam, or stuck label per iteration

  initialization/                Starting point construction
    iterate_initialization.py    Bounds push, slacks, bound duals
    multiplier_initialization.py Least squares multipliers and their refresh
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

from .interior_point.driver import Optimizer, WarmStart
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
