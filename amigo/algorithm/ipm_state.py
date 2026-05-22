"""Per-iteration state carried through the optimization loop.

IpmState holds the scalar counters, step-size history, and
filter/rejection counts that change every iteration.  StepContext
is a lightweight bag of per-iteration scratch data handed to the
barrier strategy each step.
"""

import amigo as am
from dataclasses import dataclass
from typing import Any, Optional
from .solvers import LinearSolver


@dataclass
class IpmData:
    options: dict
    problem: am.OptimizationProblem
    optimizer: am.InteriorPointOptimizer
    solver: LinearSolver  # Linear solver class instance
    vars: am.OptVector  # Variables
    grad: am.Vector  # Gradient of the problem at the current design point
    diag: am.Vector  # Diagonal contributions
    hess: am.CSRMat  # CSR matrix that stores the Hessian
    obj_scale: float = 1.0

    def zero_multipliers(self, x):
        con_indices = self.problem.get_constraint_indices()
        x.fill_at(con_indices, 0.0)

    def compute_gradient(self):
        x = self.vars.get_solution()
        alpha = self.obj_scale
        self.problem.update(x)
        self.problem.gradient(alpha, x, self.grad)
        self.optimizer.apply_gradient_scaling(self.grad)

    def compute_hessian(self):
        x = self.vars.get_solution()
        alpha = self.obj_scale
        self.problem.hessian(alpha, x, self.hess)


@dataclass
class IpmState:
    """Mutable per-iteration state of the interior-point loop."""

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


@dataclass
class StepContext:
    """Per-iteration inputs passed to BarrierStrategy.step()."""

    i: int = 0
    comm_rank: int = 0
    res_norm: float = 0.0
    tol: float = 0.0
    compl_inf_tol: float = 0.0

    # Problem structure
    x: Any = None
    diag_base: Any = None

    # Inertia correction + zero-Hessian handling
    inertia_corrector: Any = None
    zero_hessian_indices: Any = None
    zero_hessian_eps: float = 0.0

    # Classical-barrier filter monotone fallback (strategy may update
    # filter_monotone_mu in place; driver reads it back)
    filter_monotone_mode: bool = False
    filter_monotone_mu: Optional[float] = None
