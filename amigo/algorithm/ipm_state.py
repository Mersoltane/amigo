"""Per-iteration state carried through the optimization loop.

InteriorPointState holds the current point, its derivatives, the
barrier parameter, step lengths, and the error measures, along with
the currency flags that mark which of them are up to date.
"""

import amigo as am

from .barrier_strategy import fraction_to_boundary


class InteriorPointState:
    # Communicator rank
    comm_rank: int

    # Iteration counter
    iter: int

    # Barrier parameter
    mu: float

    # Fraction to the boundary parameter
    tau: float

    # Objective scaling factor and whether any constraint row is scaled
    obj_scale: float
    scaled_constraints: bool

    # Objective function value (scaled) and log-barrier term
    objective_value: float
    log_barrier_value: float
    con_infeasibility: float
    objective_current: bool

    # Current primal-dual vector
    current: am.OptVector

    # Gradient information
    gradient: am.Vector
    gradient_current: bool

    # Second-order information with a counter of Hessian evaluations
    diagonal: am.Vector
    hessian: am.CSRMat
    hessian_current: bool
    hessian_version: int

    # Max primal and max dual step lengths
    max_alpha_primal: float
    max_alpha_dual: float
    step: am.OptVector
    step_current: bool

    # Residual and step information
    residual_norm: float
    residual: am.Vector

    # Current error measures in scaled space and the unscaled constraint violation
    primal_infeas: float
    dual_infeas: float
    complementarity: float
    kkt_error: float
    constr_viol: float
    residual_current: bool

    def __init__(self, x, options, problem, optimizer):
        self.comm_rank = 0
        self.iter = 0
        # A warm start begins at a barrier consistent with its near-optimal point
        if options["warm_start"]:
            self.mu = options["warm_start_mu_init"]
        else:
            self.mu = options["initial_barrier_param"]
        self.tau = fraction_to_boundary(options, self.mu)
        self.obj_scale = 1.0
        self.scaled_constraints = False
        self.objective_value = 0.0
        self.log_barrier_value = 0.0
        self.con_infeasibility = 0.0
        self.objective_current = False

        self.current = optimizer.create_opt_vector(x)
        self.gradient = problem.create_vector()
        self.gradient_current = False

        self.diagonal = problem.create_vector()
        self.diagonal_current = False
        self.hessian = problem.create_matrix()
        self.hessian_current = False
        self.hessian_version = 0

        self.max_alpha_primal = 1.0
        self.max_alpha_dual = 1.0
        self.step = optimizer.create_opt_vector()
        self.step_current = False

        self.residual_norm = 0.0
        self.residual = problem.create_vector()
        self.primal_infeas = 0.0
        self.dual_infeas = 0.0
        self.complementarity = 0.0
        self.kkt_error = 0.0
        self.constr_viol = 0.0
        self.residual_current = False

    @property
    def barrier_objective(self):
        """Scaled objective plus the log-barrier term"""
        return self.objective_value + self.log_barrier_value

    def get_current_point(self):
        """Get the current primal-dual vector"""
        return self.current.get_solution()

    def invalidate(
        self, obj=True, grad=True, hess=True, res=True, step=True, diag=None
    ):
        # Diagonal depends on the point like the Hessian (not on mu)
        if diag is None:
            diag = hess
        if obj:
            self.objective_current = False
        if grad:
            self.gradient_current = False
        if hess:
            self.hessian_current = False
        if diag:
            self.diagonal_current = False
        if res:
            self.residual_current = False
        if step:
            self.step_current = False
