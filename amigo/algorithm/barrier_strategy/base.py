"""Base class for barrier-parameter strategies.

A BarrierStrategy owns the per-iteration barrier update:
  - decide the new mu (heuristic rule or QF oracle)
  - factorize the KKT system at the new mu
  - compute the Newton direction

Every concrete strategy reads and writes self.opt.barrier_param and uses
self.opt.{vars, grad, res, px, update, temp, optimizer, solver} plus
helpers on self.opt (_factorize_kkt, _find_direction, etc.).
"""

from abc import ABC, abstractmethod


class BarrierInfo:
    new_barrier: bool = False
    mu_new: float = 0.0
    mu_old: float = 0.0


class BarrierStrategy(ABC):
    def __init__(self, options={}):
        self.options = options

    def initialize(self, evaluator, state):
        """Initialize the barrier strategy from the initial point"""
        pass

    @abstractmethod
    def update_barrier(self, evaluator, state) -> BarrierInfo:
        """Update the barrier parameter prior to factoring the KKT matrix"""
        pass

    def add_step_correction(self, solver, evalutor, state):
        """Add the correction to the step - relevant for Mehrotra P/C steps"""
        pass

    def update_after_line_search(self, info, evaluator, state):
        """
        Update any internal state required after the results of a line search

        Default behavior is to adjust the barrier parameter if small steps are taken
        """
        initial_barrier = self.options["initial_barrier_param"]
        if state.iter > 0 and info.alpha_primal < 1e-10:
            state.mu = min(10.0 * state.mu, initial_barrier)

            # Invalidate the residual and step
            state.residual_current = False
            state.step_current = False


class MonotoneBarrierStrategy(BarrierStrategy):
    def __init__(self, options, problem, optimizer):
        super().__init__(options)
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

    def update_barrier(self, evaluator, state):
        info = BarrierInfo()

        opt_tol = self.options["convergence_tolerance"]
        relative_tol = self.options["barrier_progress_tol"]
        frac = self.options["monotone_barrier_fraction"]

        if state.residual_norm < relative_tol * state.mu:
            mu_new = max(state.mu * frac, frac * opt_tol)

            info.new_barrier = True
            info.mu_old = state.mu
            info.mu_new = mu_new

            # Update the barrier parameter. Invalidate the residuals and the step
            # (if any) because the barrier has changed
            state.mu = mu_new
            state.residual_current = False
            state.step_current = False

        return info
