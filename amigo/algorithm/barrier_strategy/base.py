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


class BarrierStrategyNew(ABC):
    @abstractmethod
    def initialize(self, evaluator, state):
        """Initialize the barrier strategy from the initial point"""
        pass

    @abstractmethod
    def update_barrier(self, evaluator, state) -> BarrierInfo:
        """Update the barrier parameter prior to factoring the KKT matrix"""
        pass

    @abstractmethod
    def add_step_correction(self, solver, evalutor, state):
        """Add the correction to the step - relevant for Mehrotra P/C steps"""
        pass

    @abstractmethod
    def update_line_search_info(self, info):
        """Update any internal state required after the results of a line search"""
        pass


class MonotoneBarrierStrategy(BarrierStrategyNew):
    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

    def initialize(self, evaluator, state):
        pass

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

    def add_step_correction(self, solver, evalutor, state):
        pass

    def update_line_search_info(self, info):
        pass


def loqo_heuristic(xi, complementarity, gamma, r, mu_floor=1e-12):
    """LOQO-style barrier parameter: mu = gamma * heuristic_factor * comp."""
    if xi > 1e-10:
        term = (1 - r) * (1 - xi) / xi
        heuristic_factor = min(term, 2.0) ** 3
    else:
        heuristic_factor = 2.0**3
    mu_new = gamma * heuristic_factor * complementarity
    return max(mu_new, mu_floor), heuristic_factor


class HeuristicBarrierStrategy(BarrierStrategyNew):
    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

    def initialize(self, evaluator, state):
        pass

    def update_barrier(self, evaluator, state):
        kappa_eps = self.options["barrier_tol_factor"]
        gamma = self.options["heuristic_barrier_gamma"]
        r = self.options["heuristic_barrier_r"]
        tol = self.options["convergence_tolerance"]
        compl_inf_tol = self.options["compl_inf_tol"]

        comp, xi = evaluator.evaluate_complementarity(state)

        mu_floor = min(tol, compl_inf_tol) / (kappa_eps + 1.0)
        mu_new, _ = loqo_heuristic(xi, comp, gamma, r, mu_floor)

        info = BarrierInfo()
        info.new_barrier = True
        info.mu_new = mu_new
        info.mu_old = state.mu

        # Update the barrier parameter
        state.mu = mu_new
        state.residual_current = False
        state.step_current = False

        return info

    def add_step_correction(self, solver, evalutor, state):
        pass

    def update_line_search_info(self, info):
        pass


class BarrierStrategy(ABC):
    """Abstract base for barrier-parameter update strategies."""

    def __init__(self, opt, options):
        self.opt = opt
        self.options = options

    @abstractmethod
    def step(self, ctx):
        """Update barrier, factorize KKT, compute Newton direction.

        Parameters
        ----------
        ctx : StepContext
            Per-iteration context (see ipm_driver).

        Returns
        -------
        factorize_ok : bool
            False if inertia correction failed.
        """

    def initialize(self, ctx):
        """One-shot init at the start of optimize().  Override if needed."""

    def on_step_rejected(self, ctx):
        """Run after the line search rejects the step."""

    def on_barrier_increased(self):
        """Run after increase_barrier_on_rejections() raised mu."""

    def handle_zero_step_recovery(
        self, i, alpha_x_prev, alpha_z_prev, zero_step_count, comm_rank
    ):
        """Escape stuck iterates by bumping mu when tiny steps repeat.

        Only active on the non-inertia-corrector path.  After three
        consecutive iterations with max(alpha_x, alpha_z) < 1e-10,
        scale mu up by 10x (capped at 1.0).
        """
        if i > 0 and max(alpha_x_prev, alpha_z_prev) < 1e-10:
            zero_step_count += 1
            if zero_step_count >= 3:
                old = self.opt.barrier_param
                self.opt.barrier_param = min(old * 10.0, 1.0)
                if comm_rank == 0 and self.opt.barrier_param != old:
                    print(
                        f"  Zero step recovery: barrier "
                        f"{old:.2e} -> {self.opt.barrier_param:.2e}"
                    )
                zero_step_count = 0
        else:
            zero_step_count = 0
        return zero_step_count

    def increase_barrier_on_rejections(
        self,
        consecutive_rejections,
        max_rejections,
        barrier_inc,
        initial_barrier,
        comm_rank,
    ):
        """Increase mu after max_rejections consecutive rejections."""
        if consecutive_rejections >= max_rejections:
            new_barrier = min(self.opt.barrier_param * barrier_inc, initial_barrier)
            if new_barrier > self.opt.barrier_param:
                if comm_rank == 0:
                    print(
                        f"  Barrier increased: {self.opt.barrier_param:.2e} -> "
                        f"{new_barrier:.2e}"
                    )
                self.opt.barrier_param = new_barrier
            elif comm_rank == 0:
                print(
                    f"  Barrier at max ({self.opt.barrier_param:.2e}), "
                    f"cannot increase further"
                )
            return 0
        return consecutive_rejections
