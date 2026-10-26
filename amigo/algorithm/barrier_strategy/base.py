"""Base class for barrier-parameter strategies.

A BarrierStrategy owns the per-iteration barrier update: pick the next mu
and apply it with set_mu, which also refreshes the coupled
fraction-to-boundary tau.
"""

from abc import ABC, abstractmethod


def fraction_to_boundary(options, mu):
    """Fraction to the boundary at barrier mu, raised toward one as mu shrinks when adaptive."""
    tau = options["fraction_to_boundary"]
    if options["adaptive_tau"]:
        tau = max(tau, 1.0 - mu)
    return tau


def barrier_floor(options, state):
    """Smallest barrier parameter, low enough to meet the tolerance and the unscaled compl_inf_tol.

    mu lives in scaled units, where the unscaled complementarity is the
    scaled one divided by obj_scale, so compl_inf_tol enters as
    obj_scale * compl_inf_tol.
    """
    tol = min(
        options["convergence_tolerance"],
        state.obj_scale * options["compl_inf_tol"],
    )
    return tol / (options["barrier_tol_factor"] + 1.0)


def monotone_mu(options, evaluator, state):
    """Barrier parameter after superlinear decreases while the barrier subproblem at the new mu is solved."""
    kappa_eps = options["barrier_tol_factor"]
    kappa_mu = options["mu_linear_decrease_factor"]
    theta_mu = options["mu_superlinear_decrease_power"]
    floor = barrier_minimum(options, state)

    evaluator.evaluate_residual(state)
    mu = state.mu
    while mu > floor and evaluator.barrier_error(state, mu) <= kappa_eps * mu:
        mu = max(min(kappa_mu * mu, mu**theta_mu), floor)
    return mu


class BarrierInfo:
    new_barrier: bool = False
    # The filter resets on new subproblems only, not adaptive retargets
    new_subproblem: bool = False
    mu_new: float = 0.0
    mu_old: float = 0.0


def barrier_minimum(options, state):
    """Smallest mu any strategy may reach."""
    return max(barrier_floor(options, state), options["mu_min"])


class BarrierStrategy(ABC):
    def __init__(self, options={}):
        self.options = options

        # Set after a tiny step so the next update lowers mu once
        self.force_decrease = False

    def set_mu(self, state, mu):
        """Set the barrier parameter and the coupled fraction-to-boundary."""
        state.mu = mu
        state.tau = fraction_to_boundary(self.options, mu)

    def initialize(self, evaluator, state):
        """Initialize the barrier strategy from the initial point"""
        pass

    @abstractmethod
    def update_barrier(self, evaluator, state) -> BarrierInfo:
        """Update the barrier parameter prior to factoring the KKT matrix"""
        pass

    def computes_step(self):
        """True when the strategy chooses mu and the step together for this iteration."""
        return False

    def compute_step(self, solver, evaluator, state):
        """Compute mu and the step in place of the Newton step when computes_step is True."""
        pass

    def update_after_line_search(self, info, evaluator, state):
        """
        Update any internal state required after the results of a line search

        Default behavior is to adjust the barrier parameter if small steps are taken
        """
        pass
