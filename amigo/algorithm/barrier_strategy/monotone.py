"""Monotone barrier strategy.

mu is reduced superlinearly once the barrier subproblem is solved to the
required accuracy. With mu_init_strategy "complementarity" the initial mu
starts from the average complementarity at the initial point.
"""

from .base import (
    BarrierStrategy,
    BarrierInfo,
    barrier_floor,
    barrier_minimum,
    monotone_mu,
)


class MonotoneBarrierStrategy(BarrierStrategy):
    def __init__(self, options, problem, optimizer):
        super().__init__(options)
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

    def initialize(self, evaluator, state):
        if self.options["mu_init_strategy"] == "complementarity":
            avg_comp, _ = evaluator.evaluate_complementarity(state)
            mu0 = min(state.mu, max(avg_comp, barrier_floor(self.options, state)))
            if mu0 != state.mu:
                self.set_mu(state, mu0)
                state.invalidate(grad=False, hess=False)

    def update_barrier(self, evaluator, state):
        info = BarrierInfo()
        mu = monotone_mu(self.options, evaluator, state)

        # A tiny step lowers mu once even when the subproblem is not solved yet
        if self.force_decrease:
            self.force_decrease = False
            kappa_mu = self.options["mu_linear_decrease_factor"]
            theta_mu = self.options["mu_superlinear_decrease_power"]
            forced = min(kappa_mu * state.mu, state.mu**theta_mu)
            mu = min(mu, max(forced, barrier_minimum(self.options, state)))

        if mu < state.mu:
            info.new_barrier = True
            info.mu_old = state.mu
            info.mu_new = mu
            self.set_mu(state, mu)
            state.invalidate(grad=False, hess=False)
        return info
