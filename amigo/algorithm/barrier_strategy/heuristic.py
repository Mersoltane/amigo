"""Classical barrier strategy: LOQO-style heuristic."""

from .base import BarrierStrategy, BarrierInfo


def loqo_heuristic(xi, complementarity, gamma, r, mu_floor=1e-12):
    """LOQO-style barrier parameter: mu = gamma * heuristic_factor * comp."""
    if xi > 1e-10:
        term = (1 - r) * (1 - xi) / xi
        heuristic_factor = min(term, 2.0) ** 3
    else:
        heuristic_factor = 2.0**3
    mu_new = gamma * heuristic_factor * complementarity
    return max(mu_new, mu_floor), heuristic_factor


class HeuristicBarrierStrategy(BarrierStrategy):
    def __init__(self, options, problem, optimizer):
        super().__init__(options)
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

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
