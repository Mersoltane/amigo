"""Starting point strategy.

Restores feasibility from the initial point and then centers the start
at the requested initial mu with the shared centering rule.
"""


class FeasibilityPresolve:
    def __init__(self, options):
        self.options = options
        self.enabled = options["feasibility_presolve"] and not options["warm_start"]

    def run(
        self,
        feasible_resto,
        multiplier_init,
        centerer,
        solver,
        evaluator,
        state,
        line_search,
    ):
        """Restore feasibility and center the start, returning True on success."""
        if not self.enabled:
            return False

        # A successful restoration resumes at the initial mu of the schedule
        resto_info = feasible_resto.restore(solver, evaluator, state, line_search)
        if not resto_info.success:
            return False

        # Center the restored point at the schedule mu
        centerer.center(evaluator, state)

        # The restored point is near feasible, where least squares multipliers are reliable
        multiplier_init.compute_least_squares_multipliers(evaluator, solver, state)
        return True
