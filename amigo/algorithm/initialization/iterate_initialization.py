"""Center the primal-dual iterate for the current barrier parameter.

Cold starts and the presolve project the primals with the standard push and set z = mu.
Warm starts keep the supplied point where the barrier admits it, every gap is held at
least mu over the dual cap times the scale of its bound, so the point moves only where
its centered dual would pass the cap in the variable's own units. Warm starts also keep
the bound duals of the previous solve when they are available, clipped into the sigma
band around mu over the gap that every iterate obeys.
"""


class IterateCenterer:
    def __init__(self, options, model, problem, optimizer):
        self.model = model
        self.options = options
        self.problem = problem
        self.optimizer = optimizer
        self.temp_con = problem.create_constraint_vector()

    def center(self, evaluator, state, warm=False, keep_bound_duals=False):
        """Center the iterate at the current mu, keeping the multipliers on a warm start."""
        x = state.get_current_point()
        con_indices = self.problem.get_constraint_indices()

        # Save the multipliers when a warm start supplies them
        if warm:
            x.get_values_at(con_indices, self.temp_con)

        # Zero the multipliers so the gradient holds the residuals below
        x.fill_at(con_indices, 0.0)
        state.invalidate()

        self._project(state, warm, keep_bound_duals)

        if self.model is not None:
            # The gradient constraint rows hold the residuals at zero multipliers
            evaluator.evaluate_gradient(state)
            slack_indices = self.model.slack_indices
            ineq_indices = self.model.ineq_constraint_indices

            x.copy_device_to_host()
            state.gradient.copy_device_to_host()
            x_array = x.get_array()
            grad_array = state.gradient.get_array()

            # Absorb each inequality residual into its slack for feasibility
            x_array[slack_indices] += grad_array[ineq_indices]
            x.copy_host_to_device()

            # Project the moved slacks back interior with the same rule
            self._project(state, warm, keep_bound_duals)

        # Restore the saved multipliers
        if warm:
            x.set_values_at(con_indices, self.temp_con)
            state.invalidate()

        return

    def _project(self, state, warm, keep_bound_duals):
        """Push the primals interior and set the bound duals for a cold or a warm start."""
        options = self.options
        if not warm:
            self.optimizer.initialize_duals(state.mu, state.current)
            state.invalidate()
            return

        # A gap below mu over the dual cap, in the scale of its bound, would need a dual above the cap to be centered
        zmax = options["warm_start_mult_init_max"]
        gap = state.mu / zmax
        if keep_bound_duals:
            # The band every iterate obeys, open when the safeguard is off
            kappa = (
                options["kappa_sigma"] if options["kappa_sigma"] > 0.0 else float("inf")
            )
            self.optimizer.initialize_duals_keep(state.mu, gap, kappa, state.current)
        else:
            self.optimizer.initialize_duals_warm(state.mu, gap, zmax, state.current)
        state.invalidate()
