"""Constraint multiplier initialization and refresh by least squares."""

# Growth of the largest multiplier in one step that marks the multipliers as diverging
MULT_JUMP = 10.0

# Factor by which the dual infeasibility must jump or dominate the other errors to call for a refresh
DUAL_JUMP = 10.0

# A refresh that keeps more than this fraction of the dual infeasibility is futile
FUTILE_FRACTION = 0.9


class MultiplierInitializer:
    def __init__(self, options, model, problem, optimizer):
        self.options = options
        self.model = model
        self.problem = problem
        self.optimizer = optimizer
        self.temp_con = problem.create_constraint_vector()
        self.saved_point = problem.create_vector()
        # Own workspace so the logged Newton step is left intact
        self.estimate = problem.create_vector()

        # Largest multiplier after the last step or estimate
        self.mult_size = None

    def initialize_multipliers(self, evaluator, solver, state):
        """Initialize the constraint multipliers in state.current."""
        # Warm starts keep the supplied multipliers
        if self.options["warm_start"]:
            return
        if self.options["init_least_squares_multipliers"]:
            self.compute_least_squares_multipliers(evaluator, solver, state)
        return

    def recompute_multipliers(self, evaluator, solver, state, dual_prev=None, force=False):
        """Refresh the multipliers by least squares after a step that left the dual infeasibility high.

        Returns True when the refresh cut the dual infeasibility by a tenth,
        False when it was futile, and None when it was not attempted. A
        forced call skips the triggers and keeps the acceptance test.
        """
        if not self.options["recompute_multipliers"]:
            return None

        # The multiplier size before the step tells a diverging update apart
        mult_prev = self.mult_size
        self.mult_size = self._mult_size(state)

        # A step that lowered the dual infeasibility keeps its multipliers
        if dual_prev is None:
            dual_prev = state.dual_infeas
        evaluator.evaluate_residual(state)
        if not force:
            if state.dual_infeas <= dual_prev:
                return None

            # Least squares multipliers only mean something near feasibility or against diverging multipliers
            diverging = mult_prev is not None and self.mult_size > MULT_JUMP * max(mult_prev, 1.0)
            if not diverging:
                if state.primal_infeas >= self.options["recompute_multiplier_tol"]:
                    return None

                # Near feasibility only a jump or a dominant dual error pays for a refresh
                jumped = state.dual_infeas > DUAL_JUMP * dual_prev
                dominant = state.dual_infeas > DUAL_JUMP * max(state.primal_infeas, state.mu)
                if not (jumped or dominant):
                    return None

        dual_before = state.dual_infeas
        if self.compute_least_squares_multipliers(evaluator, solver, state):
            if state.comm_rank == 0:
                print(
                    f"  Multiplier refresh: dual_inf {dual_before:.2e} -> "
                    f"{state.dual_infeas:.2e} at primal_infeas={state.primal_infeas:.2e}"
                )

        # A refresh that keeps more than this fraction of the dual infeasibility is futile
        return state.dual_infeas <= FUTILE_FRACTION * dual_before

    def compute_least_squares_multipliers(self, evaluator, solver, state):
        """Replace the multipliers by their least-squares estimate when that lowers the dual infeasibility.

        Solves the normal-equation system

            [ I  A^T ] [ w      ]   [ -(grad_f - zl + zu) ]
            [ A   0  ] [ lambda ] = [         0           ]

        whose (1,1) block is the identity, so lambda minimizes the dual
        infeasibility norm and w is discarded. An estimate above
        constr_mult_init_max is discarded. Returns True when taken.
        """
        x = state.get_current_point()
        primal_indices = self.problem.get_primal_indices()
        con_indices = self.problem.get_constraint_indices()

        evaluator.evaluate_residual(state)
        dual_before = state.dual_infeas
        self.saved_point.copy(x)

        # Objective gradient at the current obj_scale, constraint Hessian at obj_scale=0
        x.fill_at(con_indices, 0.0)
        state.invalidate()
        obj_scale_store = state.obj_scale
        evaluator.evaluate_gradient(state)
        state.obj_scale = 0.0
        evaluator.evaluate_hessian(state)
        state.obj_scale = obj_scale_store

        state.diagonal.zero()
        state.diagonal.fill_at(primal_indices, 1.0)
        if solver.factor(state.hessian, state.diagonal):
            x.copy_at(con_indices, self.saved_point)
            state.invalidate()
            return False

        self.optimizer.compute_dual_residual(
            state.current, state.gradient, state.residual
        )
        state.residual.scale(-1.0)
        state.residual.fill_at(con_indices, 0.0)
        solver.solve(state.residual, self.estimate)

        self.estimate.get_values_at(con_indices, self.temp_con)
        accepted = (
            self.problem.maxabs(self.temp_con) <= self.options["constr_mult_init_max"]
        )
        if accepted:
            x.copy_at(con_indices, self.estimate)
            state.invalidate()
            evaluator.evaluate_residual(state)
            accepted = state.dual_infeas <= dual_before
        if not accepted:
            x.copy_at(con_indices, self.saved_point)
            state.invalidate()
        else:
            self.mult_size = self._mult_size(state)
        return accepted

    def _mult_size(self, state):
        """Largest constraint multiplier of the current point."""
        con_indices = self.problem.get_constraint_indices()
        state.current.get_solution().get_values_at(con_indices, self.temp_con)
        return self.problem.maxabs(self.temp_con)
