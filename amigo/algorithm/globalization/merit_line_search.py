"""Augmented barrier-Lagrangian merit line search.

Backtracks on the barrier objective plus the constraint term of the
Lagrangian at the base multipliers and a fixed squared penalty on the
constraint violation.
"""

from .filter_line_search import LineSearch, LineSearchInfo, finite_vector


class MeritLineSearch(LineSearch):
    """Backtracking until the augmented barrier-Lagrangian decreases."""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        self.trial = self.optimizer.create_opt_vector()
        self.trial_grad = self.problem.create_vector()

        self.backtrack = options["backtracking_factor"]
        self.max_iters = options["max_line_search_iterations"]
        self.penalty = options["merit_penalty"]

        self.temp_con = self.problem.create_constraint_vector()
        self.base_lam = self.problem.create_constraint_vector()
        self.current_info = None

    def _merit(self, fobj, grad, barrier):
        """Merit f + barrier + lambda_k^T c + (penalty / 2) ||c||^2.

        The multipliers lambda_k stay at the base point and c holds the
        constraint rows of the given gradient.
        """
        con_indices = self.problem.get_constraint_indices()
        grad.get_values_at(con_indices, self.temp_con)
        lam_c = self.problem.dot(self.base_lam, self.temp_con)
        c_sq = self.problem.dot(self.temp_con, self.temp_con)
        return fobj + barrier + lam_c + 0.5 * self.penalty * c_sq

    def line_search(self, solver, evaluator, state):
        # Reference merit at the current point, freeze the base multipliers
        evaluator.evaluate_objective_and_infeasibility(state)
        evaluator.evaluate_gradient(state)
        con_indices = self.problem.get_constraint_indices()
        state.current.get_solution().get_values_at(con_indices, self.base_lam)
        merit_0 = self._merit(
            state.objective_value, state.gradient, state.log_barrier_value
        )

        alpha_primal = state.max_alpha_primal
        alpha_dual = state.max_alpha_dual

        info = LineSearchInfo()
        info.success = False
        info.num_search_iters = self.max_iters

        for line_iter in range(self.max_iters):
            self.optimizer.apply_step_update(
                alpha_primal, alpha_dual, state.current, state.step, self.trial
            )
            evaluator.evaluate_gradient_from_point(
                state.obj_scale, self.trial, self.trial_grad
            )
            fobj, barrier, infeas = (
                evaluator.evaluate_objective_and_infeasibility_from_point(
                    state.mu, state.obj_scale, self.trial, self.trial_grad
                )
            )
            merit_trial = self._merit(fobj, self.trial_grad, barrier)

            # A NaN merit fails the comparison, while a NaN derivative needs its own test
            if merit_trial < merit_0 and finite_vector(self.problem, self.trial_grad):
                state.invalidate()
                state.objective_value = fobj
                state.log_barrier_value = barrier
                state.con_infeasibility = infeas
                state.objective_current = True
                state.current.copy(self.trial)
                state.gradient.copy(self.trial_grad)
                state.gradient_current = True

                info.success = True
                info.num_search_iters = line_iter + 1
                info.alpha_primal = alpha_primal
                info.alpha_dual = alpha_dual
                self.current_info = info
                return info

            alpha_primal *= self.backtrack
            alpha_dual *= self.backtrack

        self.current_info = info
        return info

    def add_log_info(self, info):
        if self.current_info is not None:
            info["line_iters"] = self.current_info.num_search_iters
            info["alpha_x"] = self.current_info.alpha_primal
            info["alpha_z"] = self.current_info.alpha_dual
