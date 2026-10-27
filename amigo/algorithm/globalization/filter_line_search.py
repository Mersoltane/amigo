"""Filter line search with second-order correction.

Trial points are accepted against a two-dimensional filter over the barrier
objective and the 1-norm constraint violation. A second-order correction is
applied on the first rejection.
"""

from abc import ABC, abstractmethod

import numpy as np
from .filter_acceptance import FilterAcceptance, FilterBasePoint

# A correction above this multiple of the step it corrects is not second order
SOC_MAX_RATIO = 1e3


def finite_vector(problem, vec):
    """True when every entry of the vector is finite, since a NaN or infinity propagates into its squared norm."""
    return bool(np.isfinite(problem.dot(vec, vec)))


class LineSearch(ABC):
    def reset_on_new_barrier(self, state, barrier_info):
        pass

    def reset_after_restoration(self):
        pass

    def augment_filter(self, phi, theta):
        pass

    def restoration_acceptable(self, phi, theta):
        return True

    @abstractmethod
    def line_search(self, solver, evaluator, state):
        pass


class LineSearchInfo:
    success: bool = False
    num_search_iters: int = 0
    alpha_primal: float = 1.0
    alpha_dual: float = 1.0
    tiny_step: bool = False


class FilterLineSearch(FilterAcceptance, LineSearch):
    """Filter line search with second-order correction."""

    def __init__(self, options, problem, optimizer):
        super().__init__(options)
        self.problem = problem
        self.optimizer = optimizer

        self.trial = self.optimizer.create_opt_vector()
        self.trial_grad = self.problem.create_vector()
        self.trial_res = self.problem.create_vector()

        # Second-order correction workspace
        self.soc_rhs = self.problem.create_vector()
        self.soc_sol = self.problem.create_vector()
        self.soc_step = self.optimizer.create_opt_vector()
        self.soc_trial = self.optimizer.create_opt_vector()
        self.soc_trial_grad = self.problem.create_vector()
        self.soc_con_a = self.problem.create_constraint_vector()
        self.soc_con_b = self.problem.create_constraint_vector()

        # Step size workspace
        self.step_ratio = self.problem.create_vector()
        self.step_con = self.problem.create_constraint_vector()

        # The theta limits need the first iterate, set them lazily
        self.theta_limits_initialized = False

        self.current_info = None

        # Successive filter rejections drive the filter reset trigger
        self.successive_filter_rejections = 0
        self.filter_reset_count = 0
        self._rejections_at_start = (0, 0)

        return

    def reset_on_new_barrier(self, state, barrier_info):
        """Drop the filter when the barrier parameter changes."""

        # Entries store phi_mu, a new mu makes them incomparable
        reset = bool(barrier_info.new_barrier)

        if reset:
            self.filter.clear()
            self.successive_filter_rejections = 0
            self.filter_reset_count = 0

        return

    def reset_after_restoration(self):
        # Keep the filter and the theta limits, the filter holds the failure point
        self.successive_filter_rejections = 0

    def augment_filter(self, phi, theta):
        # Record the failure point so the search cannot return there
        if np.isfinite(phi) and np.isfinite(theta):
            self.filter.add(phi, theta)

    def restoration_acceptable(self, phi, theta):
        if self.theta_limits_initialized and theta > self.theta_max:
            return False
        return self.filter.is_acceptable(phi, theta)

    def build_base_point(self, evaluator, state):
        base = FilterBasePoint()

        # Evaluate the objective and infeasibility at this point
        evaluator.evaluate_objective_and_infeasibility(state)
        base.ref_barr = state.barrier_objective
        base.ref_theta = state.con_infeasibility

        # Evaluate the directional derivative
        base.ref_dphi = evaluator.evaluate_directional_derivative(state)

        return base

    def _second_order_correction(
        self, solver, evaluator, state, base, alpha_first, theta_first
    ):
        """Second-order correction on the first rejection.

        Each round reuses the factorization with constraint rows c_soc = alpha * c_k + c(trial).
        A round that fails to shrink the violation by kappa_soc ends the correction.
        Returns the accepted trial tuple or None with the state untouched.
        """
        max_soc = self.options["filter_max_soc"]
        kappa_soc = self.options["filter_kappa_soc"]
        if max_soc <= 0:
            return None
        self.soc_outcome = "ran"
        con_indices = self.problem.get_constraint_indices()
        step_size = self.problem.maxabs(state.step.get_solution())
        max_size = SOC_MAX_RATIO * max(1.0, step_size)

        # Residual at the current point (rhs convention of the Newton solve)
        evaluator.evaluate_residual(state)
        self.soc_rhs.copy(state.residual)
        state.residual.get_values_at(con_indices, self.soc_con_a)

        # Constraint rows of the residual at the rejected trial point
        evaluator.evaluate_residual_from_point(
            state.mu, self.trial, self.trial_grad, self.trial_res
        )
        self.trial_res.get_values_at(con_indices, self.soc_con_b)

        # c_soc = alpha * c_k + c(trial)
        self.soc_con_a.scale(alpha_first)
        self.soc_con_a.axpy(1.0, self.soc_con_b)
        self.soc_rhs.set_values_at(con_indices, self.soc_con_a)

        theta_old = theta_first
        for _ in range(max_soc):
            solver.solve(self.soc_rhs, self.soc_sol)
            self.optimizer.compute_update(
                state.mu, state.current, self.soc_sol, self.soc_step
            )
            if self.problem.maxabs(self.soc_step.get_solution()) > max_size:
                self.soc_outcome = "big"
                return None
            ax, _, az, _ = self.optimizer.compute_max_step(
                state.tau, state.current, self.soc_step
            )
            self.optimizer.apply_step_update(
                ax, az, state.current, self.soc_step, self.soc_trial
            )
            evaluator.evaluate_gradient_from_point(
                state.obj_scale, self.soc_trial, self.soc_trial_grad
            )
            fobj, barrier, infeas = (
                evaluator.evaluate_objective_and_infeasibility_from_point(
                    state.mu, state.obj_scale, self.soc_trial, self.soc_trial_grad
                )
            )
            tb, tt = fobj + barrier, infeas
            finite = (
                np.isfinite(tb)
                and np.isfinite(tt)
                and finite_vector(self.problem, self.soc_trial_grad)
            )
            # Acceptance is tested with the original trial alpha
            if finite and self.check_acceptance(base, tb, tt, alpha_first):
                self.soc_outcome = "ok"
                return (fobj, barrier, infeas, tb, tt, ax, az)
            if not finite or tt > kappa_soc * theta_old:
                self.soc_outcome = "grew"
                return None
            theta_old = tt

            # Next round: c_soc = alpha_soc * c_soc + c(trial_soc)
            evaluator.evaluate_residual_from_point(
                state.mu, self.soc_trial, self.soc_trial_grad, self.trial_res
            )
            self.trial_res.get_values_at(con_indices, self.soc_con_b)
            self.soc_con_a.scale(ax)
            self.soc_con_a.axpy(1.0, self.soc_con_b)
            self.soc_rhs.set_values_at(con_indices, self.soc_con_a)

        return None

    def line_search(self, solver, evaluator, state):
        self._rejections_at_start = (
            self.num_trial_rejections,
            self.num_filter_rejections,
        )
        self.reset_reasons()
        self.soc_outcome = ""

        # Base point for the acceptance tests
        base = self.build_base_point(evaluator, state)

        if not self.theta_limits_initialized:
            self.set_reference_values(base.ref_theta)
            self.theta_limits_initialized = True

        # Reset the filter after too many successive filter rejections
        if (
            self.successive_filter_rejections >= self.options["filter_reset_trigger"]
            and self.filter_reset_count < self.options["max_filter_resets"]
        ):
            self.filter.clear()
            self.filter_reset_count += 1
            self.successive_filter_rejections = 0

        alpha_min = self.compute_alpha_min(base)

        # The bound duals keep their own fraction to the boundary step
        alpha_primal = state.max_alpha_primal
        alpha_dual = state.max_alpha_dual

        max_line_iters = self.options["max_line_search_iterations"]

        # Failure info by default, overwritten on acceptance
        info = LineSearchInfo()
        info.success = False
        info.num_search_iters = max_line_iters
        info.alpha_primal = 0.0
        info.alpha_dual = 0.0
        self.current_info = info

        # A non-finite step has no acceptable point along it
        if not finite_vector(self.problem, state.step.get_solution()):
            info.num_search_iters = 0
            return info

        for line_iter in range(max_line_iters):
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
            trial_barr = fobj + barrier
            trial_theta = infeas

            # Non-finite trial values or derivatives can never be accepted
            trial_finite = (
                np.isfinite(trial_barr)
                and np.isfinite(trial_theta)
                and finite_vector(self.problem, self.trial_grad)
            )
            # The f-type and Armijo tests keep this alpha, also after a SOC
            alpha_test = alpha_primal
            accepted = trial_finite and self.check_acceptance(
                base, trial_barr, trial_theta, alpha_test
            )

            # Track successive filter rejections of the first trial
            if line_iter == 0:
                if accepted or not self.reject_filter:
                    self.successive_filter_rejections = 0
                else:
                    self.successive_filter_rejections += 1

            if (
                not accepted
                and line_iter == 0
                and trial_finite
                and self.options["second_order_correction"]
                and trial_theta >= base.ref_theta
            ):
                soc = self._second_order_correction(
                    solver, evaluator, state, base, alpha_primal, trial_theta
                )
                if soc is not None:
                    (
                        fobj,
                        barrier,
                        infeas,
                        trial_barr,
                        trial_theta,
                        alpha_primal,
                        alpha_dual,
                    ) = soc
                    self.trial.copy(self.soc_trial)
                    self.trial_grad.copy(self.soc_trial_grad)
                    accepted = True

            # A numerically insignificant step is taken whole without the filter test
            if (
                not accepted
                and line_iter == 0
                and trial_finite
                and self._tiny_step(state)
            ):
                info = self._accept_trial(
                    state, fobj, barrier, infeas, 1, alpha_primal, alpha_dual
                )
                info.tiny_step = True
                self.last_accept = "t"
                return info

            if not accepted and alpha_primal <= alpha_min:
                # Below alpha_min the search cannot succeed, restoration takes over
                break

            if accepted:
                self.update_filter(base, trial_barr, trial_theta, alpha_test)
                return self._accept_trial(
                    state, fobj, barrier, infeas, line_iter + 1, alpha_primal, alpha_dual
                )

            # Only the primal step backtracks
            alpha_primal = max(self.options["backtracking_factor"] * alpha_primal, alpha_min)

        return info

    def _step_sizes(self, state):
        """Largest primal step relative to 1 + |x| and the largest multiplier step."""
        x = state.current.get_solution()
        dx = state.step.get_solution()
        x.copy_device_to_host()
        dx.copy_device_to_host()
        ratio = self.step_ratio
        np.divide(
            np.abs(dx.get_array()), 1.0 + np.abs(x.get_array()), out=ratio.get_array()
        )
        ratio.copy_host_to_device()

        # Every rank reduces the same sizes so the decisions agree
        con_indices = self.problem.get_constraint_indices()
        dx.get_values_at(con_indices, self.step_con)
        multiplier_size = self.problem.maxabs(self.step_con)
        ratio.fill_at(con_indices, 0.0)
        primal_size = self.problem.maxabs(ratio)
        return primal_size, multiplier_size

    def _tiny_step(self, state):
        """True at a feasible point whose primal step is below tiny_step_tol relative to each variable."""
        tol = self.options["tiny_step_tol"]
        if tol <= 0.0 or state.constr_viol > self.options["constr_viol_tol"]:
            return False

        # The multiplier step must be small as well, as in IPOPT
        primal_size, multiplier_size = self._step_sizes(state)
        return primal_size < tol and multiplier_size < self.options["tiny_step_y_tol"]

    def _accept_trial(
        self, state, fobj, barrier, infeas, iters, alpha_primal, alpha_dual
    ):
        """Move the state to the trial point, keeping its evaluations, and return the success info."""
        state.invalidate()
        state.objective_value = fobj
        state.log_barrier_value = barrier
        state.con_infeasibility = infeas
        state.objective_current = True

        state.current.copy(self.trial)
        state.gradient.copy(self.trial_grad)
        state.gradient_current = True

        info = LineSearchInfo()
        info.success = True
        info.num_search_iters = iters
        info.alpha_primal = alpha_primal
        info.alpha_dual = alpha_dual
        self.current_info = info
        return info

    def add_log_info(self, info):
        if self.current_info is not None:
            info["filter_size"] = len(self.filter)
            info["line_iters"] = self.current_info.num_search_iters
            info["alpha_x"] = self.current_info.alpha_primal
            info["alpha_z"] = self.current_info.alpha_dual
        info["rejections"] = self.num_trial_rejections - self._rejections_at_start[0]
        info["filter_rejections"] = (
            self.num_filter_rejections - self._rejections_at_start[1]
        )
        info["total_rejections"] = self.num_trial_rejections
        info["total_filter_rejections"] = self.num_filter_rejections

        # The rejection reasons of the trials beside the type of the accepted step
        parts = [
            f"{tag}{count}"
            for tag, count in (
                ("x", self.reasons["theta_max"]),
                ("a", self.reasons["armijo"]),
                ("d", self.reasons["decrease"]),
                ("f", self.reasons["filter"]),
            )
            if count
        ]
        info["ls_why"] = "".join(parts) + (":" + self.last_accept if self.last_accept else "")
        info["soc"] = getattr(self, "soc_outcome", "")
