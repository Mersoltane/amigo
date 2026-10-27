"""Feasibility restoration on the l1 violation, as in Waechter and Biegler.

A primal-dual barrier method solves the IPOPT restoration problem
    min rho sum(p + n) + (eta / 2) ||D_R (x - x_R)||^2  s.t.  c(x) - b - p + n = 0
The phase succeeds once the violation meets its target at a point the main filter accepts.
"""

import numpy as np

from ..barrier_strategy import fraction_to_boundary
from ..solvers import InertiaCorrector
from .filter_acceptance import FilterAcceptance, FilterBasePoint

# Gentle interior push when the main phase resumes
EXIT_PUSH = 1e-6

# Scale limit of the optimality error, as in the main convergence test
SCALE_MAX = 100.0

# A violation above this multiple of the tolerance is not rounding noise
INFEASIBLE_FACTOR = 100.0


def elastic_split(r, mu, rho):
    """Elastic pair p - n = r that minimizes rho (p + n) - mu log p - mu log n.

    With a = mu / rho and s = sqrt(r^2 + a^2) the pair has the product
    p n = mu (a + s) / (2 rho), which gives each root without cancellation.
    """
    a = mu / rho
    s = np.sqrt(r * r + a * a)
    pn = mu * (a + s) / (2.0 * rho)
    p_pos = 0.5 * (a + r + s)
    n_neg = 0.5 * (a - r + s)
    p = np.where(r >= 0.0, p_pos, pn / n_neg)
    n = np.where(r >= 0.0, pn / p_pos, n_neg)
    return p, n


class RestorationInfo:
    success: bool = False
    infeasible: bool = False
    iterations: int = 0
    theta_enter: float = 0.0
    theta_final: float = 0.0


class FeasibilityRestoration:
    """Barrier method on the restoration problem, entered from a failed point."""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer
        self.rho = options["resto_penalty"]
        self.con_indices = problem.get_constraint_indices()
        self.primal_indices = problem.get_primal_indices()

        # The phase keeps its own inertia history and filter
        self.inertia = InertiaCorrector(options, problem, optimizer)
        self.acceptance = FilterAcceptance(options)

        self.entry = optimizer.create_opt_vector()
        self.step = optimizer.create_opt_vector()
        self.trial = optimizer.create_opt_vector()
        self.grad = problem.create_vector()
        self.res = problem.create_vector()
        self.sol = problem.create_vector()
        self.con_res = problem.create_vector()
        self.work = problem.create_vector()
        self.trial_grad = problem.create_vector()
        self.tmp_primal = problem.create_primal_vector()
        self.tmp_primal2 = problem.create_primal_vector()
        self.tmp_con = problem.create_constraint_vector()
        self.con_ones = problem.create_constraint_vector()
        self.con_ones.get_array()[:] = 1.0
        self.con_ones.copy_host_to_device()

        # Global counts of the constraints and of the finite bounds of x
        _, self.num_con, self.num_bounds = optimizer.get_kkt_element_counts()

        # Totals over the solve for the iteration log
        self.num_calls = 0
        self.num_iterations = 0

    def add_log_info(self, info):
        info["resto_calls"] = self.num_calls
        info["resto_iterations"] = self.num_iterations

    def restore(self, solver, evaluator, state, line_search):
        """Reduce the constraint violation from the current point, updating it in place."""
        info = RestorationInfo()
        obj_scale_store = state.obj_scale
        mu_store = state.mu

        # The failure point sets the target and returns when the phase fails
        evaluator.evaluate_objective_and_infeasibility(state)
        info.theta_enter = state.con_infeasibility
        target = max(
            self.options["resto_required_reduction"] * info.theta_enter,
            self.options["convergence_tolerance"],
        )
        self.entry.copy(state.current)

        try:
            info.success, info.infeasible, info.theta_final, info.iterations = (
                self._minimize(
                    solver,
                    evaluator,
                    state,
                    line_search,
                    target,
                    obj_scale_store,
                    mu_store,
                )
            )
        finally:
            state.obj_scale = obj_scale_store
            state.mu = mu_store
            state.tau = fraction_to_boundary(self.options, mu_store)
            if info.success:
                # The driver estimates new multipliers from the restored point
                state.get_current_point().fill_at(self.con_indices, 0.0)
                self.optimizer.initialize_duals_warm(
                    mu_store,
                    EXIT_PUSH,
                    self.options["warm_start_mult_init_max"],
                    state.current,
                )
            elif not info.infeasible:
                state.current.copy(self.entry)
            state.invalidate()

        self.num_calls += 1
        self.num_iterations += info.iterations
        if state.comm_rank == 0:
            tag = "ok" if info.success else "infeasible" if info.infeasible else "failed"
            print(
                f"  Restoration: theta {info.theta_enter:.3e} -> "
                f"{info.theta_final:.3e} in {info.iterations} iters ({tag})"
            )
        return info

    def _exit_acceptable(
        self, evaluator, state, line_search, target, obj_scale_store, mu_store
    ):
        """True when the violation meets the target and the main filter accepts the point."""
        resto_scale, resto_mu = state.obj_scale, state.mu
        state.obj_scale, state.mu = obj_scale_store, mu_store
        state.invalidate()
        evaluator.evaluate_objective_and_infeasibility(state)
        theta = state.con_infeasibility
        ok = theta <= target and line_search.restoration_acceptable(
            state.barrier_objective, theta
        )
        state.obj_scale, state.mu = resto_scale, resto_mu
        state.invalidate()
        return ok

    def _minimize(
        self, solver, evaluator, state, line_search, target, obj_scale_store, mu_store
    ):
        """Barrier iterations on the restoration problem, returns (success, infeasible, theta, iterations)."""
        opts = self.options
        tol = opts["convergence_tolerance"]
        mu_floor = max(
            opts["mu_min"],
            min(tol, opts["compl_inf_tol"]) / (opts["barrier_tol_factor"] + 1.0),
        )
        self._enter(evaluator, state, mu_store)

        theta = 0.0
        for it in range(opts["resto_max_iterations"]):
            state.mu = self.mu
            evaluator.evaluate_gradient(state)
            r = self._residual(state.current, state.gradient)
            theta = self._con_sum(np.abs(r))

            # The phase ends once the main filter accepts a point at the target violation
            if theta <= target:
                if self._exit_acceptable(
                    evaluator, state, line_search, target, obj_scale_store, mu_store
                ):
                    return True, False, theta, it
                evaluator.evaluate_gradient(state)

            y = self._con_values(state.get_current_point())
            self._gradient(state)
            if self._update_barrier(state, r, y, mu_floor):
                return False, self._infeasible(state, r, y), theta, it
            state.mu = self.mu

            if not (
                self._direction(solver, evaluator, state, r, y)
                and self._line_search(evaluator, state, r)
            ):
                return False, self._infeasible(state, r, y), theta, it

        # The budget ends the phase at the last accepted point
        evaluator.evaluate_gradient(state)
        r = self._residual(state.current, state.gradient)
        y = self._con_values(state.get_current_point())
        self._gradient(state)
        theta = self._con_sum(np.abs(r))
        return False, self._infeasible(state, r, y), theta, opts["resto_max_iterations"]

    def _infeasible(self, state, r, y):
        """True when the point nearly solves the restoration problem with a violation well above the tolerance."""
        threshold = INFEASIBLE_FACTOR * self.options["convergence_tolerance"]
        return (
            self._con_max(r) > threshold
            and self._error(state, 0.0, r, y) <= self.options["acceptable_tol"]
        )

    def _enter(self, evaluator, state, mu_store):
        """Start at the failure point with zero multipliers and elastic variables that absorb the violation."""
        x = state.get_current_point()
        state.obj_scale = 0.0
        x.fill_at(self.con_indices, 0.0)
        state.invalidate()
        evaluator.evaluate_gradient(state)
        r = self._residual(state.current, state.gradient)

        # The barrier starts at the largest violation
        self.mu = max(mu_store, self._con_max(r))
        self.eta = np.sqrt(self.mu)
        self.p, self.n = elastic_split(r, self.mu, self.rho)
        self.zp = self.mu / self.p
        self.zn = self.mu / self.n

        # The bound multipliers of x are capped at the penalty
        for z in (state.current.get_zl(), state.current.get_zu()):
            z.copy_device_to_host()
            np.minimum(z.get_array(), self.rho, out=z.get_array())
            z.copy_host_to_device()

        self.x_ref = self._primal_values(x)
        self.dr2 = np.minimum(1.0, 1.0 / np.maximum(np.abs(self.x_ref), 1e-12)) ** 2

        # The elastic variables start with zero violation for the filter limits
        self.acceptance.filter.clear()
        self.acceptance.set_reference_values(0.0)

    def _residual(self, vars, grad):
        """Constraint residual c(x) - b from a gradient that holds c in its constraint rows."""
        self.optimizer.compute_residual(0.0, vars, grad, self.con_res)
        return -self._con_values(self.con_res)

    def _gradient(self, state):
        """Primal rows J^T y plus the proximal gradient, over the gradient in the state."""
        d = self._primal_values(state.get_current_point()) - self.x_ref
        self._set_primal_values(self.work, self.eta * self.dr2 * d)
        self.grad.copy(state.gradient)
        self.grad.axpy(1.0, self.work)

    def _error(self, state, mu, r, y):
        """Scaled optimality error of the restoration barrier problem at mu."""
        dual_x, _, comp_x = self.optimizer.compute_kkt_error(mu, state.current, self.grad)
        dual = max(
            dual_x,
            self._con_max(self.rho - y - self.zp),
            self._con_max(self.rho + y - self.zn),
        )
        primal = self._con_max(r - self.p + self.n)
        comp = max(
            comp_x,
            self._con_max(self.p * self.zp - mu),
            self._con_max(self.n * self.zn - mu),
        )

        # Large multipliers scale the dual and complementarity errors down
        z_sum = (
            self.problem.abssum(state.current.get_zl())
            + self.problem.abssum(state.current.get_zu())
            + self._con_sum(self.zp + self.zn)
        )
        num_z = self.num_bounds + 2 * self.num_con
        s_d = max(SCALE_MAX, (self._con_sum(np.abs(y)) + z_sum) / (self.num_con + num_z))
        s_c = max(SCALE_MAX, z_sum / num_z)
        return max(dual * SCALE_MAX / s_d, primal, comp * SCALE_MAX / s_c)

    def _update_barrier(self, state, r, y, mu_floor):
        """Decrease mu while the barrier problem is solved, True once it is solved at the floor."""
        kappa_eps = self.options["barrier_tol_factor"]
        while self._error(state, self.mu, r, y) <= kappa_eps * self.mu:
            if self.mu <= mu_floor:
                return True
            self.mu = max(
                min(
                    self.options["mu_linear_decrease_factor"] * self.mu,
                    self.mu ** self.options["mu_superlinear_decrease_power"],
                ),
                mu_floor,
            )
            self.eta = np.sqrt(self.mu)
            self.acceptance.filter.clear()
            self._gradient(state)
        return False

    def _direction(self, solver, evaluator, state, r, y):
        """Factor the restoration KKT system and solve for the step, False on failure."""
        rho, mu = self.rho, self.mu
        p, n, zp, zn = self.p, self.n, self.zp, self.zn

        # The Hessian of the restoration Lagrangian is y^T c'' at zero objective weight
        state.hessian_current = False
        evaluator.evaluate_hessian(state)

        # Barrier and proximal terms on the primal rows and the elastic terms on the constraint rows
        self.optimizer.compute_diagonal(state.current, state.diagonal)
        self._set_primal_values(self.work, self.eta * self.dr2)
        state.diagonal.axpy(1.0, self.work)
        self._set_con_values(state.diagonal, -(p / zp + n / zn))
        state.diagonal_current = True
        if not self.inertia.factor_for_inertia(solver, evaluator, state):
            return False

        # The constraint rows carry the linearized constraint with p and n eliminated
        self.optimizer.compute_residual(mu, state.current, self.grad, self.res)
        rhs = -r + p - n + mu / zp - (p / zp) * (rho - y) - mu / zn + (n / zn) * (rho + y)
        self._set_con_values(self.res, rhs)
        solver.solve(self.res, self.sol)
        self.optimizer.compute_update(mu, state.current, self.sol, self.step)

        dy = self._con_values(self.sol)
        self.dp = (mu - p * (rho - y - dy)) / zp
        self.dn = (mu - n * (rho + y + dy)) / zn
        self.dzp = rho - y - zp - dy
        self.dzn = rho + y - zn + dy
        return True

    def _line_search(self, evaluator, state, r):
        """Filter line search on the restoration problem, moving the state on success."""
        acceptance = self.acceptance

        # The bound multipliers take the full fraction to the boundary step
        tau = fraction_to_boundary(self.options, self.mu)
        ax, _, az, _ = self.optimizer.compute_max_step(tau, state.current, self.step)
        cut_x = self._con_max(np.maximum(np.maximum(-self.dp / self.p, -self.dn / self.n), 0.0))
        cut_z = self._con_max(
            np.maximum(np.maximum(-self.dzp / self.zp, -self.dzn / self.zn), 0.0)
        )
        alpha = min(ax, tau / cut_x) if cut_x > 0.0 else ax
        alpha_z = min(az, tau / cut_z) if cut_z > 0.0 else az

        base = FilterBasePoint()
        base.ref_barr = self._objective(state.current, self.p, self.n)
        base.ref_theta = self._con_sum(np.abs(r - self.p + self.n))
        base.ref_dphi = self._slope(state)
        alpha_min = acceptance.compute_alpha_min(base)

        while True:
            self.optimizer.apply_step_update(
                alpha, alpha_z, state.current, self.step, self.trial
            )
            p = self.p + alpha * self.dp
            n = self.n + alpha * self.dn
            evaluator.evaluate_gradient_from_point(0.0, self.trial, self.trial_grad)
            r_trial = self._residual(self.trial, self.trial_grad)
            theta = self._con_sum(np.abs(r_trial - p + n))
            phi = self._objective(self.trial, p, n)
            if (
                np.isfinite(phi)
                and np.isfinite(theta)
                and acceptance.check_acceptance(base, phi, theta, alpha)
            ):
                break
            if alpha <= alpha_min:
                return False
            alpha = max(self.options["backtracking_factor"] * alpha, alpha_min)

        acceptance.update_filter(base, phi, theta, alpha)
        state.current.copy(self.trial)
        state.invalidate()
        state.gradient.copy(self.trial_grad)
        state.gradient_current = True
        self.p, self.n = p, n
        self.zp = self.zp + alpha_z * self.dzp
        self.zn = self.zn + alpha_z * self.dzn

        # The kappa_sigma safeguard keeps every bound multiplier near mu over its slack
        kappa = self.options["kappa_sigma"]
        self.optimizer.correct_bound_multipliers(self.mu, kappa, state.current)
        self.zp = np.clip(self.zp, self.mu / (kappa * self.p), kappa * self.mu / self.p)
        self.zn = np.clip(self.zn, self.mu / (kappa * self.n), kappa * self.mu / self.n)
        return True

    def _objective(self, vars, p, n):
        """Barrier objective of the restoration problem."""
        d = self._primal_values(vars.get_solution()) - self.x_ref
        prox = 0.5 * self.eta * self._primal_dot(self.dr2 * d, d)
        elastic = self._con_sum(
            self.rho * (p + n) - self.mu * (np.log(p) + np.log(n))
        )
        return elastic + prox + self.optimizer.compute_log_barrier(self.mu, vars)

    def _slope(self, state):
        """Directional derivative of the barrier objective along the step."""
        d = self._primal_values(state.get_current_point()) - self.x_ref
        dx = self._primal_values(self.step.get_solution())
        prox = self.eta * self._primal_dot(self.dr2 * d, dx)
        elastic = self._con_sum(
            self.rho * (self.dp + self.dn)
            - self.mu * (self.dp / self.p + self.dn / self.n)
        )
        barrier = self.optimizer.compute_log_barrier_derivative(
            self.mu, state.current, self.step
        )
        return elastic + prox + barrier

    def _con_values(self, vec):
        """Host copy of the constraint entries of a full vector."""
        vec.get_values_at(self.con_indices, self.tmp_con)
        self.tmp_con.copy_device_to_host()
        return self.tmp_con.get_array().copy()

    def _set_con_values(self, vec, values):
        """Write a host array into the constraint entries of a full vector."""
        self.tmp_con.get_array()[:] = values
        self.tmp_con.copy_host_to_device()
        vec.set_values_at(self.con_indices, self.tmp_con)

    def _primal_values(self, vec):
        """Host copy of the primal entries of a full vector."""
        vec.get_values_at(self.primal_indices, self.tmp_primal)
        self.tmp_primal.copy_device_to_host()
        return self.tmp_primal.get_array().copy()

    def _set_primal_values(self, vec, values):
        """Zero a full vector and write a host array into its primal entries."""
        self.tmp_primal.get_array()[:] = values
        self.tmp_primal.copy_host_to_device()
        vec.zero()
        vec.set_values_at(self.primal_indices, self.tmp_primal)

    def _con_sum(self, values):
        """Sum of a constraint array over all processors."""
        self.tmp_con.get_array()[:] = values
        self.tmp_con.copy_host_to_device()
        return self.problem.dot(self.tmp_con, self.con_ones)

    def _con_max(self, values):
        """Largest magnitude of a constraint array over all processors."""
        self.tmp_con.get_array()[:] = values
        self.tmp_con.copy_host_to_device()
        return self.problem.maxabs(self.tmp_con)

    def _primal_dot(self, a, b):
        """Dot product of two primal arrays over all processors."""
        self.tmp_primal.get_array()[:] = a
        self.tmp_primal.copy_host_to_device()
        self.tmp_primal2.get_array()[:] = b
        self.tmp_primal2.copy_host_to_device()
        return self.problem.dot(self.tmp_primal, self.tmp_primal2)
