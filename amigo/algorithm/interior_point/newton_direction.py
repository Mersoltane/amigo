"""Newton step from an already-factorized KKT system.

Solves against the factorization prepared by InertiaCorrector, forms the
full primal-dual update, and records the fraction-to-boundary step lengths
on the state. The normal step solves the same system with the dual
residual rows zeroed, so it moves onto the constraints and the central
path without descending the objective.
"""

import numpy as np


def set_step_lengths(optimizer, state):
    """Record the fraction to the boundary step lengths of state.step."""
    alpha_x, _, alpha_z, _ = optimizer.compute_max_step(
        state.tau, state.current, state.step
    )
    state.max_alpha_primal = alpha_x
    state.max_alpha_dual = alpha_z
    state.step_current = True


class NewtonStep:
    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        self.update = self.problem.create_vector()

        # Primal rows of the KKT system and the bounds of the primal variables
        self.primal = np.asarray(self.problem.get_primal_indices().get_array(), dtype=int)
        self.lower = self.problem.get_lower()
        self.upper = self.problem.get_upper()

    def compute_step(self, solver, evaluator, state):
        # Evalute the residual (may be required since the barrier may have changed)
        evaluator.evaluate_residual(state)

        # Solve the linear system to obtain the new step
        solver.solve(state.residual, self.update)

        # Compute the full step and its lengths
        self.optimizer.compute_update(state.mu, state.current, self.update, state.step)
        set_step_lengths(self.optimizer, state)

    def _centering(self, state):
        """Bound dual less barrier gradient on every bounded primal row"""
        solution = state.current.get_solution()
        zl = state.current.get_zl()
        zu = state.current.get_zu()
        solution.copy_device_to_host()
        zl.copy_device_to_host()
        zu.copy_device_to_host()

        # The primal entries of the point, its bound duals, and its bounds
        x = np.asarray(solution.get_array(), dtype=float)[self.primal]
        zl_arr = np.asarray(zl.get_array(), dtype=float)
        zu_arr = np.asarray(zu.get_array(), dtype=float)
        if len(zl_arr) == len(solution.get_array()):
            zl_arr = zl_arr[self.primal]
            zu_arr = zu_arr[self.primal]
        lower = np.asarray(self.lower.get_array(), dtype=float)[self.primal]
        upper = np.asarray(self.upper.get_array(), dtype=float)[self.primal]

        # A lower bound enters with its dual less the barrier gradient and an upper bound with the opposite sign
        centering = np.zeros(len(self.primal))
        has_lower = np.isfinite(lower)
        has_upper = np.isfinite(upper)
        centering[has_lower] += zl_arr[has_lower] - state.mu / (x[has_lower] - lower[has_lower])
        centering[has_upper] -= zu_arr[has_upper] - state.mu / (upper[has_upper] - x[has_upper])
        return centering

    def compute_normal_step(self, solver, evaluator, state):
        """Newton step on the constraints and the centering alone, with the dual residual rows zeroed"""
        evaluator.evaluate_residual(state)

        # The primal rows keep only the centering part of the residual
        state.residual.copy_device_to_host()
        residual = state.residual.get_array()
        residual[self.primal] = -self._centering(state)
        state.residual.copy_host_to_device()

        # Solve the linear system to obtain the normal step and its lengths
        solver.solve(state.residual, self.update)
        self.optimizer.compute_update(state.mu, state.current, self.update, state.step)
        set_step_lengths(self.optimizer, state)

        # The residual no longer describes the point
        state.residual_current = False
