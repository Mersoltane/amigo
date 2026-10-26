"""The dual step contract observer.

An accepted step may raise the dual residual by at most the rise its own
perturbations declare. Breaches are counted for the log and steer nothing.
"""

import numpy as np


class StepContract:
    """Watch the dual residual of each accepted step."""

    def __init__(self, options, problem):
        self.tolerance = options["convergence_tolerance"]
        self._primal = problem.get_primal_indices()
        self._duals = problem.get_constraint_indices()

        # The dual residual the current step starts from
        self.pre_step_dual = None

        self.breaches = 0
        self._breached_now = False

    def before_step(self, state):
        """Remember the dual residual the step starts from."""
        self.pre_step_dual = state.dual_infeas

    def after_step(self, evaluator, state, alpha, delta_w, delta_c):
        """Judge the step against its declared allowance."""
        self._breached_now = False
        if self.pre_step_dual is None:
            return
        evaluator.evaluate_residual(state)

        # The allowance is the increase declared by the perturbations of the solved system
        step = state.step.get_solution()
        step.copy_device_to_host()
        d = step.get_array()
        dx = np.abs(d[self._primal.get_array()]).max(initial=0.0)
        dl = np.abs(d[self._duals.get_array()]).max(initial=0.0)
        allowed = self.pre_step_dual + alpha * (delta_w * dx + delta_c * dl)

        if state.dual_infeas > allowed and state.dual_infeas > self.tolerance:
            self._breached_now = True
            self.breaches += 1

    def add_log_info(self, info):
        """Add the breach count to the iteration record."""
        info["dual_viol"] = self._breached_now
        info["dual_viol_total"] = self.breaches
