"""Funnel line search.

A monotonically contracting funnel width bounds the violation of every
accepted iterate. Backtracking, second-order correction, and restoration
plumbing are shared with the filter line search.
"""

import numpy as np

from .filter_line_search import FilterLineSearch


class Funnel:
    """Funnel width and its update rules."""

    def __init__(self, options, initial_infeasibility):
        self.width = max(
            options["funnel_ubd"], options["funnel_fact"] * initial_infeasibility
        )
        self.beta = options["funnel_beta"]
        self.kappa = options["funnel_kappa"]
        self.strategy = options["funnel_update_strategy"]
        if self.strategy not in (1, 2, 3):
            raise ValueError(f"Unknown funnel_update_strategy {self.strategy}")

    def acceptable(self, theta):
        return theta <= self.width

    def inside(self, theta):
        """Membership with the beta margin that h-type steps must meet."""
        return theta <= self.beta * self.width

    def update(self, current_theta, trial_theta):
        """Contract the width on an accepted h-type step."""
        if self.strategy == 1:
            if trial_theta < current_theta:
                self.width = max(
                    self.beta * self.width,
                    self.kappa * current_theta + (1.0 - self.kappa) * trial_theta,
                )
            else:
                self.width = self.beta * self.width
        elif self.strategy == 2:
            self.width = self.kappa * self.width + (1.0 - self.kappa) * trial_theta
        else:
            self.width = self.beta * self.width


class FunnelLineSearch(FilterLineSearch):
    """Funnel acceptance on the shared backtracking and SOC scaffolding."""

    def __init__(self, options, problem, optimizer):
        super().__init__(options, problem, optimizer)
        self.funnel = None
        self.delta = options["funnel_switching_delta"]
        self.h_exponent = options["funnel_switching_infeasibility_exponent"]
        self.min_alpha = options["funnel_min_step_length"]
        # Violation at restoration entry for the exit test
        self._resto_entry_theta = None
        # Set on acceptance so the width update sees the step type
        self._accepted_htype = False

    def reset_on_new_barrier(self, state, barrier_info):
        # No reset on mu changes, the width bounds the mu-independent violation
        return

    def set_reference_values(self, ref_theta):
        # theta_max and theta_min are filter devices replaced by the funnel
        self.theta_min = 0.0
        self.theta_max = np.inf
        if self.funnel is None:
            self.funnel = Funnel(self.options, ref_theta)

    def compute_alpha_min(self, base):
        """Smallest step length tried before restoration.

        Outside the funnel it is the step at which the linear violation model
        (1 - alpha) theta re-enters beta times the width. Inside it is the
        smallest step at which an h-type or f-type acceptance is still
        possible under the linear models of theta and phi.
        """
        alpha_min_frac = 0.05
        theta = base.ref_theta
        if not self.funnel.inside(theta):
            alpha = 1.0 - self.funnel.beta * self.funnel.width / theta
        elif base.ref_dphi < 0.0:
            alpha = min(
                self.options["filter_gamma_theta"],
                self.options["filter_gamma_phi"] * theta / (-base.ref_dphi),
                self.delta * theta**self.h_exponent / (-base.ref_dphi),
            )
        else:
            alpha = self.options["filter_gamma_theta"]
        return max(alpha_min_frac * alpha, self.min_alpha)

    def is_ftype(self, base, alpha_test):
        """Switching condition on the alpha-scaled model decrease."""
        pred = alpha_test * (-base.ref_dphi)
        return (
            base.ref_dphi < 0.0 and pred > self.delta * base.ref_theta**self.h_exponent
        )

    def _check_acceptance(self, base, trial_barr, trial_theta, alpha_test):
        self.reject_filter = False
        self._accepted_htype = False

        if not self.funnel.acceptable(trial_theta):
            return False

        if self.is_ftype(base, alpha_test):
            return self.armijo_holds(base, trial_barr, alpha_test)

        # An h-type step stays inside the funnel and makes progress on the current iterate
        if self.funnel.inside(trial_theta) and self.acceptable_to_iterate(
            base, trial_barr, trial_theta
        ):
            self._accepted_htype = True
            return True

        return False

    def update_filter(self, base, trial_barr, trial_theta, alpha_test):
        # Only h-type steps contract the funnel, nothing else is stored
        if self._accepted_htype:
            self.funnel.update(base.ref_theta, trial_theta)

    def augment_filter(self, phi, theta):
        # Restoration entry records the violation for the exit test
        if np.isfinite(theta):
            self._resto_entry_theta = theta

    def restoration_acceptable(self, phi, theta):
        """Exit test theta <= beta * min(width, theta at entry)."""
        if self.funnel is None:
            return True
        ref = self.funnel.width
        if self._resto_entry_theta is not None:
            ref = min(ref, self._resto_entry_theta)
        return theta <= self.funnel.beta * ref

    def reset_after_restoration(self):
        # The width is kept, the restored point already lies inside it
        super().reset_after_restoration()
        self._resto_entry_theta = None
