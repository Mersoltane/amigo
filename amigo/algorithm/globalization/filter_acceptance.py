"""Filter acceptance rules of the Waechter and Biegler line search.

The filter stores pairs of the barrier objective phi and the constraint
violation theta with their margins. The rules decide whether a trial point
makes enough progress over the base point and the filter.
"""

import numpy as np


class Filter:
    """Filter of (phi, theta) pairs stored with their margins."""

    def __init__(self, gamma_phi=1e-8, gamma_theta=1e-5):
        self.entries = []
        self.gamma_phi = gamma_phi
        self.gamma_theta = gamma_theta

    def is_acceptable(self, phi_trial, theta_trial):
        """True unless an entry lies strictly below the trial in both coordinates."""
        for phi_f, theta_f in self.entries:
            if phi_trial > phi_f and theta_trial > theta_f:
                return False
        return True

    def add(self, phi, theta):
        """Add the entry with its margins and drop entries it dominates."""
        phi_entry = phi - self.gamma_phi * theta
        theta_entry = (1.0 - self.gamma_theta) * theta
        self.entries = [
            (p, t) for p, t in self.entries if not (p >= phi_entry and t >= theta_entry)
        ]
        self.entries.append((phi_entry, theta_entry))

    def clear(self):
        """Remove all filter entries."""
        self.entries.clear()

    def __len__(self):
        return len(self.entries)


class FilterBasePoint:
    """Barrier objective, violation and slope at the start of a line search."""

    ref_barr: float
    ref_theta: float
    ref_dphi: float


class FilterAcceptance:
    """Acceptance rules for trial points against the base point and the filter."""

    def __init__(self, options):
        self.options = options
        self.filter = Filter(options["filter_gamma_phi"], options["filter_gamma_theta"])

        # Constant = 10 times machine precision
        self.EPS10 = 10.0 * np.finfo(float).eps

        # Whether the last acceptance check was rejected by the filter history
        self.reject_filter = False

        # Rejection counts for the iteration log
        self.num_trial_rejections = 0
        self.num_filter_rejections = 0

        # Rejection reasons of the current line search and the acceptance type
        self.reasons = {"theta_max": 0, "armijo": 0, "decrease": 0, "filter": 0}
        self.last_accept = ""

    def reset_reasons(self):
        """Start the reason tallies of a new line search."""
        for key in self.reasons:
            self.reasons[key] = 0
        self.last_accept = ""

    def set_reference_values(self, ref_theta):
        # Infeasibility floor and ceiling, fixed by the initial violation for the solve
        self.theta_min = self.options["filter_theta_min_fact"] * max(1.0, ref_theta)
        self.theta_max = self.options["filter_theta_max_fact"] * max(1.0, ref_theta)
        return

    def is_ftype(self, base, alpha_test):
        """An f-type step primarily reduces the barrier objective."""
        delta = self.options["filter_delta"]
        s_theta = self.options["filter_s_theta"]
        s_phi = self.options["filter_s_phi"]

        return (
            base.ref_dphi < 0.0
            and alpha_test * (-base.ref_dphi) ** s_phi > delta * base.ref_theta**s_theta
        )

    def armijo_holds(self, base, trial_barr, alpha_test):
        """Check if the Armijo condition holds relative to the base point"""
        eta_phi = self.options["filter_eta_phi"]

        return (
            trial_barr - base.ref_barr
        ) - eta_phi * alpha_test * base.ref_dphi <= self.EPS10 * abs(base.ref_barr)

    def acceptable_to_iterate(self, base, trial_barr, trial_theta):
        gamma_theta = self.options["filter_gamma_theta"]
        gamma_phi = self.options["filter_gamma_phi"]
        obj_max_inc = 5.0

        if trial_barr > base.ref_barr:
            basval = 1.0
            if abs(base.ref_barr) > 10.0:
                basval = np.log10(abs(base.ref_barr))
            if np.log10(max(trial_barr - base.ref_barr, 1e-300)) > obj_max_inc + basval:
                return False

        feas_reduce = trial_theta - (1.0 - gamma_theta) * base.ref_theta
        feas_check = feas_reduce <= self.EPS10 * abs(base.ref_theta)

        obj_reduce = (trial_barr - base.ref_barr) + gamma_phi * base.ref_theta
        obj_check = obj_reduce <= self.EPS10 * abs(base.ref_barr)

        return feas_check or obj_check

    def check_acceptance(self, base, trial_barr, trial_theta, alpha_test):
        """Acceptance test that also counts the rejections for the log."""
        accepted = self._check_acceptance(base, trial_barr, trial_theta, alpha_test)
        if not accepted:
            self.num_trial_rejections += 1
            if self.reject_filter:
                self.num_filter_rejections += 1
        return accepted

    def _check_acceptance(self, base, trial_barr, trial_theta, alpha_test):
        # Only a filter-history rejection counts toward the reset trigger
        self.reject_filter = False
        if trial_theta > self.theta_max:
            self.reasons["theta_max"] += 1
            return False

        # An f-type step near feasibility needs Armijo, otherwise sufficient decrease
        if (
            alpha_test > 0.0
            and self.is_ftype(base, alpha_test)
            and base.ref_theta <= self.theta_min
        ):
            if not self.armijo_holds(base, trial_barr, alpha_test):
                self.reasons["armijo"] += 1
                return False
        elif not self.acceptable_to_iterate(base, trial_barr, trial_theta):
            self.reasons["decrease"] += 1
            return False

        # Every accepted trial must also be acceptable to the filter
        accepted = self.filter.is_acceptable(trial_barr, trial_theta)
        self.reject_filter = not accepted
        if not accepted:
            self.reasons["filter"] += 1
        return accepted

    def update_filter(self, base, trial_barr, trial_theta, alpha_test):
        # The filter grows unless the switching condition and Armijo both hold
        if not (
            self.is_ftype(base, alpha_test)
            and self.armijo_holds(base, trial_barr, alpha_test)
        ):
            self.filter.add(base.ref_barr, base.ref_theta)
            self.last_accept = "h"
        else:
            self.last_accept = "f"

    def compute_alpha_min(self, base):
        # The smallest step length the search attempts before restoration
        gamma_theta = self.options["filter_gamma_theta"]
        gamma_phi = self.options["filter_gamma_phi"]
        s_theta = self.options["filter_s_theta"]
        s_phi = self.options["filter_s_phi"]
        delta = self.options["filter_delta"]
        alpha_min_frac = 0.05

        alpha_min = gamma_theta
        if base.ref_dphi < 0.0:
            alpha_min = min(gamma_theta, gamma_phi * base.ref_theta / (-base.ref_dphi))
            if base.ref_theta <= self.theta_min:
                alpha_min = min(
                    alpha_min,
                    delta * base.ref_theta**s_theta / (-base.ref_dphi) ** s_phi,
                )
        alpha_min *= alpha_min_frac

        return alpha_min
