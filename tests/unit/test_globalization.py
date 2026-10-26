"""
Unit tests for the globalization modules: the filter acceptance store, the
filter and funnel line-search acceptance tests, the funnel width logic, and
the rejection counters. Pure-logic tests, no compiled backend required: the
filter rules need only the options, and funnel line search objects are built
via __new__ with only the attributes the tested methods read.
"""

import numpy as np
import pytest

from amigo.algorithm.globalization.filter_acceptance import (
    Filter,
    FilterAcceptance,
    FilterBasePoint,
)
from amigo.algorithm.globalization.funnel_line_search import (
    Funnel,
    FunnelLineSearch,
)


FUNNEL_OPTIONS = {
    "funnel_ubd": 1.0,
    "funnel_fact": 1.5,
    "funnel_beta": 0.9999,
    "funnel_kappa": 0.5,
    "funnel_update_strategy": 1,
}

# Process constants the funnel shares with the filter line search
SHARED_OPTIONS = {
    "filter_gamma_theta": 1e-5,
    "filter_gamma_phi": 1e-8,
    "filter_eta_phi": 1e-8,
}

# Filter line search constants at their defaults
FILTER_OPTIONS = dict(
    SHARED_OPTIONS, filter_delta=1.0, filter_s_theta=1.1, filter_s_phi=2.3
)


def make_funnel(theta0=1.0, **overrides):
    options = dict(FUNNEL_OPTIONS)
    options.update(overrides)
    return Funnel(options, theta0)


def make_base_point(ref_barr=0.0, ref_theta=0.1, ref_dphi=-1.0):
    base = FilterBasePoint()
    base.ref_barr = ref_barr
    base.ref_theta = ref_theta
    base.ref_dphi = ref_dphi
    return base


def make_filter_search(theta_min=1e-4, theta_max=1e4):
    """Filter acceptance rules with the given violation floor and ceiling."""
    fls = FilterAcceptance(dict(FILTER_OPTIONS))
    fls.theta_min = theta_min
    fls.theta_max = theta_max
    return fls


def make_funnel_search(theta0=1.0, **funnel_overrides):
    """FunnelLineSearch with only the attributes the logic methods read."""
    fls = FunnelLineSearch.__new__(FunnelLineSearch)
    fls.funnel = make_funnel(theta0, **funnel_overrides)
    fls.options = dict(SHARED_OPTIONS)
    fls.EPS10 = 10.0 * np.finfo(float).eps
    fls.delta = 0.999
    fls.h_exponent = 2
    fls.min_alpha = 1e-12
    fls._resto_entry_theta = None
    fls._accepted_htype = False
    fls.reject_filter = False
    fls.num_trial_rejections = 0
    fls.num_filter_rejections = 0
    return fls


class TestFilter:
    def test_empty_filter_accepts_everything(self):
        filt = Filter()
        assert filt.is_acceptable(1e10, 1e10)

    def test_dominated_point_rejected(self):
        filt = Filter()
        filt.add(1.0, 1.0)
        assert not filt.is_acceptable(2.0, 2.0)

    def test_improvement_in_one_coordinate_accepted(self):
        filt = Filter()
        filt.add(1.0, 1.0)
        assert filt.is_acceptable(0.5, 2.0)
        assert filt.is_acceptable(2.0, 0.5)

    def test_margins_reject_marginal_point(self):
        """Points inside the stored margins count as dominated."""
        filt = Filter(gamma_phi=1e-2, gamma_theta=1e-2)
        filt.add(1.0, 1.0)
        # Slightly better than (1, 1) but within the margin
        assert not filt.is_acceptable(1.0 - 1e-3, 1.0 - 1e-3)

    def test_add_prunes_dominated_entries(self):
        filt = Filter()
        filt.add(2.0, 2.0)
        filt.add(1.0, 1.0)  # dominates the margined (2, 2) entry
        assert len(filt) == 1

    def test_clear(self):
        filt = Filter()
        filt.add(1.0, 1.0)
        filt.clear()
        assert len(filt) == 0
        assert filt.is_acceptable(2.0, 2.0)


class TestFilterAcceptance:
    def test_violation_ceiling_rejects_without_filter_blame(self):
        fls = make_filter_search(theta_max=10.0)
        base = make_base_point(ref_theta=1.0)
        assert not fls.check_acceptance(base, 0.0, 20.0, alpha_test=1.0)
        assert fls.num_trial_rejections == 1
        assert fls.num_filter_rejections == 0

    def test_sufficient_decrease_then_filter_veto_is_counted(self):
        fls = make_filter_search()
        fls.filter.add(0.0, 0.1)
        # Not f-type, the violation decreases, but the filter entry dominates
        base = make_base_point(ref_barr=1.0, ref_theta=1.0, ref_dphi=1.0)
        assert not fls.check_acceptance(base, 0.5, 0.5, alpha_test=1.0)
        assert fls.num_filter_rejections == 1

    def test_ftype_near_feasibility_requires_armijo(self):
        fls = make_filter_search()
        base = make_base_point(ref_barr=1.0, ref_theta=1e-8, ref_dphi=-1.0)
        assert fls.check_acceptance(base, 0.9, 1e-8, alpha_test=1.0)
        assert not fls.check_acceptance(base, 1.1, 1e-8, alpha_test=1.0)

    def test_armijo_ftype_step_leaves_the_filter_unchanged(self):
        fls = make_filter_search()
        base = make_base_point(ref_barr=1.0, ref_theta=1e-8, ref_dphi=-1.0)
        fls.update_filter(base, 0.9, 1e-8, alpha_test=1.0)
        assert len(fls.filter) == 0

        # A step that fails Armijo grows the filter by the base point
        fls.update_filter(base, 1.1, 1e-8, alpha_test=1.0)
        assert len(fls.filter) == 1


class TestFunnel:
    def test_initial_width_floor(self):
        funnel = make_funnel(theta0=0.0)
        assert funnel.width == FUNNEL_OPTIONS["funnel_ubd"]

    def test_initial_width_scales_with_theta(self):
        funnel = make_funnel(theta0=10.0)
        assert funnel.width == pytest.approx(15.0)

    def test_membership_applies_beta_margin(self):
        funnel = make_funnel(theta0=10.0)  # width 15
        assert funnel.inside(0.9999 * 15.0)
        assert not funnel.inside(15.0)

    def test_update_strategy_1_floors_at_beta_contraction(self):
        """The beta floor dominates: with beta near one, strategy 1
        contracts by at most a factor beta per h-step."""
        funnel = make_funnel(theta0=10.0)  # width 15
        funnel.update(current_theta=10.0, trial_theta=2.0)
        # max(beta * width, kappa*current + (1-kappa)*trial) = beta * width
        assert funnel.width == pytest.approx(0.9999 * 15.0)

    def test_update_strategy_1_interpolates_below_beta_floor(self):
        funnel = make_funnel(theta0=10.0, funnel_beta=0.3)  # width 15
        funnel.update(current_theta=10.0, trial_theta=2.0)
        # Interpolant 6.0 wins over beta * width = 4.5
        assert funnel.width == pytest.approx(6.0)

    def test_update_strategy_1_contracts_on_nondecrease(self):
        funnel = make_funnel(theta0=10.0)
        funnel.update(current_theta=2.0, trial_theta=3.0)
        assert funnel.width == pytest.approx(0.9999 * 15.0)

    def test_update_strategy_2(self):
        funnel = make_funnel(theta0=10.0, funnel_update_strategy=2)
        funnel.update(current_theta=10.0, trial_theta=1.0)
        assert funnel.width == pytest.approx(0.5 * 15.0 + 0.5 * 1.0)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            make_funnel(theta0=10.0, funnel_update_strategy=99)


class TestFunnelAcceptance:
    def test_width_rejection(self):
        fls = make_funnel_search(theta0=1.0)  # width 1.5
        base = make_base_point(ref_theta=1.0)
        assert not fls._check_acceptance(base, 0.0, 2.0, alpha_test=1.0)

    def test_htype_accepted_on_feasibility_progress(self):
        fls = make_funnel_search(theta0=1.0)
        # dphi > 0: not f-type, so acceptance is via the h-type route
        base = make_base_point(ref_theta=1.0, ref_dphi=1.0)
        assert fls._check_acceptance(base, 0.0, 0.5, alpha_test=1.0)
        assert fls._accepted_htype

    def test_htype_requires_progress_wrt_current_iterate(self):
        """Inside the funnel but neither theta nor phi decreases: reject."""
        fls = make_funnel_search(theta0=1.0)
        base = make_base_point(ref_barr=0.0, ref_theta=1.0, ref_dphi=1.0)
        assert not fls._check_acceptance(base, 0.0, 1.0, alpha_test=1.0)
        assert not fls._accepted_htype

    def test_ftype_requires_armijo(self):
        fls = make_funnel_search(theta0=1.0)
        base = make_base_point(ref_barr=1.0, ref_theta=1e-8, ref_dphi=-1.0)
        # Sufficient barrier decrease passes, and is not an h-type step
        assert fls._check_acceptance(base, 0.9, 1e-8, alpha_test=1.0)
        assert not fls._accepted_htype
        # A barrier increase fails Armijo with no h-type fallthrough
        assert not fls._check_acceptance(base, 1.1, 1e-8, alpha_test=1.0)

    def test_update_contracts_on_htype_only(self):
        fls = make_funnel_search(theta0=1.0)  # width 1.5
        base = make_base_point(ref_theta=1.0, ref_dphi=1.0)
        assert fls._check_acceptance(base, 0.0, 0.5, alpha_test=1.0)
        fls.update_filter(base, 0.0, 0.5, alpha_test=1.0)
        assert fls.funnel.width == pytest.approx(0.9999 * 1.5)

        # An f-type acceptance leaves the width unchanged
        base = make_base_point(ref_barr=1.0, ref_theta=1e-8, ref_dphi=-1.0)
        assert fls._check_acceptance(base, 0.9, 1e-8, alpha_test=1.0)
        fls.update_filter(base, 0.9, 1e-8, alpha_test=1.0)
        assert fls.funnel.width == pytest.approx(0.9999 * 1.5)

    def test_barrier_updates_leave_the_funnel_intact(self):
        """The width bounds the mu-independent violation: no reset on mu
        changes, monotone or adaptive."""
        fls = make_funnel_search(theta0=1.0)
        info = type("Info", (), {"new_barrier": True, "new_subproblem": True})
        fls.reset_on_new_barrier(None, info)
        assert fls.funnel.width == pytest.approx(1.5)

    def test_restoration_exit_below_entry_violation(self):
        fls = make_funnel_search(theta0=1.0)  # width 1.5
        fls.augment_filter(0.0, 0.8)
        assert fls.restoration_acceptable(0.0, 0.79)
        assert not fls.restoration_acceptable(0.0, 0.8)

    def test_restoration_exit_inside_funnel_without_entry(self):
        fls = make_funnel_search(theta0=1.0)  # width 1.5
        assert fls.restoration_acceptable(0.0, 1.4)
        assert not fls.restoration_acceptable(0.0, 1.5)

    def test_restoration_before_first_search_is_permissive(self):
        fls = make_funnel_search(theta0=1.0)
        fls.funnel = None
        assert fls.restoration_acceptable(0.0, 100.0)

    def test_reset_after_restoration_keeps_width(self):
        fls = make_funnel_search(theta0=1.0)
        fls.augment_filter(0.0, 0.8)
        fls.reset_after_restoration()
        assert fls._resto_entry_theta is None
        assert fls.funnel.width == pytest.approx(1.5)

    def test_counting_wrapper_counts_rejections(self):
        fls = make_funnel_search(theta0=1.0)
        base = make_base_point(ref_theta=1.0)
        fls.check_acceptance(base, 0.0, 2.0, alpha_test=1.0)
        assert fls.num_trial_rejections == 1
        assert fls.num_filter_rejections == 0


class TestFunnelAlphaMin:
    def test_feasible_point_with_descent_returns_floor(self):
        fls = make_funnel_search(theta0=1.0)
        base = make_base_point(ref_theta=0.0, ref_dphi=-1.0)
        assert fls.compute_alpha_min(base) == fls.min_alpha

    def test_feasible_point_without_descent_uses_gamma_theta(self):
        fls = make_funnel_search(theta0=1.0)
        base = make_base_point(ref_theta=0.0, ref_dphi=1.0)
        expected = 0.05 * SHARED_OPTIONS["filter_gamma_theta"]
        assert fls.compute_alpha_min(base) == pytest.approx(expected)

    def test_membership_bound_for_theta_above_width(self):
        fls = make_funnel_search(theta0=1.0)  # width 1.5
        theta = 3.0
        base = make_base_point(ref_theta=theta, ref_dphi=1.0)
        target = fls.funnel.beta * fls.funnel.width
        expected = 0.05 * (1.0 - target / theta)
        assert fls.compute_alpha_min(base) == pytest.approx(expected)

    def test_membership_gates_the_ftype_route(self):
        """A steep descent direction must not shrink alpha_min below the
        step the linear model needs to re-enter the funnel."""
        fls = make_funnel_search(theta0=1.0)  # width 1.5
        theta = 3.0
        base = make_base_point(ref_theta=theta, ref_dphi=-1e6)
        target = fls.funnel.beta * fls.funnel.width
        expected = 0.05 * (1.0 - target / theta)
        assert fls.compute_alpha_min(base) == pytest.approx(expected)

    def test_gamma_phi_route_inside_the_funnel(self):
        fls = make_funnel_search(theta0=1.0)  # width 1.5, theta inside
        base = make_base_point(ref_theta=1.0, ref_dphi=-1.0)
        expected = 0.05 * SHARED_OPTIONS["filter_gamma_phi"] * 1.0
        assert fls.compute_alpha_min(base) == pytest.approx(expected)

    def test_floor_is_respected(self):
        fls = make_funnel_search(theta0=1.0)
        base = make_base_point(ref_theta=1e-30, ref_dphi=-1.0)
        assert fls.compute_alpha_min(base) == fls.min_alpha
