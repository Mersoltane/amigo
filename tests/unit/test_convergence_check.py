"""Tests for the divergence and square system checks of the convergence test."""

from types import SimpleNamespace

import pytest

from amigo.algorithm.convergence_check import (
    CONTINUE,
    CONVERGED,
    DIVERGED,
    INVALID_NUMBER,
    STALLED,
    ConvergenceCheck,
)
from amigo.algorithm.default_options import get_default_options


class StubEvaluator:
    """Evaluator whose primal iterate has a fixed largest magnitude."""

    def __init__(self, x_max):
        self.x_max = x_max

    def evaluate_residual(self, state):
        pass

    def primal_maxabs(self, vec):
        return self.x_max


def make_state():
    """State far from convergence so only the divergence check can stop the loop."""
    return SimpleNamespace(
        comm_rank=0,
        iter=1,
        dual_infeas=1.0,
        primal_infeas=1.0,
        complementarity=1.0,
        kkt_error=1.0,
        objective_value=0.0,
        residual_norm=1.0,
        current=SimpleNamespace(get_solution=lambda: None),
    )


def make_check(**overrides):
    return ConvergenceCheck(get_default_options(overrides), None, None)


def test_large_primal_iterate_diverges():
    check = make_check(diverging_iterates_tol=1e6)
    assert check.test_convergence(StubEvaluator(1e7), make_state()) == DIVERGED


def test_zero_tolerance_disables_the_iterate_check():
    check = make_check(diverging_iterates_tol=0.0)
    assert check.test_convergence(StubEvaluator(1e30), make_state()) == CONTINUE


@pytest.mark.parametrize("field", ["kkt_error", "objective_value"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_point_is_an_invalid_number(field, value):
    state = make_state()
    setattr(state, field, value)
    assert make_check().test_convergence(StubEvaluator(1.0), state) == INVALID_NUMBER


def make_feasible_state():
    """Feasible state whose dual and complementarity errors are still large."""
    state = make_state()
    state.primal_infeas = 1e-10
    state.constr_viol = 1e-10
    return state


def test_square_system_converges_once_feasible():
    check = make_check()
    check.square = True
    assert check.test_convergence(StubEvaluator(1.0), make_feasible_state()) == CONVERGED


def test_feasibility_alone_does_not_end_a_free_problem():
    check = make_check()
    assert check.test_convergence(StubEvaluator(1.0), make_feasible_state()) == CONTINUE


def statuses(check, errors):
    """Status of the convergence test at each iteration of a sequence of errors."""
    state = make_state()
    codes = []
    for iteration, error in enumerate(errors):
        state.iter, state.kkt_error = iteration, error
        codes.append(check.test_convergence(StubEvaluator(1.0), state))
    return codes


def test_an_error_that_stops_halving_stalls_after_the_stall_iterations():
    # The error halves twice and then creeps, so three iterations after the last halving the solve stalls
    check = make_check(stall_iterations=3, stall_factor=0.5)
    assert statuses(check, [1.0, 0.5, 0.25, 0.24, 0.23, 0.22]) == [CONTINUE] * 5 + [STALLED]


def test_an_error_that_keeps_halving_never_stalls():
    check = make_check(stall_iterations=2, stall_factor=0.5)
    assert statuses(check, [1.0 / 2**k for k in range(8)]) == [CONTINUE] * 8


def test_the_stall_test_is_off_by_default():
    assert statuses(make_check(), [0.5] * 20) == [CONTINUE] * 20
