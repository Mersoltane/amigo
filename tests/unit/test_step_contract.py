"""Tests for the dual step contract observer."""

from types import SimpleNamespace

import numpy as np

from amigo.algorithm.default_options import get_default_options
from amigo.algorithm.step_contract import StepContract


class Indices:
    def __init__(self, values):
        self._array = np.array(values, dtype=np.intp)

    def get_array(self):
        return self._array


class StepVector:
    def __init__(self, values):
        self._array = np.array(values, dtype=float)

    def copy_device_to_host(self):
        pass

    def get_array(self):
        return self._array


def make_contract():
    problem = SimpleNamespace(
        get_primal_indices=lambda: Indices([0, 1]),
        get_constraint_indices=lambda: Indices([2]),
    )
    return StepContract(get_default_options(), problem)


def make_state(dual_infeas, step):
    solution = StepVector(step)
    return SimpleNamespace(
        dual_infeas=dual_infeas,
        step=SimpleNamespace(get_solution=lambda: solution),
    )


def evaluator_setting(value):
    def evaluate_residual(state):
        state.dual_infeas = value

    return SimpleNamespace(evaluate_residual=evaluate_residual)


def test_a_rise_inside_the_declared_allowance_keeps_the_contract():
    contract = make_contract()
    state = make_state(1.0, [0.5, 0.2, 3.0])
    contract.before_step(state)

    # delta_w of ten against a primal step of a half declares a rise of five
    contract.after_step(evaluator_setting(4.0), state, 1.0, 10.0, 0.0)
    assert contract.breaches == 0


def test_a_rise_beyond_the_allowance_is_counted_for_the_log():
    contract = make_contract()
    state = make_state(1e-3, [0.1, 0.0, 0.2])
    contract.before_step(state)

    # The unperturbed step grants no rise at all, so the jump is a breach
    contract.after_step(evaluator_setting(2.0), state, 1.0, 0.0, 0.0)
    info = {}
    contract.add_log_info(info)
    assert info["dual_viol"] is True
    assert info["dual_viol_total"] == 1


def test_a_converged_dual_never_breaches():
    contract = make_contract()
    state = make_state(1e-12, [0.1, 0.0, 0.2])
    contract.before_step(state)

    # A rise below the dual tolerance is quadratic remainder, not a breach
    contract.after_step(evaluator_setting(1e-11), state, 1.0, 0.0, 0.0)
    assert contract.breaches == 0
