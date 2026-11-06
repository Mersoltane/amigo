"""
Dual step contract
------------------

- The observer that counts a rise of the dual infeasibility beyond the allowance a step declares
"""

from types import SimpleNamespace

import numpy as np

from amigo.algorithm.default_options import get_default_options
from amigo.algorithm.interior_point.step_contract import StepContract


class Indices:
    """Index vector of the problem"""

    def __init__(self, values):
        self._array = np.array(values, dtype=np.intp)

    def get_array(self):
        return self._array


class StepVector:
    """Step vector held on the host"""

    def __init__(self, values):
        self._array = np.array(values, dtype=float)

    def copy_device_to_host(self):
        pass

    def get_array(self):
        return self._array


class Problem:
    """Problem with two primal entries and one constraint"""

    def get_primal_indices(self):
        return Indices([0, 1])

    def get_constraint_indices(self):
        return Indices([2])


class Step:
    """The Newton step of a state"""

    def __init__(self, values):
        self.solution = StepVector(values)

    def get_solution(self):
        return self.solution


class SettingEvaluator:
    """Evaluator whose residual sets the dual infeasibility to a value"""

    def __init__(self, value):
        self.value = value

    def evaluate_residual(self, state):
        state.dual_infeas = self.value


def make_contract():
    """Contract with the default options on the problem above"""
    return StepContract(get_default_options(), Problem())


def make_state(dual_infeas, step):
    """State with a dual infeasibility and a step"""
    return SimpleNamespace(dual_infeas=dual_infeas, step=Step(step))


def test_a_rise_inside_the_declared_allowance_keeps_the_contract():
    contract = make_contract()
    state = make_state(1.0, [0.5, 0.2, 3.0])

    contract.before_step(state)

    # delta_w of ten against a primal step of a half declares a rise of five
    contract.after_step(SettingEvaluator(4.0), state, 1.0, 10.0, 0.0)

    assert contract.breaches == 0


def test_a_rise_beyond_the_allowance_is_counted_for_the_log():
    contract = make_contract()
    state = make_state(1e-3, [0.1, 0.0, 0.2])

    contract.before_step(state)

    # The unperturbed step grants no rise at all, so the jump is a breach
    contract.after_step(SettingEvaluator(2.0), state, 1.0, 0.0, 0.0)

    info = {}
    contract.add_log_info(info)

    assert info["dual_viol"] is True
    assert info["dual_viol_total"] == 1


def test_a_converged_dual_never_breaches():
    contract = make_contract()
    state = make_state(1e-12, [0.1, 0.0, 0.2])

    contract.before_step(state)

    # A rise below the dual tolerance is quadratic remainder, not a breach
    contract.after_step(SettingEvaluator(1e-11), state, 1.0, 0.0, 0.0)

    assert contract.breaches == 0
