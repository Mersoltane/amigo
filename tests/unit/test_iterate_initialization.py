"""
Start of the iterate
--------------------

- The slack a cold start gives an inequality takes its value in the units of the row
- The pair a cold start gives an elastic row closes the row in the units of the row
"""

from types import SimpleNamespace

import numpy as np
import pytest

from amigo.algorithm.initialization.iterate_initialization import IterateCenterer


class HostVector:
    """Vector that lives on the host alone"""

    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def get_array(self):
        return self.values

    def fill_at(self, indices, value):
        self.values[indices] = value

    def copy_device_to_host(self):
        pass

    def copy_host_to_device(self):
        pass


class RowProblem:
    """Problem with one row at the last entry, one weight, and a scale on its rows"""

    def __init__(self, row, weight, scale):
        self.row = row
        self.data = HostVector([weight])
        self.scale = scale

    def create_constraint_vector(self):
        return None

    def get_constraint_indices(self):
        return np.array([self.row])

    def get_data_vector(self):
        return self.data

    def get_var_scale(self):
        return self.scale


class RowState:
    """State that holds the point and the gradient of the row"""

    def __init__(self, size, mu):
        self.mu = mu
        self.current = HostVector(np.zeros(size))
        self.gradient = HostVector(np.zeros(size))

    def get_current_point(self):
        return self.current

    def invalidate(self):
        pass


class QuietOptimizer:
    """Optimizer whose dual start leaves the point alone"""

    def initialize_duals(self, mu, point):
        pass


class SlackEvaluator:
    """Evaluator of the row c - s in the scale of the solve, with the slack at entry 0 and the row at entry 1"""

    def __init__(self, value, row_scale):
        self.value = value
        self.row_scale = row_scale

    def evaluate_gradient(self, state):
        x = state.current.get_array()
        row = self.value - x[0]
        state.gradient.get_array()[1] = self.row_scale * row


def start_slack(value, row_scale, scaled):
    """Slack of an inequality after the cold start"""
    model = SimpleNamespace(
        slack_indices=np.array([0]),
        ineq_constraint_indices=np.array([1]),
        elastic_row_indices=np.zeros(0, dtype=int),
    )

    # The scale of the solve is one on the slack and the row scale on the row
    scale = None
    if scaled:
        scale = HostVector([1.0, row_scale])

    problem = RowProblem(1, 1000.0, scale)
    state = RowState(2, 0.1)
    evaluator = SlackEvaluator(value, row_scale)

    centerer = IterateCenterer({}, model, problem, QuietOptimizer())
    centerer.center(evaluator, state)

    return state.current.get_array()[0]


def test_the_slack_takes_the_value_of_an_unscaled_inequality():
    assert start_slack(250.0, 1.0, scaled=False) == pytest.approx(250.0)


def test_the_slack_takes_the_value_of_a_scaled_inequality():
    assert start_slack(4.0e8, 1.6e-4, scaled=True) == pytest.approx(4.0e8)


class PairEvaluator:
    """Evaluator of the row c - p + n in the scale of the solve, with the pair at entries 0 and 1 and the row at entry 2"""

    def __init__(self, value, row_scale):
        self.value = value
        self.row_scale = row_scale

    def evaluate_gradient(self, state):
        x = state.current.get_array()
        row = self.value - x[0] + x[1]
        state.gradient.get_array()[2] = self.row_scale * row


def start_pair(value, row_scale, scaled):
    """Residual c - p + n of an elastic row after its start"""
    model = SimpleNamespace(
        elastic_row_indices=np.array([2]),
        elastic_p_indices=np.array([0]),
        elastic_n_indices=np.array([1]),
        elastic_weight_indices=np.array([0]),
    )

    # The scale of the solve is one on the pair and the row scale on the row
    scale = None
    if scaled:
        scale = HostVector([1.0, 1.0, row_scale])

    problem = RowProblem(2, 1000.0, scale)
    state = RowState(3, 0.1)
    evaluator = PairEvaluator(value, row_scale)

    centerer = IterateCenterer({}, model, problem, QuietOptimizer())
    centerer._start_elastic_parts(evaluator, state)

    x = state.current.get_array()

    return value - x[0] + x[1]


def test_the_pair_takes_the_violation_of_an_unscaled_row():
    assert start_pair(250.0, 1.0, scaled=False) == pytest.approx(0.0, abs=1e-9)
    assert start_pair(-250.0, 1.0, scaled=False) == pytest.approx(0.0, abs=1e-9)


def test_the_pair_takes_the_violation_of_a_scaled_row():
    assert start_pair(250.0, 0.01, scaled=True) == pytest.approx(0.0, abs=1e-9)
    assert start_pair(-250.0, 0.01, scaled=True) == pytest.approx(0.0, abs=1e-9)
