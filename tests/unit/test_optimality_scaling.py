"""
Optimality scaling
------------------

- The scaling of the optimality error reads the multipliers and the bound duals of the point
- The bound duals of the elastic parts stay out of it at any weight
"""

from types import SimpleNamespace

import numpy as np
import pytest

from amigo.algorithm.interior_point.evaluator import Evaluator


class HostVector:
    """Vector that lives on the host alone"""

    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def get_array(self):
        return self.values

    def get_values_at(self, indices, out):
        out.values = self.values[indices].copy()

    def copy_device_to_host(self):
        pass


class RowProblem:
    """Problem whose one constraint row is the last entry of the point"""

    def __init__(self, row):
        self.row = row

    def create_constraint_vector(self):
        return HostVector([0.0])

    def create_vector(self):
        return HostVector([])

    def get_constraint_indices(self):
        return np.array([self.row])

    def abssum(self, vec):
        return float(np.abs(vec.get_array()).sum())


class CountOptimizer:
    """Optimizer that reports the counts of the system"""

    def __init__(self, num_primal, num_constraints, num_bounds):
        self.num_primal = num_primal
        self.num_constraints = num_constraints
        self.num_bounds = num_bounds

    def get_kkt_element_counts(self):
        return self.num_primal, self.num_constraints, self.num_bounds


class DualPoint:
    """Point with its multipliers and its bound duals"""

    def __init__(self, solution, zl, zu):
        self.solution = HostVector(solution)
        self.zl = HostVector(zl)
        self.zu = HostVector(zu)

    def get_solution(self):
        return self.solution

    def get_zl(self):
        return self.zl

    def get_zu(self):
        return self.zu


def scaling(plain_dual, weight, leave_out):
    """Scalings of a point with two plain bounds and one elastic row of multiplier 50"""
    multiplier = 50.0

    # Two plain variables, the pair of the row, and the multiplier of the row
    solution = [0.0, 0.0, 0.0, 0.0, multiplier]
    zl = [plain_dual, plain_dual, weight - multiplier, weight + multiplier]
    zu = [0.0, 0.0, 0.0, 0.0]

    evaluator = Evaluator(RowProblem(4), CountOptimizer(4, 1, 4))
    if leave_out:
        evaluator.set_elastic_parts(np.array([2, 3]))

    state = SimpleNamespace(current=DualPoint(solution, zl, zu))

    return evaluator.compute_optimality_scaling(state)


def test_the_duals_of_the_parts_stay_out_of_the_scaling():
    factors = scaling(10.0, 1.0e6, leave_out=True)

    # Two bound duals of 10 and a multiplier of 50 average below the threshold of 100
    assert factors[0] == pytest.approx(1.0)
    assert factors[1] == pytest.approx(1.0)


def test_the_scaling_is_the_same_at_every_weight():
    low = scaling(1.0e4, 1.0e3, leave_out=True)
    high = scaling(1.0e4, 1.0e8, leave_out=True)

    assert low[0] == pytest.approx(high[0])
    assert low[1] == pytest.approx(high[1])

    # The plain duals of 1e4 still scale both errors
    assert low[0] == pytest.approx((50.0 + 2.0e4) / 3.0 / 100.0)
    assert low[1] == pytest.approx(100.0)


def test_every_bound_dual_counts_when_no_part_is_named():
    factors = scaling(10.0, 1.0e6, leave_out=False)

    assert factors[0] == pytest.approx((50.0 + 20.0 + 2.0e6) / 5.0 / 100.0)
    assert factors[1] == pytest.approx((20.0 + 2.0e6) / 4.0 / 100.0)
