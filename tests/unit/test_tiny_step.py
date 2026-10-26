"""Tests for the tiny step detection and the barrier decrease it forces."""

from types import SimpleNamespace

import numpy as np
import pytest

import amigo as am
from amigo.amigo import VectorInt
from amigo.algorithm.barrier_strategy import MonotoneBarrierStrategy
from amigo.algorithm.default_options import get_default_options
from amigo.algorithm.globalization.filter_line_search import FilterLineSearch


def index_vector(indices):
    vec = VectorInt(len(indices))
    vec.get_array()[:] = indices
    return vec


def make_line_search():
    """Line search over two primal entries and one constraint multiplier."""
    ls = FilterLineSearch.__new__(FilterLineSearch)
    ls.options = get_default_options()
    ls.problem = SimpleNamespace(
        get_primal_indices=lambda: index_vector([0, 1]),
        get_constraint_indices=lambda: index_vector([2]),
        maxabs=lambda vec: float(np.max(np.abs(vec.get_array()), initial=0.0)),
    )
    ls.step_ratio = am.Vector(3)
    ls.step_con = am.Vector(1)
    return ls


def make_state(x, dx, constr_viol=0.0):
    def solution_of(values):
        vec = am.Vector(len(values))
        vec.get_array()[:] = values
        return SimpleNamespace(get_solution=lambda: vec)

    return SimpleNamespace(
        current=solution_of(x), step=solution_of(dx), constr_viol=constr_viol
    )


@pytest.mark.parametrize(
    "dx, constr_viol, tiny",
    [
        ([1e-16, -1e-16, 1e-3], 0.0, True),
        ([1e-10, 0.0, 0.0], 0.0, False),
        ([1e-16, 0.0, 0.1], 0.0, False),
        ([1e-16, 0.0, 0.0], 1.0, False),
    ],
)
def test_tiny_step_needs_small_relative_steps_at_a_feasible_point(dx, constr_viol, tiny):
    state = make_state([10.0, -3.0, 5.0], dx, constr_viol)
    assert make_line_search()._tiny_step(state) == tiny


def test_step_sizes_separate_primal_and_multiplier_entries():
    state = make_state([10.0, -3.0, 0.5], [11.0, -8.0, 6.0])
    primal_size, multiplier_size = make_line_search()._step_sizes(state)

    # The primal steps are 1 and 2 relative to 1 + |x| and the multiplier step is 6
    assert primal_size == pytest.approx(2.0)
    assert multiplier_size == pytest.approx(6.0)


class StubEvaluator:
    """Evaluator whose barrier subproblem is never solved, so mu only moves when forced."""

    def evaluate_residual(self, state):
        pass

    def barrier_error(self, state, mu):
        return np.inf


def test_forced_decrease_lowers_mu_once():
    options = get_default_options()
    strategy = MonotoneBarrierStrategy(options, None, None)
    state = SimpleNamespace(mu=0.1, tau=0.99, obj_scale=1.0, invalidate=lambda **kw: None)

    assert not strategy.update_barrier(StubEvaluator(), state).new_barrier
    assert state.mu == 0.1

    strategy.force_decrease = True
    info = strategy.update_barrier(StubEvaluator(), state)
    assert info.new_barrier
    assert state.mu == pytest.approx(min(0.2 * 0.1, 0.1**1.5))
    assert not strategy.force_decrease

    assert not strategy.update_barrier(StubEvaluator(), state).new_barrier
