"""Tests of the homotopy path."""

import pytest

from amigo import Homotopy, HomotopyOptions
from amigo.algorithm.homotopy import midpoint


class Point:
    """A solution at one value of the parameter."""

    def __init__(self, value, converged):
        self.value = value
        self.converged = converged


def within(ratio, floor=0.0):
    """A solve that converges from a start within a ratio of its value and above a floor, with the values it was called at."""
    calls = []

    def solve(value, previous):
        calls.append(value)
        start = 1.0 if previous is None else previous.value
        return Point(value, max(value / start, start / value) <= ratio and value >= floor)

    return solve, calls


def test_values_within_reach_are_solved_in_order_each_from_the_last():
    solve, calls = within(4.0)
    history = Homotopy().run([1.0, 0.3, 0.1], solve)
    assert calls == [1.0, 0.3, 0.1]
    assert history.last.value == 0.1
    assert history.reached(0.1)


def test_a_step_out_of_reach_is_halved_geometrically_until_it_converges():
    solve, calls = within(4.0)
    history = Homotopy().run([1.0, 0.01], solve)

    # The halfway value 0.1 is still out of reach from 1, so 10 ** -0.5 bridges it
    assert calls == pytest.approx([1.0, 0.01, 0.1, 10**-0.5, 0.1, 0.01, 10**-1.5, 0.01])
    assert history.reached(0.01)


def test_a_path_stops_when_a_step_still_fails_after_its_halvings():
    solve, calls = within(100.0, floor=0.5)
    history = Homotopy(HomotopyOptions(max_bisections=2)).run([1.0, 0.1], solve)
    assert calls == pytest.approx([1.0, 0.1, 10**-0.5, 10**-0.25, 10**-0.5, 10**-0.375, 10**-0.3125])
    assert history.last.value == pytest.approx(10**-0.25)
    assert not history.reached(0.1)


def test_a_first_value_that_fails_stops_the_path_with_nothing_converged():
    solve, calls = within(4.0, floor=2.0)
    history = Homotopy().run([1.0, 0.5], solve)
    assert calls == [1.0]
    assert history.last is None


def test_the_halfway_value_is_geometric_on_one_side_of_zero_and_arithmetic_across_it():
    assert midpoint(0.04, 0.01) == pytest.approx(0.02)
    assert midpoint(-4.0, -1.0) == pytest.approx(-2.0)
    assert midpoint(1.0, 0.0) == pytest.approx(0.5)
    assert midpoint(0.5, -0.5) == pytest.approx(0.0)
