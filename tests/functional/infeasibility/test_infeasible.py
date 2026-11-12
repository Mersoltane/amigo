"""
Local infeasibility
-------------------

- An infeasible problem ends with the local infeasibility verdict
- The iteration and wall time limits of the solve hold inside its restoration
"""

import contextlib
import math
import sys

import pytest

import amigo as am
from amigo.algorithm.globalization import feasibility_restoration
from amigo.algorithm.interior_point.convergence_check import (
    LOCALLY_INFEASIBLE,
    MAX_ITERATIONS,
    TIME_LIMIT,
)


class Circle(am.Component):
    """Minimize x + y on the circle x^2 + y^2 = 4 inside a box set at solve time"""

    def __init__(self):
        super().__init__()

        self.add_input("x", value=0.5, lower=0.0, upper=2.0)
        self.add_input("y", value=0.5, lower=0.0, upper=2.0)

        self.add_objective("obj")
        self.add_constraint("circle", lower=4.0, upper=4.0)

        return

    def compute(self):
        x = self.inputs["x"]
        y = self.inputs["y"]

        self.objective["obj"] = x + y
        self.constraints["circle"] = x**2 + y**2

        return


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """The model built in a temporary directory so the repository stays clean"""
    build_dir = tmp_path_factory.mktemp("infeasible")

    model = am.Model("infeasible_test")
    model.add_component("circle", 1, Circle())

    with contextlib.chdir(build_dir):
        model.build_module()

    sys.path.insert(0, str(build_dir))
    model.initialize()

    return model


def solve(model, upper_bound, **options):
    """Solve inside the box from zero to an upper bound"""
    upper = model.get_upper()
    upper["circle.x"] = upper_bound
    upper["circle.y"] = upper_bound

    opts = {"max_iterations": 100}
    opts.update(options)

    return am.Optimizer(model).optimize(opts)


def test_infeasible_box_is_detected(model):
    # The box [0, 1]^2 keeps x^2 + y^2 at most 2
    data = solve(model, 1.0)

    assert not data["converged"]
    assert data["iterations"][-1]["status"] == LOCALLY_INFEASIBLE


def test_small_violation_is_detected(model):
    # The box keeps x^2 + y^2 at most 4 - 1e-5, far above the tolerance
    data = solve(model, math.sqrt((4.0 - 1e-5) / 2.0))

    assert not data["converged"]
    assert data["iterations"][-1]["status"] == LOCALLY_INFEASIBLE


def test_budget_ending_at_the_solution_keeps_the_verdict(model):
    # The iterations of the solve and of its restoration when nothing limits them
    free = solve(model, 1.0)["iterations"]
    taken = free[-1]["iteration"]
    restored = free[-1]["resto_iterations"]

    # The phase draws on the budget of the solve and runs out where it would stop as solved
    data = solve(model, 1.0, max_iterations=taken + restored)

    assert data["iterations"][-1]["resto_iterations"] == restored
    assert data["iterations"][-1]["status"] == LOCALLY_INFEASIBLE


def test_budget_ending_inside_the_restoration_is_the_iteration_limit(model):
    # The iteration at which the solve calls its restoration
    free = solve(model, 1.0)["iterations"]
    taken = free[-1]["iteration"]

    # One iteration is left for the phase, which cannot solve its problem in it
    data = solve(model, 1.0, max_iterations=taken + 1)

    assert data["iterations"][-1]["resto_iterations"] == 1
    assert data["iterations"][-1]["status"] == MAX_ITERATIONS


class LateClock:
    """Clock that reads past every deadline"""

    def perf_counter(self):
        return math.inf


def test_wall_time_ending_inside_the_restoration_is_the_time_limit(model, monkeypatch):
    # Only the clock of the phase is late, so the solve reaches its restoration
    monkeypatch.setattr(feasibility_restoration, "time", LateClock())

    data = solve(model, 1.0)

    assert data["iterations"][-1]["resto_iterations"] == 0
    assert data["iterations"][-1]["status"] == TIME_LIMIT


def test_feasible_box_converges(model):
    data = solve(model, 2.0)

    assert data["converged"]
