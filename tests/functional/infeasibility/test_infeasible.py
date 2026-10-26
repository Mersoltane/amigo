"""Functional test that an infeasible problem ends with the local infeasibility verdict."""

import math
import os
import sys

import pytest

import amigo as am
from amigo.algorithm.convergence_check import LOCALLY_INFEASIBLE


class Circle(am.Component):
    """Minimize x + y on the circle x^2 + y^2 = 4 inside a box set at solve time."""

    def __init__(self):
        super().__init__()
        self.add_input("x", value=0.5, lower=0.0, upper=2.0)
        self.add_input("y", value=0.5, lower=0.0, upper=2.0)
        self.add_objective("obj")
        self.add_constraint("circle", lower=4.0, upper=4.0)

    def compute(self):
        x = self.inputs["x"]
        y = self.inputs["y"]
        self.objective["obj"] = x + y
        self.constraints["circle"] = x**2 + y**2


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    # Build in a temporary directory so the repository stays clean
    build_dir = tmp_path_factory.mktemp("infeasible")
    model = am.Model("infeasible_test")
    model.add_component("circle", 1, Circle())
    orig_dir = os.getcwd()
    os.chdir(build_dir)
    try:
        model.build_module()
    finally:
        os.chdir(orig_dir)
    sys.path.insert(0, str(build_dir))
    model.initialize()
    return model


def solve(model, upper_bound, **options):
    upper = model.get_upper()
    upper["circle.x"] = upper_bound
    upper["circle.y"] = upper_bound
    return am.Optimizer(model).optimize({"max_iterations": 100, **options})


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
    # The phase runs out of iterations at the point where it would stop as solved
    budget = solve(model, 1.0)["iterations"][-1]["resto_iterations"]
    data = solve(model, 1.0, resto_max_iterations=budget)
    assert data["iterations"][-1]["status"] == LOCALLY_INFEASIBLE


def test_feasible_box_converges(model):
    data = solve(model, 2.0)
    assert data["converged"]
