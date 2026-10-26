"""Functional tests that every way a solve ends is reported with its own status."""

import os
import sys

import pytest

import amigo as am
from amigo.algorithm.convergence_check import (
    CONVERGED,
    DIVERGED,
    INVALID_NUMBER,
    MAX_ITERATIONS,
    STATUS_MESSAGES,
    TIME_LIMIT,
)


class Probe(am.Component):
    """Minimize a (x - 1)^2 - b x + c sqrt(x) over a free x with weights set as data."""

    def __init__(self):
        super().__init__()
        self.add_data("a", value=1.0)
        self.add_data("b", value=0.0)
        self.add_data("c", value=0.0)
        self.add_input("x", value=0.0, lower=-am.inf, upper=am.inf)
        self.add_objective("obj")

    def compute(self):
        x = self.inputs["x"]
        a, b, c = self.data["a"], self.data["b"], self.data["c"]
        self.objective["obj"] = a * (x - 1.0) ** 2 - b * x + c * am.sqrt(x)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    # Build in a temporary directory so the repository stays clean
    build_dir = tmp_path_factory.mktemp("exit_status")
    model = am.Model("exit_status_test")
    model.add_component("probe", 1, Probe())
    orig_dir = os.getcwd()
    os.chdir(build_dir)
    try:
        model.build_module()
    finally:
        os.chdir(orig_dir)
    sys.path.insert(0, str(build_dir))
    model.initialize()
    return model


def solve(model, weights, start=0.5, **options):
    # The square root term needs a positive start even when its weight is zero
    data = model.get_data_vector()
    for name, value in zip("abc", weights):
        data[f"probe.{name}"] = value
    model.get_initial_point()["probe.x"] = start
    return am.Optimizer(model).optimize(options)


def assert_status(data, status):
    assert data["status"] == status
    assert data["status_message"] == STATUS_MESSAGES[status]
    assert data["iterations"][-1]["status"] == status
    assert data["converged"] == (status == CONVERGED)


def test_convex_problem_converges(model):
    assert_status(solve(model, (1.0, 0.0, 0.0)), CONVERGED)


def test_iteration_limit_is_reported(model):
    assert_status(solve(model, (0.0, 1.0, 0.0), max_iterations=1), MAX_ITERATIONS)


def test_unbounded_objective_diverges(model):
    # A linear objective without curvature takes constant steps, so the test lowers the threshold
    data = solve(model, (0.0, 1.0, 0.0), diverging_iterates_tol=1e10)
    assert_status(data, DIVERGED)


def test_invalid_start_names_the_variable(model, capsys):
    data = solve(model, (0.0, 0.0, 1.0), start=-1.0)
    assert_status(data, INVALID_NUMBER)
    assert "probe.x[0]" in capsys.readouterr().out


def test_wall_time_limit_is_reported(model):
    assert_status(solve(model, (1.0, 0.0, 0.0), max_wall_time=0.0), TIME_LIMIT)
