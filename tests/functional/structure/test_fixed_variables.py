"""
Fixed variables
---------------

- A variable held at equal bounds stays put whenever the bounds become equal
"""

import contextlib
import sys

import numpy as np
import pytest

import amigo as am


class Chain(am.Component):
    """One free input the objective curves in and two variables the equalities define from it"""

    def __init__(self):
        super().__init__()

        self.add_input("x", value=0.5)
        self.add_input("y", value=1.0)
        self.add_input("z", value=1.0)

        self.add_constraint("def_y")
        self.add_constraint("def_z")
        self.add_constraint("cap", lower=-float("inf"), upper=3.0)

        self.add_objective("obj")

        return

    def compute(self):
        x = self.inputs["x"]
        y = self.inputs["y"]
        z = self.inputs["z"]

        self.constraints["def_y"] = y - 2.0 * x - 1.0
        self.constraints["def_z"] = z - y * y
        self.constraints["cap"] = z

        self.objective["obj"] = x * x + z

        return


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """The model built in a temporary directory so the repository stays clean"""
    build_dir = tmp_path_factory.mktemp("fixed_variables")

    model = am.Model("fixed_variables")
    model.add_component("chain", 1, Chain())

    with contextlib.chdir(build_dir):
        model.build_module()

    sys.path.insert(0, str(build_dir))
    model.initialize()

    return model


def value(x, name):
    """Value of a scalar variable"""
    return float(np.ravel(x[name])[0])


def bounds(model, lower_x, upper_x):
    """Lower and upper bound vectors with bounds on x alone"""
    lower = model.create_vector()
    upper = model.create_vector()

    lower["chain.x"] = lower_x
    upper["chain.x"] = upper_x

    return lower, upper


def test_a_matrix_created_before_the_bounds_meet_still_holds_the_variable(model):
    # The Hessian pattern exists before the solve fixes x, as a simulation of the start creates it
    model.create_matrix()

    limits = bounds(model, 0.3, 0.3)

    opt = am.Optimizer(model, lower=limits[0], upper=limits[1])
    data = opt.optimize({"max_iterations": 100})
    x = opt.get_optimized_point()

    assert data["converged"], data["status_message"]
    assert value(x, "chain.x") == pytest.approx(0.3, abs=1e-12)
    assert value(x, "chain.y") == pytest.approx(1.6, abs=1e-6)


def test_the_variable_is_free_again_once_its_bounds_part(model):
    # Solve once with x fixed
    limits = bounds(model, 0.3, 0.3)

    am.Optimizer(model, lower=limits[0], upper=limits[1]).optimize(
        {"max_iterations": 100}
    )

    # Then solve with x free
    limits = bounds(model, -am.inf, am.inf)

    opt = am.Optimizer(model, lower=limits[0], upper=limits[1])
    data = opt.optimize({"max_iterations": 100})
    x = opt.get_optimized_point()

    # The minimum of x^2 + (2x + 1)^2 sits at x = -0.4
    assert data["converged"], data["status_message"]
    assert value(x, "chain.x") == pytest.approx(-0.4, abs=1e-6)
    assert data["iterations"][-1]["objective"] == pytest.approx(0.2, abs=1e-6)
