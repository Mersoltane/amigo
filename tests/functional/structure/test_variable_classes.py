"""The defined, free, and shifted variables of a model read from its equality structure."""

import os
import sys

import numpy as np
import pytest

import amigo as am
from amigo.algorithm.initialization import classify_variables, hessian_perturbation_rows


class Chain(am.Component):
    """Two free inputs, three variables the equalities define, and a capped inequality.

    The row def_y defines y from x, def_z defines z from y with curvature, and
    def_v defines v from w linearly. The objective curves in x alone, so x is
    the one free variable that carries curvature, while w only feeds a linear
    row and sits on its bound.
    """

    def __init__(self):
        super().__init__()
        self.add_input("x", value=0.5)
        self.add_input("w", value=0.5, lower=0.0)
        self.add_input("y", value=1.0)
        self.add_input("z", value=1.0)
        self.add_input("v", value=1.0)
        self.add_constraint("def_y")
        self.add_constraint("def_z")
        self.add_constraint("def_v")
        self.add_constraint("cap", lower=-float("inf"), upper=3.0)
        self.add_objective("obj")

    def compute(self):
        x, w, y, z, v = (self.inputs[name] for name in ("x", "w", "y", "z", "v"))
        self.constraints["def_y"] = y - 2.0 * x - 1.0
        self.constraints["def_z"] = z - y * y
        self.constraints["def_v"] = v - 2.0 * w
        self.constraints["cap"] = z
        self.objective["obj"] = x * x + z + w


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    build_dir = tmp_path_factory.mktemp("variable_classes")
    model = am.Model("variable_classes")
    model.add_component("chain", 1, Chain())
    orig_dir = os.getcwd()
    os.chdir(build_dir)
    try:
        model.build_module()
    finally:
        os.chdir(orig_dir)
    sys.path.insert(0, str(build_dir))
    model.initialize()
    return model


def index(model, name):
    return int(np.ravel(model.get_indices(name))[0])


def test_the_equalities_define_their_unit_variables_and_leave_the_rest_free(model):
    defined, free, shifted = classify_variables(model)
    x, w, y, z, v = (index(model, f"chain.{name}") for name in ("x", "w", "y", "z", "v"))
    slack = int(np.ravel(model.slack_indices)[0])

    assert set(defined.tolist()) == {y, z, v, slack}
    assert set(free.tolist()) == {x, w}

    # Only x reaches curvature, directly in the objective and through y into the row of z
    assert shifted.tolist() == [x]
    assert hessian_perturbation_rows(model).tolist() == [x]


@pytest.mark.parametrize("rows", ["free", "named"])
def test_the_free_shift_solves_the_chain(model, rows):
    named = hessian_perturbation_rows(model).tolist() if rows == "named" else "free"
    data = am.Optimizer(model).optimize({"hessian_perturbation_indices": named, "max_iterations": 100})
    assert data["converged"], data["status_message"]

    # The minimum of x^2 + (2x + 1)^2 at x = -0.4 with w on its bound
    assert data["iterations"][-1]["objective"] == pytest.approx(0.2, abs=1e-6)
