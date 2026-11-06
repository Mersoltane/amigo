"""
Variable classes
----------------

- The defined, free, and shifted variables of a model read from its equality structure
"""

import contextlib
import sys

import numpy as np
import pytest

import amigo as am
from amigo.algorithm.initialization import classify_variables, hessian_perturbation_rows


class Chain(am.Component):
    """Two free inputs, three variables the equalities define, and a capped inequality"""

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

        return

    def compute(self):
        x = self.inputs["x"]
        w = self.inputs["w"]
        y = self.inputs["y"]
        z = self.inputs["z"]
        v = self.inputs["v"]

        # The equalities define y from x, z from y with curvature, and v from w linearly
        self.constraints["def_y"] = y - 2.0 * x - 1.0
        self.constraints["def_z"] = z - y * y
        self.constraints["def_v"] = v - 2.0 * w
        self.constraints["cap"] = z

        # The objective curves in x alone, while w only feeds a linear row and sits on its bound
        self.objective["obj"] = x * x + z + w

        return


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """The model built in a temporary directory so the repository stays clean"""
    build_dir = tmp_path_factory.mktemp("variable_classes")

    model = am.Model("variable_classes")
    model.add_component("chain", 1, Chain())

    with contextlib.chdir(build_dir):
        model.build_module()

    sys.path.insert(0, str(build_dir))
    model.initialize()

    return model


def index(model, name):
    """Global index of a scalar variable"""
    return int(np.ravel(model.get_indices(name))[0])


def test_the_equalities_define_their_unit_variables_and_leave_the_rest_free(model):
    classes = classify_variables(model)
    defined = classes[0]
    free = classes[1]
    shifted = classes[2]

    x = index(model, "chain.x")
    w = index(model, "chain.w")
    y = index(model, "chain.y")
    z = index(model, "chain.z")
    v = index(model, "chain.v")

    slack = int(np.ravel(model.slack_indices)[0])

    assert set(defined.tolist()) == {y, z, v, slack}
    assert set(free.tolist()) == {x, w}

    # Only x reaches curvature, directly in the objective and through y into the row of z
    assert shifted.tolist() == [x]
    assert hessian_perturbation_rows(model).tolist() == [x]


@pytest.mark.parametrize("rows", ["free", "named"])
def test_the_free_shift_solves_the_chain(model, rows):
    # The shift goes on the free rows by name or on the rows listed by index
    named = "free"
    if rows == "named":
        named = hessian_perturbation_rows(model).tolist()

    data = am.Optimizer(model).optimize(
        {"hessian_perturbation_indices": named, "max_iterations": 100}
    )

    assert data["converged"], data["status_message"]

    # The minimum of x^2 + (2x + 1)^2 at x = -0.4 with w on its bound
    assert data["iterations"][-1]["objective"] == pytest.approx(0.2, abs=1e-6)
