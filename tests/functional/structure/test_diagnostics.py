"""
Start diagnostics
-----------------

- The residual of every constraint group is read from its own rows in a model with an inequality
"""

import contextlib
import sys

import numpy as np
import pytest

import amigo as am


class Pair(am.Component):
    """Two inputs under one equality and one inequality"""

    def __init__(self):
        super().__init__()

        self.add_input("x", value=2.0)
        self.add_input("y", value=3.0)

        self.add_constraint("balance")
        self.add_constraint("cap", lower=-float("inf"), upper=10.0)

        self.add_objective("obj")

        return

    def compute(self):
        x = self.inputs["x"]
        y = self.inputs["y"]

        self.constraints["balance"] = y - x * x
        self.constraints["cap"] = x + y

        self.objective["obj"] = x * x + y * y

        return


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """The model built in a temporary directory so the repository stays clean"""
    build_dir = tmp_path_factory.mktemp("diagnostics")

    model = am.Model("diagnostics_pair")
    model.add_component("pair", 3, Pair())

    with contextlib.chdir(build_dir):
        model.build_module()

    sys.path.insert(0, str(build_dir))
    model.initialize()

    return model


def test_every_group_reads_its_own_rows_with_an_inequality_in_the_model(model):
    diagnostics = am.Diagnostics(model)
    residuals = diagnostics._eval_constraints(model.get_initial_point())

    # The rows of the slacks are the rows of the inequality, so only the two groups are named
    assert sorted(residuals) == ["pair.balance", "pair.cap"]

    # The equality reads y - x^2 at the start of every instance
    assert residuals["pair.balance"] == pytest.approx(np.full(3, -1.0))
    assert residuals["pair.cap"].shape == (3,)
