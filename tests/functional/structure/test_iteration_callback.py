"""The iteration callback sees every logged iterate with its count."""

import os
import sys

import numpy as np
import pytest

import amigo as am


class Bowl(am.Component):
    """One free input with a bounded one the equality ties to it."""

    def __init__(self):
        super().__init__()
        self.add_input("x", value=2.0)
        self.add_input("y", value=0.0, lower=-1.0)
        self.add_constraint("tie")
        self.add_objective("obj")

    def compute(self):
        x, y = self.inputs["x"], self.inputs["y"]
        self.constraints["tie"] = y - 0.5 * x
        self.objective["obj"] = (x - 1.0) ** 2 + y * y


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    build_dir = tmp_path_factory.mktemp("iteration_callback")
    model = am.Model("iteration_callback")
    model.add_component("bowl", 1, Bowl())
    orig_dir = os.getcwd()
    os.chdir(build_dir)
    try:
        model.build_module()
    finally:
        os.chdir(orig_dir)
    sys.path.insert(0, str(build_dir))
    model.initialize()
    return model


def test_the_callback_runs_once_per_logged_iterate(model):
    calls = []

    def record(k, x):
        x.copy_device_to_host()
        calls.append((k, float(np.ravel(x["bowl.x"])[0])))

    opt = am.Optimizer(model)
    data = opt.optimize({"max_iterations": 50, "iteration_callback": record})
    x = opt.get_optimized_point()

    assert data["converged"], data["status_message"]
    assert [k for k, _ in calls] == list(range(len(data["iterations"])))
    assert calls[0][1] == pytest.approx(2.0)
    assert calls[-1][1] == pytest.approx(float(np.ravel(x["bowl.x"])[0]))
