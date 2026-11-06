"""
Iteration callback
------------------

- The iteration callback sees every logged iterate with its count
"""

import contextlib
import sys

import numpy as np
import pytest

import amigo as am


class Bowl(am.Component):
    """One free input with a bounded one the equality ties to it"""

    def __init__(self):
        super().__init__()

        self.add_input("x", value=2.0)
        self.add_input("y", value=0.0, lower=-1.0)

        self.add_constraint("tie")
        self.add_objective("obj")

        return

    def compute(self):
        x = self.inputs["x"]
        y = self.inputs["y"]

        self.constraints["tie"] = y - 0.5 * x
        self.objective["obj"] = (x - 1.0) ** 2 + y * y

        return


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """The model built in a temporary directory so the repository stays clean"""
    build_dir = tmp_path_factory.mktemp("iteration_callback")

    model = am.Model("iteration_callback")
    model.add_component("bowl", 1, Bowl())

    with contextlib.chdir(build_dir):
        model.build_module()

    sys.path.insert(0, str(build_dir))
    model.initialize()

    return model


class Recorder:
    """Iteration callback that records the count and the value of x at every logged iterate"""

    def __init__(self):
        self.calls = []

    def __call__(self, k, x):
        x.copy_device_to_host()

        self.calls.append((k, float(np.ravel(x["bowl.x"])[0])))


def test_the_callback_runs_once_per_logged_iterate(model):
    record = Recorder()

    opt = am.Optimizer(model)
    data = opt.optimize({"max_iterations": 50, "iteration_callback": record})
    x = opt.get_optimized_point()

    # The counts of the calls in order
    counts = []

    for call in record.calls:
        counts.append(call[0])

    assert data["converged"], data["status_message"]
    assert counts == list(range(len(data["iterations"])))
    assert record.calls[0][1] == pytest.approx(2.0)
    assert record.calls[-1][1] == pytest.approx(float(np.ravel(x["bowl.x"])[0]))
