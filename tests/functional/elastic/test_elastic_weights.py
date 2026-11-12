"""The elastic parts a model adds to the constraints it marks and the weight the solve raises."""

import os
import sys

import numpy as np
import pytest

import amigo as am


class Relaxed(am.Component):
    """Minimize (x - 2)^2 under x <= 1 with the cap marked elastic.

    The multiplier of the cap at its solution is 2, so a weight of 1 leaves
    the part at 0.5 and the exact penalty needs a weight above 2. The start
    violates the cap by 2.
    """

    def __init__(self):
        super().__init__()
        self.add_input("x", value=3.0)
        self.add_constraint("cap", lower=-float("inf"), upper=1.0, elastic=True)
        self.add_objective("obj")

    def compute(self):
        x = self.inputs["x"]
        self.constraints["cap"] = x
        self.objective["obj"] = (x - 2.0) ** 2


class Blocked(am.Component):
    """Minimize x^2 on 0 <= x <= 1 under x = 2 with the row marked elastic.

    No point closes the row, so its parts keep a sum of at least 1 at every
    weight.
    """

    def __init__(self):
        super().__init__()
        self.add_input("x", value=0.5, lower=0.0, upper=1.0)
        self.add_constraint("pin", elastic=True)
        self.add_objective("obj")

    def compute(self):
        x = self.inputs["x"]
        self.constraints["pin"] = x - 2.0
        self.objective["obj"] = x * x


def build_model(tmp_path_factory, name, comp_name, comp):
    """Built and initialized model of one component"""
    build_dir = tmp_path_factory.mktemp(name)
    model = am.Model(name)
    model.add_component(comp_name, 1, comp)
    orig_dir = os.getcwd()
    os.chdir(build_dir)
    try:
        model.build_module()
    finally:
        os.chdir(orig_dir)
    sys.path.insert(0, str(build_dir))
    model.initialize()
    return model


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return build_model(tmp_path_factory, "elastic_weights", "relaxed", Relaxed())


@pytest.fixture(scope="module")
def blocked(tmp_path_factory):
    return build_model(tmp_path_factory, "elastic_blocked", "blocked", Blocked())


def test_only_a_constraint_can_be_elastic():
    comp = am.Component()
    with pytest.raises(ValueError):
        comp.add_input("p", elastic=True)
    with pytest.raises(ValueError):
        comp.add_data("rho", elastic=True)


def test_the_model_adds_a_pair_of_parts_and_a_weight_to_the_elastic_row(model):
    parts, weights = model.get_elastic()
    assert parts.size == 2 and weights.size == 1
    assert (
        model.elastic_row_indices.tolist()
        == np.ravel(model.get_indices("relaxed.cap")).tolist()
    )

    # The parts are bounded below by zero and nothing else
    lower = np.asarray(model.get_lower().get_vector().get_array())
    upper = np.asarray(model.get_upper().get_vector().get_array())
    assert lower[parts].tolist() == [0.0, 0.0]
    assert np.all(np.isinf(upper[parts]))


def test_the_start_puts_the_violation_in_the_part_and_the_solve_grows_the_weight_once(
    model,
):
    opt = am.Optimizer(model)
    data = opt.optimize(
        {"convergence_tolerance": 1e-8, "max_iterations": 200, "elastic_weight": 1.0}
    )
    assert data["converged"], data["status_message"]

    # The weight of 1 was raised to 10, above the multiplier of 2, and the parts vanished
    parts, weights = model.get_elastic()
    rho = np.asarray(model.get_data_vector().get_vector().get_array())[weights]
    assert rho.tolist() == pytest.approx([10.0])
    x = opt.get_optimized_point()
    assert float(np.ravel(x["relaxed.x"])[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.all(np.asarray(x.get_vector().get_array())[parts] < 1e-6)


def test_parts_left_at_the_last_weight_end_the_solve_locally_infeasible(blocked):
    opt = am.Optimizer(blocked)
    data = opt.optimize({"max_iterations": 300, "elastic_max_updates": 2})

    assert not data["converged"]
    assert data["status_message"] == "Converged to a locally infeasible point"

    # The weight of 1e3 was raised twice and the parts still carry the gap of 1
    parts, weights = blocked.get_elastic()
    rho = np.asarray(blocked.get_data_vector().get_vector().get_array())[weights]
    assert rho.tolist() == pytest.approx([1.0e5])
    x = opt.get_optimized_point()
    assert float(np.ravel(x["blocked.x"])[0]) == pytest.approx(1.0, abs=1e-5)
    left = np.asarray(x.get_vector().get_array())[parts]
    assert float(np.sum(left)) == pytest.approx(1.0, abs=1e-4)
