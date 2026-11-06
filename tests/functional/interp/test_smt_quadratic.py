"""
SMT surrogate constraint
------------------------

    minimize    x
    subject to  KRG(x) = y
                y >= 0.25
                0 <= x <= 2

- A one input KRG surrogate of y = x^2 wired into a model through ExternalSMTComponent
- The optimizer recovers x = 0.5 and y = 0.25
"""

import contextlib
import sys

import numpy as np
import pytest
from smt.surrogate_models import KRG

import amigo as am
from amigo.interp import ExternalSMTComponent


class Source(am.Component):
    """The input x and the surrogate value y"""

    def __init__(self):
        super().__init__()

        self.add_input("x")
        self.add_input("y")

        return


class SMTConstraintPlaceholder(am.Component):
    """The residual row that the external surrogate fills"""

    def __init__(self):
        super().__init__()

        self.add_constraint("res")

        return


class Objective(am.Component):
    """Minimize x"""

    def __init__(self):
        super().__init__()

        self.add_input("x")
        self.add_objective("obj")

        return

    def compute(self):
        x = self.inputs["x"]

        self.objective["obj"] = x

        return


@pytest.fixture(scope="module")
def trained_krg():
    """KRG surrogate trained on y = x^2 over [0, 2]"""
    x_train = np.linspace(0.0, 2.0, 20).reshape(-1, 1)
    y_train = x_train**2

    sm = KRG(theta0=[1e-2], print_global=False)
    sm.set_training_values(x_train, y_train)
    sm.train()

    return sm


@pytest.fixture(scope="module", params=["amigo"])
def opt_result(request, trained_krg, tmp_path_factory):
    """Build and solve the model once, giving the solution vector and the solve data"""
    solver = request.param
    sm = trained_krg

    smt_ext = ExternalSMTComponent(num_points=1, num_inputs=1, smt_model=sm)

    model = am.Model("smt_quadratic_test")

    model.add_component("src", 1, Source())
    model.add_component("smt_con", 1, SMTConstraintPlaceholder())
    model.add_component("obj", 1, Objective())

    # Link the input to the objective and the surrogate to its residual
    model.link("src.x", "obj.x")
    model.add_external_component(
        "smt",
        smt_ext,
        inputs=["src.x", "src.y"],
        constraints=["smt_con.res"],
    )

    # The start sits on the surrogate at x = 1
    values = model.get_meta_view("value")
    lower = model.get_meta_view("lower")
    upper = model.get_meta_view("upper")

    values["src.x"] = 1.0
    values["src.y"] = float(sm.predict_values(np.array([[1.0]]))[0, 0])

    # The bounds on x and y
    lower["src.x"] = 0.0
    upper["src.x"] = 2.0
    lower["src.y"] = 0.25
    upper["src.y"] = float("inf")

    # Build in a temporary directory so the repository stays clean
    build_dir = tmp_path_factory.mktemp("smt_quadratic")

    with contextlib.chdir(build_dir):
        model.build_module()

    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))

    model.initialize()

    opt_options = {
        "solver": solver,
        "max_iterations": 200,
        "initial_barrier_param": 1.0,
        "convergence_tolerance": 1e-5,
        "max_line_search_iterations": 10,
        "init_least_squares_multipliers": False,
    }

    xvec = model.create_vector()
    opt = am.Optimizer(model, xvec)
    data = opt.optimize(opt_options)

    return xvec, data


def test_surrogate_accuracy(trained_krg):
    """KRG interpolates y = x^2 to machine precision at the training points"""
    sm = trained_krg

    x_check = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
    pred = sm.predict_values(x_check).flatten()
    expected = x_check.flatten() ** 2

    np.testing.assert_allclose(pred, expected, atol=1e-6)


def test_converged(opt_result):
    """The optimizer reports convergence"""
    data = opt_result[1]

    assert data["converged"], "Optimizer did not converge"


def test_optimal_x(opt_result):
    """The optimal x is 0.5, where the bound y = 0.25 is active"""
    xvec = opt_result[0]
    x_opt = xvec["src.x"][0]

    assert abs(x_opt - 0.5) < 1e-4, f"x* = {x_opt:.6f}, expected 0.5"


def test_optimal_y(opt_result):
    """The optimal y is 0.25 on its active bound"""
    xvec = opt_result[0]
    y_opt = xvec["src.y"][0]

    assert abs(y_opt - 0.25) < 1e-4, f"y* = {y_opt:.6f}, expected 0.25"


def test_surrogate_constraint_satisfied(opt_result, trained_krg):
    """The surrogate residual vanishes at the solution"""
    xvec = opt_result[0]
    x_opt = xvec["src.x"][0]
    y_opt = xvec["src.y"][0]

    pred = trained_krg.predict_values(np.array([[x_opt]]))[0, 0]
    residual = abs(pred - y_opt)

    assert residual < 1e-6, f"Surrogate residual = {residual:.2e}, expected < 1e-6"
