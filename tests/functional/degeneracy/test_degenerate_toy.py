"""A small saddle with nearly parallel constraints, the degeneracy that jams the large transfers.

Two equality rows collapse onto each other as eps falls, so the duals lose
uniqueness while the indefinite objective keeps the Hessian perturbation
busy, the cocktail behind the jams and the divergent spirals seen on the
transfer problems. The solver must converge across the whole eps dial in a
bounded number of iterations, so any change to the regularization law gets
its verdict here in seconds instead of a family run.
"""

import os
import sys

import numpy as np
import pytest

import amigo as am

MODULE = "degenerate_toy_test"

Q = [1.0, 1.0, -3.0, 1.0, -2.0, 1.0]
C = [1.0, -2.0, 0.5, 1.0, -1.0, 0.5]


class Saddle(am.Component):
    """An indefinite quadratic with two equality rows that eps pulls onto each other."""

    def __init__(self):
        super().__init__()
        self.add_data("eps", value=1.0)
        self.add_input("x", shape=6, lower=-5.0, upper=5.0)
        self.add_constraint("close", label="the pair of nearly parallel rows", shape=2)
        self.add_constraint("chain", label="the row that couples the rest")
        self.add_objective("obj")

    def compute(self):
        x = self.inputs["x"]
        eps = self.data["eps"]
        self.constraints["close"] = [
            x[0] + x[1] + x[2] - 1.0,
            x[0] + x[1] + x[2] + eps * x[3] - 1.0,
        ]
        self.constraints["chain"] = x[2] + x[3] + x[4] + x[5] - 1.0
        obj = 0
        for i in range(6):
            obj = obj + 0.5 * Q[i] * x[i] * x[i] + C[i] * x[i]
        self.objective["obj"] = obj


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    # Build the module in a temporary directory so the repository stays untouched
    tmp_path = tmp_path_factory.mktemp("degenerate_toy")
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        model = am.Model(MODULE)
        model.add_component("pt", 1, Saddle())
        model.build_module()
    finally:
        os.chdir(orig_dir)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))


def solve(eps, options=None):
    """One solve at a value of eps, giving its point and its iteration count."""
    model = am.Model(MODULE)
    model.add_component("pt", 1, Saddle())
    model.initialize()
    data = model.get_data_vector()
    data["pt.eps"] = eps
    data.get_vector().copy_host_to_device()
    x = model.create_vector()
    x["pt.x"] = np.full(6, 0.1)
    opt = am.Optimizer(model, x)
    result = opt.optimize(dict({"convergence_tolerance": 1e-8, "max_iterations": 200}, **(options or {})))
    return result, opt.get_optimized_point()


def test_the_solver_crosses_the_degeneracy_dial_in_bounded_iterations(built):
    worst = 0
    for eps in (1.0, 1e-2, 1e-4, 1e-6, 1e-8):
        result, x = solve(eps)
        iterations = len(result["iterations"])
        worst = max(worst, iterations)
        assert result["converged"], f"eps {eps:g} failed after {iterations} iterations"

        # The nearly parallel rows force x4 to zero and the feasible rows hold
        point = np.asarray(x["pt.x"]).ravel()
        assert abs(point[0] + point[1] + point[2] - 1.0) < 1e-6
        assert abs(eps * point[3]) < 1e-6
    assert worst <= 100, f"the dial took up to {worst} iterations"
