"""Functional tests that the optional solver paths reach known solutions."""

import os
import sys

import numpy as np
import pytest

import amigo as am
from amigo.algorithm.convergence_check import CONVERGED
from amigo.algorithm.solvers.linear_solver import SINGULAR_MATRIX
from amigo.algorithm.solvers.mumps_solver import _load_mumps_library

HS71_OPTIMUM = 17.0140173


class HS71(am.Component):
    """Hock-Schittkowski problem 71 with its standard start."""

    def __init__(self):
        super().__init__()
        self.add_input("x1", value=1.0, lower=1.0, upper=5.0)
        self.add_input("x2", value=5.0, lower=1.0, upper=5.0)
        self.add_input("x3", value=5.0, lower=1.0, upper=5.0)
        self.add_input("x4", value=1.0, lower=1.0, upper=5.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=25.0, upper=float("inf"))
        self.add_constraint("c2", lower=40.0, upper=40.0)

    def compute(self):
        x1, x2, x3, x4 = (self.inputs[n] for n in ("x1", "x2", "x3", "x4"))
        self.objective["obj"] = x1 * x4 * (x1 + x2 + x3) + x3
        self.constraints["c1"] = x1 * x2 * x3 * x4
        self.constraints["c2"] = x1**2 + x2**2 + x3**2 + x4**2


class Quadratic(am.Component):
    """Unconstrained quadratic whose optimum depends on the data a and b."""

    def __init__(self):
        super().__init__()
        self.add_input("x1", value=0.0, lower=-10.0, upper=10.0)
        self.add_input("x2", value=0.0, lower=-10.0, upper=10.0)
        self.add_data("a", value=1.0)
        self.add_data("b", value=3.0)
        self.add_objective("obj")
        self.add_output("f")

    def compute(self):
        x1, x2 = self.inputs["x1"], self.inputs["x2"]
        a, b = self.data["a"], self.data["b"]
        self.objective["obj"] = x1**2 + x1 * x2 + x2**2 - a * x1 - b * x2

    def compute_output(self):
        x1, x2 = self.inputs["x1"], self.inputs["x2"]
        self.outputs["f"] = x1**2 + x2**2


class Square(am.Component):
    """Two equations in two unknowns with the root x = y = sqrt(2)."""

    def __init__(self):
        super().__init__()
        self.add_input("x", value=1.0, lower=-am.inf, upper=am.inf)
        self.add_input("y", value=1.0, lower=-am.inf, upper=am.inf)
        self.add_constraint("circle", lower=4.0, upper=4.0)
        self.add_constraint("diagonal", lower=0.0, upper=0.0)

    def compute(self):
        x, y = self.inputs["x"], self.inputs["y"]
        self.constraints["circle"] = x**2 + y**2
        self.constraints["diagonal"] = x - y


class Duplicate(am.Component):
    """Minimize x^2 + y^2 + z^2 with the equality x + y = 1 written twice."""

    def __init__(self):
        super().__init__()
        self.add_input("x", value=0.3, lower=-10.0, upper=10.0)
        self.add_input("y", value=0.1, lower=-10.0, upper=10.0)
        self.add_input("z", value=0.2, lower=-10.0, upper=10.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=1.0, upper=1.0)
        self.add_constraint("c2", lower=1.0, upper=1.0)

    def compute(self):
        x, y, z = self.inputs["x"], self.inputs["y"], self.inputs["z"]
        self.objective["obj"] = x**2 + y**2 + z**2
        self.constraints["c1"] = x + y
        self.constraints["c2"] = x + y


def build(tmp_path_factory, name, component):
    # Build in a temporary directory so the repository stays clean
    build_dir = tmp_path_factory.mktemp(name)
    model = am.Model(name)
    model.add_component("comp", 1, component)
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
def hs71(tmp_path_factory):
    return build(tmp_path_factory, "paths_hs71", HS71())


@pytest.fixture(scope="module")
def quadratic(tmp_path_factory):
    return build(tmp_path_factory, "paths_quadratic", Quadratic())


@pytest.fixture(scope="module")
def square(tmp_path_factory):
    return build(tmp_path_factory, "paths_square", Square())


@pytest.fixture(scope="module")
def duplicate(tmp_path_factory):
    return build(tmp_path_factory, "paths_duplicate", Duplicate())


def solve_with_mumps(model):
    try:
        _load_mumps_library()
    except ImportError:
        pytest.skip("MUMPS library not found")
    opt = am.Optimizer(model)
    return opt, opt.optimize({"solver": "mumps", "max_iterations": 100})


def solve_hs71(model, **options):
    start = model.get_initial_point()
    for name, value in zip(("x1", "x2", "x3", "x4"), (1.0, 5.0, 5.0, 1.0)):
        start[f"comp.{name}"] = value
    opt = am.Optimizer(model)
    return opt, opt.optimize({"max_iterations": 300, **options})


def final_objective(data):
    return data["iterations"][-1]["objective"]


@pytest.mark.parametrize("line_search", ["filter", "funnel", "merit"])
@pytest.mark.parametrize("barrier", ["monotone", "quality_function", "heuristic"])
def test_strategy_pair_reaches_the_optimum(hs71, barrier, line_search):
    _, data = solve_hs71(hs71, barrier_strategy=barrier, line_search=line_search)
    assert data["converged"], data["status_message"]
    assert final_objective(data) == pytest.approx(HS71_OPTIMUM, rel=1e-6)


def test_feasibility_presolve_reaches_the_optimum(hs71):
    _, data = solve_hs71(hs71, feasibility_presolve=True)
    assert data["converged"], data["status_message"]
    assert final_objective(data) == pytest.approx(HS71_OPTIMUM, rel=1e-6)


def test_warm_start_from_the_solution_is_quick(hs71):
    opt, cold = solve_hs71(hs71)
    assert cold["converged"]

    # The solution becomes the start and the bound duals carry over
    hs71.get_initial_point().get_vector().copy(opt.x)
    warm = opt.optimize({"max_iterations": 300, "warm_start": True})
    assert warm["converged"], warm["status_message"]
    assert final_objective(warm) == pytest.approx(HS71_OPTIMUM, rel=1e-6)
    assert len(warm["iterations"]) < len(cold["iterations"])


def test_guess_set_in_x_becomes_the_start_and_converges(hs71):
    solve_hs71(hs71)
    x = hs71.create_vector()
    x["comp.x2"] = 4.0
    opt = am.Optimizer(hs71, x)
    start = hs71.get_initial_point()
    assert np.ravel(start["comp.x2"])[0] == 4.0
    assert np.ravel(start["comp.x1"])[0] == 1.0

    data = opt.optimize({"max_iterations": 300})
    assert data["converged"], data["status_message"]
    assert final_objective(data) == pytest.approx(HS71_OPTIMUM, rel=1e-6)
    assert np.ravel(x["comp.x1"])[0] == pytest.approx(1.0, abs=1e-6)


def test_optimized_point_restarts_from_the_solution(hs71):
    opt, cold = solve_hs71(hs71)
    assert cold["converged"]
    solution = np.array(opt.x.get_array())

    # The wrapped solution counts as fully set and becomes the next start
    restart = am.Optimizer(hs71, opt.get_optimized_point())
    start = np.array(hs71.get_initial_point().get_vector().get_array())
    assert np.array_equal(start, solution)

    data = restart.optimize({"max_iterations": 300})
    assert data["converged"], data["status_message"]
    assert final_objective(data) == pytest.approx(HS71_OPTIMUM, rel=1e-6)


def test_blank_x_keeps_the_model_start(hs71):
    solve_hs71(hs71)
    am.Optimizer(hs71, hs71.create_vector())
    start = hs71.get_initial_point()
    assert np.ravel(start["comp.x2"])[0] == 5.0


def test_partial_bounds_change_only_the_entries_set(hs71):
    lower_before = np.array(hs71.get_lower().get_vector().get_array())
    upper_before = np.array(hs71.get_upper().get_vector().get_array())
    try:
        lower = hs71.create_vector()
        lower["comp.x1"] = 1.5
        opt = am.Optimizer(hs71, lower=lower)
        after = np.array(hs71.get_lower().get_vector().get_array())
        changed = np.flatnonzero(after != lower_before)
        assert changed.tolist() == np.ravel(hs71.get_indices("comp.x1")).tolist()
        assert after[changed[0]] == 1.5

        # A zeroed bound vector must not pin the entries it leaves unset
        data = opt.optimize({"max_iterations": 300})
        assert data["converged"], data["status_message"]
        assert np.ravel(opt.get_optimized_point()["comp.x1"])[0] >= 1.5 - 1e-8
    finally:
        hs71.get_lower().get_vector().get_array()[:] = lower_before
        hs71.get_upper().get_vector().get_array()[:] = upper_before


def test_square_system_converges_to_its_root(square):
    opt = am.Optimizer(square)
    data = opt.optimize({"max_iterations": 50})
    assert data["status"] == CONVERGED
    point = opt.get_optimized_point()
    assert np.ravel(point["comp.x"])[0] == pytest.approx(np.sqrt(2.0), rel=1e-8)
    assert np.ravel(point["comp.y"])[0] == pytest.approx(np.sqrt(2.0), rel=1e-8)


def test_mumps_regularizes_a_singular_kkt_matrix(duplicate):
    # The repeated constraint makes the unregularized KKT matrix singular
    _, data = solve_with_mumps(duplicate)
    assert data["converged"], data["status_message"]
    assert final_objective(data) == pytest.approx(0.5, rel=1e-8)


@pytest.mark.parametrize(
    "codes, flag, doublings",
    [([-9, -8], 0, 2), ([-10], SINGULAR_MATRIX, 0), ([-13], -13, 0)],
)
def test_mumps_errors_become_factorization_flags(hs71, monkeypatch, codes, flag, doublings):
    opt, data = solve_with_mumps(hs71)
    assert data["converged"], data["status_message"]
    solver = opt.solver
    call = solver._call_mumps
    pending = list(codes)

    # Each factorization call reports the next scripted error code
    def scripted_call():
        call()
        if solver._mumps.job == 2 and pending:
            solver._mumps.infog[0] = pending.pop(0)

    monkeypatch.setattr(solver, "_call_mumps", scripted_call)
    workspace = solver._mumps.icntl[13]
    assert solver.factor(opt.state.hessian, opt.state.diagonal) == flag
    assert solver._mumps.icntl[13] == workspace * 2**doublings


def test_post_optimality_derivatives_match_the_analytic_values(quadratic):
    opt = am.Optimizer(quadratic)
    assert opt.optimize()["converged"]
    dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(
        of="comp.f", wrt=["comp.a", "comp.b"]
    )

    # The optimum f* = x1^2 + x2^2 has the derivatives (10 a - 8 b) / 9 and (10 b - 8 a) / 9
    a, b = 1.0, 3.0
    assert dfdx[of_map["comp.f"], wrt_map["comp.a"]] == pytest.approx((10 * a - 8 * b) / 9, abs=1e-8)
    assert dfdx[of_map["comp.f"], wrt_map["comp.b"]] == pytest.approx((10 * b - 8 * a) / 9, abs=1e-8)
