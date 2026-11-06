"""Compiled checks of the B-spline, its basis functions, and the inverse of a coordinate spline."""

import os
import sys

import numpy as np
import pytest
from scipy.interpolate import BSpline

import amigo as am

COEF = np.array([0.3, -1.2, 0.8, 2.5, -0.4, 1.1, 0.6, -0.9, 1.7, 0.2])
TABLE = np.array(
    [[0.5, -0.3, 1.0], [1.5, 0.7, -0.4], [-0.2, 2.0, 0.3], [0.9, -1.1, 0.8],
     [0.1, 0.4, -0.6], [1.2, -0.8, 0.5], [-0.5, 1.3, 0.9], [0.7, 0.2, -1.4]]
)
CX = np.array([0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.4, 2.9, 3.3, 3.6])
CX2 = np.array([[-1.0, 0.1], [-0.4, 0.2], [0.3, 0.35], [0.9, 0.5], [1.2, 0.8], [2.0, 0.9]])
TENSOR = np.array(
    [[0.4, -0.2, 1.1, 0.3, -0.7], [1.0, 0.6, -0.5, 0.8, 0.2], [-0.3, 1.4, 0.9, -1.0, 0.5],
     [0.7, -0.9, 0.2, 1.2, -0.4], [0.1, 0.8, -0.6, 0.4, 1.3]]
)

PIECES = ["spline", "spline_column", "basis", "inverse", "inverse_column", "clamped", "tensor"]


class Splines(am.Component):
    def __init__(self):
        super().__init__()
        self.add_data("coef", shape=len(COEF))
        self.add_data("table", shape=TABLE.shape)
        self.add_data("cx", shape=len(CX))
        self.add_data("cx2", shape=CX2.shape)
        self.add_data("grid", shape=TENSOR.shape)
        self.add_input("x", shape=3)
        self.add_input("s")
        self.add_input("y")
        self.add_objective("obj")
        for name in PIECES:
            self.add_output(name)

    def pieces(self):
        x, s, y = self.inputs["x"], self.inputs["s"], self.inputs["y"]
        tensor = 0.0
        for j in range(TENSOR.shape[1]):
            column = am.bspline(self.data["grid"], s, 4, column=j)
            tensor = tensor + am.bspline_basis(y, j, TENSOR.shape[1], 4) * column
        return {
            "spline": am.bspline(self.data["coef"], s, 4) * y,
            "spline_column": am.bspline(self.data["table"], s * y, 3, column=1),
            "basis": am.bspline_basis(s, 3, 10, 4) * y + am.bspline_basis(y, 0, 6, 3),
            "inverse": am.bspline_inverse(self.data["cx"], 0.2 + 3.0 * y, 4) * s,
            "inverse_column": am.bspline_inverse(self.data["cx2"], x[0], 3, column=1),
            "clamped": am.bspline_inverse(self.data["cx"], x[1] - 5.0, 4)
            + am.bspline_inverse(self.data["cx"], x[2] + 50.0, 4),
            "tensor": tensor,
        }

    def compute(self):
        pieces = self.pieces()
        total = pieces[PIECES[0]]
        for name in PIECES[1:]:
            total = total + pieces[name]
        self.objective["obj"] = total

    def compute_output(self):
        for name, value in self.pieces().items():
            self.outputs[name] = value


def knots(k, m):
    """Open uniform knots of m control points of order k on the unit interval."""
    return np.concatenate([np.zeros(k), np.linspace(0.0, 1.0, m - k + 2)[1:-1], np.ones(k)])


def spline(coef, k, t):
    return float(BSpline(knots(k, len(coef)), coef, k - 1)(t))


def basis(index, m, k, t):
    return spline(np.eye(m)[index], k, t)


def inverse(coef, k, x):
    """Root of the increasing coordinate spline by bisection, held at the ends outside its range."""
    if x <= coef[0]:
        return 0.0
    if x >= coef[-1]:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if spline(coef, k, mid) < x:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reference(x, s, y):
    tensor = sum(basis(j, 5, 4, y) * spline(TENSOR[:, j], 4, s) for j in range(5))
    return {
        "spline": spline(COEF, 4, s) * y,
        "spline_column": spline(TABLE[:, 1], 3, s * y),
        "basis": basis(3, 10, 4, s) * y + basis(0, 6, 3, y),
        "inverse": inverse(CX, 4, 0.2 + 3.0 * y) * s,
        "inverse_column": inverse(CX2[:, 1], 3, x[0]),
        "clamped": inverse(CX, 4, x[1] - 5.0) + inverse(CX, 4, x[2] + 50.0),
        "tensor": tensor,
    }


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("bspline")
    model = am.Model("bspline_test")
    model.add_component("ops", 1, Splines())

    # Build in a temporary directory so the repository CMakeLists.txt stays untouched
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        model.build_module()
    finally:
        os.chdir(orig_dir)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    model.initialize()
    data = model.get_data_vector()
    data["ops.coef"] = COEF
    data["ops.table"] = TABLE
    data["ops.cx"] = CX
    data["ops.cx2"] = CX2
    data["ops.grid"] = TENSOR
    data.get_vector().copy_host_to_device()
    return model


def point(model, x, s, y):
    vec = model.create_vector()
    vec["ops.x"] = x
    vec["ops.s"] = s
    vec["ops.y"] = y
    return vec


def gradient(model, x, s, y):
    g = model.create_vector()
    model.eval_gradient(point(model, x, s, y), g)
    return np.concatenate([g["ops.x"].ravel(), g["ops.s"].ravel(), g["ops.y"].ravel()])


def state_indices(model):
    names = ["ops.x", "ops.s", "ops.y"]
    return np.concatenate([np.asarray(model.get_indices(n)).ravel() for n in names])


def hessian(model, x, s, y):
    mat = model.create_matrix()
    model.eval_hessian(point(model, x, s, y), mat)
    idx = state_indices(model)
    return am.tocsr(mat).toarray()[np.ix_(idx, idx)]


def total(v):
    return sum(reference(v[:3], v[3], v[4]).values())


@pytest.mark.parametrize(
    "x, s, y",
    [
        # Interior spans of every spline
        ([0.3, -0.4, 1.2], 0.37, 0.62),
        # The first and last spans and an argument on an interior knot
        ([0.85, 0.2, -0.3], 0.02, 0.5),
    ],
)
def test_values_match_scipy(model, x, s, y):
    out = model.create_output_vector()
    model.compute_output(point(model, x, s, y), out)
    expected = reference(np.array(x), s, y)
    for name in PIECES:
        assert abs(out[f"ops.{name}"] - expected[name]) < 1e-12, name


def test_gradient_matches_differences(model):
    v = np.array([0.3, -0.4, 1.2, 0.37, 0.62])
    h = 1e-6
    fd = np.array([(total(v + h * e) - total(v - h * e)) / (2 * h) for e in np.eye(5)])
    assert np.allclose(gradient(model, v[:3], v[3], v[4]), fd, rtol=1e-7, atol=1e-8)


def test_hessian_matches_differences_of_the_gradient(model):
    v = np.array([0.3, -0.4, 1.2, 0.37, 0.62])
    h = 1e-6
    H = hessian(model, v[:3], v[3], v[4])
    for j, e in enumerate(np.eye(5)):
        plus = gradient(model, *np.split(v + h * e, [3, 4]))
        minus = gradient(model, *np.split(v - h * e, [3, 4]))
        column = (plus - minus) / (2 * h)
        assert np.allclose(H[:, j], column.ravel(), rtol=1e-6, atol=1e-6), j


def test_clamped_inverse_has_zero_derivatives(model):
    g = gradient(model, np.array([0.3, -0.4, 1.2]), 0.37, 0.62)
    H = hessian(model, np.array([0.3, -0.4, 1.2]), 0.37, 0.62)
    assert g[1] == 0.0 and g[2] == 0.0
    assert np.all(H[1, :] == 0.0) and np.all(H[2, :] == 0.0)
