"""Compiled checks of the norm, table interpolation, Fourier series, binary functions with a number, and a long chain."""

import os
import sys

import numpy as np
import pytest

import amigo as am

# A chain this long must be cut into temporaries by the code generator
N_CHAIN = 60

TABLE = np.array([0.3, -1.2, 0.8, 2.5, -0.4, 1.1])
TABLE2 = np.array([[0.5, -0.3], [1.5, 0.7], [-0.2, 2.0], [0.9, -1.1], [0.1, 0.4]])

PIECES = ["norm", "max", "min", "atan2", "periodic", "column", "series", "series_column", "chain"]


class Operations(am.Component):
    def __init__(self):
        super().__init__()
        self.add_data("table", shape=len(TABLE))
        self.add_data("table2", shape=TABLE2.shape)
        self.add_input("x", shape=3)
        self.add_input("s")
        self.add_input("y")
        self.add_objective("obj")
        for name in PIECES:
            self.add_output(name)

    def pieces(self):
        x, s, y = self.inputs["x"], self.inputs["s"], self.inputs["y"]
        table, table2 = self.data["table"], self.data["table2"]
        chain = x[0] * s
        for k in range(1, N_CHAIN):
            chain = chain + (0.01 * k) * x[k % 3] * s
        return {
            "norm": am.norm(x),
            "max": am.max2(0.0, y) * s,
            "min": am.min2(y, 0.5),
            "atan2": am.atan2(0.3, s) + am.atan2(y, 2.0),
            "periodic": am.interp1d(s, table, dx=0.2, periodic=True) * y,
            "column": am.interp1d(y, table2, column=1, x0=-0.5, dx=0.25),
            "series": am.fourier(s, table, period=0.7) * y,
            "series_column": am.fourier(s * y, table2, column=0, period=2.0),
            "chain": y * chain,
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


def linear(table, u, periodic):
    """Linear interpolation at grid coordinate u, wrapped or extended past the ends."""
    n = len(table)
    k = np.floor(u)
    if periodic:
        lo = int(k) % n
        hi = (lo + 1) % n
    else:
        k = min(max(k, 0), n - 2)
        lo, hi = int(k), int(k) + 1
    return table[lo] + (u - k) * (table[hi] - table[lo])


def series(coef, x, period):
    """Fourier series with an even count of coefficients ending at the Nyquist cosine."""
    w = 2.0 * np.pi / period
    n = len(coef)
    value = coef[0]
    for k in range(1, (n + 1) // 2):
        value += coef[2 * k - 1] * np.cos(k * w * x) + coef[2 * k] * np.sin(k * w * x)
    if n % 2 == 0:
        value += coef[n - 1] * np.cos(n // 2 * w * x)
    return value


def reference(x, s, y):
    chain = x[0] * s + sum(0.01 * k * x[k % 3] * s for k in range(1, N_CHAIN))
    return {
        "norm": np.linalg.norm(x),
        "max": max(0.0, y) * s,
        "min": min(y, 0.5),
        "atan2": np.arctan2(0.3, s) + np.arctan2(y, 2.0),
        "periodic": linear(TABLE, s / 0.2, True) * y,
        "column": linear(TABLE2[:, 1], (y + 0.5) / 0.25, False),
        "series": series(TABLE, s, 0.7) * y,
        "series_column": series(TABLE2[:, 0], s * y, 2.0),
        "chain": y * chain,
    }


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("operations")
    model = am.Model("operations_test")
    model.add_component("ops", 1, Operations())

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
    data["ops.table"] = TABLE
    data["ops.table2"] = TABLE2
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


def hessian(model, x, s, y):
    mat = model.create_matrix()
    model.eval_hessian(point(model, x, s, y), mat)
    names = ["ops.x", "ops.s", "ops.y"]
    idx = np.concatenate([np.asarray(model.get_indices(n)).ravel() for n in names])
    return am.tocsr(mat).toarray()[np.ix_(idx, idx)]


def total(v):
    return sum(reference(v[:3], v[3], v[4]).values())


@pytest.mark.parametrize(
    "x, s, y",
    [
        # Inside the periodic table and past the end of the column table
        ([0.3, -0.4, 1.2], 0.37, 0.62),
        # The periodic argument wrapped past one period
        ([-0.7, 0.2, 0.5], 1.37, -0.3),
    ],
)
def test_values_match_numpy(model, x, s, y):
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


def test_norm_at_the_origin_has_finite_zero_derivatives(model):
    zero = np.zeros(3)
    g = gradient(model, zero, 0.37, 0.62)
    H = hessian(model, zero, 0.37, 0.62)
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(H))

    # At the origin only the chain depends on x
    weights = np.zeros(3)
    weights[0] = 1.0
    for k in range(1, N_CHAIN):
        weights[k % 3] += 0.01 * k
    assert np.allclose(g[:3], 0.62 * 0.37 * weights, rtol=1e-12, atol=1e-14)
