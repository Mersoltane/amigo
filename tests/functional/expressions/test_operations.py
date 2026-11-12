"""
Operations
----------

- Compiled checks of the norm, the inverse tangents, and binary functions with a number
- The tangent, the hyperbolic functions, and their inverses to second order
- A long chain that the code generator cuts into temporaries
- The gradient, the Hessian, and the output Jacobian against finite differences
"""

import contextlib
import sys

import numpy as np
import pytest

import amigo as am

# A chain this long must be cut into temporaries by the code generator
N_CHAIN = 60

PIECES = [
    "norm",
    "max",
    "min",
    "atan",
    "atan2",
    "atan2_vars",
    "tan",
    "tanh",
    "sinh_cosh",
    "asinh",
    "acosh",
    "atanh",
    "chain",
]


class Operations(am.Component):
    """Every operation as an output, with their sum as the objective"""

    def __init__(self):
        super().__init__()

        self.add_input("x", shape=3)
        self.add_input("s")
        self.add_input("y")

        self.add_objective("obj")

        for name in PIECES:
            self.add_output(name)

        return

    def pieces(self):
        """Every operation by name"""
        x = self.inputs["x"]
        s = self.inputs["s"]
        y = self.inputs["y"]

        # A long chain of products
        chain = x[0] * s

        for k in range(1, N_CHAIN):
            chain = chain + (0.01 * k) * x[k % 3] * s

        return {
            "norm": am.norm(x),
            "max": am.max2(0.0, y) * s,
            "min": am.min2(y, 0.5),
            "atan": am.atan(s * y),
            "atan2": am.atan2(0.3, s) + am.atan2(y, 2.0),
            "atan2_vars": am.atan2(y, s) * s,
            "tan": am.tan(0.5 * y) * s,
            "tanh": am.tanh(s * y),
            "sinh_cosh": am.sinh(s) * am.cosh(y),
            "asinh": am.asinh(s - y),
            "acosh": am.acosh(2.0 + s * s),
            "atanh": am.atanh(0.5 * y) * s,
            "chain": y * chain,
        }

    def compute(self):
        pieces = self.pieces()

        total = pieces[PIECES[0]]

        for name in PIECES[1:]:
            total = total + pieces[name]

        self.objective["obj"] = total

        return

    def compute_output(self):
        pieces = self.pieces()

        for name in PIECES:
            self.outputs[name] = pieces[name]

        return


def reference(x, s, y):
    """Every operation by name from numpy"""
    # The tail of the chain summed from zero
    tail = 0

    for k in range(1, N_CHAIN):
        tail = tail + 0.01 * k * x[k % 3] * s

    chain = x[0] * s + tail

    return {
        "norm": np.linalg.norm(x),
        "max": max(0.0, y) * s,
        "min": min(y, 0.5),
        "atan": np.arctan(s * y),
        "atan2": np.arctan2(0.3, s) + np.arctan2(y, 2.0),
        "atan2_vars": np.arctan2(y, s) * s,
        "tan": np.tan(0.5 * y) * s,
        "tanh": np.tanh(s * y),
        "sinh_cosh": np.sinh(s) * np.cosh(y),
        "asinh": np.arcsinh(s - y),
        "acosh": np.arccosh(2.0 + s * s),
        "atanh": np.arctanh(0.5 * y) * s,
        "chain": y * chain,
    }


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """The model built with data derivatives in a temporary directory"""
    tmp_path = tmp_path_factory.mktemp("operations")

    model = am.Model("operations_test")
    model.add_component("ops", 1, Operations())

    # Build in a temporary directory so the repository CMakeLists.txt stays untouched
    with contextlib.chdir(tmp_path):
        model.build_module(data_derivatives=True)

    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    model.initialize()

    return model


def point(model, x, s, y):
    """Design vector at x, s, and y"""
    vec = model.create_vector()

    vec["ops.x"] = x
    vec["ops.s"] = s
    vec["ops.y"] = y

    return vec


def gradient(model, x, s, y):
    """Gradient of the objective with respect to x, s, and y"""
    g = model.create_vector()
    model.eval_gradient(point(model, x, s, y), g)

    return np.concatenate([g["ops.x"].ravel(), g["ops.s"].ravel(), g["ops.y"].ravel()])


def state_indices(model):
    """Global indices of x, s, and y in order"""
    indices = []

    for name in ["ops.x", "ops.s", "ops.y"]:
        indices.append(np.asarray(model.get_indices(name)).ravel())

    return np.concatenate(indices)


def hessian(model, x, s, y):
    """Hessian of the objective with respect to x, s, and y"""
    mat = model.create_matrix()
    model.eval_hessian(point(model, x, s, y), mat)

    idx = state_indices(model)

    return am.tocsr(mat).toarray()[np.ix_(idx, idx)]


def outputs(model, v):
    """Every output in order at the point v of x, s, and y"""
    out = model.create_output_vector()
    model.compute_output(point(model, v[:3], v[3], v[4]), out)

    values = []

    for name in PIECES:
        values.append(np.ravel(out[f"ops.{name}"]))

    return np.concatenate(values)


def total(v):
    """Sum of every reference operation at the point v of x, s, and y"""
    pieces = reference(v[:3], v[3], v[4])

    value = 0

    for name in PIECES:
        value = value + pieces[name]

    return value


@pytest.mark.parametrize(
    "x, s, y",
    [
        # A positive y, so max2 takes y and min2 takes 0.5
        ([0.3, -0.4, 1.2], 0.37, 0.62),
        # A negative y, so max2 takes 0 and min2 takes y
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

    # Central differences of the reference total along every coordinate
    fd = np.zeros(5)

    for j in range(5):
        e = np.eye(5)[j]
        fd[j] = (total(v + h * e) - total(v - h * e)) / (2 * h)

    assert np.allclose(gradient(model, v[:3], v[3], v[4]), fd, rtol=1e-7, atol=1e-8)


def test_hessian_matches_differences_of_the_gradient(model):
    v = np.array([0.3, -0.4, 1.2, 0.37, 0.62])
    h = 1e-6

    H = hessian(model, v[:3], v[3], v[4])

    # Central differences of the compiled gradient along every coordinate
    for j in range(5):
        e = np.eye(5)[j]

        up = v + h * e
        down = v - h * e

        plus = gradient(model, up[:3], up[3], up[4])
        minus = gradient(model, down[:3], down[3], down[4])
        column = (plus - minus) / (2 * h)

        assert np.allclose(H[:, j], column.ravel(), rtol=1e-6, atol=1e-6), j


def test_output_jacobian_matches_differences(model):
    v = np.array([0.3, -0.4, 1.2, 0.37, 0.62])

    problem = model.get_problem()
    jac = problem.create_output_jacobian_wrt_input()
    problem.output_jacobian_wrt_input(point(model, v[:3], v[3], v[4]).get_vector(), jac)

    # The rows of the outputs and the columns of x, s, and y
    rows = []

    for name in PIECES:
        rows.append(np.ravel(model.get_indices(f"ops.{name}")))

    rows = np.concatenate(rows)
    J = am.tocsr(jac).toarray()[np.ix_(rows, state_indices(model))]

    # Central differences of the outputs along every coordinate
    h = 1e-6
    fd = []

    for j in range(5):
        e = np.eye(5)[j]
        fd.append((outputs(model, v + h * e) - outputs(model, v - h * e)) / (2 * h))

    assert np.allclose(J, np.column_stack(fd), rtol=1e-7, atol=1e-8)


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
