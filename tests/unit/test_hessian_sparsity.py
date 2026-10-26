"""Tests for the Hessian sparsity analysis and its compressed recovery."""

import numpy as np
import pytest

import amigo as am
from amigo.hessian_sparsity import HessianSparsity, compress_hessian


def lagrangian_pattern(comp):
    comp._initialize_expressions()
    mult = comp.constraints.get_multipliers()
    inputs = [comp.inputs[n] for n in comp.inputs] + [mult[n] for n in mult]
    sparsity = HessianSparsity(inputs)
    return sparsity.pattern(comp.lagrangian[0]), sparsity.ncomp


def recovery_error(pattern, n, seed=0):
    """Largest error recovering a random symmetric matrix with this pattern."""
    rng = np.random.default_rng(seed)
    H = np.zeros((n, n))
    for i, j in pattern:
        H[i, j] = H[j, i] = rng.standard_normal()

    seeds, entries = compress_hessian(pattern, n)
    R = np.zeros((n, n))
    for cols, found in zip(seeds, entries):
        p = np.zeros(n)
        p[cols] = 1.0
        h = H @ p
        for r, c, src in found:
            R[r, c] = h[src]
    return np.abs(R - H).max(), len(seeds)


class Defect(am.Component):
    """Linear in x with a bilinear time scaling, like a collocation defect."""

    def __init__(self):
        super().__init__()
        self.add_input("t")
        self.add_input("x", shape=3)
        self.add_input("f", shape=2)
        self.add_constraint("res", shape=2)

    def compute(self):
        t, x, f = self.inputs["t"], self.inputs["x"], self.inputs["f"]
        self.constraints["res"] = [x[1] - x[0] - t * f[0], x[2] - x[1] - t * f[1]]


class Nonlinear(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("a")
        self.add_input("b")
        self.add_input("c")
        self.add_objective("obj")

    def compute(self):
        a, b, c = self.inputs["a"], self.inputs["b"], self.inputs["c"]
        self.objective["obj"] = am.sin(a) + b * c + c / b


class Lookup(am.Component):
    """Interpolated tables read at an input and at a product."""

    def __init__(self):
        super().__init__()
        self.add_data("table", shape=4)
        self.add_input("a")
        self.add_input("b")
        self.add_input("c")
        self.add_objective("obj")

    def compute(self):
        a, b, c = self.inputs["a"], self.inputs["b"], self.inputs["c"]
        table = self.data["table"]
        self.objective["obj"] = am.interp1d(a, table) * b + am.interp1d(b * c, table, dx=0.5)


class Series(am.Component):
    """Fourier series read at an input and at a product."""

    def __init__(self):
        super().__init__()
        self.add_data("coef", shape=4)
        self.add_input("a")
        self.add_input("b")
        self.add_input("c")
        self.add_objective("obj")

    def compute(self):
        a, b, c = self.inputs["a"], self.inputs["b"], self.inputs["c"]
        coef = self.data["coef"]
        self.objective["obj"] = am.fourier(a, coef) * b + am.fourier(b * c, coef, period=0.5)


class Magnitude(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("v", shape=3)
        self.add_input("w")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = am.norm(self.inputs["v"]) + self.inputs["w"]


def test_interpolation_adds_no_curvature_of_its_own():
    # a is 0, b is 1, and c is 2
    pattern, _ = lagrangian_pattern(Lookup())
    assert pattern == {(0, 1), (1, 2)}


def test_fourier_series_curves_in_its_argument():
    # a is 0, b is 1, and c is 2
    pattern, _ = lagrangian_pattern(Series())
    assert pattern == {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)}


def test_norm_couples_every_entry_of_its_vector():
    pattern, _ = lagrangian_pattern(Magnitude())
    assert pattern == {(i, j) for i in range(3) for j in range(i, 3)}


def test_defect_pattern_keeps_only_the_multiplier_and_time_couplings():
    # t is 0, x is 1 to 3, f is 4 and 5, and the multipliers are 6 and 7
    pattern, n = lagrangian_pattern(Defect())
    assert n == 8
    assert pattern == {
        (0, 4), (0, 6), (1, 6), (2, 6), (4, 6),
        (0, 5), (0, 7), (2, 7), (3, 7), (5, 7),
    }


def test_nonlinear_rules():
    pattern, _ = lagrangian_pattern(Nonlinear())
    assert pattern == {(0, 0), (1, 1), (1, 2)}


def test_defect_needs_few_directions():
    pattern, n = lagrangian_pattern(Defect())
    error, ndirs = recovery_error(pattern, n)
    assert error == 0.0
    assert ndirs <= 3


@pytest.mark.parametrize("density", [0.02, 0.1, 0.3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_patterns_recover_exactly(density, seed):
    rng = np.random.default_rng(seed)
    n = 40
    pattern = {
        (i, j) for i in range(n) for j in range(i, n) if rng.random() < density
    }
    error, ndirs = recovery_error(pattern, n, seed)
    assert error == 0.0
    assert ndirs <= n


def test_arrow_pattern_uses_the_dense_column():
    # One variable couples to all others, the rest form a diagonal
    n = 30
    pattern = {(0, j) for j in range(n)} | {(j, j) for j in range(n)}
    error, ndirs = recovery_error(pattern, n)
    assert error == 0.0
    assert ndirs == 2


def test_dense_pattern_needs_every_direction():
    n = 12
    pattern = {(i, j) for i in range(n) for j in range(i, n)}
    error, ndirs = recovery_error(pattern, n)
    assert error == 0.0
    assert ndirs == n
