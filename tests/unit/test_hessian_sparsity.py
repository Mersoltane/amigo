"""
Hessian sparsity
----------------

- The structural Hessian pattern of a component Lagrangian
- The compressed recovery of the Hessian from a few directions
"""

import numpy as np
import pytest

import amigo as am
from amigo.modeling.hessian_sparsity import HessianSparsity, compress_hessian


def lagrangian_pattern(comp):
    """Hessian pattern of the Lagrangian of a component and the number of its inputs and multipliers"""
    comp._initialize_expressions()
    mult = comp.constraints.get_multipliers()

    # The inputs first and then the multipliers
    inputs = []

    for name in comp.inputs:
        inputs.append(comp.inputs[name])

    for name in mult:
        inputs.append(mult[name])

    sparsity = HessianSparsity(inputs)

    return sparsity.pattern(comp.lagrangian[0]), sparsity.ncomp


def recovery_error(pattern, n, seed=0):
    """Largest error recovering a random symmetric matrix with this pattern and the directions it took"""
    rng = np.random.default_rng(seed)

    # A random symmetric matrix on the pattern
    H = np.zeros((n, n))

    for pair in pattern:
        i = pair[0]
        j = pair[1]

        value = rng.standard_normal()
        H[i, j] = value
        H[j, i] = value

    compressed = compress_hessian(pattern, n)
    seeds = compressed[0]
    entries = compressed[1]

    # Every direction recovers the entries it holds
    R = np.zeros((n, n))

    for d in range(len(seeds)):
        p = np.zeros(n)
        p[seeds[d]] = 1.0
        h = H @ p

        for found in entries[d]:
            r = found[0]
            c = found[1]
            src = found[2]

            R[r, c] = h[src]

    return np.abs(R - H).max(), len(seeds)


class Defect(am.Component):
    """Linear in x with a bilinear time scaling, like a collocation defect"""

    def __init__(self):
        super().__init__()

        self.add_input("t")
        self.add_input("x", shape=3)
        self.add_input("f", shape=2)

        self.add_constraint("res", shape=2)

        return

    def compute(self):
        t = self.inputs["t"]
        x = self.inputs["x"]
        f = self.inputs["f"]

        self.constraints["res"] = [x[1] - x[0] - t * f[0], x[2] - x[1] - t * f[1]]

        return


class Nonlinear(am.Component):
    """A sine, a product, and a quotient of three inputs"""

    def __init__(self):
        super().__init__()

        self.add_input("a")
        self.add_input("b")
        self.add_input("c")

        self.add_objective("obj")

        return

    def compute(self):
        a = self.inputs["a"]
        b = self.inputs["b"]
        c = self.inputs["c"]

        self.objective["obj"] = am.sin(a) + b * c + c / b

        return


class Lookup(am.Component):
    """Interpolated tables read at an input and at a product"""

    def __init__(self):
        super().__init__()

        self.add_data("table", shape=4)

        self.add_input("a")
        self.add_input("b")
        self.add_input("c")

        self.add_objective("obj")

        return

    def compute(self):
        table = self.data["table"]

        a = self.inputs["a"]
        b = self.inputs["b"]
        c = self.inputs["c"]

        self.objective["obj"] = am.interp1d(a, table) * b + am.interp1d(b * c, table, dx=0.5)

        return


class Series(am.Component):
    """Fourier series read at an input and at a product"""

    def __init__(self):
        super().__init__()

        self.add_data("coef", shape=4)

        self.add_input("a")
        self.add_input("b")
        self.add_input("c")

        self.add_objective("obj")

        return

    def compute(self):
        coef = self.data["coef"]

        a = self.inputs["a"]
        b = self.inputs["b"]
        c = self.inputs["c"]

        self.objective["obj"] = am.fourier(a, coef) * b + am.fourier(b * c, coef, period=0.5)

        return


class Magnitude(am.Component):
    """The norm of a vector plus a scalar"""

    def __init__(self):
        super().__init__()

        self.add_input("v", shape=3)
        self.add_input("w")

        self.add_objective("obj")

        return

    def compute(self):
        v = self.inputs["v"]
        w = self.inputs["w"]

        self.objective["obj"] = am.norm(v) + w

        return


def upper_pairs(n):
    """Every pair of the upper triangle of an n by n matrix"""
    pairs = set()

    for i in range(n):
        for j in range(i, n):
            pairs.add((i, j))

    return pairs


def test_interpolation_adds_no_curvature_of_its_own():
    # a is 0, b is 1, and c is 2
    pattern = lagrangian_pattern(Lookup())[0]

    assert pattern == {(0, 1), (1, 2)}


def test_fourier_series_curves_in_its_argument():
    # a is 0, b is 1, and c is 2
    pattern = lagrangian_pattern(Series())[0]

    assert pattern == {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)}


def test_norm_couples_every_entry_of_its_vector():
    pattern = lagrangian_pattern(Magnitude())[0]

    assert pattern == upper_pairs(3)


def test_defect_pattern_keeps_only_the_multiplier_and_time_couplings():
    # t is 0, x is 1 to 3, f is 4 and 5, and the multipliers are 6 and 7
    result = lagrangian_pattern(Defect())
    pattern = result[0]
    n = result[1]

    assert n == 8
    assert pattern == {
        (0, 4),
        (0, 6),
        (1, 6),
        (2, 6),
        (4, 6),
        (0, 5),
        (0, 7),
        (2, 7),
        (3, 7),
        (5, 7),
    }


def test_nonlinear_rules():
    pattern = lagrangian_pattern(Nonlinear())[0]

    assert pattern == {(0, 0), (1, 1), (1, 2)}


def test_defect_needs_few_directions():
    result = lagrangian_pattern(Defect())
    pattern = result[0]
    n = result[1]

    recovery = recovery_error(pattern, n)
    error = recovery[0]
    ndirs = recovery[1]

    assert error == 0.0
    assert ndirs <= 3


@pytest.mark.parametrize("density", [0.02, 0.1, 0.3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_patterns_recover_exactly(density, seed):
    rng = np.random.default_rng(seed)
    n = 40

    # Each pair of the upper triangle is in the pattern with the given density
    pattern = set()

    for i in range(n):
        for j in range(i, n):
            if rng.random() < density:
                pattern.add((i, j))

    recovery = recovery_error(pattern, n, seed)
    error = recovery[0]
    ndirs = recovery[1]

    assert error == 0.0
    assert ndirs <= n


def test_arrow_pattern_uses_the_dense_column():
    n = 30

    # One variable couples to all others, the rest form a diagonal
    pattern = set()

    for j in range(n):
        pattern.add((0, j))
        pattern.add((j, j))

    recovery = recovery_error(pattern, n)
    error = recovery[0]
    ndirs = recovery[1]

    assert error == 0.0
    assert ndirs == 2


def test_dense_pattern_needs_every_direction():
    n = 12
    pattern = upper_pairs(n)

    recovery = recovery_error(pattern, n)
    error = recovery[0]
    ndirs = recovery[1]

    assert error == 0.0
    assert ndirs == n
