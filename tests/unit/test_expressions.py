"""
Expressions
-----------

- The zero, activity, cost, and C++ rules of every expression node
- The zero shortcuts of the Expr operators and the serialize and deserialize round trip
- The chunked sums, the depth limit, the binary functions with a number, the norm, and the tables
"""

import pkgutil

import numpy as np
import pytest

import amigo as am
import amigo.modeling.unary_operations as operations
from amigo.modeling.expressions import (
    _MAX_DEPTH,
    _SUM_CHUNK,
    BinaryNode,
    ConstNode,
    Expr,
    ExprBuilder,
    IndexNode,
    PassiveNode,
    UnaryNode,
    VarNode,
    _balanced_sum,
    _normalize_shape,
    _sum_terms,
)

# is_zero on each node type


class TestIsZero:
    def test_const_zero_int(self):
        assert ConstNode(value=0).is_zero() is True

    def test_const_zero_float(self):
        assert ConstNode(value=0.0).is_zero() is True

    def test_const_nonzero(self):
        assert ConstNode(value=1).is_zero() is False

    def test_var_is_never_zero(self):
        assert VarNode("x").is_zero() is False

    def test_unary_neg_propagates_zero(self):
        # A UnaryNode wraps an Expr, not a bare node
        zero_expr = Expr(ConstNode(value=0))
        node = UnaryNode("-", zero_expr)

        assert node.is_zero() is True

    def test_unary_neg_nonzero(self):
        nonzero_expr = Expr(VarNode("x"))
        node = UnaryNode("-", nonzero_expr)

        assert node.is_zero() is False


class TestUnsupportedUnary:
    @pytest.mark.parametrize("op", UnaryNode.unsupported_ops)
    def test_unsupported_op_raises_before_code_generation(self, op):
        with pytest.raises(NotImplementedError, match=op):
            UnaryNode(op, Expr(VarNode("x")))

    def test_unknown_op_names_the_op(self):
        with pytest.raises(ValueError, match="erf"):
            UnaryNode("erf", Expr(VarNode("x")))


# is_active on each node type


class TestIsActive:
    def test_const_is_never_active(self):
        assert ConstNode(value=1.0).is_active() is False

    def test_const_named_is_never_active(self):
        assert ConstNode(name="c").is_active() is False

    def test_var_active_true(self):
        assert VarNode("x", active=True).is_active() is True

    def test_var_active_false(self):
        assert VarNode("x", active=False).is_active() is False

    def test_passive_node_is_never_active_even_when_inner_is_active(self):
        active_expr = Expr(VarNode("x", active=True))
        node = PassiveNode(active_expr)

        assert node.is_active() is False

    def test_passive_node_is_never_active_when_inner_is_inactive(self):
        inactive_expr = Expr(VarNode("x", active=False))
        node = PassiveNode(inactive_expr)

        assert node.is_active() is False


# compute_cost on each node type


class TestComputeCost:
    def test_const_cost_is_zero(self):
        assert ConstNode(value=5.0).compute_cost() == 0

    def test_var_cost_is_zero(self):
        assert VarNode("x").compute_cost() == 0

    def test_unary_sqrt_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("sqrt", base_expr)

        assert node.compute_cost() == base_expr.compute_cost() + 20

    def test_unary_sin_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("sin", base_expr)

        assert node.compute_cost() == base_expr.compute_cost() + 100

    def test_unary_log_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("log", base_expr)

        assert node.compute_cost() == base_expr.compute_cost() + 200

    def test_unary_neg_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("-", base_expr)

        assert node.compute_cost() == base_expr.compute_cost() + 1

    def test_unary_fabs_cost(self):
        base_expr = Expr(VarNode("x"))
        node = UnaryNode("fabs", base_expr)

        assert node.compute_cost() == base_expr.compute_cost() + 1

    def test_binary_div_cost(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("/", a, b)

        assert node.compute_cost() == a.compute_cost() + b.compute_cost() + 10

    def test_binary_add_cost(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("+", a, b)

        assert node.compute_cost() == a.compute_cost() + b.compute_cost() + 1

    def test_compute_cost_accumulates_over_nested_tree(self):
        # The root costs 20 and the logarithm of it 200 more
        x = Expr(VarNode("x"))
        sqrt_node = UnaryNode("sqrt", x)
        sqrt_expr = Expr(sqrt_node)
        log_node = UnaryNode("log", sqrt_expr)

        assert log_node.compute_cost() == 220


# to_cpp strings for each node type


class TestToCpp:
    def test_const_named(self):
        assert ConstNode(name="c").to_cpp() == "c"

    def test_const_value(self):
        assert ConstNode(value=3.0).to_cpp() == "3.0"

    def test_var(self):
        assert VarNode("x").to_cpp() == "x"

    def test_index_1d(self):
        arr_expr = Expr(VarNode("arr", shape=(5,)))
        node = IndexNode(arr_expr, 2)

        assert node.to_cpp() == "arr[2]"

    def test_index_2d(self):
        arr_expr = Expr(VarNode("arr", shape=(3, 2)))
        node = IndexNode(arr_expr, (1, 0))

        assert node.to_cpp() == "arr(1, 0)"

    def test_passive_node(self):
        inner_expr = Expr(VarNode("x"))
        node = PassiveNode(inner_expr)

        assert node.to_cpp() == "A2D::get_data(x)"

    def test_unary_neg(self):
        expr = Expr(VarNode("x"))
        node = UnaryNode("-", expr)

        assert node.to_cpp() == "-(x)"

    def test_binary_add(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("+", a, b)

        assert node.to_cpp() == "(a + b)"

    def test_binary_pow(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("**", a, b)

        assert node.to_cpp() == "A2D::pow(a, b)"

    def test_binary_atan2(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("atan2", a, b)

        assert node.to_cpp() == "A2D::atan2(a, b)"

    def test_binary_min2(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("min2", a, b)

        assert node.to_cpp() == "A2D::min2(a, b)"

    def test_binary_max2(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        node = BinaryNode("max2", a, b)

        assert node.to_cpp() == "A2D::max2(a, b)"


# Zero propagation in the Expr operators


class TestExprArithmetic:
    """Zero shortcuts and operator overloads of the Expr wrapper"""

    @pytest.fixture(autouse=True)
    def _fixtures(self):
        self.zero = Expr(ConstNode(value=0.0))
        self.e = Expr(VarNode("x"))

    # Addition

    def test_zero_plus_e_returns_non_zero_operand(self):
        """Zero plus e is e itself with no binary node"""
        result = self.zero + self.e

        assert result.serialize() == self.e.serialize()

    def test_e_plus_zero_returns_non_zero_operand(self):
        """e plus zero is e itself"""
        result = self.e + self.zero

        assert result.serialize() == self.e.serialize()

    def test_nonzero_plus_nonzero_produces_binary_node(self):
        """The sum of two variables is a binary node"""
        e2 = Expr(VarNode("y"))

        result = self.e + e2

        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "+"

    # Multiplication

    def test_zero_times_e_is_zero(self):
        """Zero times e is zero"""
        result = self.zero * self.e

        assert result.is_zero() is True

    def test_e_times_zero_is_zero(self):
        """e times zero is zero"""
        result = self.e * self.zero

        assert result.is_zero() is True

    # Power

    def test_e_pow_zero_int_is_one(self):
        """e to the integer zero is the constant one"""
        result = self.e**0

        assert result.is_zero() is False
        assert result.to_cpp() == "1.0"

    # Negation

    def test_neg_e_wraps_unary_minus(self):
        """Negating e wraps it in a unary minus"""
        result = -self.e

        assert isinstance(result.node, UnaryNode)
        assert result.node.op == "-"

    # Subtraction

    def test_e_minus_zero_returns_non_zero_operand(self):
        """e minus zero is e itself"""
        result = self.e - self.zero

        assert result.serialize() == self.e.serialize()

    def test_zero_minus_e_wraps_unary_minus(self):
        """Zero minus e is a unary minus of e"""
        result = self.zero - self.e

        assert isinstance(result.node, UnaryNode)
        assert result.node.op == "-"
        assert result.node.expr.serialize() == self.e.serialize()

    # Division

    def test_zero_divided_by_d_is_zero(self):
        """Zero over d is zero"""
        d = Expr(VarNode("d"))

        result = self.zero / d

        assert result.is_zero() is True

    # Reflected addition

    def test_radd_int_produces_binary_node(self):
        """An integer plus e is a binary node with the constant on the left"""
        result = 1 + self.e

        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "+"
        assert isinstance(result.node.left.node, ConstNode)
        assert result.node.left.node.value == 1
        assert result.node.right.serialize() == self.e.serialize()

    # Reflected subtraction

    def test_rsub_float_produces_binary_node(self):
        """A float minus e is a binary node with the constant on the left"""
        result = 1.0 - self.e

        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "-"
        assert isinstance(result.node.left.node, ConstNode)
        assert result.node.left.node.value == 1.0
        assert result.node.right.serialize() == self.e.serialize()

    # Reflected multiplication

    def test_rmul_int_produces_binary_node(self):
        """An integer times e is a binary node with the constant on the left"""
        result = 2 * self.e

        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "*"
        assert isinstance(result.node.left.node, ConstNode)
        assert result.node.left.node.value == 2
        assert result.node.right.serialize() == self.e.serialize()

    # Reflected division

    def test_rtruediv_float_produces_binary_node(self):
        """A float over e is a binary node with the constant on the left"""
        result = 1.0 / self.e

        assert isinstance(result.node, BinaryNode)
        assert result.node.op == "/"
        assert isinstance(result.node.left.node, ConstNode)
        assert result.node.left.node.value == 1.0
        assert result.node.right.serialize() == self.e.serialize()

    def test_rtruediv_zero_numerator_is_zero(self):
        """Zero over e is zero"""
        result = 0.0 / self.e

        assert result.is_zero() is True


# Serialize and deserialize round trip


class TestExprRoundTrip:
    """The serialize and deserialize round trip of every node type"""

    def test_const_node_int_type_preserved(self):
        """An integer constant keeps its type"""
        original = Expr(ConstNode(value=5, type=int))
        deserialized = Expr.deserialize(original.serialize())

        assert isinstance(deserialized.node, ConstNode)
        assert deserialized.node.type is int
        assert deserialized.node.value == 5

    def test_var_node_active_false_preserved(self):
        """A passive variable stays passive"""
        original = Expr(VarNode("x", active=False))
        deserialized = Expr.deserialize(original.serialize())

        assert isinstance(deserialized.node, VarNode)
        assert deserialized.node.active is False
        assert deserialized.node.name == "x"

    def test_nested_expression_round_trip(self):
        """A nested expression serializes the same after a round trip"""
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        c = Expr(VarNode("c"))

        expr = (a + b) * c
        once = expr.serialize()
        twice = Expr.deserialize(once).serialize()

        assert twice == once

    def test_index_node_2d_tuple_preserved(self):
        """A two dimensional index keeps its tuple"""
        arr = Expr(VarNode("arr", shape=(3, 2)))
        original = Expr(IndexNode(arr, (1, 0)))
        deserialized = Expr.deserialize(original.serialize())

        assert isinstance(deserialized.node, IndexNode)
        assert deserialized.node.index == (1, 0)

    def test_unrecognized_op_returns_value_error(self):
        """An unknown op comes back as a ValueError instead of raising"""
        result = Expr.deserialize(("unknown_op", "foo", "bar"))

        assert isinstance(result, ValueError)


# Shape normalization and validation


class TestNormalizeShape:
    """_normalize_shape conversions and errors"""

    def test_none_returns_none(self):
        assert _normalize_shape(None) is None

    def test_int_returns_1d_tuple(self):
        assert _normalize_shape(3) == (3,)

    def test_list_returns_tuple(self):
        assert _normalize_shape([2, 3]) == (2, 3)

    def test_1d_tuple_unchanged(self):
        assert _normalize_shape((4,)) == (4,)

    def test_3d_tuple_raises_value_error(self):
        with pytest.raises(ValueError):
            _normalize_shape((1, 2, 3))

    def test_float_raises_type_error(self):
        with pytest.raises(TypeError):
            _normalize_shape(3.14)


# Long sums in the generated derivative code are built in chunks


def _names(n):
    """Variables t0 to t(n - 1)"""
    names = []

    for i in range(n):
        names.append(Expr(VarNode(f"t{i}")))

    return names


def cpp_of(exprs):
    """The C++ text of every expression in order"""
    texts = []

    for expr in exprs:
        texts.append(expr.to_cpp())

    return texts


def _lagrangian_lines(nterms, mode="hprod", passive=None):
    """Generated lines for a Lagrangian built term by term like a component builds it"""
    x = Expr(VarNode("x", shape=nterms))
    lam = Expr(VarNode("lam", shape=nterms))

    rhs = Expr(ConstNode(value=0.0))

    for i in range(nterms):
        rhs = rhs + x[i] * lam[i]

    if passive is not None:
        rhs = rhs + passive

    lhs = Expr(VarNode("lagrangian__"))
    builder = ExprBuilder([], [], [x, lam], [], rhs=rhs, lhs=lhs)

    return builder.get_cpp_lines(mode=mode, template_name="R__")


class TestSplitSum:
    def test_sum_terms_returns_the_terms_in_order(self):
        names = _names(4)
        a = names[0]
        b = names[1]
        c = names[2]
        d = names[3]

        terms = _sum_terms(((a + b) + c) + d)

        assert cpp_of(terms) == ["t0", "t1", "t2", "t3"]

    def test_sum_terms_keeps_a_difference_whole(self):
        names = _names(3)
        a = names[0]
        b = names[1]
        c = names[2]

        terms = _sum_terms(a + (b - c))

        assert cpp_of(terms) == ["t0", "(t1 - t2)"]

    def test_balanced_sum_pairs_the_terms(self):
        total = _balanced_sum(_names(5))

        assert total.to_cpp() == "(((t0 + t1) + (t2 + t3)) + t4)"
        assert cpp_of(_sum_terms(total)) == cpp_of(_names(5))

    def test_short_sum_keeps_one_eval(self):
        active = _lagrangian_lines(_SUM_CHUNK)[2]

        assert len(active) == 1
        assert active[0].startswith("A2D::Eval(")
        assert "call_noinline" not in active[0]

    def test_long_sum_builds_each_chunk_in_its_own_frame(self):
        nterms = 3 * _SUM_CHUNK + 1

        lines = _lagrangian_lines(nterms)
        decl = lines[0]
        active = lines[2]

        chunks = active[:-1]
        total = active[-1]

        assert len(chunks) == 4

        for line in chunks:
            assert line.startswith("amigo::call_noinline([&]() {")

        for g in range(4):
            assert f"A2D::A2DObj<R__> lagrangian{g}__" in decl
            assert f"lagrangian{g}__" in total

        # Every term lands in exactly one chunk
        count = 0

        for line in chunks:
            count += line.count("x[")

        assert count == nterms
        assert "x[" not in total
        assert total.startswith("A2D::Eval(") and total.endswith(", lagrangian__)")

    def test_eval_mode_is_never_split(self):
        active = _lagrangian_lines(4 * _SUM_CHUNK, mode="eval")[2]

        assert len(active) == 1
        assert active[0].startswith("lagrangian__ = ")

    def test_passive_terms_join_the_final_sum(self):
        data = Expr(VarNode("d", active=False))

        active = _lagrangian_lines(2 * _SUM_CHUNK + 1, passive=data * data)[2]

        assert "(d * d)" in active[-1]

        for line in active[:-1]:
            assert "(d * d)" not in line


# Binary functions with a number, the norm, and table interpolation

FUNCTIONS = {"max2": am.max2, "min2": am.min2, "atan2": am.atan2}


class TestNumberArguments:
    @pytest.mark.parametrize("op", ["max2", "min2", "atan2"])
    def test_number_first_becomes_a_float_constant(self, op):
        a = Expr(VarNode("a"))

        assert FUNCTIONS[op](0, a).to_cpp() == f"A2D::{op}(0.0, a)"

    @pytest.mark.parametrize("op", ["max2", "min2", "atan2"])
    def test_number_second_becomes_a_float_constant(self, op):
        a = Expr(VarNode("a"))

        assert FUNCTIONS[op](a, 2).to_cpp() == f"A2D::{op}(a, 2.0)"

    @pytest.mark.parametrize("op", ["max2", "min2", "atan2"])
    def test_two_numbers_raise(self, op):
        with pytest.raises(ValueError):
            FUNCTIONS[op](1.0, 2.0)

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            am.max2("a", Expr(VarNode("a")))

    def test_code_generation_with_a_number(self):
        x = Expr(VarNode("x"))
        lhs = Expr(VarNode("lagrangian__"))

        builder = ExprBuilder([], [], [x], [], rhs=am.max2(0.0, x) * x, lhs=lhs)
        active = builder.get_cpp_lines(mode="hprod", template_name="R__")[2]

        assert "A2D::max2(0.0, x)" in active[-1]


class TestNorm:
    def test_norm_of_scalars_is_the_root_of_a_balanced_sum(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))
        c = Expr(VarNode("c"))

        expected = "A2D::norm_sqrt((((a * a) + (b * b)) + (c * c)))"

        assert am.norm(a, b, c).to_cpp() == expected

    def test_norm_of_a_vector_reads_its_entries(self):
        v = Expr(VarNode("v", shape=3))

        assert am.norm(v).to_cpp() == am.norm(v[0], v[1], v[2]).to_cpp()

    def test_norm_of_a_list(self):
        a = Expr(VarNode("a"))
        b = Expr(VarNode("b"))

        assert am.norm([a, b]).to_cpp() == am.norm(a, b).to_cpp()

    def test_norm_needs_an_argument(self):
        with pytest.raises(ValueError):
            am.norm()

    def test_norm_sqrt_costs_like_sqrt(self):
        a = Expr(VarNode("a"))

        assert (
            UnaryNode("norm_sqrt", a).compute_cost()
            == UnaryNode("sqrt", a).compute_cost()
        )


def make_table(shape=5):
    """A passive table of values"""
    return Expr(VarNode("tab", shape=shape, active=False))


class TestInterp:
    def test_vector_table(self):
        s = Expr(VarNode("s"))

        e = am.interp1d(s, make_table(), dx=0.25, periodic=True)

        assert e.to_cpp() == "A2D::interp(tab, s, 0.0, 0.25, true)"

    def test_matrix_column(self):
        s = Expr(VarNode("s"))

        e = am.interp1d(s, make_table((5, 3)), column=2, x0=-1.0)

        assert e.to_cpp() == "A2D::interp(tab, 2, s, -1.0, 1.0, false)"

    def test_active_table_raises(self):
        with pytest.raises(TypeError):
            am.interp1d(Expr(VarNode("s")), Expr(VarNode("tab", shape=5)))

    @pytest.mark.parametrize(
        "shape, kwargs",
        [
            ((5, 3), {}),
            ((5, 3), {"column": 3}),
            (5, {"column": 0}),
            (5, {"dx": 0.0}),
            (1, {}),
        ],
    )
    def test_invalid_tables_raise(self, shape, kwargs):
        with pytest.raises(ValueError):
            am.interp1d(Expr(VarNode("s")), make_table(shape), **kwargs)

    def test_activity_follows_the_argument(self):
        assert am.interp1d(Expr(VarNode("s")), make_table()).is_active()
        assert not am.interp1d(Expr(ConstNode(value=0.3)), make_table()).is_active()

    def test_round_trip(self):
        e = am.interp1d(
            Expr(VarNode("s")), make_table((5, 2)), column=1, x0=0.5, dx=0.1
        )

        assert Expr.deserialize(e.serialize()).to_cpp() == e.to_cpp()

    def test_code_generation(self):
        s = Expr(VarNode("s"))
        table = make_table()
        lhs = Expr(VarNode("lagrangian__"))

        rhs = am.interp1d(s, table, dx=0.5) * s
        builder = ExprBuilder([], [table], [s], [], rhs=rhs, lhs=lhs)

        lines = builder.get_cpp_lines(mode="hprod", template_name="R__")
        decl = lines[0]
        active = lines[2]

        assert "A2D::VecRef<R__, 5> tab" in decl[0]
        assert "A2D::interp(tab, s, 0.0, 0.5, false)" in active[-1]

    def test_no_operation_shares_its_name_with_a_subpackage(self):
        # Importing a subpackage binds its name on amigo over any operation of that name
        shared = []

        for module in pkgutil.iter_modules(am.__path__):
            if callable(operations.__dict__.get(module.name)):
                shared.append(module.name)

        assert shared == []


class TestFourier:
    def test_vector_table(self):
        s = Expr(VarNode("s"))

        assert am.fourier(s, make_table()).to_cpp() == "A2D::fourier(tab, s, 1.0)"

    def test_matrix_column(self):
        s = Expr(VarNode("s"))

        e = am.fourier(s, make_table((5, 3)), column=2, period=0.5)

        assert e.to_cpp() == "A2D::fourier(tab, 2, s, 0.5)"

    def test_active_table_raises(self):
        with pytest.raises(TypeError):
            am.fourier(Expr(VarNode("s")), Expr(VarNode("tab", shape=5)))

    @pytest.mark.parametrize(
        "shape, kwargs",
        [
            ((5, 3), {}),
            ((5, 3), {"column": 3}),
            (5, {"column": 0}),
            (5, {"period": 0.0}),
        ],
    )
    def test_invalid_tables_raise(self, shape, kwargs):
        with pytest.raises(ValueError):
            am.fourier(Expr(VarNode("s")), make_table(shape), **kwargs)

    def test_activity_follows_the_argument(self):
        assert am.fourier(Expr(VarNode("s")), make_table()).is_active()
        assert not am.fourier(Expr(ConstNode(value=0.3)), make_table()).is_active()

    def test_round_trip(self):
        e = am.fourier(Expr(VarNode("s")), make_table((5, 2)), column=1, period=2.0)

        assert Expr.deserialize(e.serialize()).to_cpp() == e.to_cpp()

    def test_code_generation(self):
        s = Expr(VarNode("s"))
        table = make_table()
        lhs = Expr(VarNode("lagrangian__"))

        rhs = am.fourier(s, table, period=0.5) * s
        builder = ExprBuilder([], [table], [s], [], rhs=rhs, lhs=lhs)

        lines = builder.get_cpp_lines(mode="hprod", template_name="R__")
        decl = lines[0]
        active = lines[2]

        assert "A2D::VecRef<R__, 5> tab" in decl[0]
        assert "A2D::fourier(tab, s, 0.5)" in active[-1]

    @pytest.mark.parametrize("n", [4, 5, 100])
    def test_coefficients_give_the_curve_through_the_samples(self, n):
        rng = np.random.default_rng(n)
        samples = rng.standard_normal((n, 2))

        coef = am.fourier_coefficients(samples)

        # The cosine and sine basis at the samples
        s = np.arange(n) / n
        basis = [np.ones(n)]

        for k in range(1, (n + 1) // 2):
            basis.append(np.cos(2 * np.pi * k * s))
            basis.append(np.sin(2 * np.pi * k * s))

        if n % 2 == 0:
            basis.append(np.cos(np.pi * n * s))

        assert np.allclose(np.array(basis).T @ coef, samples, atol=1e-12)


# Long chains inside a term end in shallow temporaries


def _paren_depth(code):
    """Deepest nesting of parentheses in a line of code"""
    best = 0
    current = 0

    for ch in code:
        if ch == "(":
            current += 1
            best = max(best, current)
        elif ch == ")":
            current -= 1

    return best


def deepest(lines):
    """Deepest nesting of parentheses over lines of code"""
    depths = []

    for line in lines:
        depths.append(_paren_depth(line))

    return max(depths)


def _chain_builder(nterms):
    """A term that multiplies a chain of nterms additions, like a long series in a constraint"""
    x = Expr(VarNode("x", shape=nterms))
    lam = Expr(VarNode("lam"))

    chain = x[0]

    for i in range(1, nterms):
        chain = chain + x[i] * x[i]

    lhs = Expr(VarNode("lagrangian__"))

    return ExprBuilder([], [], [x, lam], [], rhs=lam * chain, lhs=lhs)


class TestDepthLimit:
    def test_short_chains_keep_one_expression(self):
        builder = _chain_builder(_MAX_DEPTH // 2)

        assert builder.temp_exprs == []

    def test_long_chain_is_cut_into_shallow_temporaries(self):
        builder = _chain_builder(100)

        active = builder.get_cpp_lines(mode="hprod", template_name="R__")[2]

        assert len(builder.temp_exprs) >= 100 // _MAX_DEPTH
        assert deepest(active) <= _MAX_DEPTH + 4

    def test_deep_temporaries_are_built_in_their_own_frame(self):
        builder = _chain_builder(100)

        active = builder.get_cpp_lines(mode="hprod", template_name="R__")[2]
        temps = active[:-1]

        assert temps

        for line in temps:
            assert line.startswith("amigo::call_noinline([&]() {")

        assert all(builder.temp_frames)

    def test_eval_mode_uses_plain_temporaries(self):
        builder = _chain_builder(100)

        active = builder.get_cpp_lines(mode="eval", template_name="R__")[2]

        for line in active:
            assert "call_noinline" not in line

        assert deepest(active) <= _MAX_DEPTH + 2

    def test_top_level_sum_is_left_to_the_chunks(self):
        nterms = 4 * _MAX_DEPTH

        x = Expr(VarNode("x", shape=nterms))
        lam = Expr(VarNode("lam", shape=nterms))

        rhs = x[0] * lam[0]

        for i in range(1, nterms):
            rhs = rhs + x[i] * lam[i]

        lhs = Expr(VarNode("lagrangian__"))
        builder = ExprBuilder([], [], [x, lam], [], rhs=rhs, lhs=lhs)

        assert builder.temp_exprs == []

    def test_very_deep_chain_generates_code(self):
        builder = _chain_builder(3000)

        active = builder.get_cpp_lines(mode="grad", template_name="R__")[2]

        assert deepest(active) <= _MAX_DEPTH + 4
