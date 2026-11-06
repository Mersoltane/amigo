import numbers

from .expressions import (
    Expr,
    VarNode,
    UnaryNode,
    BinaryNode,
    ConstNode,
    PassiveNode,
    BSplineNode,
    BSplineBasisNode,
    _balanced_sum,
    _to_expr,
)


def abs(expr: Expr):
    return Expr(UnaryNode("fabs", expr))


def fabs(expr: Expr):
    return Expr(UnaryNode("fabs", expr))


def sqrt(expr: Expr):
    return Expr(UnaryNode("sqrt", expr))


def sin(expr: Expr):
    return Expr(UnaryNode("sin", expr))


def asin(expr: Expr):
    return Expr(UnaryNode("asin", expr))


def cos(expr: Expr):
    return Expr(UnaryNode("cos", expr))


def acos(expr: Expr):
    return Expr(UnaryNode("acos", expr))


def tan(expr: Expr):
    return Expr(UnaryNode("tan", expr))


def atan(expr: Expr):
    return Expr(UnaryNode("atan", expr))


def sinh(expr: Expr):
    return Expr(UnaryNode("sinh", expr))


def asinh(expr: Expr):
    return Expr(UnaryNode("asinh", expr))


def cosh(expr: Expr):
    return Expr(UnaryNode("cosh", expr))


def acosh(expr: Expr):
    return Expr(UnaryNode("acosh", expr))


def tanh(expr: Expr):
    return Expr(UnaryNode("tanh", expr))


def atanh(expr: Expr):
    return Expr(UnaryNode("atanh", expr))


def exp(expr: Expr):
    return Expr(UnaryNode("exp", expr))


def log(expr: Expr):
    return Expr(UnaryNode("log", expr))


def _binary(op, a, b):
    """Binary operation of two expressions, or of an expression and a number."""
    a_number = isinstance(a, numbers.Real)
    b_number = isinstance(b, numbers.Real)
    if a_number and b_number:
        raise ValueError("Neither argument is active")
    if not (a_number or isinstance(a, Expr)) or not (b_number or isinstance(b, Expr)):
        raise TypeError(f"Types not recognized for {op}")

    # A number becomes a floating constant so the plain C++ overload sees one type
    if a_number:
        a = _to_expr(float(a))
    if b_number:
        b = _to_expr(float(b))
    return Expr(BinaryNode(op, a, b))


def atan2(a: Expr, b: Expr):
    return _binary("atan2", a, b)


def min2(a: Expr, b: Expr):
    return _binary("min2", a, b)


def max2(a: Expr, b: Expr):
    return _binary("max2", a, b)


def norm(*args):
    """Euclidean norm of scalar expressions or of the entries of one vector input.

    The square root of the sum of squares has a zero derivative at the origin,
    the smallest subgradient of the norm, where sqrt would give zero times
    infinity.
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = tuple(args[0])
    elif len(args) == 1 and isinstance(args[0], Expr) and args[0].shape is not None:
        vector = args[0]
        if len(vector.shape) != 1:
            raise ValueError("norm takes scalars or one vector")
        args = tuple(vector[i] for i in range(vector.shape[0]))
    if len(args) == 0:
        raise ValueError("norm needs at least one argument")

    squares = [_to_expr(a) * _to_expr(a) for a in args]
    return Expr(UnaryNode("norm_sqrt", _balanced_sum(squares)))


def bspline(table, t, order, column=None):
    """B-spline with the coefficients of a table of component data at the parameter t.

    A spline of order k with m coefficients lives on the open uniform knots of
    the unit interval that the MBI package uses, the knot 0 repeated k times,
    the interior knots j / (m - k + 1), and the knot 1 repeated k times. A 2D
    table is read along one column and the end spans extend the spline outside
    the unit interval.
    """
    return Expr(BSplineNode(table, _to_expr(t), order, column))


def bspline_basis(t, index, size, order):
    """Basis function index of the B-spline of order k with size coefficients at the parameter t."""
    return Expr(BSplineBasisNode(_to_expr(t), index, size, order))


def bspline_inverse(table, x, order, column=None):
    """Parameter where the coordinate spline with increasing coefficients takes the value x.

    The coefficients come from a table of component data as in bspline. An x
    below the first or above the last coefficient holds the parameter at 0 or
    1 with zero derivatives, as the clamped inverse map of MBI does.
    """
    return Expr(BSplineNode(table, _to_expr(x), order, column, inverse=True))


def passive(expr: Expr):
    """Force an expression to be passive"""
    if not isinstance(expr.node, VarNode):
        raise TypeError("Passive type must be a variable")
    return Expr(PassiveNode(expr))
