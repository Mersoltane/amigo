import numbers

import numpy as np

from .expressions import (
    Expr,
    VarNode,
    UnaryNode,
    BinaryNode,
    ConstNode,
    PassiveNode,
    InterpNode,
    FourierNode,
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


def interp1d(x, table, column=None, x0=0.0, dx=1.0, periodic=False):
    """Linear interpolation in a table of component data on a uniform grid.

    Sample k sits at x0 + k * dx. A 2D table is read along one column. A
    periodic table holds one period of samples and wraps the argument with
    that period, otherwise the end intervals extend the table linearly. The
    derivative is the slope of the interval that holds x and the second
    derivative is zero.
    """
    return Expr(InterpNode(table, _to_expr(x), column, x0, dx, periodic))


def fourier(x, coefficients, column=None, period=1.0):
    """Real Fourier series with coefficients from a table of component data.

    The coefficients are a0, a1, b1, a2, b2, and so on, where ak multiplies
    cos(2 pi k x / period) and bk multiplies sin(2 pi k x / period). An even
    number of coefficients ends with the cosine at the Nyquist frequency, so
    fourier_coefficients of equally spaced samples gives the smooth periodic
    curve through them. A 2D table is read along one column.
    """
    return Expr(FourierNode(coefficients, _to_expr(x), column, period))


def fourier_coefficients(samples):
    """Coefficients of the trigonometric polynomial through samples equally spaced over one period.

    Sample j sits at the fraction j / n of the period. The first axis runs over
    the samples and every other axis holds a separate function.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0]
    spectrum = np.fft.fft(samples, axis=0) / n
    coef = np.empty_like(samples)
    coef[0] = spectrum[0].real
    for k in range(1, (n + 1) // 2):
        coef[2 * k - 1] = 2.0 * spectrum[k].real
        coef[2 * k] = -2.0 * spectrum[k].imag
    if n % 2 == 0:
        coef[n - 1] = spectrum[n // 2].real
    return coef


def passive(expr: Expr):
    """Force an expression to be passive"""
    if not isinstance(expr.node, VarNode):
        raise TypeError("Passive type must be a variable")
    return Expr(PassiveNode(expr))
