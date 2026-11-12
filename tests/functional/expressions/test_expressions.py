"""
Unary functions
---------------

- Every unary function that A2D differentiates to second order matches numpy in a solved model
"""

import contextlib
import sys

import numpy as np

import amigo as am

# Functions that A2D differentiates to second order
unary_expressions = [
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
]


def numpy_acosh(x):
    """Inverse hyperbolic cosine of 1 + x, which keeps it inside its domain"""
    return np.acosh(1.0 + x)


def amigo_acosh(x):
    """Inverse hyperbolic cosine of 1 + x, which keeps it inside its domain"""
    return am.acosh(1.0 + x)


# The numpy and the amigo function of every name
NUMPY = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.asin,
    "acos": np.acos,
    "atan": np.atan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asinh": np.asinh,
    "acosh": numpy_acosh,
    "atanh": np.atanh,
}

AMIGO = {
    "sqrt": am.sqrt,
    "exp": am.exp,
    "log": am.log,
    "sin": am.sin,
    "cos": am.cos,
    "tan": am.tan,
    "asin": am.asin,
    "acos": am.acos,
    "atan": am.atan,
    "sinh": am.sinh,
    "cosh": am.cosh,
    "tanh": am.tanh,
    "asinh": am.asinh,
    "acosh": amigo_acosh,
    "atanh": am.atanh,
}


class Expressions(am.Component):
    """Every unary function of one input as a constraint near its value and as an output"""

    def __init__(self):
        super().__init__()

        value = 0.34

        self.add_input("x", value=value, lower=-am.inf, upper=am.inf)
        self.add_objective("obj")

        # Each function stays within a band around its value at the start
        delta = 0.1

        for expr in unary_expressions:
            fval = NUMPY[expr](value)

            lower = fval - delta
            upper = fval + delta

            self.add_constraint(f"{expr}_value", lower=lower, upper=upper)

        for expr in unary_expressions:
            self.add_output(f"{expr}_output")

        return

    def compute(self):
        x = self.inputs["x"]

        self.objective["obj"] = x**2

        for expr in unary_expressions:
            self.constraints[f"{expr}_value"] = AMIGO[expr](x)

        return

    def compute_output(self):
        x = self.inputs["x"]

        for expr in unary_expressions:
            self.outputs[f"{expr}_output"] = AMIGO[expr](x)

        return


def test_expressions(tmp_path):
    model = am.Model("expr_test")
    model.add_component("expr", 1, Expressions())

    # Build in a scratch directory so the project CMakeLists.txt stays untouched
    with contextlib.chdir(tmp_path):
        model.build_module()

    # The directory goes on sys.path for the compiled module import
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    model.initialize()

    x = model.create_vector()
    opt = am.Optimizer(model, x)
    opt.optimize()

    g = model.create_vector()
    model.eval_gradient(x, g)

    output = model.create_output_vector()
    model.compute_output(x, output)

    # Every output matches numpy at the solution
    tol = 1e-8
    xval = x["expr.x"]

    for expr in unary_expressions:
        am_val = output[f"expr.{expr}_output"]
        np_val = NUMPY[expr](xval)

        assert abs(am_val - np_val) < tol
