import os
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
    "asin",
    "acos",
    "atan",
]


class Expressions(am.Component):
    def __init__(self):
        super().__init__()

        value = 0.34
        self.add_input("x", value=value, lower=-am.inf, upper=am.inf)
        self.add_objective("obj")

        delta = 0.1
        for expr in unary_expressions:
            fval = getattr(np, expr)(value)

            lower = fval - delta
            upper = fval + delta
            self.add_constraint(f"{expr}_value", lower=lower, upper=upper)

        for expr in unary_expressions:
            fval = getattr(np, expr)(value)
            self.add_output(f"{expr}_output")

    def compute(self):
        x = self.inputs["x"]
        self.objective["obj"] = x**2

        for expr in unary_expressions:
            self.constraints[f"{expr}_value"] = getattr(am, expr)(x)

    def compute_output(self):
        x = self.inputs["x"]
        for expr in unary_expressions:
            self.outputs[f"{expr}_output"] = getattr(am, expr)(x)


def test_expressions(tmp_path):

    model = am.Model("expr_test")
    expr = Expressions()
    model.add_component("expr", 1, expr)

    # build_module() generates expr_test.cpp/.h and a CMakeLists.txt in the
    # current working directory. Build in a temporary directory so that
    # running pytest from the repository root does not overwrite the
    # project's own CMakeLists.txt, then put that directory on sys.path so
    # initialize() can import the compiled module.
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        model.build_module()
    finally:
        os.chdir(orig_dir)

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

    tol = 1e-8
    xval = x["expr.x"]
    for expr in unary_expressions:
        am_val = output[f"expr.{expr}_output"]
        np_val = getattr(np, expr)(xval)
        assert abs(am_val - np_val) < tol


if __name__ == "__main__":
    test_expressions()
