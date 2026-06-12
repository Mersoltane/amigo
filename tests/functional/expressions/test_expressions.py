import numpy as np
import amigo as am

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


class Expressions(am.Component):
    def __init__(self):
        super().__init__()

        value = 0.34
        self.add_input("x", value=value, lower=-am.inf, upper=am.inf)
        self.add_objective("obj")

        delta = 0.1
        for expr in unary_expressions:
            if expr == "acosh":
                fval = getattr(np, expr)(value + 1)
            else:
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
            if expr == "acosh":
                self.constraints[f"{expr}_value"] = getattr(am, expr)(x + 1)
            else:
                self.constraints[f"{expr}_value"] = getattr(am, expr)(x)

    def compute_output(self):
        x = self.inputs["x"]
        for expr in unary_expressions:
            if expr == "acosh":
                self.outputs[f"{expr}_output"] = getattr(am, expr)(x + 1)
            else:
                self.outputs[f"{expr}_output"] = getattr(am, expr)(x)


def test_expressions():

    model = am.Model("expr_test")
    expr = Expressions()
    model.add_component("expr", 1, expr)

    model.build_module()
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
        if expr == "acosh":
            np_val = getattr(np, expr)(xval + 1)
        else:
            np_val = getattr(np, expr)(xval)

        assert abs(am_val - np_val) < tol


if __name__ == "__main__":
    test_expressions()
