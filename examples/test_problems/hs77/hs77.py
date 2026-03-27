"""
HS77: Nonlinear equalities, Hessian accuracy test (5 vars, 2 eq)
  min  (x1-1)^2 + (x1-x2)^2 + (x3-1)^2 + (x4-1)^4 + (x5-1)^6
  s.t. x1^2*x4 + sin(x4-x5) - 2*sqrt(2) = 0
       x2 + x3^4*x4^2 - 8 - sqrt(2) = 0
  x0 = (2, 2, 2, 2, 2), f* = 0.2415051
"""

import amigo as am
import argparse
import numpy as np


class HS77(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=2.0)
        self.add_input("x2", value=2.0)
        self.add_input("x3", value=2.0)
        self.add_input("x4", value=2.0)
        self.add_input("x5", value=2.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=0.0)
        self.add_constraint("c2", lower=0.0, upper=0.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        x5 = self.inputs["x5"]
        self.objective["obj"] = (
            (x1 - 1) ** 2
            + (x1 - x2) ** 2
            + (x3 - 1) ** 2
            + (x4 - 1) ** 4
            + (x5 - 1) ** 6
        )
        self.constraints["c1"] = x1**2 * x4 + am.sin(x4 - x5) - 2 * np.sqrt(2)
        self.constraints["c2"] = x2 + x3**4 * x4**2 - 8 - np.sqrt(2)


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs77")
model.add_component("hs77", 1, HS77())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 200,
        "filter_line_search": True,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = 0.2415051
