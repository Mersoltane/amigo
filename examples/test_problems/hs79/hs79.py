"""
HS79: Nonlinear equalities (5 vars, 3 eq)
  min  (x1-1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^4 + (x4-x5)^4
  s.t. x1 + x2^2 + x3^3 - 2 - 3*sqrt(2) = 0
       x2 - x3^2 + x4 + 2 - 2*sqrt(2) = 0
       x1*x5 - 2 = 0
  x0 = (2, 2, 2, 2, 2), f* = 0.0787768209
"""

import amigo as am
import argparse
import numpy as np


class HS79(am.Component):
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
        self.add_constraint("c3", lower=0.0, upper=0.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        x5 = self.inputs["x5"]
        self.objective["obj"] = (
            (x1 - 1) ** 2
            + (x1 - x2) ** 2
            + (x2 - x3) ** 2
            + (x3 - x4) ** 4
            + (x4 - x5) ** 4
        )
        self.constraints["c1"] = x1 + x2**2 + x3**3 - 2 - 3 * np.sqrt(2)
        self.constraints["c2"] = x2 - x3**2 + x4 + 2 - 2 * np.sqrt(2)
        self.constraints["c3"] = x1 * x5 - 2


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs79")
model.add_component("hs79", 1, HS79())
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
# f* = 0.0787768209
