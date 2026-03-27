"""
HS40: Cubic constraints (4 vars, 3 eq)
  min  -x1*x2*x3*x4
  s.t. x1^3 + x2^2 - 1 = 0
       x1^2*x4 - x3 = 0
       x4^2 - x2 = 0
  x0 = (0.8, 0.8, 0.8, 0.8), f* = -0.25
"""

import amigo as am
import argparse


class HS40(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=0.8)
        self.add_input("x2", value=0.8)
        self.add_input("x3", value=0.8)
        self.add_input("x4", value=0.8)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=0.0)
        self.add_constraint("c2", lower=0.0, upper=0.0)
        self.add_constraint("c3", lower=0.0, upper=0.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        self.objective["obj"] = -x1 * x2 * x3 * x4
        self.constraints["c1"] = x1**3 + x2**2 - 1
        self.constraints["c2"] = x1**2 * x4 - x3
        self.constraints["c3"] = x4**2 - x2


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs40")
model.add_component("hs40", 1, HS40())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 100,
        "filter_line_search": True,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = -0.25
