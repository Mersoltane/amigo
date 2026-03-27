"""
HS39: Near-singular Jacobian (4 vars, 2 eq)
  min  -x1
  s.t. x2 - x1^3 - x3^2 = 0
       x1^2 - x2 - x4^2 = 0
  x0 = (2, 2, 2, 2), f* = -1
"""

import amigo as am
import argparse


class HS39(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=2.0)
        self.add_input("x2", value=2.0)
        self.add_input("x3", value=2.0)
        self.add_input("x4", value=2.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=0.0)
        self.add_constraint("c2", lower=0.0, upper=0.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        self.objective["obj"] = -x1
        self.constraints["c1"] = x2 - x1**3 - x3**2
        self.constraints["c2"] = x1**2 - x2 - x4**2


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs39")
model.add_component("hs39", 1, HS39())
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
# f* = -1
