"""
HS53: Over-determined equalities (5 vars, 3 eq)
  min  (x1-x2)^2 + (x2+x3-2)^2 + (x4-1)^2 + (x5-1)^2
  s.t. x1 + 3*x2 = 0
       x3 + x4 - 2*x5 = 0
       x2 - x5 = 0
       -10 <= xi <= 10
  x0 = (2, 2, 2, 2, 2), f* = 176/43 ≈ 4.0930233
"""

import amigo as am
import argparse


class HS53(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=2.0, lower=-10.0, upper=10.0)
        self.add_input("x2", value=2.0, lower=-10.0, upper=10.0)
        self.add_input("x3", value=2.0, lower=-10.0, upper=10.0)
        self.add_input("x4", value=2.0, lower=-10.0, upper=10.0)
        self.add_input("x5", value=2.0, lower=-10.0, upper=10.0)
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
            (x1 - x2) ** 2 + (x2 + x3 - 2) ** 2 + (x4 - 1) ** 2 + (x5 - 1) ** 2
        )
        self.constraints["c1"] = x1 + 3 * x2
        self.constraints["c2"] = x3 + x4 - 2 * x5
        self.constraints["c3"] = x2 - x5


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs53")
model.add_component("hs53", 1, HS53())
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
# f* = 4.0930233
