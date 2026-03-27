"""
HS29: Nonconvex with multiple local minima (3 vars, 1 ineq)
  min  -x1*x2*x3
  s.t. -x1^2 - 2*x2^2 - 4*x3^2 + 48 >= 0
  x0 = (1, 1, 1), f* = -22.627417
"""

import amigo as am
import argparse


class HS29(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=1.0)
        self.add_input("x2", value=1.0)
        self.add_input("x3", value=1.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=float("inf"))

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        self.objective["obj"] = -x1 * x2 * x3
        self.constraints["c1"] = -(x1**2) - 2 * x2**2 - 4 * x3**2 + 48


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs29")
model.add_component("hs29", 1, HS29())
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
# f* = -22.627417
