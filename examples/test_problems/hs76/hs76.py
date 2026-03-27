"""
HS76: Standard QP test (4 vars, 3 ineq)
  min  x1^2 + 0.5*x2^2 + x3^2 + 0.5*x4^2 - x1*x3 - x3*x4
       - x1 - 3*x2 + x3 - x4
  s.t. 5 - x1 - x2 - x3 - x4 >= 0
       8 + x1 + x2 - x3 - x4 >= 0
       6 - x1 + x2 - x3 + x4 >= 0
       x1, x2, x3, x4 >= 0
  x0 = (0.5, 0.5, 0.5, 0.5), f* = -4.681818
"""

import amigo as am
import argparse


class HS76(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=0.5, lower=0.0)
        self.add_input("x2", value=0.5, lower=0.0)
        self.add_input("x3", value=0.5, lower=0.0)
        self.add_input("x4", value=0.5, lower=0.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=float("inf"))
        self.add_constraint("c2", lower=0.0, upper=float("inf"))
        self.add_constraint("c3", lower=0.0, upper=float("inf"))

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        self.objective["obj"] = (
            x1**2
            + 0.5 * x2**2
            + x3**2
            + 0.5 * x4**2
            - x1 * x3
            - x3 * x4
            - x1
            - 3 * x2
            + x3
            - x4
        )
        self.constraints["c1"] = 5 - x1 - x2 - x3 - x4
        self.constraints["c2"] = 8 + x1 + x2 - x3 - x4
        self.constraints["c3"] = 6 - x1 + x2 - x3 + x4


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs76")
model.add_component("hs76", 1, HS76())
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
# f* = -4.681818
