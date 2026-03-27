"""
HS6: Equality constraint on Rosenbrock valley (2 vars, 1 eq)
  min  (1 - x1)^2
  s.t. 10*(x2 - x1^2) = 0
  x0 = (-1.2, 1), f* = 0
"""

import amigo as am
import argparse


class HS6(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=-1.2)
        self.add_input("x2", value=1.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=0.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        self.objective["obj"] = (1 - x1) ** 2
        self.constraints["c1"] = 10 * (x2 - x1**2)


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs6")
model.add_component("hs6", 1, HS6())
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
# f* = 0
