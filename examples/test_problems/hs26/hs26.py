"""
HS26: Rosenbrock-like, notoriously slow for IPM (3 vars, 1 eq)
  min  (x1 - x2)^2 + (x2 - x3)^4
  s.t. (1 + x2^2)*x1 + x3^4 - 3 = 0
  x0 = (-2.6, 2, 2), f* = 0
"""

import amigo as am
import argparse


class HS26(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=-2.6)
        self.add_input("x2", value=2.0)
        self.add_input("x3", value=2.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=0.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        self.objective["obj"] = (x1 - x2) ** 2 + (x2 - x3) ** 4
        self.constraints["c1"] = (1 + x2**2) * x1 + x3**4 - 3


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs26")
model.add_component("hs26", 1, HS26())
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
# f* = 0
