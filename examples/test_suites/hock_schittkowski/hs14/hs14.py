"""
Hock-Schittkowski problem 14
----------------------------

    minimize    (x1 - 2)^2 + (x2 - 1)^2
    subject to  x1 - 2 x2 + 1 = 0
                -x1^2 / 4 - x2^2 + 1 >= 0
"""

import amigo as am
import argparse


class HS14(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=2.0)
        self.add_input("x2", value=2.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=0.0)
        self.add_constraint("c2", lower=0.0, upper=float("inf"))

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        self.objective["obj"] = (x1 - 2) ** 2 + (x2 - 1) ** 2
        self.constraints["c1"] = x1 - 2 * x2 + 1
        self.constraints["c2"] = -(x1**2) / 4 - x2**2 + 1


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs14")
model.add_component("hs14", 1, HS14())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 100,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = 1.3934651
