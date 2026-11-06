"""
Hock-Schittkowski problem 35
----------------------------

    minimize    9 - 8 x1 - 6 x2 - 4 x3 + 2 x1^2 + 2 x2^2 + x3^2
                + 2 x1 x2 + 2 x1 x3
    subject to  x1 + x2 + 2 x3 <= 3
                x1, x2, x3 >= 0
"""

import amigo as am
import argparse


class HS35(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=0.5, lower=0.0)
        self.add_input("x2", value=0.5, lower=0.0)
        self.add_input("x3", value=0.5, lower=0.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=-float("inf"), upper=3.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        self.objective["obj"] = (
            9
            - 8 * x1
            - 6 * x2
            - 4 * x3
            + 2 * x1**2
            + 2 * x2**2
            + x3**2
            + 2 * x1 * x2
            + 2 * x1 * x3
        )
        self.constraints["c1"] = x1 + x2 + 2 * x3


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs35")
model.add_component("hs35", 1, HS35())
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
# f* = 0.1111111
