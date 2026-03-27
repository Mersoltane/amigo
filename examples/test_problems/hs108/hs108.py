"""
HS108: Polygon area maximization (9 vars, 13 inequality constraints).
  min  -0.5*(x1*x4 - x2*x3 + x3*x9 - x5*x9 + x5*x8 - x6*x7)
  s.t. 13 nonlinear inequality constraints (see below)
  No variable bounds (unbounded)
  x0 = (1, 1, 1, 1, 1, 1, 1, 1, 1)
  f* = -0.8660254038

  Note: Near-degenerate constraints at solution make this a good
  stress test for inertia correction.
"""

import amigo as am
import argparse


class HS108(am.Component):
    def __init__(self):
        super().__init__()
        for i in range(9):
            self.add_input(f"x{i+1}", value=1.0)
        self.add_objective("obj")
        for i in range(13):
            self.add_constraint(f"c{i+1}", lower=0.0, upper=float("inf"))

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(9)]

        # Objective: maximize polygon area (minimize negative)
        self.objective["obj"] = -0.5 * (
            x[0] * x[3]
            - x[1] * x[2]
            + x[2] * x[8]
            - x[4] * x[8]
            + x[4] * x[7]
            - x[5] * x[6]
        )

        # Constraints (all >= 0)
        self.constraints["c1"] = 1 - x[2] ** 2 - x[3] ** 2
        self.constraints["c2"] = 1 - x[4] ** 2 - x[5] ** 2
        self.constraints["c3"] = 1 - x[8] ** 2
        self.constraints["c4"] = 1 - x[0] ** 2 - (x[1] - x[8]) ** 2
        self.constraints["c5"] = 1 - (x[0] - x[4]) ** 2 - (x[1] - x[5]) ** 2
        self.constraints["c6"] = 1 - (x[0] - x[6]) ** 2 - (x[1] - x[7]) ** 2
        self.constraints["c7"] = 1 - (x[2] - x[6]) ** 2 - (x[3] - x[7]) ** 2
        self.constraints["c8"] = 1 - (x[2] - x[4]) ** 2 - (x[3] - x[5]) ** 2
        self.constraints["c9"] = 1 - x[6] ** 2 - (x[7] - x[8]) ** 2
        self.constraints["c10"] = x[0] * x[3] - x[1] * x[2]
        self.constraints["c11"] = x[2] * x[8]
        self.constraints["c12"] = -x[4] * x[8]
        self.constraints["c13"] = x[4] * x[7] - x[5] * x[6]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs108")
model.add_component("hs108", 1, HS108())
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
# f* = -0.8660254038
