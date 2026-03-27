"""
HS113: 10 variables, 8 inequalities: many active constraints
  min  x1^2 + x2^2 + x1*x2 - 14*x1 - 16*x2 + (x3-10)^2 + 4*(x4-5)^2
       + (x5-3)^2 + 2*(x6-1)^2 + 5*x7^2 + 7*(x8-11)^2 + 2*(x9-10)^2
       + (x10-7)^2 + 45
  s.t. 8 linear/quadratic inequalities
  x0 = (2, 3, 5, 5, 1, 2, 7, 3, 6, 10), f* = 24.3062091
"""

import amigo as am
import argparse


class HS113(am.Component):
    def __init__(self):
        super().__init__()
        x0 = [2.0, 3.0, 5.0, 5.0, 1.0, 2.0, 7.0, 3.0, 6.0, 10.0]
        for i in range(10):
            self.add_input(f"x{i+1}", value=x0[i])
        self.add_objective("obj")
        for i in range(8):
            self.add_constraint(f"c{i+1}", lower=0.0, upper=float("inf"))

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(10)]
        self.objective["obj"] = (
            x[0] ** 2
            + x[1] ** 2
            + x[0] * x[1]
            - 14 * x[0]
            - 16 * x[1]
            + (x[2] - 10) ** 2
            + 4 * (x[3] - 5) ** 2
            + (x[4] - 3) ** 2
            + 2 * (x[5] - 1) ** 2
            + 5 * x[6] ** 2
            + 7 * (x[7] - 11) ** 2
            + 2 * (x[8] - 10) ** 2
            + (x[9] - 7) ** 2
            + 45
        )
        self.constraints["c1"] = 105 - 4 * x[0] - 5 * x[1] + 3 * x[6] - 9 * x[7]
        self.constraints["c2"] = -10 * x[0] + 8 * x[1] + 17 * x[6] - 2 * x[7]
        self.constraints["c3"] = 8 * x[0] - 2 * x[1] - 5 * x[8] + 2 * x[9] + 12
        self.constraints["c4"] = (
            -3 * (x[0] - 2) ** 2 - 4 * (x[1] - 3) ** 2 - 2 * x[2] ** 2 + 7 * x[3] + 120
        )
        self.constraints["c5"] = (
            -5 * x[0] ** 2 - 8 * x[1] - (x[2] - 6) ** 2 + 2 * x[3] + 40
        )
        self.constraints["c6"] = (
            -0.5 * (x[0] - 8) ** 2 - 2 * (x[1] - 4) ** 2 - 3 * x[4] ** 2 + x[5] + 30
        )
        self.constraints["c7"] = (
            -x[0] ** 2 - 2 * (x[1] - 2) ** 2 + 2 * x[0] * x[1] - 14 * x[4] + 6 * x[5]
        )
        self.constraints["c8"] = 3 * x[0] - 6 * x[1] - 12 * (x[8] - 8) ** 2 + 7 * x[9]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs113")
model.add_component("hs113", 1, HS113())
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
# f* = 24.3062091
