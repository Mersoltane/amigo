"""
HS116 (revised): 13 variables, 15 nonlinear inequality constraints.
  min  x11 + x12 + x13
  s.t. c_i >= 0, i = 1..15
  f* = 97.5875
  Reference: Hock & Schittkowski (revised), tp116r
"""

import amigo as am
import argparse


class HS116(am.Component):
    def __init__(self):
        super().__init__()
        # Variable bounds (1-indexed in the reference, 0-indexed here)
        lb = [0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1, 0.1, 500, 0.1, 1, 0.0001, 0.0001]
        ub = [1, 1, 1, 0.1, 0.9, 0.9, 1000, 1000, 1000, 500, 150, 150, 150]
        x0 = [0.5, 0.8, 0.9, 0.1, 0.14, 0.5, 489, 80, 650, 450, 150, 150, 150]
        for i in range(13):
            self.add_input(f"x{i+1}", value=x0[i], lower=lb[i], upper=ub[i])
        self.add_objective("obj")
        for i in range(15):
            self.add_constraint(f"c{i+1}", lower=0.0, upper=float("inf"))

    def compute(self):
        # x[0..12] = x1..x13 in the reference
        x = [self.inputs[f"x{i+1}"] for i in range(13)]

        # Objective: min x11 + x12 + x13
        self.objective["obj"] = x[10] + x[11] + x[12]

        # All constraints c_i >= 0
        # c1:  x3 - x2
        self.constraints["c1"] = x[2] - x[1]
        # c2:  x2 - x1
        self.constraints["c2"] = x[1] - x[0]
        # c3:  1 - 0.002*x7 + 0.002*x8
        self.constraints["c3"] = 1 - 0.002 * x[6] + 0.002 * x[7]
        # c4:  x11 + x12 + x13 - 50
        self.constraints["c4"] = x[10] + x[11] + x[12] - 50
        # c5:  250 - x11 - x12 - x13
        self.constraints["c5"] = 250 - x[10] - x[11] - x[12]
        # c6:  x13 - 1.262626*x10 + 1.231059*x3*x10
        self.constraints["c6"] = x[12] - 1.262626 * x[9] + 1.231059 * x[2] * x[9]
        # c7:  x5 - 0.03475*x2 - 0.975*x2*x5 + 0.00975*x2^2
        self.constraints["c7"] = (
            x[4] - 0.03475 * x[1] - 0.975 * x[1] * x[4] + 0.00975 * x[1] ** 2
        )
        # c8:  x6 - 0.03475*x3 - 0.975*x3*x6 + 0.00975*x3^2
        self.constraints["c8"] = (
            x[5] - 0.03475 * x[2] - 0.975 * x[2] * x[5] + 0.00975 * x[2] ** 2
        )
        # c9:  x5*x7 - x1*x8 - x4*x7 + x4*x8
        self.constraints["c9"] = x[4] * x[6] - x[0] * x[7] - x[3] * x[6] + x[3] * x[7]
        # c10: 1 - 0.002*(x2*x9 + x5*x8 - x1*x8 - x6*x9) - x5 - x6
        self.constraints["c10"] = (
            1
            - 0.002 * (x[1] * x[8] + x[4] * x[7] - x[0] * x[7] - x[5] * x[8])
            - x[4]
            - x[5]
        )
        # c11: x2*x9 - x3*x10 - x6*x9 - 500*x2 + 500*x6 + x2*x10
        self.constraints["c11"] = (
            x[1] * x[8]
            - x[2] * x[9]
            - x[5] * x[8]
            - 500 * x[1]
            + 500 * x[5]
            + x[1] * x[9]
        )
        # c12: x2 - 0.9 - 0.002*(x2*x10 - x3*x10)
        self.constraints["c12"] = x[1] - 0.9 - 0.002 * (x[1] * x[9] - x[2] * x[9])
        # c13: x4 - 0.03475*x1 - 0.975*x1*x4 + 0.00975*x1^2
        self.constraints["c13"] = (
            x[3] - 0.03475 * x[0] - 0.975 * x[0] * x[3] + 0.00975 * x[0] ** 2
        )
        # c14: x11 - 1.262626*x8 + 1.231059*x1*x8
        self.constraints["c14"] = x[10] - 1.262626 * x[7] + 1.231059 * x[0] * x[7]
        # c15: x12 - 1.262626*x9 + 1.231059*x2*x9
        self.constraints["c15"] = x[11] - 1.262626 * x[8] + 1.231059 * x[1] * x[8]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs116")
model.add_component("hs116", 1, HS116())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 50,
        "filter_line_search": True,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = 97.5875
