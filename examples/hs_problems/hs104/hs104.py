"""
HS104: Large nonconvex (8 vars, 6 ineq)
  min  0.4*x1^0.67 * x7^(-0.67) + 0.4*x2^0.67 * x8^(-0.67) + 10 - x1 - x2
  s.t. 1 - 0.0588*x5*x7 - 0.1*x1 >= 0
       1 - 0.0588*x6*x8 - 0.1*x1 - 0.1*x2 >= 0
       1 - 4*x3/x5 - 2/(x3^0.71 * x5) - 0.0588*x7/x3^1.3 >= 0
       1 - 4*x4/x6 - 2/(x4^0.71 * x6) - 0.0588*x8/x4^1.3 >= 0
       0.1 <= x1,x2 <= 10
       1 <= x3,x4 <= 10
       1 <= x5,x6 <= 5
       0.1 <= x7,x8 <= 10
  x0 = (6,3,0.4,0.2,6,6,1,0.5), f* = 3.9511634396
"""

import amigo as am
import argparse


class HS104(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=6.0, lower=0.1, upper=10.0)
        self.add_input("x2", value=3.0, lower=0.1, upper=10.0)
        self.add_input("x3", value=0.4, lower=0.1, upper=10.0)
        self.add_input("x4", value=0.2, lower=0.1, upper=10.0)
        self.add_input("x5", value=6.0, lower=0.1, upper=10.0)
        self.add_input("x6", value=6.0, lower=0.1, upper=10.0)
        self.add_input("x7", value=1.0, lower=0.1, upper=10.0)
        self.add_input("x8", value=0.5, lower=0.1, upper=10.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=float("inf"))
        self.add_constraint("c2", lower=0.0, upper=float("inf"))
        self.add_constraint("c3", lower=0.0, upper=float("inf"))
        self.add_constraint("c4", lower=0.0, upper=float("inf"))
        self.add_constraint("c5", lower=0.1, upper=4.2)  # objective bounds

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        x5 = self.inputs["x5"]
        x6 = self.inputs["x6"]
        x7 = self.inputs["x7"]
        x8 = self.inputs["x8"]
        obj_val = (
            0.4 * x1**0.67 * x7 ** (-0.67)
            + 0.4 * x2**0.67 * x8 ** (-0.67)
            + 10
            - x1
            - x2
        )
        self.objective["obj"] = obj_val
        self.constraints["c5"] = obj_val
        self.constraints["c1"] = 1 - 0.0588 * x5 * x7 - 0.1 * x1
        self.constraints["c2"] = 1 - 0.0588 * x6 * x8 - 0.1 * x1 - 0.1 * x2
        self.constraints["c3"] = (
            1 - 4 * x3 / x5 - 2 / (x3**0.71 * x5) - 0.0588 * x7 / x3**1.3
        )
        self.constraints["c4"] = (
            1 - 4 * x4 / x6 - 2 / (x4**0.71 * x6) - 0.0588 * x8 / x4**1.3
        )


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs104")
model.add_component("hs104", 1, HS104())
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
# f* = 3.9511634396
