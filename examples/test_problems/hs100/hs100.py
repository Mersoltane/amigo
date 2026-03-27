"""
HS100: Medium nonconvex (7 vars, 4 ineq)
  min  (x1-10)^2 + 5*(x2-12)^2 + x3^4 + 3*(x4-11)^2
       + 10*x5^6 + 7*x6^2 + x7^4 - 4*x6*x7 - 10*x6 - 8*x7
  s.t. 127 - 2*x1^2 - 3*x2^4 - x3 - 4*x4^2 - 5*x5 >= 0
       282 - 7*x1 - 3*x2 - 10*x3^2 - x4 + x5 >= 0
       196 - 23*x1 - x2^2 - 6*x6^2 + 8*x7 >= 0
       -4*x1^2 - x2^2 + 3*x1*x2 - 2*x3^2 - 5*x6 + 11*x7 >= 0
  x0 = (1,2,0,4,0,1,1), f* = 680.6300573
"""

import amigo as am
import argparse


class HS100(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=1.0)
        self.add_input("x2", value=2.0)
        self.add_input("x3", value=0.0)
        self.add_input("x4", value=4.0)
        self.add_input("x5", value=0.0)
        self.add_input("x6", value=1.0)
        self.add_input("x7", value=1.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=float("inf"))
        self.add_constraint("c2", lower=0.0, upper=float("inf"))
        self.add_constraint("c3", lower=0.0, upper=float("inf"))
        self.add_constraint("c4", lower=0.0, upper=float("inf"))

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        x5 = self.inputs["x5"]
        x6 = self.inputs["x6"]
        x7 = self.inputs["x7"]
        self.objective["obj"] = (
            (x1 - 10) ** 2
            + 5 * (x2 - 12) ** 2
            + x3**4
            + 3 * (x4 - 11) ** 2
            + 10 * x5**6
            + 7 * x6**2
            + x7**4
            - 4 * x6 * x7
            - 10 * x6
            - 8 * x7
        )
        self.constraints["c1"] = 127 - 2 * x1**2 - 3 * x2**4 - x3 - 4 * x4**2 - 5 * x5
        self.constraints["c2"] = 282 - 7 * x1 - 3 * x2 - 10 * x3**2 - x4 + x5
        self.constraints["c3"] = 196 - 23 * x1 - x2**2 - 6 * x6**2 + 8 * x7
        self.constraints["c4"] = (
            -4 * x1**2 - x2**2 + 3 * x1 * x2 - 2 * x3**2 - 5 * x6 + 11 * x7
        )


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs100")
model.add_component("hs100", 1, HS100())
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
# f* = 680.6300573
