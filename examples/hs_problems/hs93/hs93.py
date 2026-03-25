"""
HS93: Challenging nonlinear (6 vars, 2 ineq)
  min  0.0204*x1*x4*(x1+x2+x3) + 0.0187*x2*x3*(x1+1.57*x2+x4)
       + 0.0607*x1*x4*x5^2*(x1+x2+x3) + 0.0437*x2*x3*x6^2*(x1+1.57*x2+x4)
  s.t. 0.001*x1*x2*x3*x4*x5*x6 >= 2.07
       0.00062*x1*x4*x5^2*(x1+x2+x3) + 0.00058*x2*x3*x6^2*(x1+1.57*x2+x4) <= 1
       0 <= xi
  x0 = (5.54, 4.4, 12.02, 11.82, 0.702, 0.852), f* = 135.076
"""

import amigo as am
import argparse


class HS93(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=5.54, lower=0.0)
        self.add_input("x2", value=4.4, lower=0.0)
        self.add_input("x3", value=12.02, lower=0.0)
        self.add_input("x4", value=11.82, lower=0.0)
        self.add_input("x5", value=0.702, lower=0.0)
        self.add_input("x6", value=0.852, lower=0.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=float("inf"))
        self.add_constraint("c2", lower=0.0, upper=float("inf"))

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        x5 = self.inputs["x5"]
        x6 = self.inputs["x6"]
        s1 = x1 + x2 + x3
        s2 = x1 + 1.57 * x2 + x4
        self.objective["obj"] = (
            0.0204 * x1 * x4 * s1
            + 0.0187 * x2 * x3 * s2
            + 0.0607 * x1 * x4 * x5**2 * s1
            + 0.0437 * x2 * x3 * x6**2 * s2
        )
        self.constraints["c1"] = 0.001 * x1 * x2 * x3 * x4 * x5 * x6 - 2.07
        self.constraints["c2"] = (
            1 - 0.00062 * x1 * x4 * x5**2 * s1 - 0.00058 * x2 * x3 * x6**2 * s2
        )


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs93")
model.add_component("hs93", 1, HS93())
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
# f* = 135.076
