"""
HS71: Canonical NLP test (4 vars, 2 constraints)
  min  x1*x4*(x1+x2+x3) + x3
  s.t. x1*x2*x3*x4 >= 25
       x1^2 + x2^2 + x3^2 + x4^2 = 40
       1 <= xi <= 5
  x0 = (1, 5, 5, 1), f* = 17.0140173
"""

import amigo as am
import argparse


class HS71(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x1", value=1.0, lower=1.0, upper=5.0)
        self.add_input("x2", value=5.0, lower=1.0, upper=5.0)
        self.add_input("x3", value=5.0, lower=1.0, upper=5.0)
        self.add_input("x4", value=1.0, lower=1.0, upper=5.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=25.0, upper=float("inf"))
        self.add_constraint("c2", lower=40.0, upper=40.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        self.objective["obj"] = x1 * x4 * (x1 + x2 + x3) + x3
        self.constraints["c1"] = x1 * x2 * x3 * x4
        self.constraints["c2"] = x1**2 + x2**2 + x3**2 + x4**2


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

model = am.Model("hs71")
model.add_component("hs71", 1, HS71())

if args.build:
    model.build_module()

model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "initial_barrier_param": 0.1,
        "max_iterations": 100,
        "max_line_search_iterations": 30,
        "convergence_tolerance": 1e-8,
        "filter_line_search": True,
    }
)
