"""
HS87: Electrical network design (6 vars + 6 aux, 7 eq constraints).
  Piecewise linear cost with nonlinear trig constraints.
  Multiple local minima — stress test for convergence to correct basin.
  f* = 8827.5977

  Ref: Hock & Schittkowski, Test Examples for Nonlinear Programming Codes,
       Lecture Notes in Economics and Mathematical Systems, v. 187.
"""

import amigo as am
import argparse
import math


class HS87(am.Component):
    def __init__(self):
        super().__init__()
        # Parameters
        self._pa = 131.078
        self._pb = 1.48477
        self._pc = 0.90798
        self._pd = math.cos(1.47588)
        self._pe = math.sin(1.47588)

        # Design variables x[1:6]
        self.add_input("x1", value=390.0, lower=0.0, upper=400.0)
        self.add_input("x2", value=1000.0, lower=0.0, upper=1000.0)
        self.add_input("x3", value=419.5, lower=340.0, upper=420.0)
        self.add_input("x4", value=340.5, lower=340.0, upper=420.0)
        self.add_input("x5", value=198.175, lower=-1000.0, upper=1000.0)
        self.add_input("x6", value=0.5, lower=0.0, upper=0.5236)

        # Auxiliary variables for piecewise linear cost
        self.add_input("add1", value=90.0, lower=0.0)  # max(0, x1-300)
        self.add_input("add2", value=900.0, lower=0.0)  # max(0, x2-100)
        self.add_input("add3", value=800.0, lower=0.0)  # max(0, x2-200)
        self.add_input("slk1", value=0.0)  # slack (free)
        self.add_input("slk2", value=0.0)  # slack (free)
        self.add_input("slk3", value=0.0)  # slack (free)

        self.add_objective("obj")
        # 7 equality constraints
        for i in range(7):
            self.add_constraint(f"h{i+1}", lower=0.0, upper=0.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        x4 = self.inputs["x4"]
        x5 = self.inputs["x5"]
        x6 = self.inputs["x6"]
        add1 = self.inputs["add1"]
        add2 = self.inputs["add2"]
        add3 = self.inputs["add3"]
        slk1 = self.inputs["slk1"]
        slk2 = self.inputs["slk2"]
        slk3 = self.inputs["slk3"]

        pa = self._pa
        pb = self._pb
        pc = self._pc
        pd = self._pd
        pe = self._pe

        # Objective: piecewise linear cost
        # rate = [30, 31, 28, 29, 30]
        self.objective["obj"] = (
            30 * x1
            + 1 * add1  # rate[1]*x1 + (rate[2]-rate[1])*add1
            + 28 * x2
            + 1 * add2
            + 1 * add3  # rate[3]*x2 + (rate[4]-rate[3])*add2 + (rate[5]-rate[4])*add3
        )

        # Auxiliary variable definitions
        self.constraints["h1"] = add1 - x1 + 300 + slk1
        self.constraints["h2"] = add2 - x2 + 100 + slk2
        self.constraints["h3"] = add3 - x2 + 200 + slk3

        # Nonlinear trig constraints
        self.constraints["h4"] = (
            x1 - 300 + x3 * x4 * am.cos(pb - x6) / pa - pc * x3**2 * pd / pa
        )
        self.constraints["h5"] = (
            x2 + x3 * x4 * am.cos(pb + x6) / pa - pc * x4**2 * pd / pa
        )
        self.constraints["h6"] = (
            x5 + x3 * x4 * am.sin(pb + x6) / pa - pc * x4**2 * pe / pa
        )
        self.constraints["h7"] = (
            200 - x3 * x4 * am.sin(pb - x6) / pa + pc * x3**2 * pe / pa
        )


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs87")
model.add_component("hs87", 1, HS87())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 300,
        "filter_line_search": True,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = 8827.5977
