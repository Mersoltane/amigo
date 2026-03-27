"""
HS99: Resource allocation / optimal control (7 vars, 2 eq constraints).
  Discretized trajectory optimization with cumulative state constraints.
  Largest standard HS variant by effective complexity (ODE-like structure).
  f* = -831079892

  Ref: Hock & Schittkowski, Test Examples for Nonlinear Programming Codes,
       Lecture Notes in Economics and Mathematical Systems, v. 187.
"""

import amigo as am
import argparse

# Problem parameters
_A = [0, 50, 50, 75, 75, 75, 100, 100]  # a[1:8] (index 0 unused)
_T = [0, 25, 50, 100, 150, 200, 290, 380]  # t[1:8]
_B = 32


class HS99(am.Component):
    def __init__(self):
        super().__init__()
        # 7 control variables (angles)
        for i in range(7):
            self.add_input(f"x{i+1}", value=0.5, lower=0.0, upper=1.58)
        self.add_objective("obj")
        # 2 endpoint equality constraints: q[8] = 1e5, s[8] = 1e3
        self.add_constraint("h_q", lower=0.0, upper=0.0)
        self.add_constraint("h_s", lower=0.0, upper=0.0)

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(7)]

        # Forward simulation of state recurrences
        # r[1]=0, q[1]=0, s[1]=0
        # For j=2..8 (k=0..6 in 0-indexed):
        #   r[j] = a[j]*dt*cos(x[j-1]) + r[j-1]
        #   q[j] = 0.5*dt^2*(a[j]*sin(x[j-1]) - b) + dt*s[j-1] + q[j-1]
        #   s[j] = dt*(a[j]*sin(x[j-1]) - b) + s[j-1]
        s_val = 0.0
        q_val = 0.0
        obj = 0.0

        for k in range(7):
            dt = _T[k + 1] - _T[k]
            ak = _A[k + 1]

            # q uses OLD s_val, so compute q before updating s
            q_val = q_val + 0.5 * dt**2 * (ak * am.sin(x[k]) - _B) + dt * s_val
            s_val = s_val + dt * (ak * am.sin(x[k]) - _B)
            obj = obj - (ak * dt * am.cos(x[k])) ** 2

        self.objective["obj"] = obj

        # Endpoint constraints
        self.constraints["h_q"] = q_val - 1e5
        self.constraints["h_s"] = s_val - 1e3


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs99")
model.add_component("hs99", 1, HS99())
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
# f* = -831079892
