"""
HS117: Nonconvex QP with matrix structure (15 vars, 5 ineq constraints).
  min  -b^T y + x^T C x + 2*sum_i d_i*x_i^3
  s.t. sum_j c[k][j]*2*x[j] + 3*d[k]*x[k]^2 + e[k]
       - sum_j a[j][k]*y[j] >= 0,  k=1..5
  x_i > 0, y_j > 0
  f* = 32.34867897

  Variables: x[11..15] = x (5 design), x[1..10] = y (10 auxiliary)
"""

import amigo as am
import argparse


class HS117(am.Component):
    def __init__(self):
        super().__init__()
        # x[1..10] = y variables, x[11..15] = x variables
        for i in range(1, 7):
            self.add_input(f"x{i}", value=0.001, lower=1e-8)
        self.add_input("x7", value=60.0, lower=1e-8)
        for i in range(8, 16):
            self.add_input(f"x{i}", value=0.001, lower=1e-8)
        self.add_objective("obj")
        for i in range(5):
            self.add_constraint(f"c{i+1}", lower=0.0, upper=float("inf"))

    def compute(self):
        # y[0..9] = x[1..10], xv[0..4] = x[11..15]
        y = [self.inputs[f"x{i+1}"] for i in range(10)]
        xv = [self.inputs[f"x{i+11}"] for i in range(5)]

        # Parameters
        b = [-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]
        d = [4, 8, 10, 6, 2]
        e = [-15, -27, -36, -18, -12]
        C = [
            [30, -20, -10, 32, -10],
            [-20, 39, -6, -31, 32],
            [-10, -6, 10, -6, -10],
            [32, -31, -6, 39, -20],
            [-10, 32, -10, -20, 30],
        ]
        a = [
            [-16, 2, 0, 1, 0],
            [0, -2, 0, 4, 2],
            [-3.5, 0, 2, 0, 0],
            [0, -2, 0, -4, -1],
            [0, -9, -2, 1, -2.8],
            [2, 0, -4, 0, 0],
            [-1, -1, -1, -1, -1],
            [-1, -2, -3, -2, -1],
            [1, 2, 3, 4, 5],
            [1, 1, 1, 1, 1],
        ]

        # Objective: -b^T y + x^T C x + 2*sum(d_i*x_i^3)
        obj_by = sum(-b[j] * y[j] for j in range(10))
        obj_xCx = sum(C[i][j] * xv[i] * xv[j] for i in range(5) for j in range(5))
        obj_cubic = 2 * sum(d[i] * xv[i] ** 3 for i in range(5))
        self.objective["obj"] = obj_by + obj_xCx + obj_cubic

        # Constraints: sum1[k] - sum2[k] >= 0 for k=0..4
        for k in range(5):
            # sum1[k] = 2*sum(C[k][j]*xv[j]) + 3*d[k]*xv[k]^2 + e[k]
            s1 = 2 * sum(C[k][j] * xv[j] for j in range(5))
            s1 += 3 * d[k] * xv[k] ** 2 + e[k]
            # sum2[k] = sum(a[j][k]*y[j]) over j=0..9
            s2 = sum(a[j][k] * y[j] for j in range(10))
            self.constraints[f"c{k+1}"] = s1 - s2


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs117")
model.add_component("hs117", 1, HS117())
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
# f* = 32.34867897
