"""
Bryson and Denham problem
-------------------------

    minimize    integral of u^2 / 2 from 0 to 1
    subject to  x' = v,  v' = u,  x <= 1/9
                (x, v) = (0, 1) at t = 0 and (0, -1) at t = 1
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 60

# Position limit and the fixed final time
ELL = 1.0 / 9.0
TF = 1.0

# Boundary values of the position and the velocity with a free final cost
X0 = [0.0, 1.0, 0.0]
XF = [0.0, -1.0, None]


class DoubleIntegrator(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x", shape=3, upper=[ELL, am.inf, am.inf])
        self.add_input("xdot", shape=3)
        self.add_input("u")

        self.add_constraint("res", shape=3)

        return

    def compute(self):
        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        v = x[1]

        res = 3 * [None]
        res[0] = xdot[0] - v
        res[1] = xdot[1] - u
        res[2] = xdot[2] - 0.5 * u * u

        self.constraints["res"] = res

        return


# The boundary values of the position and the velocity and the final cost
class Ends(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x0", shape=3)
        self.add_input("xf", shape=3)

        self.add_constraint("res", shape=5)

        self.add_objective("obj")

        return

    def compute(self):
        x0 = self.inputs["x0"]
        xf = self.inputs["xf"]

        self.constraints["res"] = [
            x0[0] - X0[0],
            x0[1] - X0[1],
            x0[2] - X0[2],
            xf[0] - XF[0],
            xf[1] - XF[1],
        ]
        self.objective["obj"] = xf[2]

        return


def create_bryson_denham_model(module_name="bench_bryson_denham", num_intervals=60):
    dyn = DoubleIntegrator()
    dynm = DoubleIntegrator()
    hs = common.HermiteSimpsonFixed(TF / num_intervals)
    ends = Ends()

    model = am.Model(module_name)

    model.add_component("dyn", num_intervals + 1, dyn)
    model.add_component("dynm", num_intervals, dynm)
    model.add_component("hs", 3 * num_intervals, hs)
    model.add_component("ends", 1, ends)

    # Link the states and rates to the Hermite Simpson rule
    common.link_collocation(model, "dyn", "dynm", "hs", 3, num_intervals)

    # Link the initial and final states
    model.link("dyn.x[0, :]", "ends.x0[0, :]")
    model.link(f"dyn.x[{num_intervals}, :]", "ends.xf[0, :]")

    return model


def double_integrator_rates(states, u):
    """Rates of the position, the velocity, and the cost"""
    v = states[:, 1]

    rates = np.zeros(states.shape)
    rates[:, 0] = v
    rates[:, 1] = u
    rates[:, 2] = 0.5 * u * u

    return rates


def set_initial_guess(model, num_intervals=60, guess="linear"):
    """Initial point of the linear or the constant family with the rates of the dynamics"""
    # Normalized times of the nodes and the interval midpoints
    tau = np.linspace(0.0, 1.0, num_intervals + 1)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])

    # The states of the family with no acceleration
    states = common.family_states(guess, X0, XF, tau)
    mid_states = common.family_states(guess, X0, XF, tau_mid)
    u = np.zeros(num_intervals + 1)
    mid_u = np.zeros(num_intervals)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = double_integrator_rates(states, u)
    x["dynm.xdot"] = double_integrator_rates(mid_states, mid_u)

    x["dyn.u"] = u
    x["dynm.u"] = mid_u

    return


def get_objective(x):
    """Objective of a solution with the accumulated cost to minimize"""
    return x["dyn.x"][-1, 2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=False,
        help="Enable building",
    )
    args = parser.parse_args()

    model = create_bryson_denham_model(num_intervals=num_intervals)
    set_initial_guess(model, num_intervals=num_intervals)

    if args.build:
        model.build_module()

    model.initialize()

    print(f"Num variables:              {model.num_variables}")
    print(f"Num constraints:            {model.num_constraints}")

    opt_options = {
        "max_iterations": 1000,
        "max_wall_time": 600.0,
    }

    # Get the design variables
    x = model.create_vector()

    # Set up the optimizer and optimize
    opt = am.Optimizer(model, x=x)
    opt_data = opt.optimize(opt_options)

    # Copy the solution from the device to the host
    x.copy_device_to_host()

    print(f"Objective:                  {get_objective(x):.6f}")
