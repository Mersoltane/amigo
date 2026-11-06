"""
Hypersensitive problem
----------------------

    minimize    integral of (y^2 + u^2) / 2 from 0 to 50
    subject to  y' = -y^3 + u,  |y| <= 50,  |u| <= 50
                y = 1.5 at t = 0 and y = 1 at t = 50
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 200

# Boundary values of the state with a free final cost
X0 = [1.5, 0.0]
XF = [1.0, None]

# Fixed final time and the bound on the state and the control
TF = 50.0
BOUND = 50.0


class Plant(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x", shape=2, lower=[-BOUND, -am.inf], upper=[BOUND, am.inf])
        self.add_input("xdot", shape=2)
        self.add_input("u", lower=-BOUND, upper=BOUND)

        self.add_constraint("res", shape=2)

        return

    def compute(self):
        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        y = x[0]

        res = 2 * [None]
        res[0] = xdot[0] - (-y * y * y + u)
        res[1] = xdot[1] - 0.5 * (y * y + u * u)

        self.constraints["res"] = res

        return


# The boundary values of the state and the final cost
class Ends(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x0", shape=2)
        self.add_input("xf", shape=2)

        self.add_constraint("res", shape=3)

        self.add_objective("obj")

        return

    def compute(self):
        x0 = self.inputs["x0"]
        xf = self.inputs["xf"]

        self.constraints["res"] = [x0[0] - X0[0], x0[1] - X0[1], xf[0] - XF[0]]
        self.objective["obj"] = xf[1]

        return


def create_hypersensitive_model(module_name="bench_hypersensitive", num_intervals=200):
    dyn = Plant()
    dynm = Plant()
    hs = common.HermiteSimpsonFixed(TF / num_intervals)
    ends = Ends()

    model = am.Model(module_name)

    model.add_component("dyn", num_intervals + 1, dyn)
    model.add_component("dynm", num_intervals, dynm)
    model.add_component("hs", 2 * num_intervals, hs)
    model.add_component("ends", 1, ends)

    # Link the states and rates to the Hermite Simpson rule
    common.link_collocation(model, "dyn", "dynm", "hs", 2, num_intervals)

    # Link the initial and final states
    model.link("dyn.x[0, :]", "ends.x0[0, :]")
    model.link(f"dyn.x[{num_intervals}, :]", "ends.xf[0, :]")

    return model


def literature_states(tau):
    """State held at one with no cost"""
    return np.column_stack([np.ones_like(tau), np.zeros_like(tau)])


def plant_rates(states, u):
    """Rates of the state and the cost"""
    y = states[:, 0]

    rates = np.zeros(states.shape)
    rates[:, 0] = -y * y * y + u
    rates[:, 1] = 0.5 * (y * y + u * u)

    return rates


def set_initial_guess(model, num_intervals=200, guess="literature"):
    """Initial point of one guess family with the rates of the dynamics"""
    # Normalized times of the nodes and the interval midpoints
    tau = np.linspace(0.0, 1.0, num_intervals + 1)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])

    # The literature start or one of the families
    if guess == "literature":
        states = literature_states(tau)
        mid_states = literature_states(tau_mid)
    else:
        states = common.family_states(guess, X0, XF, tau)
        mid_states = common.family_states(guess, X0, XF, tau_mid)

    # Every start has no control
    u = np.zeros(num_intervals + 1)
    mid_u = np.zeros(num_intervals)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = plant_rates(states, u)
    x["dynm.xdot"] = plant_rates(mid_states, mid_u)

    x["dyn.u"] = u
    x["dynm.u"] = mid_u

    return


def get_objective(x):
    """Objective of a solution with the accumulated cost to minimize"""
    return x["dyn.x"][-1, 1]


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

    model = create_hypersensitive_model(num_intervals=num_intervals)
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
