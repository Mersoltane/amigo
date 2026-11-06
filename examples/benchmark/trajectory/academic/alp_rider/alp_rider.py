"""
Alp rider
---------

- Four stable linear states driven by two controls between fixed ends over a fixed horizon
- The squared norm of the states must clear a sum of four narrow Gaussian peaks
- Minimizes the integral of the squared states with a small weight on the controls
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 200

# Boundary values with a free final cost
X0 = [2.0, 1.0, 2.0, 1.0, 0.0]
XF = [2.0, 3.0, 1.0, -2.0, None]

# Fixed final time and the bounds on the states and the controls
TF = 20.0
STATE_BOUND = 4.0
CONTROL_BOUND = 500.0

# Height, center, and decay rate of each of the four peaks
PEAKS = [(3.0, 3.0, 12.0), (3.0, 6.0, 10.0), (3.0, 10.0, 6.0), (8.0, 15.0, 4.0)]


class Rider(am.Component):
    def __init__(self):
        super().__init__()

        self.add_data("floor")

        lower = [-STATE_BOUND] * 4 + [-am.inf]
        upper = [STATE_BOUND] * 4 + [am.inf]

        self.add_input("x", shape=5, lower=lower, upper=upper)
        self.add_input("xdot", shape=5)
        self.add_input("u", shape=2, lower=-CONTROL_BOUND, upper=CONTROL_BOUND)

        self.add_constraint("res", shape=5)
        self.add_constraint("path", lower=0.0, upper=100.0)

        return

    def compute(self):
        floor = self.data["floor"]

        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        y1 = x[0]
        y2 = x[1]
        y3 = x[2]
        y4 = x[3]

        u1 = u[0]
        u2 = u[1]

        norm2 = y1 * y1 + y2 * y2 + y3 * y3 + y4 * y4

        res = 5 * [None]
        res[0] = xdot[0] - (-10.0 * y1 + u1 + u2)
        res[1] = xdot[1] - (-2.0 * y2 + u1 + 2.0 * u2)
        res[2] = xdot[2] - (-3.0 * y3 + 5.0 * y4 + u1 - u2)
        res[3] = xdot[3] - (5.0 * y3 - 3.0 * y4 + u1 + 3.0 * u2)
        res[4] = xdot[4] - (100.0 * norm2 + 0.01 * (u1 * u1 + u2 * u2))

        self.constraints["res"] = res
        self.constraints["path"] = norm2 - floor

        return


# The boundary values of the four states and the final cost
class Ends(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x0", shape=5)
        self.add_input("xf", shape=5)

        self.add_constraint("res", shape=9)

        self.add_objective("obj")

        return

    def compute(self):
        x0 = self.inputs["x0"]
        xf = self.inputs["xf"]

        res = 9 * [None]
        for i in range(5):
            res[i] = x0[i] - X0[i]
        for i in range(4):
            res[5 + i] = xf[i] - XF[i]

        self.constraints["res"] = res
        self.objective["obj"] = xf[4]

        return


def terrain(t):
    """Height of the peaks that the squared state norm must clear"""
    height = 0.0

    for peak in PEAKS:
        h = peak[0]
        a = peak[1]
        b = peak[2]
        height = height + h * np.exp(-b * (t - a) ** 2)

    return height + 0.01


def create_alp_rider_model(module_name="bench_alp_rider", num_intervals=200):
    dyn = Rider()
    dynm = Rider()
    hs = common.HermiteSimpsonFixed(TF / num_intervals)
    ends = Ends()

    model = am.Model(module_name)

    model.add_component("dyn", num_intervals + 1, dyn)
    model.add_component("dynm", num_intervals, dynm)
    model.add_component("hs", 5 * num_intervals, hs)
    model.add_component("ends", 1, ends)

    # Link the states and rates to the Hermite Simpson rule
    common.link_collocation(model, "dyn", "dynm", "hs", 5, num_intervals)

    # Link the initial and final states
    model.link("dyn.x[0, :]", "ends.x0[0, :]")
    model.link(f"dyn.x[{num_intervals}, :]", "ends.xf[0, :]")

    # The peak heights at the nodes and the midpoints
    tau = np.linspace(0.0, 1.0, num_intervals + 1)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])
    model.set_meta("value", "dyn.floor", terrain(TF * tau), vtype="data")
    model.set_meta("value", "dynm.floor", terrain(TF * tau_mid), vtype="data")

    return model


def rider_rates(states, controls):
    """Rates of the four states and the cost"""
    y1 = states[:, 0]
    y2 = states[:, 1]
    y3 = states[:, 2]
    y4 = states[:, 3]

    u1 = controls[:, 0]
    u2 = controls[:, 1]

    norm2 = y1 * y1 + y2 * y2 + y3 * y3 + y4 * y4

    rates = np.zeros(states.shape)
    rates[:, 0] = -10.0 * y1 + u1 + u2
    rates[:, 1] = -2.0 * y2 + u1 + 2.0 * u2
    rates[:, 2] = -3.0 * y3 + 5.0 * y4 + u1 - u2
    rates[:, 3] = 5.0 * y3 - 3.0 * y4 + u1 + 3.0 * u2
    rates[:, 4] = 100.0 * norm2 + 0.01 * (u1 * u1 + u2 * u2)

    return rates


def set_initial_guess(model, num_intervals=200, guess="literature"):
    """Initial point of one guess family with the rates of the dynamics"""
    # Normalized times of the nodes and the interval midpoints
    tau = np.linspace(0.0, 1.0, num_intervals + 1)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])

    # The literature start is the linear family
    family = guess
    if guess == "literature":
        family = "linear"

    states = common.family_states(family, X0, XF, tau)
    mid_states = common.family_states(family, X0, XF, tau_mid)

    # Every start has no control
    controls = np.zeros((num_intervals + 1, 2))
    mid_controls = np.zeros((num_intervals, 2))

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = rider_rates(states, controls)
    x["dynm.xdot"] = rider_rates(mid_states, mid_controls)

    x["dyn.u"] = controls
    x["dynm.u"] = mid_controls

    return


def get_objective(x):
    """Objective of a solution with the accumulated cost to minimize"""
    return x["dyn.x"][-1, 4]


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

    model = create_alp_rider_model(num_intervals=num_intervals)
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
