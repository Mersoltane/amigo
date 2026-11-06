"""
Moon lander
-----------

    minimize    integral of T from 0 to tf
    subject to  h' = v,  v' = -g + T / m,  m' = -T / E,  0 <= T <= Tmax
                |h| <= 20,  |v| <= 20,  0.01 <= m <= 1
                (h, v, m) = (1, -0.783, 1) at t = 0 and (h, v) = (0, 0) at a free tf
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 100

# Gravity, exhaust velocity, and thrust limit of the normalized lander
G = 1.0
EXHAUST = 2.349
T_MAX = 1.227

# Boundary values with a free final mass and a free final cost
X0 = [1.0, -0.783, 1.0, 0.0]
XF = [0.0, 0.0, None, None]
TF_GUESS = 1.5

# State bounds and the mass floors of the problem and of the infeasible variant
STATE_BOUND = 20.0
MASS_FLOOR = 0.01
MASS_FLOOR_INFEASIBLE = 0.99


class Lander(am.Component):
    def __init__(self, mass_floor):
        super().__init__()

        lower = [-STATE_BOUND, -STATE_BOUND, mass_floor, -am.inf]
        upper = [STATE_BOUND, STATE_BOUND, 1.0, am.inf]

        self.add_input("x", shape=4, lower=lower, upper=upper)
        self.add_input("xdot", shape=4)
        self.add_input("u", lower=0.0, upper=T_MAX)

        self.add_constraint("res", shape=4)

        return

    def compute(self):
        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        v = x[1]
        mass = x[2]

        res = 4 * [None]
        res[0] = xdot[0] - v
        res[1] = xdot[1] - (-G + u / mass)
        res[2] = xdot[2] - (-u / EXHAUST)
        res[3] = xdot[3] - u

        self.constraints["res"] = res

        return


# The initial state, the touchdown at rest, and the fuel over a free final time
class Ends(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", lower=0.0, upper=1000.0)
        self.add_input("x0", shape=4)
        self.add_input("xf", shape=4)

        self.add_constraint("res", shape=6)

        self.add_objective("obj")

        return

    def compute(self):
        x0 = self.inputs["x0"]
        xf = self.inputs["xf"]

        res = 6 * [None]
        for i in range(4):
            res[i] = x0[i] - X0[i]
        res[4] = xf[0] - XF[0]
        res[5] = xf[1] - XF[1]

        self.constraints["res"] = res
        self.objective["obj"] = xf[3]

        return


def create_moon_lander_model(
    module_name="bench_moon_lander", num_intervals=100, infeasible=False
):
    # The infeasible variant raises the mass floor
    mass_floor = MASS_FLOOR
    if infeasible:
        mass_floor = MASS_FLOOR_INFEASIBLE

    dyn = Lander(mass_floor)
    dynm = Lander(mass_floor)
    hs = common.HermiteSimpson(num_intervals)
    ends = Ends()

    model = am.Model(module_name)

    model.add_component("dyn", num_intervals + 1, dyn)
    model.add_component("dynm", num_intervals, dynm)
    model.add_component("hs", 4 * num_intervals, hs)
    model.add_component("ends", 1, ends)

    # Link the states and rates to the Hermite Simpson rule
    common.link_collocation(model, "dyn", "dynm", "hs", 4, num_intervals)

    # Link the initial and final states
    model.link("dyn.x[0, :]", "ends.x0[0, :]")
    model.link(f"dyn.x[{num_intervals}, :]", "ends.xf[0, :]")

    # Link the final time to every interval
    model.link("ends.tf[0]", "hs.tf[:]")

    return model


def literature_states(tau):
    """Linear descent to rest with the mass down to its floor and no fuel"""
    return np.column_stack(
        [
            1.0 - tau,
            -0.783 * (1.0 - tau),
            1.0 - (1.0 - MASS_FLOOR) * tau,
            np.zeros_like(tau),
        ]
    )


def lander_rates(states, thrust):
    """Rates of the altitude, the velocity, the mass, and the fuel"""
    v = states[:, 1]
    mass = states[:, 2]

    rates = np.zeros(states.shape)
    rates[:, 0] = v
    rates[:, 1] = -G + thrust / mass
    rates[:, 2] = -thrust / EXHAUST
    rates[:, 3] = thrust

    return rates


def set_initial_guess(model, num_intervals=100, guess="literature"):
    """Initial point of one guess family with the rates of the dynamics"""
    # Normalized times of the nodes and the interval midpoints
    tau = np.linspace(0.0, 1.0, num_intervals + 1)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])

    # The literature start or one of the families
    if guess == "literature":
        states = literature_states(tau)
        mid_states = literature_states(tau_mid)
        tf = TF_GUESS
    else:
        states = common.family_states(guess, X0, XF, tau)
        mid_states = common.family_states(guess, X0, XF, tau_mid)
        tf = common.family_final_time(guess, TF_GUESS)

    # Every start burns at half the thrust limit
    thrust = np.full(num_intervals + 1, 0.5 * T_MAX)
    mid_thrust = np.full(num_intervals, 0.5 * T_MAX)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["ends.tf[0]"] = tf

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = lander_rates(states, thrust)
    x["dynm.xdot"] = lander_rates(mid_states, mid_thrust)

    x["dyn.u"] = thrust
    x["dynm.u"] = mid_thrust

    return


def get_objective(x):
    """Objective of a solution with the fuel to minimize"""
    return x["dyn.x"][-1, 3]


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

    model = create_moon_lander_model(num_intervals=num_intervals)
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

    tf = x["ends.tf"][0]

    print(f"Objective:                  {get_objective(x):.6f}")
    print(f"Final time:                 {tf:.6f}")
