"""
Goddard rocket
--------------

    maximize    h(tf)
    subject to  h' = v,  v' = (T - D) / m - 1 / h^2,  m' = -T / c,  0 <= T <= Tmax
                D = Dc v^2 exp(-hc (h - 1)),  h >= 1,  v >= 0,  0.6 <= m <= 1
                (h, v, m) = (1, 0, 1) at t = 0 and m = 0.6 at a free final time tf
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 200

# Initial state and the constants of the normalized rocket
H0 = 1.0
V0 = 0.0
M0 = 1.0
G0 = 1.0
HC = 500.0
MC = 0.6
VC = 620.0
T_MAX = 3.5 * G0 * M0
D_C = 0.5 * VC * M0 / G0
C = 0.5 * (G0 * H0) ** 0.5
M_F = MC * M0

# Boundary values with a free final altitude and velocity
X0 = [H0, V0, M0]
XF = [None, None, M_F]


class Rocket(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x", shape=3, lower=[H0, 0.0, M_F], upper=[am.inf, am.inf, M0])
        self.add_input("xdot", shape=3)
        self.add_input("u", lower=0.0, upper=T_MAX)

        self.add_constraint("res", shape=3)

        return

    def compute(self):
        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        h = x[0]
        v = x[1]
        mass = x[2]

        # Drag that decays with the altitude and the inverse square gravity
        drag = D_C * v * v * am.exp(-HC * (h - H0) / H0)
        gravity = G0 * (H0 / h) ** 2

        res = 3 * [None]
        res[0] = xdot[0] - v
        res[1] = xdot[1] - ((u - drag) / mass - gravity)
        res[2] = xdot[2] - (-u / C)

        self.constraints["res"] = res

        return


# The initial state and the final mass
class Ends(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x0", shape=3)
        self.add_input("mf")

        self.add_constraint("res", shape=4)

        return

    def compute(self):
        x0 = self.inputs["x0"]
        mf = self.inputs["mf"]

        self.constraints["res"] = [x0[0] - H0, x0[1] - V0, x0[2] - M0, mf - M_F]

        return


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", lower=0.0, upper=am.inf)
        self.add_input("hf")

        self.add_objective("obj")

        return

    def compute(self):
        hf = self.inputs["hf"]

        self.objective["obj"] = -hf

        return


def create_goddard_model(module_name="bench_goddard", num_intervals=200):
    dyn = Rocket()
    dynm = Rocket()
    hs = common.HermiteSimpson(num_intervals)
    ends = Ends()
    obj = Objective()

    model = am.Model(module_name)

    model.add_component("dyn", num_intervals + 1, dyn)
    model.add_component("dynm", num_intervals, dynm)
    model.add_component("hs", 3 * num_intervals, hs)
    model.add_component("ends", 1, ends)
    model.add_component("obj", 1, obj)

    # Link the states and rates to the Hermite Simpson rule
    common.link_collocation(model, "dyn", "dynm", "hs", 3, num_intervals)

    # Link the initial state and the final mass
    model.link("dyn.x[0, :]", "ends.x0[0, :]")
    model.link(f"dyn.x[{num_intervals}, 2]", "ends.mf[0]")

    # Link the final altitude to the objective
    model.link(f"dyn.x[{num_intervals}, 0]", "obj.hf[0]")

    # Link the final time to every interval
    model.link("obj.tf[0]", "hs.tf[:]")

    return model


def literature_states(tau):
    """Unit altitude, a parabolic velocity, and a linear mass"""
    return np.column_stack(
        [np.ones_like(tau), tau * (1.0 - tau), (M_F - M0) * tau + M0]
    )


def rocket_rates(states, thrust):
    """Rates of the altitude, the velocity, and the mass"""
    h = states[:, 0]
    v = states[:, 1]
    mass = states[:, 2]

    drag = D_C * v * v * np.exp(-HC * (h - H0) / H0)
    gravity = G0 * (H0 / h) ** 2

    rates = np.zeros(states.shape)
    rates[:, 0] = v
    rates[:, 1] = (thrust - drag) / mass - gravity
    rates[:, 2] = -thrust / C

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
        tf = 1.0
    else:
        states = common.family_states(guess, X0, XF, tau)
        mid_states = common.family_states(guess, X0, XF, tau_mid)
        tf = common.family_final_time(guess, 1.0)

    # Every start burns at half the thrust limit
    thrust = np.full(num_intervals + 1, 0.5 * T_MAX)
    mid_thrust = np.full(num_intervals, 0.5 * T_MAX)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["obj.tf[0]"] = tf

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = rocket_rates(states, thrust)
    x["dynm.xdot"] = rocket_rates(mid_states, mid_thrust)

    x["dyn.u"] = thrust
    x["dynm.u"] = mid_thrust

    return


def get_objective(x):
    """Objective of a solution with the final altitude to maximize"""
    return -x["dyn.x"][-1, 0]


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

    model = create_goddard_model(num_intervals=num_intervals)
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

    xs = x["dyn.x"]
    tf = x["obj.tf"][0]

    print(f"Objective:                  {get_objective(x):.6f}")
    print(f"Final altitude:             {xs[-1, 0]:.6f}")
    print(f"Final time:                 {tf:.6f}")
