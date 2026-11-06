"""
Robot arm
---------

    minimize    tf
    subject to  L rho'' = u1,  I_theta theta'' = u2,  I_phi phi'' = u3,  |u_i| <= 1
                I_phi = ((L - rho)^3 + rho^3) / 3,  I_theta = I_phi sin(phi)^2
                0 <= rho <= L,  |theta| <= pi,  0 <= phi <= pi
                rest at (rho, theta, phi) = (4.5, 0, pi/4) at t = 0
                rest at (rho, theta, phi) = (4.5, 2 pi/3, pi/4) at a free tf
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 100

# Arm length and the rest states at both ends
L = 5.0
X0 = [4.5, 0.0, np.pi / 4.0, 0.0, 0.0, 0.0]
XF = [4.5, 2.0 * np.pi / 3.0, np.pi / 4.0, 0.0, 0.0, 0.0]


class Arm(am.Component):
    def __init__(self):
        super().__init__()

        lower = [0.0, -np.pi, 0.0, -am.inf, -am.inf, -am.inf]
        upper = [L, np.pi, np.pi, am.inf, am.inf, am.inf]

        self.add_input("x", shape=6, lower=lower, upper=upper)
        self.add_input("xdot", shape=6)
        self.add_input("u", shape=3, lower=-1.0, upper=1.0)

        self.add_constraint("res", shape=6)

        return

    def compute(self):
        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        rho = x[0]
        phi = x[2]

        # Moment of inertia of the arm about the pivot
        inertia = ((L - rho) ** 3 + rho**3) / 3.0
        sin_phi = am.sin(phi)

        res = 6 * [None]
        res[0] = xdot[0] - x[3]
        res[1] = xdot[1] - x[4]
        res[2] = xdot[2] - x[5]
        res[3] = xdot[3] - u[0] / L
        res[4] = xdot[4] - u[1] / (inertia * sin_phi * sin_phi)
        res[5] = xdot[5] - u[2] / inertia

        self.constraints["res"] = res

        return


# Rest at both ends and the free final time to minimize
class Ends(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", lower=0.0, upper=am.inf)
        self.add_input("x0", shape=6)
        self.add_input("xf", shape=6)

        self.add_constraint("res", shape=12)

        self.add_objective("obj")

        return

    def compute(self):
        tf = self.inputs["tf"]
        x0 = self.inputs["x0"]
        xf = self.inputs["xf"]

        res = 12 * [None]
        for i in range(6):
            res[i] = x0[i] - X0[i]
            res[6 + i] = xf[i] - XF[i]

        self.constraints["res"] = res
        self.objective["obj"] = tf

        return


def create_robot_arm_model(module_name="bench_robot_arm", num_intervals=100):
    dyn = Arm()
    dynm = Arm()
    hs = common.HermiteSimpson(num_intervals)
    ends = Ends()

    model = am.Model(module_name)

    model.add_component("dyn", num_intervals + 1, dyn)
    model.add_component("dynm", num_intervals, dynm)
    model.add_component("hs", 6 * num_intervals, hs)
    model.add_component("ends", 1, ends)

    # Link the states and rates to the Hermite Simpson rule
    common.link_collocation(model, "dyn", "dynm", "hs", 6, num_intervals)

    # Link the initial and final states
    model.link("dyn.x[0, :]", "ends.x0[0, :]")
    model.link(f"dyn.x[{num_intervals}, :]", "ends.xf[0, :]")

    # Link the final time to every interval
    model.link("ends.tf[0]", "hs.tf[:]")

    return model


def literature_states(tau):
    """Parabolic swing of the arm with the length and the elevation held"""
    swing = 2.0 * np.pi / 3.0
    zero = np.zeros_like(tau)

    return np.column_stack(
        [
            np.full_like(tau, 4.5),
            swing * tau**2,
            np.full_like(tau, np.pi / 4.0),
            zero,
            2.0 * swing * tau,
            zero,
        ]
    )


def arm_rates(states, controls):
    """Rates of the length, the two angles, and their velocities"""
    rho = states[:, 0]
    phi = states[:, 2]

    inertia = ((L - rho) ** 3 + rho**3) / 3.0
    sin_phi = np.sin(phi)

    rates = np.zeros(states.shape)
    rates[:, 0] = states[:, 3]
    rates[:, 1] = states[:, 4]
    rates[:, 2] = states[:, 5]
    rates[:, 3] = controls[:, 0] / L
    rates[:, 4] = controls[:, 1] / (inertia * sin_phi * sin_phi)
    rates[:, 5] = controls[:, 2] / inertia

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
        tf = 1.0
    else:
        states = common.family_states(guess, X0, XF, tau)
        mid_states = common.family_states(guess, X0, XF, tau_mid)
        tf = common.family_final_time(guess, 1.0)

    # Every start has no control
    controls = np.zeros((num_intervals + 1, 3))
    mid_controls = np.zeros((num_intervals, 3))

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["ends.tf[0]"] = tf

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = arm_rates(states, controls)
    x["dynm.xdot"] = arm_rates(mid_states, mid_controls)

    x["dyn.u"] = controls
    x["dynm.u"] = mid_controls

    return


def get_objective(x):
    """Objective of a solution with the final time to minimize"""
    return x["ends.tf"][0]


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

    model = create_robot_arm_model(num_intervals=num_intervals)
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

    print(f"Final time:                 {get_objective(x):.6f}")
