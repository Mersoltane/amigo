"""
Spacecraft reorientation
------------------------

- Rest to rest slew of an asymmetric rigid spacecraft by 150 degrees about its x axis
- The attitude is a unit quaternion whose scalar part is an algebraic variable
- The unit norm of the quaternion holds as a path constraint
- Minimizes the final time with bounded torques and bounded body rates
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 100

# Principal inertias and the torque and rate limits
IX = 5621.0
IY = 4547.0
IZ = 2364.0
TORQUE = 50.0
RATE = 0.5

# Rest to rest slew about the x axis
PHI = np.radians(150.0)
X0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
XF = [np.sin(PHI / 2.0), 0.0, 0.0, 0.0, 0.0, 0.0]
Q4_0 = 1.0
Q4_F = np.cos(PHI / 2.0)

# Bounds on the final time and the final time of the literature start
TF_LOWER = 25.0
TF_UPPER = 35.0
TF_GUESS = 40.0


class Spacecraft(am.Component):
    def __init__(self):
        super().__init__()

        lower = [-1.0, -1.0, -1.0, -RATE, -RATE, -RATE]
        upper = [1.0, 1.0, 1.0, RATE, RATE, RATE]

        self.add_input("x", shape=6, lower=lower, upper=upper)
        self.add_input("xdot", shape=6)
        self.add_input(
            "u", shape=4, lower=[-TORQUE] * 3 + [-1.0], upper=[TORQUE] * 3 + [1.0]
        )

        self.add_constraint("res", shape=6)
        self.add_constraint("unit")

        return

    def compute(self):
        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        q1 = x[0]
        q2 = x[1]
        q3 = x[2]
        w1 = x[3]
        w2 = x[4]
        w3 = x[5]
        q4 = u[3]

        # Quaternion kinematics and the Euler equations of the body rates
        res = 6 * [None]
        res[0] = xdot[0] - 0.5 * (w1 * q4 - w2 * q3 + w3 * q2)
        res[1] = xdot[1] - 0.5 * (w1 * q3 + w2 * q4 - w3 * q1)
        res[2] = xdot[2] - 0.5 * (-w1 * q2 + w2 * q1 + w3 * q4)
        res[3] = xdot[3] - (u[0] / IX - ((IZ - IY) / IX) * w2 * w3)
        res[4] = xdot[4] - (u[1] / IY - ((IX - IZ) / IY) * w1 * w3)
        res[5] = xdot[5] - (u[2] / IZ - ((IY - IX) / IZ) * w1 * w2)

        self.constraints["res"] = res
        self.constraints["unit"] = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4 - 1.0

        return


# Rest at both attitudes and the free final time to minimize
class Ends(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", lower=TF_LOWER, upper=TF_UPPER)
        self.add_input("x0", shape=6)
        self.add_input("xf", shape=6)
        self.add_input("q40")
        self.add_input("q4f")

        self.add_constraint("res", shape=14)

        self.add_objective("obj")

        return

    def compute(self):
        tf = self.inputs["tf"]
        x0 = self.inputs["x0"]
        xf = self.inputs["xf"]
        q40 = self.inputs["q40"]
        q4f = self.inputs["q4f"]

        res = 14 * [None]
        for i in range(6):
            res[i] = x0[i] - X0[i]
            res[6 + i] = xf[i] - XF[i]
        res[12] = q40 - Q4_0
        res[13] = q4f - Q4_F

        self.constraints["res"] = res
        self.objective["obj"] = tf

        return


def create_reorientation_model(module_name="bench_reorientation", num_intervals=100):
    dyn = Spacecraft()
    dynm = Spacecraft()
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

    # Link the scalar part of the attitude at both ends
    model.link("dyn.u[0, 3]", "ends.q40[0]")
    model.link(f"dyn.u[{num_intervals}, 3]", "ends.q4f[0]")

    # Link the final time to every interval
    model.link("ends.tf[0]", "hs.tf[:]")

    return model


def attitude_controls(torque, q4_end, tau):
    """Equal torques with the scalar part of the quaternion moving linearly"""
    return np.column_stack(
        [np.full((len(tau), 3), torque), Q4_0 + (q4_end - Q4_0) * tau]
    )


def spacecraft_rates(states, controls):
    """Rates of the vector part of the quaternion and of the body rates"""
    q1 = states[:, 0]
    q2 = states[:, 1]
    q3 = states[:, 2]
    w1 = states[:, 3]
    w2 = states[:, 4]
    w3 = states[:, 5]
    q4 = controls[:, 3]

    rates = np.zeros(states.shape)
    rates[:, 0] = 0.5 * (w1 * q4 - w2 * q3 + w3 * q2)
    rates[:, 1] = 0.5 * (w1 * q3 + w2 * q4 - w3 * q1)
    rates[:, 2] = 0.5 * (-w1 * q2 + w2 * q1 + w3 * q4)
    rates[:, 3] = controls[:, 0] / IX - ((IZ - IY) / IX) * w2 * w3
    rates[:, 4] = controls[:, 1] / IY - ((IX - IZ) / IY) * w1 * w3
    rates[:, 5] = controls[:, 2] / IZ - ((IY - IX) / IZ) * w1 * w2

    return rates


def set_initial_guess(model, num_intervals=100, guess="literature"):
    """Initial point of one guess family with the rates of the dynamics"""
    # Normalized times of the nodes and the interval midpoints
    tau = np.linspace(0.0, 1.0, num_intervals + 1)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])

    # The literature start applies full torques
    family = "linear"
    torque = TORQUE
    q4_end = Q4_F
    tf = TF_GUESS

    # The families apply no torque
    if guess != "literature":
        torque = 0.0
        tf = common.family_final_time(guess, 30.0)

    # The constant family holds the attitude
    if guess == "constant":
        family = "constant"
        q4_end = Q4_0

    states = common.family_states(family, X0, XF, tau)
    mid_states = common.family_states(family, X0, XF, tau_mid)
    controls = attitude_controls(torque, q4_end, tau)
    mid_controls = attitude_controls(torque, q4_end, tau_mid)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["ends.tf[0]"] = tf

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = spacecraft_rates(states, controls)
    x["dynm.xdot"] = spacecraft_rates(mid_states, mid_controls)

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

    model = create_reorientation_model(num_intervals=num_intervals)
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
