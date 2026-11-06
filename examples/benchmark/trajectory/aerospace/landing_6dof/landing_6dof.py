"""
6-DoF rocket landing
--------------------

- Minimum fuel landing of a rigid body rocket over a free final time
- A gimbaled engine below the center of mass with a nonzero lower bound on the thrust
- The attitude is a unit quaternion with the scalar part first
- Mass, position, velocity, and body rates are fixed at ignition and the attitude is free
- The rocket lands upright at rest except for a small descent rate
- Glide slope, gimbal, tilt, and body rate limits hold along the path
- Units are nondimensional with the up axis first
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 30

# Vehicle and constraint parameters
INERTIA = 0.168 * np.array([2e-2, 1.0, 1.0])
ARM = 0.25
ALPHA = 1.0 / 30.0
M_DRY = 1.0
M_IG = 2.0
T_MIN = 1.5
T_MAX = 6.5
GIMBAL = np.radians(20.0)
GLIDE = np.radians(75.0)
RATE = np.radians(28.6)

# State order is mass, position, velocity, quaternion with the scalar first, and body rates
X0 = [M_IG, 4.33, 3.5, 0.0, -0.5, -2.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
XF = [None, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# States fixed at ignition and at touchdown
FIXED_START = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13]
FIXED_END = list(range(1, 14))
TF_GUESS = 5.0


class Rocket(am.Component):
    def __init__(self):
        super().__init__()

        lower = [M_DRY, 0.0] + [-am.inf] * 12
        upper = [M_IG] + [am.inf] * 13

        self.add_input("x", shape=14, lower=lower, upper=upper)
        self.add_input("xdot", shape=14)
        self.add_input("u", shape=3, lower=-T_MAX, upper=T_MAX)

        self.add_constraint("res", shape=14)
        self.add_constraint("thrust", lower=T_MIN, upper=T_MAX)
        self.add_constraint("gimbal", lower=0.0, upper=am.inf)
        self.add_constraint("tilt", lower=0.0, upper=am.inf)
        self.add_constraint("rate", lower=-am.inf, upper=RATE**2)

        return

    def compute(self):
        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        mass = x[0]
        q0 = x[7]
        q1 = x[8]
        q2 = x[9]
        q3 = x[10]
        w1 = x[11]
        w2 = x[12]
        w3 = x[13]

        t1 = u[0]
        t2 = u[1]
        t3 = u[2]

        j1 = INERTIA[0]
        j2 = INERTIA[1]
        j3 = INERTIA[2]

        # Body to inertial rotation as the transpose of the inertial to body matrix
        c = [
            [
                1.0 - 2.0 * (q2 * q2 + q3 * q3),
                2.0 * (q1 * q2 + q0 * q3),
                2.0 * (q1 * q3 - q0 * q2),
            ],
            [
                2.0 * (q1 * q2 - q0 * q3),
                1.0 - 2.0 * (q1 * q1 + q3 * q3),
                2.0 * (q2 * q3 + q0 * q1),
            ],
            [
                2.0 * (q1 * q3 + q0 * q2),
                2.0 * (q2 * q3 - q0 * q1),
                1.0 - 2.0 * (q1 * q1 + q2 * q2),
            ],
        ]

        # Thrust in the inertial frame
        force = 3 * [None]
        for i in range(3):
            force[i] = c[0][i] * t1 + c[1][i] * t2 + c[2][i] * t3

        thrust = am.sqrt(t1 * t1 + t2 * t2 + t3 * t3)

        res = 14 * [None]
        res[0] = xdot[0] - (-ALPHA * thrust)
        res[1] = xdot[1] - x[4]
        res[2] = xdot[2] - x[5]
        res[3] = xdot[3] - x[6]

        # Translation under the thrust and the unit gravity along the up axis
        res[4] = xdot[4] - (force[0] / mass - 1.0)
        res[5] = xdot[5] - force[1] / mass
        res[6] = xdot[6] - force[2] / mass

        # Quaternion kinematics with the scalar part first
        res[7] = xdot[7] - 0.5 * (-w1 * q1 - w2 * q2 - w3 * q3)
        res[8] = xdot[8] - 0.5 * (w1 * q0 + w3 * q2 - w2 * q3)
        res[9] = xdot[9] - 0.5 * (w2 * q0 - w3 * q1 + w1 * q3)
        res[10] = xdot[10] - 0.5 * (w3 * q0 + w2 * q1 - w1 * q2)

        # Engine torque from the arm below the center of mass minus the gyroscopic term
        res[11] = xdot[11] - (-(j3 - j2) * w2 * w3 / j1)
        res[12] = xdot[12] - (ARM * t3 - (j1 - j3) * w3 * w1) / j2
        res[13] = xdot[13] - (-ARM * t2 - (j2 - j1) * w1 * w2) / j3

        self.constraints["res"] = res
        self.constraints["thrust"] = thrust
        self.constraints["gimbal"] = u[0] - np.cos(GIMBAL) * thrust
        self.constraints["tilt"] = 1.0 - 2.0 * (x[9] * x[9] + x[10] * x[10])
        self.constraints["rate"] = x[11] * x[11] + x[12] * x[12] + x[13] * x[13]

        return


# The glide slope cone in squared form away from the pad
class GlideSlope(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("r", shape=3)

        self.add_constraint("cone", lower=0.0, upper=am.inf)

        return

    def compute(self):
        r = self.inputs["r"]

        self.constraints["cone"] = (
            np.tan(GLIDE) ** 2 * r[0] * r[0] - r[1] * r[1] - r[2] * r[2]
        )

        return


# The ignition state, the upright landing, and the fuel over a free final time
class Ends(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", lower=0.0, upper=am.inf)
        self.add_input("x0", shape=14)
        self.add_input("xf", shape=14)

        self.add_constraint("res", shape=len(FIXED_START) + len(FIXED_END))

        self.add_objective("obj")

        return

    def compute(self):
        x0 = self.inputs["x0"]
        xf = self.inputs["xf"]

        res = []
        for i in FIXED_START:
            res.append(x0[i] - X0[i])
        for i in FIXED_END:
            res.append(xf[i] - XF[i])

        self.constraints["res"] = res
        self.objective["obj"] = M_IG - xf[0]

        return


def create_landing_6dof_model(module_name="bench_landing_6dof", num_intervals=30):
    dyn = Rocket()
    dynm = Rocket()
    hs = common.HermiteSimpson(num_intervals)
    gs = GlideSlope()
    gsm = GlideSlope()
    ends = Ends()

    model = am.Model(module_name)

    model.add_component("dyn", num_intervals + 1, dyn)
    model.add_component("dynm", num_intervals, dynm)
    model.add_component("hs", 14 * num_intervals, hs)
    model.add_component("gs", num_intervals, gs)
    model.add_component("gsm", num_intervals, gsm)
    model.add_component("ends", 1, ends)

    # Link the states and rates to the Hermite Simpson rule
    common.link_collocation(model, "dyn", "dynm", "hs", 14, num_intervals)

    # Link the positions away from the pad to the glide slope
    model.link(f"dyn.x[:{num_intervals}, 1:4]", "gs.r[:, :]")
    model.link("dynm.x[:, 1:4]", "gsm.r[:, :]")

    # Link the initial and final states
    model.link("dyn.x[0, :]", "ends.x0[0, :]")
    model.link(f"dyn.x[{num_intervals}, :]", "ends.xf[0, :]")

    # Link the final time to every interval
    model.link("ends.tf[0]", "hs.tf[:]")

    return model


def literature_states(tau):
    """Straight line descent with the mass falling to dry and the attitude upright"""
    mass = M_IG + (M_DRY - M_IG) * tau
    r = np.outer(1.0 - tau, X0[1:4])
    v = np.outer(1.0 - tau, X0[4:7]) + np.outer(tau, XF[4:7])
    q = np.tile([1.0, 0.0, 0.0, 0.0], (len(tau), 1))

    return np.column_stack([mass, r, v, q, np.zeros((len(tau), 3))])


def literature_controls(tau):
    """Thrust along the up axis that hovers the falling mass"""
    mass = M_IG + (M_DRY - M_IG) * tau

    return np.column_stack([mass, np.zeros_like(tau), np.zeros_like(tau)])


def rocket_rates(states, controls):
    """Rates of the mass, the translation, the attitude, and the body rates"""
    mass = states[:, 0]
    q0 = states[:, 7]
    q1 = states[:, 8]
    q2 = states[:, 9]
    q3 = states[:, 10]
    w1 = states[:, 11]
    w2 = states[:, 12]
    w3 = states[:, 13]

    t1 = controls[:, 0]
    t2 = controls[:, 1]
    t3 = controls[:, 2]

    j1 = INERTIA[0]
    j2 = INERTIA[1]
    j3 = INERTIA[2]

    # Body to inertial rotation as the transpose of the inertial to body matrix
    c = [
        [
            1.0 - 2.0 * (q2 * q2 + q3 * q3),
            2.0 * (q1 * q2 + q0 * q3),
            2.0 * (q1 * q3 - q0 * q2),
        ],
        [
            2.0 * (q1 * q2 - q0 * q3),
            1.0 - 2.0 * (q1 * q1 + q3 * q3),
            2.0 * (q2 * q3 + q0 * q1),
        ],
        [
            2.0 * (q1 * q3 + q0 * q2),
            2.0 * (q2 * q3 - q0 * q1),
            1.0 - 2.0 * (q1 * q1 + q2 * q2),
        ],
    ]

    # Thrust in the inertial frame
    force = 3 * [None]
    for i in range(3):
        force[i] = c[0][i] * t1 + c[1][i] * t2 + c[2][i] * t3

    thrust = np.sqrt(t1 * t1 + t2 * t2 + t3 * t3)

    rates = np.zeros(states.shape)
    rates[:, 0] = -ALPHA * thrust
    rates[:, 1] = states[:, 4]
    rates[:, 2] = states[:, 5]
    rates[:, 3] = states[:, 6]
    rates[:, 4] = force[0] / mass - 1.0
    rates[:, 5] = force[1] / mass
    rates[:, 6] = force[2] / mass
    rates[:, 7] = 0.5 * (-w1 * q1 - w2 * q2 - w3 * q3)
    rates[:, 8] = 0.5 * (w1 * q0 + w3 * q2 - w2 * q3)
    rates[:, 9] = 0.5 * (w2 * q0 - w3 * q1 + w1 * q3)
    rates[:, 10] = 0.5 * (w3 * q0 + w2 * q1 - w1 * q2)
    rates[:, 11] = -(j3 - j2) * w2 * w3 / j1
    rates[:, 12] = (ARM * t3 - (j1 - j3) * w3 * w1) / j2
    rates[:, 13] = (-ARM * t2 - (j2 - j1) * w1 * w2) / j3

    return rates


def set_initial_guess(model, num_intervals=30, guess="literature"):
    """Initial point of one guess family with the rates of the dynamics"""
    # Normalized times of the nodes and the interval midpoints
    tau = np.linspace(0.0, 1.0, num_intervals + 1)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])

    # The literature start or one of the families with the ignition thrust
    if guess == "literature":
        states = literature_states(tau)
        mid_states = literature_states(tau_mid)
        controls = literature_controls(tau)
        mid_controls = literature_controls(tau_mid)
        tf = TF_GUESS
    else:
        states = common.family_states(guess, X0, XF, tau)
        mid_states = common.family_states(guess, X0, XF, tau_mid)
        controls = np.tile([M_IG, 0.0, 0.0], (num_intervals + 1, 1))
        mid_controls = np.tile([M_IG, 0.0, 0.0], (num_intervals, 1))
        tf = common.family_final_time(guess, TF_GUESS)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["ends.tf[0]"] = tf

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = rocket_rates(states, controls)
    x["dynm.xdot"] = rocket_rates(mid_states, mid_controls)

    x["dyn.u"] = controls
    x["dynm.u"] = mid_controls

    return


def get_objective(x):
    """Objective of a solution with the fuel to minimize"""
    return M_IG - x["dyn.x"][-1, 0]


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

    model = create_landing_6dof_model(num_intervals=num_intervals)
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

    print(f"Fuel used:                  {get_objective(x):.6f}")
    print(f"Final time:                 {tf:.6f}")
