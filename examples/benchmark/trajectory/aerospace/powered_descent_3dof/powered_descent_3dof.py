"""
3-DoF powered descent
---------------------

- Minimum fuel landing on Mars in the original nonconvex form
- The lander flies from a fixed state to rest at the pad over a free final time
- The thrust magnitude lies between a nonzero minimum and a maximum
- The thrust points within a cone about the vertical
- A glide slope cone, a speed limit, and the rotation of Mars shape the path
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 60

# Planet, vehicle, and constraint parameters
GRAVITY = 3.71
OMEGA = np.radians(np.array([3.5, 0.0, 2.0]) * 1e-3)
M_DRY = 1505.0
M_WET = 1905.0
ALPHA = 1.0 / (225.0 * 9.807)
RHO_MIN = 4971.0
RHO_MAX = 13258.0
GLIDE = np.radians(86.0)
POINTING = np.radians(40.0)
V_MAX = 500.0 / 3.6

# Dry mass of the infeasible variant
M_DRY_INFEASIBLE = 1850.0

# Initial position, velocity, and mass with the landing at rest
X0 = [2000.0, 0.0, 1500.0, 80.0, 30.0, -75.0, M_WET]
XF = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None]
TF_GUESS = 60.0


class Lander(am.Component):
    def __init__(self, dry_mass):
        super().__init__()

        lower = [-am.inf, -am.inf, 0.0, -am.inf, -am.inf, -am.inf, dry_mass]
        upper = [am.inf] * 6 + [M_WET]

        self.add_input("x", shape=7, lower=lower, upper=upper)
        self.add_input("xdot", shape=7)
        self.add_input("u", shape=3, lower=-RHO_MAX, upper=RHO_MAX)

        self.add_constraint("res", shape=7)
        self.add_constraint("thrust", lower=RHO_MIN, upper=RHO_MAX)
        self.add_constraint("pointing", lower=0.0, upper=am.inf)
        self.add_constraint("speed", lower=-am.inf, upper=V_MAX**2)

        return

    def compute(self):
        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        r = [x[0], x[1], x[2]]
        v = [x[3], x[4], x[5]]
        mass = x[6]

        wx = OMEGA[0]
        wy = OMEGA[1]
        wz = OMEGA[2]

        # Centripetal and Coriolis accelerations of the rotating frame
        wr = [wy * r[2] - wz * r[1], wz * r[0] - wx * r[2], wx * r[1] - wy * r[0]]
        wwr = [
            wy * wr[2] - wz * wr[1],
            wz * wr[0] - wx * wr[2],
            wx * wr[1] - wy * wr[0],
        ]
        wv = [wy * v[2] - wz * v[1], wz * v[0] - wx * v[2], wx * v[1] - wy * v[0]]
        gravity = [0.0, 0.0, -GRAVITY]

        thrust = am.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

        res = 7 * [None]
        for i in range(3):
            res[i] = xdot[i] - v[i]
            accel = gravity[i] + u[i] / mass - wwr[i] - 2.0 * wv[i]
            res[3 + i] = xdot[3 + i] - accel
        res[6] = xdot[6] - (-ALPHA * thrust)

        self.constraints["res"] = res
        self.constraints["thrust"] = thrust
        self.constraints["pointing"] = u[2] - np.cos(POINTING) * thrust
        self.constraints["speed"] = x[3] * x[3] + x[4] * x[4] + x[5] * x[5]

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
            np.tan(GLIDE) ** 2 * r[2] * r[2] - r[0] * r[0] - r[1] * r[1]
        )

        return


# The initial state, the landing at rest, and the fuel over a free final time
class Ends(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", lower=0.0, upper=am.inf)
        self.add_input("x0", shape=7)
        self.add_input("xf", shape=7)

        self.add_constraint("res", shape=13)

        self.add_objective("obj")

        return

    def compute(self):
        x0 = self.inputs["x0"]
        xf = self.inputs["xf"]

        res = 13 * [None]
        for i in range(7):
            res[i] = x0[i] - X0[i]
        for i in range(6):
            res[7 + i] = xf[i] - XF[i]

        self.constraints["res"] = res
        self.objective["obj"] = M_WET - xf[6]

        return


def create_powered_descent_3dof_model(
    module_name="bench_powered_descent_3dof", num_intervals=60, infeasible=False
):
    # The infeasible variant raises the dry mass
    dry_mass = M_DRY
    if infeasible:
        dry_mass = M_DRY_INFEASIBLE

    dyn = Lander(dry_mass)
    dynm = Lander(dry_mass)
    hs = common.HermiteSimpson(num_intervals)
    gs = GlideSlope()
    gsm = GlideSlope()
    ends = Ends()

    model = am.Model(module_name)

    model.add_component("dyn", num_intervals + 1, dyn)
    model.add_component("dynm", num_intervals, dynm)
    model.add_component("hs", 7 * num_intervals, hs)
    model.add_component("gs", num_intervals, gs)
    model.add_component("gsm", num_intervals, gsm)
    model.add_component("ends", 1, ends)

    # Link the states and rates to the Hermite Simpson rule
    common.link_collocation(model, "dyn", "dynm", "hs", 7, num_intervals)

    # Link the positions away from the pad to the glide slope
    model.link(f"dyn.x[:{num_intervals}, 0:3]", "gs.r[:, :]")
    model.link("dynm.x[:, 0:3]", "gsm.r[:, :]")

    # Link the initial and final states
    model.link("dyn.x[0, :]", "ends.x0[0, :]")
    model.link(f"dyn.x[{num_intervals}, :]", "ends.xf[0, :]")

    # Link the final time to every interval
    model.link("ends.tf[0]", "hs.tf[:]")

    return model


def lander_rates(states, controls):
    """Rates of the position, the velocity, and the mass in the rotating frame"""
    r = [states[:, 0], states[:, 1], states[:, 2]]
    v = [states[:, 3], states[:, 4], states[:, 5]]
    mass = states[:, 6]
    u = [controls[:, 0], controls[:, 1], controls[:, 2]]

    wx = OMEGA[0]
    wy = OMEGA[1]
    wz = OMEGA[2]

    # Centripetal and Coriolis accelerations of the rotating frame
    wr = [wy * r[2] - wz * r[1], wz * r[0] - wx * r[2], wx * r[1] - wy * r[0]]
    wwr = [wy * wr[2] - wz * wr[1], wz * wr[0] - wx * wr[2], wx * wr[1] - wy * wr[0]]
    wv = [wy * v[2] - wz * v[1], wz * v[0] - wx * v[2], wx * v[1] - wy * v[0]]
    gravity = [0.0, 0.0, -GRAVITY]

    thrust = np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

    rates = np.zeros(states.shape)
    for i in range(3):
        rates[:, i] = v[i]
        rates[:, 3 + i] = gravity[i] + u[i] / mass - wwr[i] - 2.0 * wv[i]
    rates[:, 6] = -ALPHA * thrust

    return rates


def set_initial_guess(model, num_intervals=60, guess="linear"):
    """Initial point of one guess family with the rates of the dynamics"""
    # Normalized times of the nodes and the interval midpoints
    tau = np.linspace(0.0, 1.0, num_intervals + 1)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])

    # The states and the final time of the family
    states = common.family_states(guess, X0, XF, tau)
    mid_states = common.family_states(guess, X0, XF, tau_mid)
    tf = common.family_final_time(guess, TF_GUESS)

    # The nominal thrust hovers the wet vehicle
    hover = [0.0, 0.0, M_WET * GRAVITY]
    controls = np.tile(hover, (num_intervals + 1, 1))
    mid_controls = np.tile(hover, (num_intervals, 1))

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["ends.tf[0]"] = tf

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = lander_rates(states, controls)
    x["dynm.xdot"] = lander_rates(mid_states, mid_controls)

    x["dyn.u"] = controls
    x["dynm.u"] = mid_controls

    return


def get_objective(x):
    """Objective of a solution with the fuel to minimize"""
    return M_WET - x["dyn.x"][-1, 6]


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

    model = create_powered_descent_3dof_model(num_intervals=num_intervals)
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

    print(f"Fuel used:                  {get_objective(x):.6f} kg")
    print(f"Final time:                 {tf:.6f} s")
