"""
Minimum time to climb
---------------------

    minimize    tf
    subject to  m v' = T cos alpha - D - m g sin gamma
                m v gamma' = T sin alpha + L - m g cos gamma
                h' = v sin gamma,  r' = v cos gamma,  m' = -T / (g Isp)
                (v, gamma, h, r, m) = (136, 0, 100, 0, 19030) at t = 0
                (v, gamma, h) = (340, 0, 20000) at t = tf
"""

import argparse
import json
import amigo as am
from amigo.interp import BSpline
import numpy as np

num_time_steps = 500

# Number of bspline control points
nctrl = 10

# Units of the scaled velocity, altitude, range, and mass
V_REF = 100.0
H_REF = 1000.0
R_REF = 1000.0
M_REF = 1000.0


class TrapezoidRule(am.Component):
    def __init__(self, num_time_steps=500):
        super().__init__()

        self.add_constant("num_time_steps", value=num_time_steps)

        self.add_input("tf", lower=1.0, upper=am.inf)

        self.add_input("q1")
        self.add_input("q2")

        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_constraint("res")

        return

    def compute(self):
        num_time_steps = self.constants["num_time_steps"]

        tf = self.inputs["tf"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        dt = tf / num_time_steps

        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return


class AircraftDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("S", value=49.2386)
        self.add_constant("CL_alpha", value=3.44)
        self.add_constant("CD0", value=0.013)
        self.add_constant("kappa", value=0.54)

        self.add_constant("Isp", value=1600.0)
        self.add_constant("TtoW", value=0.9)
        self.add_constant("m0", value=19030.0)

        self.add_constant("g", value=9.81)
        self.add_constant("conv", value=np.pi / 180.0)

        self.add_input("alpha", label="angle of attack (degrees)")
        self.add_input("q", shape=5, label="state variables")
        self.add_input("qdot", shape=5, label="state derivatives")

        self.add_constraint("res", shape=5, label="dynamics residual")

        return

    def compute(self):
        S = self.constants["S"]
        CL_alpha = self.constants["CL_alpha"]
        CD0 = self.constants["CD0"]
        kappa = self.constants["kappa"]
        Isp = self.constants["Isp"]
        TtoW = self.constants["TtoW"]
        m0 = self.constants["m0"]
        g = self.constants["g"]
        conv = self.constants["conv"]

        alpha = self.inputs["alpha"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Velocity in m/s, flight path angle in degrees, and mass in kg
        v = V_REF * q[0]
        gamma = q[1]
        m = M_REF * q[4]

        # Sea level air density
        rho = 1.225

        # Lift and drag coefficients
        CL = CL_alpha * conv * alpha
        CD = CD0 + kappa * CL**2

        # Drag, lift, and thrust forces
        qinfty = 0.5 * rho * v**2
        D = qinfty * S * CD
        L = qinfty * S * CL
        T = TtoW * m0 * g

        # The angles in radians
        alpha_rad = conv * alpha
        gamma_rad = conv * gamma

        sin_alpha = am.sin(alpha_rad)
        cos_alpha = am.cos(alpha_rad)
        sin_gamma = am.sin(gamma_rad)
        cos_gamma = am.cos(gamma_rad)

        # Rates of the velocity and of the flight path angle in radians
        vdot = (T / m) * cos_alpha - (D / m) - g * sin_gamma
        gammadot = T / (m * v) * sin_alpha + (L / (m * v)) - (g / v) * cos_gamma

        # Rates of the altitude and the range
        hdot = v * sin_gamma
        rdot = v * cos_gamma

        # Fuel flow of the engine
        mflow = T / (g * Isp)

        # Residuals of the scaled states with the flight path angle in degrees
        res = 5 * [None]
        res[0] = qdot[0] - vdot / V_REF
        res[1] = conv * qdot[1] - gammadot
        res[2] = qdot[2] - hdot / H_REF
        res[3] = qdot[3] - rdot / R_REF
        res[4] = qdot[4] + mflow / M_REF

        self.constraints["res"] = res

        return


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", label="final time")

        self.add_objective("obj")

        return

    def compute(self):
        tf = self.inputs["tf"]

        self.objective["obj"] = tf

        return


# The aircraft starts level at 136 m/s and 100 m with a mass of 19030 kg
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=5)

        self.add_constraint("res", shape=5)

        return

    def compute(self):
        q = self.inputs["q"]

        self.constraints["res"] = [
            q[0] - 136.0 / V_REF,
            q[1] - 0.0,
            q[2] - 100.0 / H_REF,
            q[3] - 0.0,
            q[4] - 19030.0 / M_REF,
        ]

        return


# The aircraft ends level at 340 m/s and 20 km
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=5)

        self.add_constraint("res", shape=3)

        return

    def compute(self):
        q = self.inputs["q"]

        self.constraints["res"] = [
            q[0] - 340.0 / V_REF,
            q[1] - 0.0,
            q[2] - 20000.0 / H_REF,
        ]

        return


def create_time_to_climb_model(
    module_name="time_to_climb", num_time_steps=500, nctrl=10
):
    ac = AircraftDynamics()
    trap = TrapezoidRule(num_time_steps=num_time_steps)
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()

    # Interpolate from the x control points to the angle of attack
    bspline = BSpline(
        input_name="x",
        output_name="alpha",
        num_interp_points=(num_time_steps + 1),
        num_ctrl_points=nctrl,
    )

    model = am.Model(module_name)

    model.add_component("ac", num_time_steps + 1, ac)
    model.add_component("trap", 5 * num_time_steps, trap)
    model.add_model("bspline", bspline.create_model())
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link the states and rates of every state to the trapezoid rule
    for i in range(5):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link the states at both ends of every interval
        model.link(f"ac.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"ac.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates at both ends of every interval
        model.link(f"ac.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"ac.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link the angle of attack to the bspline values
    model.link("ac.alpha", "bspline.interp_values.alpha")

    # Link the final time to every interval
    model.link("obj.tf[0]", "trap.tf[:]")

    # Link the boundary conditions
    model.link("ac.q[0, :]", "ic.q[0, :]")
    model.link(f"ac.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Bound the flight path angle and the altitude
    model.set_meta("lower", "ac.q[:, 1]", -90.0)
    model.set_meta("upper", "ac.q[:, 1]", 90.0)
    model.set_meta("lower", "ac.q[:, 2]", 0.0)
    model.set_meta("upper", "ac.q[:, 2]", 25.0)

    # Bound the angle of attack through its control points
    model.set_meta("lower", "bspline.control_points.x", -25.0)
    model.set_meta("upper", "bspline.control_points.x", 25.0)

    return model


def set_initial_guess(model, num_time_steps=500):
    """Initial point from simple state profiles over a guessed final time"""
    tf = 200.0
    t = np.linspace(0, tf, num_time_steps + 1)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["obj.tf"] = tf

    # The states move linearly in time with an arc in the flight path angle
    x["ac.q[:, 0]"] = 1.36 + (3.40 - 1.36) * t / tf
    x["ac.q[:, 1]"] = 5.0 * np.sin(np.pi * t / tf)
    x["ac.q[:, 2]"] = 1.0 + (20.0 - 1.0) * t / tf
    x["ac.q[:, 3]"] = 1.36 * t
    x["ac.q[:, 4]"] = 19.03 - 0.2 * t / tf

    # The angle of attack starts at one degree
    x["ac.alpha"] = 1.0

    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

model = create_time_to_climb_model(num_time_steps=num_time_steps, nctrl=nctrl)
set_initial_guess(model, num_time_steps=num_time_steps)

if args.build:
    model.build_module()

model.initialize()

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

opt_options = {
    "convergence_tolerance": 1e-10,
    "max_line_search_iterations": 5,
    "max_iterations": 1000,
}

# Get the design variables
x = model.create_vector()

# Set up the optimizer and optimize
opt = am.Optimizer(model, x=x)
opt_data = opt.optimize(opt_options)

# Copy the solution from the device to the host
x.copy_device_to_host()

tf = x["obj.tf"][0]
q = x["ac.q"]
alpha = x["ac.alpha"]

print(f"Optimal time:               {tf:.2f} seconds")
print(f"Final altitude:             {q[-1, 2]:.0f} m")
print(f"Final velocity:             {q[-1, 0]:.1f} m/s")
print(f"Final mass:                 {q[-1, 4]:.0f} kg")

# Save the optimization data
with open("time_to_climb_opt_data.json", "w") as fp:
    json.dump(opt_data, fp, indent=2)

# Save the solution for plot_time_to_climb.py
t = np.linspace(0, tf, num_time_steps + 1)

with open("time_to_climb_solution.json", "w") as fp:
    json.dump({"t": t.tolist(), "q": q.tolist(), "alpha": alpha.tolist()}, fp)
