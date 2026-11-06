"""
Maximum range of a hang glider
------------------------------

- The lift coefficient is the control
- The glider descends with the same speed at both ends
"""

import argparse
import json
import amigo as am
import numpy as np

num_time_steps = 100

# Scales of the distance, the velocity, and the time
DISTANCE_SCALE = 100.0
VELOCITY_SCALE = 10.0
TIME_SCALE = 10.0

# Launch point and velocity in m and m/s
X0 = 0.0
Y0 = 1000.0
VX0 = 13.227567500
VY0 = -1.2875005200

# Final altitude and velocity in m and m/s
YF = 900.0
VXF = 13.227567500
VYF = -1.2875005200


class TrapezoidRule(am.Component):
    def __init__(self, num_time_steps=100):
        super().__init__()

        self.add_constant("num_time_steps", value=num_time_steps)

        self.add_input("tf", lower=50.0 / TIME_SCALE, upper=200.0 / TIME_SCALE)

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

        dt = TIME_SCALE * tf / num_time_steps

        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return


class GliderDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("uM", value=2.5)
        self.add_constant("m", value=100.0)
        self.add_constant("R", value=100.0)
        self.add_constant("S", value=14.0)
        self.add_constant("C0", value=0.034)
        self.add_constant("rho", value=1.13)
        self.add_constant("k", value=0.069662)
        self.add_constant("g", value=9.80665)

        self.add_input("CL", label="lift coefficient", lower=0.0, upper=1.4)
        self.add_input("q", shape=4, label="state")
        self.add_input("qdot", shape=4, label="rate")

        self.add_constraint("res", shape=4, label="dynamics residuals")

        return

    def compute(self):
        uM = self.constants["uM"]
        m = self.constants["m"]
        R = self.constants["R"]
        S = self.constants["S"]
        C0 = self.constants["C0"]
        rho = self.constants["rho"]
        k = self.constants["k"]
        g = self.constants["g"]

        CL = self.inputs["CL"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Horizontal distance in m and velocities in m/s
        x = DISTANCE_SCALE * q[0]
        vx = VELOCITY_SCALE * q[2]
        vy = VELOCITY_SCALE * q[3]

        # Updraft of the thermal centered at 2.5 R
        X_term = x / R - 2.5
        X = X_term * X_term
        ua = uM * (1.0 - X) * am.exp(-X)

        # Speed and flight path angle relative to the rising air
        Vy = vy - ua
        vr_sq = vx * vx + Vy * Vy
        vr = vr_sq**0.5
        sin_eta = Vy / vr
        cos_eta = vx / vr

        # Drag polar, aerodynamic forces, and weight
        CD = C0 + k * CL * CL
        half_rho_S = 0.5 * rho * S
        D = half_rho_S * CD * vr_sq
        L = half_rho_S * CL * vr_sq
        W = m * g

        # Accelerations of the planar motion
        vxdot = (-L * sin_eta - D * cos_eta) / m
        vydot = (L * cos_eta - D * sin_eta - W) / m

        res = 4 * [None]
        res[0] = qdot[0] - vx / DISTANCE_SCALE
        res[1] = qdot[1] - vy / DISTANCE_SCALE
        res[2] = qdot[2] - vxdot / VELOCITY_SCALE
        res[3] = qdot[3] - vydot / VELOCITY_SCALE

        self.constraints["res"] = res

        return


class RangeObjective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("xf", label="final horizontal distance")

        self.add_objective("obj")

        return

    def compute(self):
        xf = self.inputs["xf"]

        self.objective["obj"] = -xf

        return


# The launch point and the launch velocity
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=4)

        self.add_constraint("res", shape=4)

        return

    def compute(self):
        q = self.inputs["q"]

        self.constraints["res"] = [
            q[0] - X0 / DISTANCE_SCALE,
            q[1] - Y0 / DISTANCE_SCALE,
            q[2] - VX0 / VELOCITY_SCALE,
            q[3] - VY0 / VELOCITY_SCALE,
        ]

        return


# The final altitude and velocity with a free range
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=4)

        self.add_constraint("res", shape=3)

        return

    def compute(self):
        q = self.inputs["q"]

        self.constraints["res"] = [
            q[1] - YF / DISTANCE_SCALE,
            q[2] - VXF / VELOCITY_SCALE,
            q[3] - VYF / VELOCITY_SCALE,
        ]

        return


def create_hang_glider_model(module_name="max_range_glide", num_time_steps=100):
    gd = GliderDynamics()
    trap = TrapezoidRule(num_time_steps=num_time_steps)
    obj = RangeObjective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    model.add_component("gd", num_time_steps + 1, gd)
    model.add_component("trap", 4 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link the states and rates of every state to the trapezoid rule
    for i in range(4):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link the states at both ends of every interval
        model.link(f"gd.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"gd.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates at both ends of every interval
        model.link(f"gd.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"gd.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link the boundary conditions
    model.link("gd.q[0, :]", "ic.q[0, :]")
    model.link(f"gd.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Link the final horizontal distance to the objective
    model.link(f"gd.q[{num_time_steps}, 0]", "obj.xf[0]")

    # Link the final time of every interval to the first one
    model.link("trap.tf[1:]", "trap.tf[0]")

    return model


def set_initial_guess(model, num_time_steps=100):
    """Initial point from a straight glide that holds the launch velocity"""
    tf = 100.0
    xf = 1250.0
    t = np.linspace(0.0, tf, num_time_steps + 1)
    s = t / tf

    # The distance and the altitude change linearly in time
    x_pos = X0 + s * xf
    y_pos = Y0 + s * (YF - Y0)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["gd.q[:, 0]"] = x_pos / DISTANCE_SCALE
    x["gd.q[:, 1]"] = y_pos / DISTANCE_SCALE
    x["gd.q[:, 2]"] = VX0 / VELOCITY_SCALE
    x["gd.q[:, 3]"] = VY0 / VELOCITY_SCALE

    # The rates of the straight line
    x["gd.qdot[:, 0]"] = (xf / tf) / DISTANCE_SCALE
    x["gd.qdot[:, 1]"] = ((YF - Y0) / tf) / DISTANCE_SCALE
    x["gd.qdot[:, 2]"] = 0.0
    x["gd.qdot[:, 3]"] = 0.0

    # The lift coefficient and the final time
    x["gd.CL"] = 1.0
    x["trap.tf"] = tf / TIME_SCALE

    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

model = create_hang_glider_model(num_time_steps=num_time_steps)
set_initial_guess(model, num_time_steps=num_time_steps)

if args.build:
    model.build_module()

model.initialize()

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

opt_options = {
    "initial_barrier_param": 0.1,
    "max_line_search_iterations": 30,
    "max_iterations": 500,
    "fraction_to_boundary": 0.995,
}

# Get the design variables
x = model.create_vector()

# Set up the optimizer and optimize
opt = am.Optimizer(model, x=x)
opt_data = opt.optimize(opt_options)

# Copy the solution from the device to the host
x.copy_device_to_host()

q = x["gd.q"]
CL = x["gd.CL"]
tf = x["trap.tf"][0]

print(f"Flight time:                {tf * TIME_SCALE:.2f} s")
print(f"Range:                      {q[-1, 0] * DISTANCE_SCALE:.2f} m")
print(f"Lift coefficient at start:  {CL[0]:.4f}")
print(f"Lift coefficient at end:    {CL[-1]:.4f}")

# Save the solution in m, m/s, and s for plot_hang_glider.py
t = np.linspace(0.0, tf * TIME_SCALE, num_time_steps + 1)
states = np.zeros((num_time_steps + 1, 4))
states[:, 0] = q[:, 0] * DISTANCE_SCALE
states[:, 1] = q[:, 1] * DISTANCE_SCALE
states[:, 2] = q[:, 2] * VELOCITY_SCALE
states[:, 3] = q[:, 3] * VELOCITY_SCALE

with open("hang_glider_solution.json", "w") as fp:
    json.dump({"t": t.tolist(), "q": states.tolist(), "CL": CL.tolist()}, fp)
