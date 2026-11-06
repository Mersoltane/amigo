"""
Free flying robot
-----------------

    minimize    integral of (u1 + u2 + u3 + u4) from 0 to 12
    subject to  y1' = y4,  y2' = y5,  y3' = y6,  y6' = alpha T1 - beta T2
                y4' = (T1 + T2) cos y3,  y5' = (T1 + T2) sin y3
                T1 = u1 - u2,  T2 = u3 - u4,  0 <= uk <= 1
                y(0) = (-10, -10, pi/2, 0, 0, 0),  y(12) = 0
"""

import argparse
import json
import amigo as am
import numpy as np

final_time = 12.0
num_time_steps = 100

# Distances of the two thrusters from the center of mass
alpha = 0.2
beta = 0.2


class TrapezoidRule(am.Component):
    def __init__(self, final_time=12.0, num_time_steps=100):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)

        self.add_input("q1")
        self.add_input("q2")

        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_constraint("res")

        return

    def compute(self):
        dt = self.constants["dt"]

        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return


class FreeFlyingRobotDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("alpha", value=alpha)
        self.add_constant("beta", value=beta)

        self.add_input("q", shape=6, label="state")
        self.add_input("qdot", shape=6, label="rate")
        self.add_input("u", shape=4, label="control", lower=0.0, upper=1.0)

        self.add_constraint("res", shape=6, label="dynamics residuals")

        return

    def compute(self):
        alpha = self.constants["alpha"]
        beta = self.constants["beta"]

        q = self.inputs["q"]
        qdot = self.inputs["qdot"]
        u = self.inputs["u"]

        y3 = q[2]
        y4 = q[3]
        y5 = q[4]
        y6 = q[5]

        u1 = u[0]
        u2 = u[1]
        u3 = u[2]
        u4 = u[3]

        # Net thrust of each thruster
        T1 = u1 - u2
        T2 = u3 - u4
        T_total = T1 + T2

        cos_y3 = am.cos(y3)
        sin_y3 = am.sin(y3)

        res = 6 * [None]
        res[0] = qdot[0] - y4
        res[1] = qdot[1] - y5
        res[2] = qdot[2] - y6
        res[3] = qdot[3] - T_total * cos_y3
        res[4] = qdot[4] - T_total * sin_y3
        res[5] = qdot[5] - (alpha * T1 - beta * T2)

        self.constraints["res"] = res

        return


class Objective(am.Component):
    def __init__(self, final_time=12.0, num_time_steps=100):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)

        self.add_input("u1", shape=4, label="control at the start of the interval")
        self.add_input("u2", shape=4, label="control at the end of the interval")

        self.add_objective("obj")

        return

    def compute(self):
        dt = self.constants["dt"]

        u1 = self.inputs["u1"]
        u2 = self.inputs["u2"]

        sum_u1 = u1[0] + u1[1] + u1[2] + u1[3]
        sum_u2 = u2[0] + u2[1] + u2[2] + u2[3]

        self.objective["obj"] = 0.5 * dt * (sum_u1 + sum_u2)

        return


# The robot starts at (-10, -10) facing up and at rest
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("pi", value=np.pi)

        self.add_input("q", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        pi = self.constants["pi"]
        q = self.inputs["q"]

        self.constraints["res"] = [
            q[0] + 10.0,
            q[1] + 10.0,
            q[2] - pi / 2.0,
            q[3] - 0.0,
            q[4] - 0.0,
            q[5] - 0.0,
        ]


# The robot ends at the origin facing right and at rest
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        q = self.inputs["q"]

        self.constraints["res"] = [
            q[0] - 0.0,
            q[1] - 0.0,
            q[2] - 0.0,
            q[3] - 0.0,
            q[4] - 0.0,
            q[5] - 0.0,
        ]


def create_free_flying_robot_model(
    module_name="freeflyingrobot", final_time=12.0, num_time_steps=100
):
    robot = FreeFlyingRobotDynamics()
    trap = TrapezoidRule(final_time=final_time, num_time_steps=num_time_steps)
    obj = Objective(final_time=final_time, num_time_steps=num_time_steps)
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    model.add_component("robot", num_time_steps + 1, robot)
    model.add_component("trap", 6 * num_time_steps, trap)
    model.add_component("obj", num_time_steps, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link the states and rates of every state to the trapezoid rule
    for i in range(6):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link the states at both ends of every interval
        model.link(f"robot.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"robot.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates at both ends of every interval
        model.link(f"robot.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"robot.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link the controls at both ends of every interval to the fuel
    model.link("robot.u[:-1, :]", "obj.u1[:, :]")
    model.link("robot.u[1:, :]", "obj.u2[:, :]")

    # Link the boundary conditions
    model.link("robot.q[0, :]", "ic.q[0, :]")
    model.link(f"robot.q[{num_time_steps}, :]", "fc.q[0, :]")

    return model


def set_initial_guess(model, final_time=12.0, num_time_steps=100):
    """Initial point from smooth position profiles and a switching thrust"""
    dt = final_time / num_time_steps
    s = np.linspace(0.0, 1.0, num_time_steps + 1)

    # A profile from 0 to 1 with zero slope at both ends
    position = 3 * s**2 - 2 * s**3
    velocity = (6 * s - 6 * s**2) / final_time

    # The positions and the angle move from the initial to the final state
    q = np.zeros((num_time_steps + 1, 6))
    q[:, 0] = -10.0 + 10.0 * position
    q[:, 1] = -10.0 + 10.0 * position
    q[:, 2] = np.pi / 2.0 * (1.0 - position)

    # The velocities follow the slope of the profile
    q[:, 3] = 10.0 * velocity
    q[:, 4] = 10.0 * velocity
    q[:, 5] = -np.pi / 2.0 * velocity

    # The thrust switches from accelerating to braking halfway
    switch = 0.5 * (1 + np.tanh(10 * (s - 0.5)))

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["robot.q"] = q
    x["robot.qdot"] = np.gradient(q, dt, axis=0)

    x["robot.u[:, 0]"] = 0.4 * (1 - switch)
    x["robot.u[:, 1]"] = 0.4 * switch
    x["robot.u[:, 2]"] = 0.4 * (1 - switch)
    x["robot.u[:, 3]"] = 0.4 * switch

    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

model = create_free_flying_robot_model(final_time=final_time, num_time_steps=num_time_steps)
set_initial_guess(model, final_time=final_time, num_time_steps=num_time_steps)

if args.build:
    model.build_module()

model.initialize()

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

opt_options = {
    "initial_barrier_param": 10.0,
    "max_line_search_iterations": 50,
    "max_iterations": 1000,
    "convergence_tolerance": 1e-6,
}

# Get the design variables
x = model.create_vector()

# Set up the optimizer and optimize
opt = am.Optimizer(model, x=x)
opt_data = opt.optimize(opt_options)

# Copy the solution from the device to the host
x.copy_device_to_host()

q = x["robot.q"]
u = x["robot.u"]

# Fuel used over the maneuver by the trapezoid rule
dt = final_time / num_time_steps
fuel = 0.0
for i in range(num_time_steps):
    fuel += 0.5 * dt * (np.sum(u[i]) + np.sum(u[i + 1]))

print(f"Fuel:                       {fuel:.4f}")

# Save the solution for plot_free_flying_robot.py
t = np.linspace(0.0, final_time, num_time_steps + 1)

with open("free_flying_robot_solution.json", "w") as fp:
    json.dump({"t": t.tolist(), "q": q.tolist(), "u": u.tolist()}, fp)
