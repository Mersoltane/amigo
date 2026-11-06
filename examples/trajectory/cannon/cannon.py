"""
Cannon shot with quadratic drag
-------------------------------

    minimize    vx(0)^2 + vy(0)^2
    subject to  x' = vx,  y' = vy,  vx' = -c vx v,  vy' = -c vy v - g
                v = sqrt(vx^2 + vy^2)
                x(0) = y(0) = 0,  x(3) = 6,  y(3) = 0
"""

import argparse
import json
import amigo as am
import numpy as np

num_time_steps = 50
final_time = 3.0


class TrapezoidRule(am.Component):
    def __init__(self, final_time=3.0, num_time_steps=50):
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


class CannonDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("c", value=0.4, label="drag")
        self.add_constant("g", value=1.0, label="gravity")

        self.add_input("q", shape=(4), label="state")
        self.add_input("qdot", shape=(4), label="rate")

        self.add_constraint("res", shape=(4), label="residual")

        return

    def compute(self):
        c = self.constants["c"]
        g = self.constants["g"]

        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        vx = q[2]
        vy = q[3]

        # Speed of the ball
        v = am.sqrt(vx * vx + vy * vy)

        res = 4 * [None]

        # The position changes with the velocity
        res[0] = qdot[0] - vx
        res[1] = qdot[1] - vy

        # Quadratic drag slows the ball and gravity pulls it down
        res[2] = qdot[2] + c * vx * v
        res[3] = qdot[3] + c * vy * v + g

        self.constraints["res"] = res

        return


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=(4), label="state")

        self.add_objective("obj")

        return

    def compute(self):
        q = self.inputs["q"]

        vx = q[2]
        vy = q[3]

        # The squared speed at the launch
        self.objective["obj"] = vx * vx + vy * vy

        return


# The ball starts at the origin
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=4)

        self.add_constraint("res", shape=2)

        return

    def compute(self):
        q = self.inputs["q"]

        self.constraints["res"] = [q[0], q[1]]

        return


# The ball reaches the target at the final time
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=4)

        self.add_constraint("res", shape=2)

        return

    def compute(self):
        q = self.inputs["q"]

        self.constraints["res"] = [q[0] - 6.0, q[1]]

        return


def create_cannon_model(module_name="cannon", final_time=3.0, num_time_steps=50):
    cannon = CannonDynamics()
    trap = TrapezoidRule(final_time=final_time, num_time_steps=num_time_steps)
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    model.add_component("cannon", num_time_steps + 1, cannon)
    model.add_component("trap", 4 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link the states and rates of every state to the trapezoid rule
    for i in range(4):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link the states at both ends of every interval
        model.link(f"cannon.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"cannon.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates at both ends of every interval
        model.link(f"cannon.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"cannon.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link the boundary conditions
    model.link("cannon.q[0, :]", "ic.q[0, :]")
    model.link(f"cannon.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Link the launch state to the objective
    model.link("cannon.q[0, :]", "obj.q[0, :]")

    # Bounds on the positions
    model.set_meta("lower", "cannon.q[:, 0]", -10)
    model.set_meta("upper", "cannon.q[:, 0]", 50)
    model.set_meta("lower", "cannon.q[:, 1]", -50)
    model.set_meta("upper", "cannon.q[:, 1]", 50)

    # Bounds on the velocities
    model.set_meta("lower", "cannon.q[:, 2]", -50)
    model.set_meta("upper", "cannon.q[:, 2]", 50)
    model.set_meta("lower", "cannon.q[:, 3]", -50)
    model.set_meta("upper", "cannon.q[:, 3]", 50)

    return model


def set_initial_guess(model, final_time=3.0, num_time_steps=50):
    """Initial point from the ballistic shot without drag"""
    target_x = 6.0
    target_y = 0.0
    g = 1.0

    # Launch velocity that reaches the target without drag
    vx_init = target_x / final_time
    vy_init = target_y / final_time + 0.5 * g * final_time

    t = np.linspace(0.0, final_time, num_time_steps + 1)

    # Positions of the ballistic shot
    x_pos = vx_init * t
    y_pos = vy_init * t - 0.5 * g * t * t

    # Velocities of the ballistic shot
    vx = np.full(num_time_steps + 1, vx_init)
    vy = vy_init - g * t

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["cannon.q[:, 0]"] = x_pos
    x["cannon.q[:, 1]"] = y_pos
    x["cannon.q[:, 2]"] = vx
    x["cannon.q[:, 3]"] = vy

    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

model = create_cannon_model(final_time=final_time, num_time_steps=num_time_steps)
set_initial_guess(model, final_time=final_time, num_time_steps=num_time_steps)

if args.build:
    model.build_module()

model.initialize()

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

opt_options = {
    "max_iterations": 100,
}

# Get the design variables
x = model.create_vector()

# Set up the optimizer and optimize
opt = am.Optimizer(model, x=x)
opt_data = opt.optimize(opt_options)

# Copy the solution from the device to the host
x.copy_device_to_host()

q = x["cannon.q"]

print(f"Launch velocity x:          {q[0, 2]:.4f}")
print(f"Launch velocity y:          {q[0, 3]:.4f}")

# Save the optimization data
with open("cannon_opt_data.json", "w") as fp:
    json.dump(opt_data, fp, indent=2)

# Save the solution for plot_cannon.py
t = np.linspace(0.0, final_time, num_time_steps + 1)

with open("cannon_solution.json", "w") as fp:
    json.dump({"t": t.tolist(), "q": q.tolist()}, fp)
