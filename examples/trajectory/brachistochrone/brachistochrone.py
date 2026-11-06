"""
Brachistochrone
---------------

    minimize    tf
    subject to  x' = v sin theta,  y' = -v cos theta,  v' = g cos theta
                x(0) = 0,  y(0) = 10,  v(0) = 0,  x(tf) = 10,  y(tf) = 5
                0 <= theta <= pi
"""

import argparse
import json
import amigo as am
import numpy as np

num_time_steps = 100


class TrapezoidRule(am.Component):
    def __init__(self, num_time_steps=100):
        super().__init__()

        self.add_constant("num_steps", value=num_time_steps)

        self.add_input("tf", lower=1.0, upper=am.inf)

        self.add_input("q1")
        self.add_input("q2")

        self.add_input("q1dot")
        self.add_input("q2dot")

        self.add_constraint("res")

        return

    def compute(self):
        num_steps = self.constants["num_steps"]

        tf = self.inputs["tf"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        dt = tf / num_steps

        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)

        return


class ParticleDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("g", value=9.80655, label="gravitational acceleration")

        self.add_input("theta", label="control", lower=0.0, upper=np.pi)
        self.add_input("q", shape=(3), label="state")
        self.add_input("qdot", shape=(3), label="rate")

        self.add_constraint("res", shape=(3), label="residual")

        return

    def compute(self):
        g = self.constants["g"]

        theta = self.inputs["theta"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        v = q[2]

        sint = am.sin(theta)
        cost = am.cos(theta)

        res = 3 * [None]
        res[0] = qdot[0] - v * sint
        res[1] = qdot[1] + v * cost
        res[2] = qdot[2] - g * cost

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


# The particle starts at rest at x = 0 and y = 10
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=3)
        self.add_constraint("res", shape=3)

    def compute(self):
        q = self.inputs["q"]

        self.constraints["res"] = [q[0], q[1] - 10.0, q[2]]


# The particle reaches x = 10 and y = 5 at any speed
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=3)
        self.add_constraint("res", shape=2)

    def compute(self):
        q = self.inputs["q"]

        self.constraints["res"] = [q[0] - 10.0, q[1] - 5.0]


def create_brachistochrone_model(module_name="brachistochrone", num_time_steps=100):
    dynamics = ParticleDynamics()
    trap = TrapezoidRule(num_time_steps=num_time_steps)
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    model.add_component("dynamics", num_time_steps + 1, dynamics)
    model.add_component("trap", 3 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link the states and rates of every state to the trapezoid rule
    for i in range(3):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link the states at both ends of every interval
        model.link(f"dynamics.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"dynamics.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates at both ends of every interval
        model.link(f"dynamics.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"dynamics.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link the boundary conditions
    model.link("dynamics.q[0, :]", "ic.q[0, :]")
    model.link(f"dynamics.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Link the final time to every interval
    model.link("obj.tf[0]", "trap.tf")

    return model


def set_initial_guess(model, num_time_steps=100):
    """Initial point from a straight path at a growing speed"""
    tf = 3.0

    # A straight path from the start to the end point at a growing speed
    q = np.zeros((num_time_steps + 1, 3))
    q[:, 0] = np.linspace(0.0, 10.0, num_time_steps + 1)
    q[:, 1] = np.linspace(10.0, 5.0, num_time_steps + 1)
    q[:, 2] = np.linspace(0.0, 9.9, num_time_steps + 1)

    # Every rate grows from 0 to 1 along the path
    qdot = np.outer(np.linspace(0.0, 1.0, num_time_steps + 1), [1, 1, 1])

    # The angle from the vertical grows from 0.1 to 0.1 pi
    theta = np.linspace(0.1, 0.1 * np.pi, num_time_steps + 1)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["obj.tf[0]"] = tf

    x["dynamics.q"] = q
    x["dynamics.qdot"] = qdot
    x["dynamics.theta"] = theta

    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

model = create_brachistochrone_model(num_time_steps=num_time_steps)
set_initial_guess(model, num_time_steps=num_time_steps)

if args.build:
    model.build_module()

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

opt_options = {
    "max_iterations": 500,
    "max_line_search_iterations": 5,
}

# Get the design variables
x = model.create_vector()

# Set up the optimizer and optimize
opt = am.Optimizer(model, x=x)
opt_data = opt.optimize(opt_options)

# Copy the solution from the device to the host
x.copy_device_to_host()

q = x["dynamics.q"]
theta = x["dynamics.theta"]
tf = x["obj.tf"][0]

print(f"Final time:                 {tf:.6f} s")

# Save the model and the optimizer data
with open("brachistochrone_model.json", "w") as fp:
    json.dump(model.serialize(), fp, indent=2)

with open("brachistochrone_opt_data.json", "w") as fp:
    json.dump(opt_data, fp, indent=2)

# Save the solution for plot_brachistochrone.py
t = np.linspace(0.0, tf, num_time_steps + 1)

with open("brachistochrone_solution.json", "w") as fp:
    json.dump({"t": t.tolist(), "q": q.tolist(), "theta": theta.tolist()}, fp)
