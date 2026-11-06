"""
Cart pole swing up
------------------

    minimize    integral of u^2 from 0 to 2
    subject to  (m1 + m2 s^2) d'' = u + m2 L s theta'^2 + m2 g c s
                L (m1 + m2 s^2) theta'' = -u c - m2 L c s theta'^2 - (m1 + m2) g s
                s = sin theta,  c = cos theta,  -50 <= u <= 50
                (d, theta, d', theta') = 0 at t = 0 and (2, pi, 0, 0) at t = 2
"""

import argparse
import json
import amigo as am
import numpy as np

final_time = 2.0
num_time_steps = 100


class TrapezoidRule(am.Component):
    def __init__(self, final_time=2.0, num_time_steps=100):
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


class CartComponent(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("g", value=9.81)
        self.add_data("L", value=0.5)
        self.add_data("m1", value=1.0)
        self.add_data("m2", value=0.3)

        self.add_input("x", label="control", lower=-50, upper=50)
        self.add_input("q", shape=(4), label="state")
        self.add_input("qdot", shape=(4), label="rate")

        self.add_constraint("res", shape=(4), label="residual")

        return

    def compute(self):
        g = self.constants["g"]
        L0 = self.data["L"]
        m1 = self.data["m1"]
        m2 = self.data["m2"]

        x = self.inputs["x"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        sint = am.sin(q[1])
        cost = am.cos(q[1])

        res = 4 * [None]
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]
        res[2] = (m1 + m2 * sint**2) * qdot[2] - (
            x + m2 * L0 * sint * q[3] ** 2 + m2 * g * cost * sint
        )
        res[3] = L0 * (m1 + m2 * sint**2) * qdot[3] - (
            -x * cost - m2 * L0 * cost * sint * q[3] ** 2 - (m1 + m2) * g * sint
        )

        self.constraints["res"] = res

        return


class KineticEnergyOutput(am.Component):
    def __init__(self, final_time=2.0, num_time_steps=100):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)
        self.add_data("m1")

        self.add_input("u1dot")
        self.add_input("u2dot")

        self.add_output("ke")

        return

    def compute_output(self):
        dt = self.constants["dt"]
        m1 = self.data["m1"]

        u1dot = self.inputs["u1dot"]
        u2dot = self.inputs["u2dot"]

        self.outputs["ke"] = m1 * dt * (u1dot * u1dot + u2dot * u2dot)


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x1", label="control")
        self.add_input("x2", label="control")

        self.add_objective("obj")

        return

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]

        self.objective["obj"] = (x1**2 + x2**2) / 2

        return


# Initial conditions are q = 0
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=4)
        self.add_constraint("res", shape=4)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [q[0], q[1], q[2], q[3]]


# Set the final conditions
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("pi", value=np.pi)

        self.add_input("q", shape=4)
        self.add_constraint("res", shape=4)

    def compute(self):
        pi = self.constants["pi"]
        q = self.inputs["q"]
        self.constraints["res"] = [q[0] - 2.0, q[1] - pi, q[2], q[3]]


def create_cart_model(module_name="cart_pole", final_time=2.0, num_time_steps=100):
    cart = CartComponent()
    trap = TrapezoidRule(final_time=final_time, num_time_steps=num_time_steps)
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()
    kin = KineticEnergyOutput(final_time=final_time, num_time_steps=num_time_steps)

    model = am.Model(module_name)

    model.add_component("cart", num_time_steps + 1, cart)
    model.add_component("trap", 4 * num_time_steps, trap)
    model.add_component("obj", num_time_steps, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)
    model.add_component("kin", num_time_steps, kin)

    # Add the links
    for i in range(4):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps
        # Link the state variables
        model.link(f"cart.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"cart.q[1:, {i}]", f"trap.q2[{start}:{end}]")

        # Link the state rates
        model.link(f"cart.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"cart.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    model.link(f"cart.x[:-1]", f"obj.x1[:]")
    model.link(f"cart.x[1:]", f"obj.x2[:]")

    model.link("cart.q[0, :]", "ic.q[0, :]")
    model.link(f"cart.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Link the data
    model.link("cart.m1[1:]", "cart.m1[0]")
    model.link("cart.m2[1:]", "cart.m2[0]")
    model.link("cart.L[1:]", "cart.L[0]")

    # Link the kinetic energy computation
    model.link("kin.m1", "cart.m1[0]")
    model.link("kin.u1dot", "cart.qdot[:-1, 0]")
    model.link("kin.u2dot", "cart.qdot[1:, 0]")

    # Link the outputs
    model.link("kin.ke[1:]", "kin.ke[0]")

    # Set the initial point by using a view of the initial values. This view acts
    # like a vector, but sets the values into the staged model
    x = model.get_meta_view("value")
    x["cart.q[:, 0]"] = np.linspace(0, 2.0, num_time_steps + 1)
    x["cart.q[:, 1]"] = np.linspace(0, np.pi, num_time_steps + 1)
    x["cart.q[:, 2]"] = 1.0
    x["cart.q[:, 3]"] = 1.0

    return model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

model = create_cart_model(num_time_steps=num_time_steps)

if args.build:
    model.build_module()

model.initialize()

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

opt_options = {
    "initial_barrier_param": 0.1,
    "convergence_tolerance": 5e-7,
    "max_line_search_iterations": 30,
    "max_iterations": 500,
}

# Get the design variables
x = model.create_vector()

# Serialize the entire model
with open("cart_pole_model.json", "w") as fp:
    json.dump(model.serialize(), fp, indent=2)

# Set up the optimizer
opt = am.Optimizer(model, x=x)

# Optimize
opt_data = opt.optimize(opt_options)

opt_data["num_time_steps"] = num_time_steps
opt_data["num_variables"] = model.num_variables
opt_data["num_constraints"] = model.num_constraints

# Copy the solution from the device to host if we've been
# running the model on the GPU
x.copy_device_to_host()

# Print objective value
u = x["cart.x[:]"]  # control force at each timestep
obj_value = 0.0
dt = final_time / num_time_steps
for i in range(num_time_steps):
    obj_value += 0.5 * (u[i] ** 2 + u[i + 1] ** 2) * dt

print("Objective value: ", obj_value)

with open("cart_opt_data.json", "w") as fp:
    json.dump(opt_data, fp, indent=2)

q = x["cart.q"]

norms = []
for iter_data in opt_data["iterations"]:
    norms.append(iter_data["residual"])

# Save the solution and the residual norms for plot_cart_pole.py
t = np.linspace(0.0, final_time, num_time_steps + 1)

with open("cart_pole_solution.json", "w") as fp:
    json.dump(
        {"t": t.tolist(), "q": q.tolist(), "u": u.tolist(), "residual": norms}, fp
    )
