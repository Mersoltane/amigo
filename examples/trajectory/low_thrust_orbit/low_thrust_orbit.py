"""
Low thrust orbit transfer
-------------------------

- Many revolution transfer from a circular park orbit to a high eccentricity mission orbit
- Modified equinoctial elements with J2 to J4 gravity
- Maximizes the final weight over the thrust direction and the throttle
"""

import argparse
import json
import math
import amigo as am
import numpy as np

num_time_steps = 1000

# Physical constants in ft, s, and lb
MU = 1.407645794e16
RE = 20925662.73
G0 = 32.174
ISP = 450.0
THRUST = 4.446618e-3
W0 = 1.0
TAU_L = -50.0
J2 = 1082.639e-6
J3 = -2.565e-6
J4 = -1.608e-6

# Canonical units with the Earth radius as length and mu = 1
L_REF = RE
T_REF = math.sqrt(L_REF**3 / MU)
C_T = G0 * THRUST * T_REF**2 / L_REF
C_W = THRUST * T_REF / ISP
RE_ND = RE / L_REF

# Initial orbit
P0 = 21837080.052835 / L_REF
H0 = -0.25396764647494
L0 = math.pi

# Final orbit
PF = 40007346.015232 / L_REF
EF = 0.73550320568829
CHIF = 0.61761258786099


class HermiteSimpson(am.Component):
    def __init__(self, num_time_steps=1000):
        super().__init__()

        self.add_constant("inv_steps", value=1.0 / num_time_steps)

        self.add_input("tf", lower=0.1, upper=am.inf)

        self.add_input("q1")
        self.add_input("q2")
        self.add_input("qm")

        self.add_input("q1dot")
        self.add_input("q2dot")
        self.add_input("qmdot")

        self.add_constraint("herm")
        self.add_constraint("simp")

        return

    def compute(self):
        inv_steps = self.constants["inv_steps"]

        tf = self.inputs["tf"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        qm = self.inputs["qm"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]
        qmdot = self.inputs["qmdot"]

        dt = tf * inv_steps

        # Midpoint interpolation and Simpson quadrature over one interval
        self.constraints["herm"] = qm - 0.5 * (q1 + q2) - (dt / 8.0) * (q1dot - q2dot)
        self.constraints["simp"] = q2 - q1 - (dt / 6.0) * (q1dot + 4.0 * qmdot + q2dot)

        return


class OrbitDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("mu", value=1.0)
        self.add_constant("Re", value=RE_ND)
        self.add_constant("J3", value=J3)
        self.add_constant("J4", value=J4)

        self.add_data("thrust", value=THRUST)
        self.add_data("isp", value=ISP)
        self.add_data("J2", value=J2)

        self.add_input("x", shape=(7), label="state")
        self.add_input("xdot", shape=(7), label="rate")
        self.add_input("u", shape=(3), label="thrust")
        self.add_input("tau", label="throttle")

        self.add_constraint("res", shape=(7), label="dynamics residuals")
        self.add_constraint("unit", label="unit thrust direction")

        return

    def compute(self):
        mu = self.constants["mu"]
        Re = self.constants["Re"]
        J3 = self.constants["J3"]
        J4 = self.constants["J4"]

        thrust = self.data["thrust"]
        isp = self.data["isp"]
        J2 = self.data["J2"]

        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]
        tau = self.inputs["tau"]

        cT = G0 * thrust * T_REF**2 / L_REF
        cw = thrust * T_REF / isp

        p = x[0]
        f = x[1]
        g = x[2]
        h = x[3]
        k = x[4]
        L = x[5]
        w = x[6]

        ur = u[0]
        ut = u[1]
        uh = u[2]

        cosL = am.cos(L)
        sinL = am.sin(L)
        q = 1.0 + f * cosL + g * sinL
        s2 = 1.0 + h * h + k * k
        sqrt_pm = am.sqrt(p / mu)
        hk = h * sinL - k * cosL

        # Oblate gravity in the radial, transverse, and normal frame
        r = p / q
        sphi = 2.0 * hk / s2
        P2 = 0.5 * (3.0 * sphi**2 - 1.0)
        P3 = 0.5 * (5.0 * sphi**3 - 3.0 * sphi)
        P4 = (35.0 * sphi**4 - 30.0 * sphi**2 + 3.0) / 8.0
        dP2 = 3.0 * sphi
        dP3 = 0.5 * (15.0 * sphi**2 - 3.0)
        dP4 = 0.5 * (35.0 * sphi**3 - 15.0 * sphi)
        ror = Re / r
        mu_r2 = mu / (r * r)
        Sn = ror**2 * dP2 * J2 + ror**3 * dP3 * J3 + ror**4 * dP4 * J4
        Sr = 3.0 * ror**2 * P2 * J2 + 4.0 * ror**3 * P3 * J3 + 5.0 * ror**4 * P4 * J4
        dgn = -mu_r2 * Sn
        dgr = -mu_r2 * Sr
        ihz = (1.0 - h * h - k * k) / s2
        ithz = 2.0 * (h * cosL + k * sinL) / s2
        grav_r = -dgr
        grav_t = dgn * ithz
        grav_n = dgn * ihz

        # Thrust plus gravity perturbation
        thr = cT * (1.0 + 0.01 * tau) / w
        dr = thr * ur + grav_r
        dt = thr * ut + grav_t
        dn = thr * uh + grav_n

        # Rates of the elements and the weight
        pdot = (2.0 * p / q) * sqrt_pm * dt
        fdot = sqrt_pm * (dr * sinL + ((q + 1.0) * cosL + f) * dt / q - hk * g * dn / q)
        gdot = sqrt_pm * (
            -dr * cosL + ((q + 1.0) * sinL + g) * dt / q + hk * f * dn / q
        )
        hdot = sqrt_pm * (s2 * cosL / (2.0 * q)) * dn
        kdot = sqrt_pm * (s2 * sinL / (2.0 * q)) * dn
        Ldot = am.sqrt(mu * p) * (q / p) ** 2 + sqrt_pm * (hk / q) * dn
        wdot = -cw * (1.0 + 0.01 * tau)

        res = 7 * [None]
        res[0] = xdot[0] - pdot
        res[1] = xdot[1] - fdot
        res[2] = xdot[2] - gdot
        res[3] = xdot[3] - hdot
        res[4] = xdot[4] - kdot
        res[5] = xdot[5] - Ldot
        res[6] = xdot[6] - wdot

        self.constraints["res"] = res
        self.constraints["unit"] = ur * ur + ut * ut + uh * uh - 1.0

        return


# The initial orbit and weight
class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x", shape=7)
        self.add_constraint("res", shape=7)

    def compute(self):
        x = self.inputs["x"]
        self.constraints["res"] = [
            x[0] - P0,
            x[1] - 0.0,
            x[2] - 0.0,
            x[3] - H0,
            x[4] - 0.0,
            x[5] - L0,
            x[6] - W0,
        ]


# The mission orbit through p, f, g, h, and k at the final node
class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x", shape=5)
        self.add_constraint("res", shape=4)
        self.add_constraint("ineq", lower=-am.inf, upper=0.0)

    def compute(self):
        x = self.inputs["x"]

        p = x[0]
        f = x[1]
        g = x[2]
        h = x[3]
        k = x[4]

        self.constraints["res"] = [
            p - PF,
            am.sqrt(f * f + g * g) - EF,
            am.sqrt(h * h + k * k) - CHIF,
            f * h + g * k,
        ]
        self.constraints["ineq"] = g * h - k * f


class Objective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", lower=0.1, upper=am.inf)
        self.add_input("tau", lower=TAU_L, upper=0.0)
        self.add_input("wf", label="final weight")

        self.add_objective("obj")

        return

    def compute(self):
        wf = self.inputs["wf"]

        self.objective["obj"] = -wf

        return


def create_low_thrust_model(module_name="low_thrust_orbit", num_time_steps=1000):
    dyn = OrbitDynamics()
    dynm = OrbitDynamics()
    hs = HermiteSimpson(num_time_steps=num_time_steps)
    ic = InitialConditions()
    fc = FinalConditions()
    obj = Objective()

    model = am.Model(module_name)

    model.add_component("dyn", num_time_steps + 1, dyn)
    model.add_component("dynm", num_time_steps, dynm)
    model.add_component("hs", 7 * num_time_steps, hs)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link the states and rates of every element to the Hermite-Simpson rule
    for i in range(7):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        # Link the states at the nodes and the midpoints
        model.link(f"dyn.x[:{num_time_steps}, {i}]", f"hs.q1[{start}:{end}]")
        model.link(f"dyn.x[1:, {i}]", f"hs.q2[{start}:{end}]")
        model.link(f"dynm.x[:, {i}]", f"hs.qm[{start}:{end}]")

        # Link the state rates at the nodes and the midpoints
        model.link(f"dyn.xdot[:-1, {i}]", f"hs.q1dot[{start}:{end}]")
        model.link(f"dyn.xdot[1:, {i}]", f"hs.q2dot[{start}:{end}]")
        model.link(f"dynm.xdot[:, {i}]", f"hs.qmdot[{start}:{end}]")

    # Link the initial and final orbits
    model.link("dyn.x[0, :]", "ic.x[0, :]")
    model.link(f"dyn.x[{num_time_steps}, 0:5]", "fc.x[0, :]")

    # Link the final time to every interval
    model.link("obj.tf[0]", "hs.tf[:]")

    # Link the throttle to every node and midpoint
    model.link("obj.tau[0]", "dyn.tau[:]")
    model.link("obj.tau[0]", "dynm.tau[:]")

    # Link the final weight to the objective
    model.link(f"dyn.x[{num_time_steps}, 6]", "obj.wf[0]")

    return model


def tangential_thrust_rates(s, tau):
    """Rates of the elements and the weight under tangential thrust without J2"""
    p = s[0]
    f = s[1]
    g = s[2]
    L = s[5]
    w = s[6]

    cosL = np.cos(L)
    sinL = np.sin(L)
    q = 1.0 + f * cosL + g * sinL
    sqrt_p = np.sqrt(p)
    at = C_T * (1.0 + 0.01 * tau) / w

    pdot = (2.0 * p / q) * sqrt_p * at
    fdot = sqrt_p * ((q + 1.0) * cosL + f) * at / q
    gdot = sqrt_p * ((q + 1.0) * sinL + g) * at / q
    Ldot = sqrt_p * (q / p) ** 2
    wdot = -C_W * (1.0 + 0.01 * tau)

    return np.array([pdot, fdot, gdot, 0.0, 0.0, Ldot, wdot])


def rk4_step(s, dt, tau):
    """One RK4 step of the tangential thrust rates"""
    k1 = tangential_thrust_rates(s, tau)
    k2 = tangential_thrust_rates(s + 0.5 * dt * k1, tau)
    k3 = tangential_thrust_rates(s + 0.5 * dt * k2, tau)
    k4 = tangential_thrust_rates(s + dt * k3, tau)

    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def set_initial_guess(model, num_time_steps=1000):
    """Initial point from a throttled tangential thrust trajectory"""
    tau = -25.0
    tf = 90000.0 / T_REF
    dt = tf / num_time_steps

    # Fly the transfer from the initial orbit
    states = np.zeros((num_time_steps + 1, 7))
    states[0] = [P0, 0.0, 0.0, H0, 0.0, L0, W0]

    for n in range(num_time_steps):
        states[n + 1] = rk4_step(states[n], dt, tau)

    # The midpoints sit halfway between the nodes
    mid_states = 0.5 * (states[:-1] + states[1:])

    # The rates at the nodes
    rates = np.zeros((num_time_steps + 1, 7))

    for n in range(num_time_steps + 1):
        rates[n] = tangential_thrust_rates(states[n], tau)

    # The rates at the midpoints
    mid_rates = np.zeros((num_time_steps, 7))

    for n in range(num_time_steps):
        mid_rates[n] = tangential_thrust_rates(mid_states[n], tau)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["obj.tf[0]"] = tf
    x["obj.tau[0]"] = tau

    x["dyn.x"] = states
    x["dynm.x"] = mid_states

    x["dyn.xdot"] = rates
    x["dynm.xdot"] = mid_rates

    # The thrust starts along the transverse direction
    x["dyn.u[:, 1]"] = 1.0
    x["dynm.u[:, 1]"] = 1.0

    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

model = create_low_thrust_model(num_time_steps=num_time_steps)
set_initial_guess(model, num_time_steps=num_time_steps)

if args.build:
    model.build_module()

model.initialize()

print(f"Num variables:              {model.num_variables}")
print(f"Num constraints:            {model.num_constraints}")

opt_options = {
    "initial_barrier_param": 0.1,
    "max_iterations": 200,
    "fraction_to_boundary": 0.995,
}

# Get the design variables
x = model.create_vector()

# Set up the optimizer and optimize
opt = am.Optimizer(model, x=x)
opt_data = opt.optimize(opt_options)

# Copy the solution from the device to the host
x.copy_device_to_host()

xs = x["dyn.x"]
u = x["dyn.u"]
tf = x["obj.tf"][0]
tau = x["obj.tau"][0]

print(f"Transfer time:              {tf * T_REF:.1f} s")
print(f"Throttle:                   {tau:.4f}")
print(f"Final weight:               {xs[-1, 6]:.6f} lb")
print(f"Revolutions:                {(xs[-1, 5] - L0) / (2 * np.pi):.2f}")

# Save the trajectory with p in ft for the plot scripts
t = np.linspace(0.0, tf * T_REF, num_time_steps + 1)
q = xs.copy()
q[:, 0] = xs[:, 0] * L_REF

with open("low_thrust_trajectory.json", "w") as fp:
    json.dump({"t": t.tolist(), "q": q.tolist(), "u": u.tolist()}, fp)
