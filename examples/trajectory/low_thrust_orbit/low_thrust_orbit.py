import amigo as am
import numpy as np
import math
import argparse
import json

# Problem parameters
num_time_steps = 1000

"""
Low-Thrust Orbit Transfer
=========================

Many-revolution transfer from a circular park orbit to a high-eccentricity
mission orbit, formulated in modified equinoctial elements (MEE).  The
spacecraft maximizes its final weight (minimum fuel) under continuous low
thrust, so the optimal steering must be found over many revolutions.

State:   [p, f, g, h, k, L, w]   (MEE elements + weight)
Control: u = (u_r, u_theta, u_h)  unit thrust direction, |u| = 1
Param:   tau    scalar throttle factor, tau_L <= tau <= 0

Canonical units (length = Re, mu = 1) keep the elements O(1) over the whole
transfer.  
"""

# Physical constants
MU = 1.407645794e16  # Gravitational parameter [ft^3/s^2]
RE = 20925662.73  # Earth equatorial radius [ft]
G0 = 32.174  # Mass-to-weight conversion [ft/s^2]
ISP = 450.0  # Specific impulse [s]
THRUST = 4.446618e-3  # Maximum thrust [lb]
W0 = 1.0  # Initial weight [lb]
TAU_L = -50.0  # Throttle lower bound
J2, J3, J4 = 1082.639e-6, -2.565e-6, -1.608e-6  # Zonal harmonics

# Canonical scaling (length = Re, mu = 1)
L_REF = RE
T_REF = math.sqrt(L_REF**3 / MU)  # Time scale [s]
C_T = G0 * THRUST * T_REF**2 / L_REF  # Thrust acceleration coefficient
C_W = THRUST * T_REF / ISP  # Weight flow coefficient
RE_ND = RE / L_REF  # Earth radius [nd]

# Initial orbit
P0 = 21837080.052835 / L_REF  # Semi-parameter
H0 = -0.25396764647494  # h element
L0 = math.pi  # True longitude [rad]

# Final orbit
PF = 40007346.015232 / L_REF  # Semi-parameter
EF = 0.73550320568829  # Eccentricity
CHIF = 0.61761258786099  # sqrt(h^2 + k^2)


class HermiteSimpson(am.Component):
    # Separated Hermite-Simpson defect over one interval
    def __init__(self):
        super().__init__()
        self.add_input("tf", lower=0.1, upper=am.inf)
        self.add_input("q1")
        self.add_input("q2")
        self.add_input("qm")
        self.add_input("q1dot")
        self.add_input("q2dot")
        self.add_input("qmdot")
        self.add_constraint("herm")  # midpoint interpolation defect
        self.add_constraint("simp")  # Simpson collocation defect

    def compute(self):
        tf = self.inputs["tf"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        qm = self.inputs["qm"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]
        qmdot = self.inputs["qmdot"]
        dt = tf / num_time_steps
        self.constraints["herm"] = qm - 0.5 * (q1 + q2) - (dt / 8.0) * (q1dot - q2dot)
        self.constraints["simp"] = q2 - q1 - (dt / 6.0) * (q1dot + 4.0 * qmdot + q2dot)


class OrbitDynamics(am.Component):
    def __init__(self):
        super().__init__()
        self.add_constant("mu", value=1.0)
        self.add_constant("cT", value=C_T)
        self.add_constant("cw", value=C_W)
        self.add_constant("Re", value=RE_ND)
        self.add_constant("J2", value=J2)
        self.add_constant("J3", value=J3)
        self.add_constant("J4", value=J4)

        self.add_input("x", shape=(7), label="state")  # p, f, g, h, k, L, w
        self.add_input("xdot", shape=(7), label="rate")
        self.add_input("u", shape=(3), label="thrust")  # u_r, u_theta, u_h
        self.add_input("tau", label="throttle")  # shared scalar parameter

        self.add_constraint("res", shape=(7), label="dynamics residuals")
        self.add_constraint("unit", label="unit thrust direction")

    def compute(self):
        mu = self.constants["mu"]
        cT = self.constants["cT"]
        cw = self.constants["cw"]
        Re = self.constants["Re"]
        J2 = self.constants["J2"]
        J3 = self.constants["J3"]
        J4 = self.constants["J4"]

        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]
        tau = self.inputs["tau"]

        p, f, g, h, k, L, w = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        ur, ut, uh = u[0], u[1], u[2]

        cosL = am.cos(L)
        sinL = am.sin(L)
        q = 1.0 + f * cosL + g * sinL
        s2 = 1.0 + h * h + k * k
        sqrt_pm = am.sqrt(p / mu)
        hk = h * sinL - k * cosL

        # Oblate gravity (J2-J4) in the radial/transverse/normal frame
        r = p / q
        sphi = 2.0 * hk / s2  # sin geocentric latitude
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

        # Total disturbing acceleration = thrust + oblate gravity
        thr = cT * (1.0 + 0.01 * tau) / w
        dr = thr * ur + grav_r
        dt = thr * ut + grav_t
        dn = thr * uh + grav_n

        pdot = (2.0 * p / q) * sqrt_pm * dt
        fdot = sqrt_pm * (dr * sinL + ((q + 1.0) * cosL + f) * dt / q - hk * g * dn / q)
        gdot = sqrt_pm * (
            -dr * cosL + ((q + 1.0) * sinL + g) * dt / q + hk * f * dn / q
        )
        hdot = sqrt_pm * (s2 * cosL / (2.0 * q)) * dn
        kdot = sqrt_pm * (s2 * sinL / (2.0 * q)) * dn
        Ldot = am.sqrt(mu * p) * (q / p) ** 2 + sqrt_pm * (hk / q) * dn
        wdot = -cw * (1.0 + 0.01 * tau)

        rhs = [pdot, fdot, gdot, hdot, kdot, Ldot, wdot]
        self.constraints["res"] = [xdot[i] - rhs[i] for i in range(7)]
        self.constraints["unit"] = ur * ur + ut * ut + uh * uh - 1.0


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


class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x", shape=5)  # p, f, g, h, k at the final node
        self.add_constraint("res", shape=4)  # equality conditions
        self.add_constraint("ineq", lower=-am.inf, upper=0.0)  # gh - kf <= 0

    def compute(self):
        x = self.inputs["x"]
        p, f, g, h, k = x[0], x[1], x[2], x[3], x[4]
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
        self.add_input("tf", lower=0.1, upper=am.inf)  # free final time (owner)
        self.add_input("tau", lower=TAU_L, upper=0.0)  # throttle factor (owner)
        self.add_input("wf", label="final weight")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = -self.inputs["wf"]  # maximize final weight


# State rates for the initial-guess integration (tangential thrust, no J2)
def mee_rhs(s, tau_g):
    p, f, g, _, _, L, w = s
    cosL, sinL = np.cos(L), np.sin(L)
    qq = 1.0 + f * cosL + g * sinL
    spm = np.sqrt(p)  # mu = 1
    at = C_T * (1.0 + 0.01 * tau_g) / w  # tangential thrust accel
    pdot = (2.0 * p / qq) * spm * at
    fdot = spm * ((qq + 1.0) * cosL + f) * at / qq
    gdot = spm * ((qq + 1.0) * sinL + g) * at / qq
    Ldot = spm * (qq / p) ** 2
    wdot = -C_W * (1.0 + 0.01 * tau_g)
    return np.array([pdot, fdot, gdot, 0.0, 0.0, Ldot, wdot])


# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--solver",
    dest="solver",
    choices=["amigo", "mumps", "cuda"],
    default="amigo",
    help="Solver type",
)
args = parser.parse_args()

# Create component instances
dyn = OrbitDynamics()  # node-point dynamics
dynm = OrbitDynamics()  # midpoint dynamics (same model, separate instance)
hs = HermiteSimpson()
ic = InitialConditions()
fc = FinalConditions()
obj = Objective()

# Create the model
model = am.Model("low_thrust_orbit")
model.add_component("dyn", num_time_steps + 1, dyn)
model.add_component("dynm", num_time_steps, dynm)
model.add_component("hs", 7 * num_time_steps, hs)
model.add_component("obj", 1, obj)
model.add_component("ic", 1, ic)
model.add_component("fc", 1, fc)

# Hermite-Simpson links for each state
for i in range(7):
    start = i * num_time_steps
    end = (i + 1) * num_time_steps
    model.link(f"dyn.x[:{num_time_steps}, {i}]", f"hs.q1[{start}:{end}]")
    model.link(f"dyn.x[1:, {i}]", f"hs.q2[{start}:{end}]")
    model.link(f"dynm.x[:, {i}]", f"hs.qm[{start}:{end}]")
    model.link(f"dyn.xdot[:-1, {i}]", f"hs.q1dot[{start}:{end}]")
    model.link(f"dyn.xdot[1:, {i}]", f"hs.q2dot[{start}:{end}]")
    model.link(f"dynm.xdot[:, {i}]", f"hs.qmdot[{start}:{end}]")

# Boundary conditions
model.link("dyn.x[0, :]", "ic.x[0, :]")
model.link(f"dyn.x[{num_time_steps}, 0:5]", "fc.x[0, :]")

# Shared parameters
model.link("obj.tf[0]", "hs.tf[:]")
model.link("obj.tau[0]", "dyn.tau[:]")
model.link("obj.tau[0]", "dynm.tau[:]")
model.link(f"dyn.x[{num_time_steps}, 6]", "obj.wf[0]")

# Initial guess: forward-integrate a throttled tangential-thrust trajectory
tau_g = -25.0  # guess throttle
tf_g = 90000.0 / T_REF  # ~9e4 s
dt = tf_g / num_time_steps

xg = np.zeros((num_time_steps + 1, 7))
xg[0] = [P0, 0.0, 0.0, H0, 0.0, L0, W0]
for n in range(num_time_steps):
    s = xg[n]
    a = mee_rhs(s, tau_g)
    b = mee_rhs(s + 0.5 * dt * a, tau_g)
    c = mee_rhs(s + 0.5 * dt * b, tau_g)
    d = mee_rhs(s + dt * c, tau_g)
    xg[n + 1] = s + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)

xg_mid = 0.5 * (xg[:-1] + xg[1:])  # midpoints
xgdot = np.array([mee_rhs(x, tau_g) for x in xg])
xgdot_mid = np.array([mee_rhs(x, tau_g) for x in xg_mid])

model.set_meta("value", "obj.tf[0]", tf_g)
model.set_meta("value", "obj.tau[0]", tau_g)
for i in range(7):
    model.set_meta("value", f"dyn.x[:,{i}]", xg[:, i])
    model.set_meta("value", f"dyn.xdot[:,{i}]", xgdot[:, i])
    model.set_meta("value", f"dynm.x[:,{i}]", xg_mid[:, i])
    model.set_meta("value", f"dynm.xdot[:,{i}]", xgdot_mid[:, i])

for c in ("dyn", "dynm"):
    model.set_meta("value", f"{c}.u[:,0]", 0.0)
    model.set_meta("value", f"{c}.u[:,1]", 1.0)
    model.set_meta("value", f"{c}.u[:,2]", 0.0)

# Build the module if requested
if args.build:
    model.build_module()

# Initialize the model
model.initialize()
print(f"Num variables:    {model.num_variables}")
print(f"Num constraints:  {model.num_constraints}")

# Optimize
x = model.create_vector()
opt = am.Optimizer(model, x)
data = opt.optimize(
    {
        "solver": args.solver,
        "initial_barrier_param": 0.1,
        "max_iterations": 200,
        "fraction_to_boundary": 0.995,
        "init_least_squares_multipliers": True,
    }
)

# Extract results
xs = x["dyn.x"]
u = x["dyn.u"]
tf_opt = float(x["obj.tf"][0])
t = np.linspace(0.0, tf_opt * T_REF, num_time_steps + 1)  # seconds

# Save the trajectory for plotting (p in ft, L in rad, w in lb)
q = xs.copy()
q[:, 0] = xs[:, 0] * L_REF
with open("low_thrust_trajectory.json", "w") as fp:
    json.dump({"t": t.tolist(), "q": q.tolist(), "u": u.tolist()}, fp)

print(f"\nResults:")
print(f"Transfer time:  {tf_opt * T_REF:.1f} s")
print(f"Throttle tau:   {float(x['obj.tau'][0]):.4f}")
print(f"Final weight:   {xs[-1, 6]:.6f} lb")
print(f"Final p:        {xs[-1, 0] * L_REF:.1f} ft  (target {PF * L_REF:.1f})")
print(f"Revolutions:    {(xs[-1, 5] - L0) / (2 * np.pi):.2f}")
