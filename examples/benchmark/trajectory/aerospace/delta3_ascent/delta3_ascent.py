"""
Delta III ascent
----------------

- Ascent of a Delta III launcher to a geosynchronous transfer orbit in four phases
- Six strap on boosters burn with the first stage, then three more, then the first stage alone
- The second stage flies a last phase of free length
- The mass drops at each junction by the jettisoned hardware
- The thrust direction is a unit vector and the orbit is reached through five elements
- Maximizes the final mass
"""

import argparse
import sys
from pathlib import Path
import amigo as am
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

num_intervals = 20

# Earth, atmosphere, and drag constants
MU = 3.986012e14
RE = 6378145.0
OMEGA_E = 7.29211585e-5
RHO0 = 1.225
SCALE_HEIGHT = 7200.0
DRAG = 0.5 * 4.0 * np.pi

# Squared speed floor in m2/s2 that keeps the drag differentiable at lift off
SPEED_FLOOR = 1e-6

# Masses, propellant masses, and thrusts of the boosters and the two stages
M_SRB = 19290.0
PROP_SRB = 17010.0
T_SRB = 628500.0
M_FIRST = 104380.0
PROP_FIRST = 95550.0
T_FIRST = 1083100.0
M_SECOND = 19300.0
PROP_SECOND = 16820.0
T_SECOND = 110094.0
PAYLOAD = 4164.0

# Mass flows of the boosters and the two stages
FLOW_SRB = PROP_SRB / 75.2
FLOW_FIRST = PROP_FIRST / 261.0
FLOW_SECOND = PROP_SECOND / 700.0

# Phase durations with the free last phase bounded by the second stage burn
DURATIONS = [75.2, 75.2, 110.6]
LAST_MAX = 700.0
LAST_GUESS = 700.0

# Thrust and mass flow of each phase
THRUST = [6 * T_SRB + T_FIRST, 3 * T_SRB + T_FIRST, T_FIRST, T_SECOND]
FLOW = [6 * FLOW_SRB + FLOW_FIRST, 3 * FLOW_SRB + FLOW_FIRST, FLOW_FIRST, FLOW_SECOND]

# Mass at the start and end of each phase and the drops at the junctions
DROPS = [6 * (M_SRB - PROP_SRB), 3 * (M_SRB - PROP_SRB), M_FIRST - PROP_FIRST]
M_START = [PAYLOAD + M_SECOND + M_FIRST + 9 * M_SRB]
M_END = []
for p in range(3):
    M_END.append(M_START[p] - FLOW[p] * DURATIONS[p])
    M_START.append(M_END[p] - DROPS[p])
M_END.append(PAYLOAD)

# Launch site at rest on the rotating Earth
LAT = np.radians(28.5)
R0 = np.array([RE * np.cos(LAT), 0.0, RE * np.sin(LAT)])
V0 = np.array([-OMEGA_E * R0[1], OMEGA_E * R0[0], 0.0])

# Semi major axis, eccentricity, inclination, node, and periapsis of the target orbit
A_F = 24361140.0
E_F = 0.7308
INC_F = np.radians(28.5)
RAAN_F = np.radians(269.8)
PERI_F = np.radians(130.5)


# Length, time, and mass units with every constant expressed in them
class Units:
    def __init__(self, canonical=False):
        self.length = 1.0
        self.time = 1.0
        self.mass = 1.0

        # The canonical units are the Earth radius, its orbital time, and the lift off mass
        if canonical:
            self.length = RE
            self.time = np.sqrt(RE**3 / MU)
            self.mass = M_START[0]

        speed = self.length / self.time
        force = self.mass * self.length / self.time**2

        self.mu = MU * self.time**2 / self.length**3
        self.re = RE / self.length
        self.omega = OMEGA_E * self.time
        self.drag = RHO0 * DRAG * self.length / self.mass
        self.scale_height = SCALE_HEIGHT / self.length
        self.speed_floor = SPEED_FLOOR / speed**2

        self.thrust = []
        self.flow = []
        for p in range(4):
            self.thrust.append(THRUST[p] / force)
            self.flow.append(FLOW[p] * self.time / self.mass)

        self.durations = []
        for p in range(3):
            self.durations.append(DURATIONS[p] / self.time)

        self.last_max = LAST_MAX / self.time
        self.last_guess = LAST_GUESS / self.time

        self.m_start = []
        self.m_end = []
        for p in range(4):
            self.m_start.append(M_START[p] / self.mass)
            self.m_end.append(M_END[p] / self.mass)

        self.drops = []
        for p in range(3):
            self.drops.append(DROPS[p] / self.mass)

        self.r0 = R0 / self.length
        self.v0 = V0 / speed
        self.a_f = A_F / self.length
        self.pos_bound = 2.0 * RE / self.length
        self.vel_bound = 1e4 / speed

        return


class Vehicle(am.Component):
    def __init__(self, units, phase):
        super().__init__()

        self.units = units

        self.add_data("thrust", value=units.thrust[phase])
        self.add_data("flow", value=units.flow[phase])

        lower = [-units.pos_bound] * 3 + [-units.vel_bound] * 3 + [units.m_end[phase]]
        upper = [units.pos_bound] * 3 + [units.vel_bound] * 3 + [units.m_start[phase]]

        self.add_input("x", shape=7, lower=lower, upper=upper)
        self.add_input("xdot", shape=7)
        self.add_input("u", shape=3, lower=-1.0, upper=1.0)

        self.add_constraint("res", shape=7)
        self.add_constraint("unit")

        return

    def compute(self):
        mu = self.units.mu
        re = self.units.re
        omega = self.units.omega
        drag_factor = self.units.drag
        scale_height = self.units.scale_height
        speed_floor = self.units.speed_floor

        thrust = self.data["thrust"]
        flow = self.data["flow"]

        x = self.inputs["x"]
        xdot = self.inputs["xdot"]
        u = self.inputs["u"]

        r = [x[0], x[1], x[2]]
        v = [x[3], x[4], x[5]]
        mass = x[6]

        # Drag on the velocity relative to the rotating atmosphere
        rad = am.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        vrel = [v[0] + omega * r[1], v[1] - omega * r[0], v[2]]
        speed = am.sqrt(
            vrel[0] * vrel[0] + vrel[1] * vrel[1] + vrel[2] * vrel[2] + speed_floor
        )
        drag = drag_factor * am.exp(-(rad - re) / scale_height) * speed / (2.0 * mass)

        # Point mass gravity
        grav = mu / (rad * rad * rad)

        res = 7 * [None]
        for i in range(3):
            res[i] = xdot[i] - v[i]
            accel = -grav * r[i] + (thrust / mass) * u[i] - drag * vrel[i]
            res[3 + i] = xdot[3 + i] - accel
        res[6] = xdot[6] - (-flow)

        self.constraints["res"] = res
        self.constraints["unit"] = u[0] * u[0] + u[1] * u[1] + u[2] * u[2] - 1.0

        return


# Continuity of the position and the velocity and the mass drop between two phases
class Junction(am.Component):
    def __init__(self, drop):
        super().__init__()

        self.add_data("drop", value=drop)

        self.add_input("a", shape=7)
        self.add_input("b", shape=7)

        self.add_constraint("res", shape=7)

        return

    def compute(self):
        drop = self.data["drop"]

        a = self.inputs["a"]
        b = self.inputs["b"]

        res = 7 * [None]
        for i in range(6):
            res[i] = b[i] - a[i]
        res[6] = a[6] - b[6] - drop

        self.constraints["res"] = res

        return


# Lift off at rest on the rotating Earth with the full mass
class Launch(am.Component):
    def __init__(self, units):
        super().__init__()

        self.start = list(units.r0) + list(units.v0) + [units.m_start[0]]

        self.add_input("x0", shape=7)

        self.add_constraint("res", shape=7)

        return

    def compute(self):
        x0 = self.inputs["x0"]

        res = 7 * [None]
        for i in range(7):
            res[i] = x0[i] - self.start[i]

        self.constraints["res"] = res

        return


# Five orbital elements of the final state and the final mass to maximize
class Orbit(am.Component):
    def __init__(self, units):
        super().__init__()

        self.units = units

        self.add_input("tf", lower=0.0, upper=units.last_max)
        self.add_input("xf", shape=7)

        self.add_constraint("res", shape=5)
        self.add_constraint("quadrant", shape=2, lower=0.0, upper=am.inf)

        self.add_objective("obj")

        return

    def compute(self):
        mu = self.units.mu
        a_f = self.units.a_f

        xf = self.inputs["xf"]

        r = [xf[0], xf[1], xf[2]]
        v = [xf[3], xf[4], xf[5]]

        # Angular momentum, node, and eccentricity vectors of the final orbit
        rad = am.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        hnorm = am.sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2])
        node = [-h[1], h[0]]
        nnorm = am.sqrt(node[0] * node[0] + node[1] * node[1])
        ecc = 3 * [None]
        for i in range(3):
            ecc[i] = ((v2 - mu / rad) * r[i] - rv * v[i]) / mu
        enorm = am.sqrt(ecc[0] * ecc[0] + ecc[1] * ecc[1] + ecc[2] * ecc[2])
        semi_major = 1.0 / (2.0 / rad - v2 / mu)

        # The node lies below the x axis and periapsis above the equator
        self.constraints["res"] = [
            semi_major - a_f,
            enorm - E_F,
            am.acos(h[2] / hnorm) - INC_F,
            am.acos(node[0] / nnorm) - (2.0 * np.pi - RAAN_F),
            am.acos((node[0] * ecc[0] + node[1] * ecc[1]) / (nnorm * enorm)) - PERI_F,
        ]
        self.constraints["quadrant"] = [-node[1], ecc[2]]
        self.objective["obj"] = -xf[6]

        return


def create_delta3_ascent_model(
    module_name="bench_delta3_ascent", num_intervals=20, canonical=False
):
    units = Units(canonical)

    # The canonical units compile into a module of their own
    if canonical:
        module_name = module_name + "_canonical"

    model = am.Model(module_name)

    # Every phase with its dynamics at the nodes and the midpoints
    for p in range(4):
        dyn = Vehicle(units, p)
        dynm = Vehicle(units, p)

        # The first three phases have fixed durations and the last one a free one
        if p < 3:
            hs = common.HermiteSimpsonFixed(units.durations[p] / num_intervals)
        else:
            hs = common.HermiteSimpson(num_intervals)

        model.add_component(f"dyn{p}", num_intervals + 1, dyn)
        model.add_component(f"dynm{p}", num_intervals, dynm)
        model.add_component(f"hs{p}", 7 * num_intervals, hs)

        # Link the states and rates of the phase to the Hermite Simpson rule
        common.link_collocation(
            model, f"dyn{p}", f"dynm{p}", f"hs{p}", 7, num_intervals
        )

    # Link the end of each phase to the start of the next through a junction
    for p in range(3):
        junc = Junction(units.drops[p])

        model.add_component(f"junc{p}", 1, junc)

        model.link(f"dyn{p}.x[{num_intervals}, :]", f"junc{p}.a[0, :]")
        model.link(f"dyn{p + 1}.x[0, :]", f"junc{p}.b[0, :]")

    launch = Launch(units)
    orbit = Orbit(units)

    model.add_component("launch", 1, launch)
    model.add_component("orbit", 1, orbit)

    # Link the lift off state and the final state
    model.link("dyn0.x[0, :]", "launch.x0[0, :]")
    model.link(f"dyn3.x[{num_intervals}, :]", "orbit.xf[0, :]")

    # Link the final time to every interval of the last phase
    model.link("orbit.tf[0]", "hs3.tf[:]")

    return model


def rotation_z(a):
    """Rotation matrix about the z axis"""
    return np.array(
        [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


def rotation_x(a):
    """Rotation matrix about the x axis"""
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]]
    )


def target_state(units):
    """Position and velocity at the periapsis of the target orbit"""
    semi = A_F / units.length
    p = semi * (1.0 - E_F * E_F)
    r_pf = np.array([p / (1.0 + E_F), 0.0, 0.0])
    v_pf = np.sqrt(units.mu / p) * np.array([0.0, E_F + 1.0, 0.0])

    rot = rotation_z(RAAN_F) @ rotation_x(INC_F) @ rotation_z(PERI_F)

    return np.concatenate([rot @ r_pf, rot @ v_pf])


def phase_states(units, profile, p, last, tau):
    """Position, velocity, and mass of one phase along one profile"""
    start = np.concatenate([units.r0, units.v0])
    target = target_state(units)
    times = np.cumsum([0.0] + units.durations + [last])

    # The literature start sits at the launch site and in the last phase at the periapsis
    if profile == "literature" and p < 3:
        base = np.tile(start, (len(tau), 1))
    elif profile == "literature":
        base = np.tile(target, (len(tau), 1))
    elif profile == "constant":
        base = np.tile(start, (len(tau), 1))
    else:
        s = (times[p] + tau * (times[p + 1] - times[p])) / times[-1]
        base = start[None, :] + np.outer(s, target - start)

    # The mass falls linearly over the phase unless the constant profile holds it
    frac = tau
    if profile == "constant":
        frac = 0.0 * tau

    mass = units.m_start[p] + (units.m_end[p] - units.m_start[p]) * frac

    return np.column_stack([base, mass])


def vehicle_rates(states, controls, units, thrust, flow):
    """Rates of the position, the velocity, and the mass under one propulsion set"""
    r = [states[:, 0], states[:, 1], states[:, 2]]
    v = [states[:, 3], states[:, 4], states[:, 5]]
    mass = states[:, 6]
    u = [controls[:, 0], controls[:, 1], controls[:, 2]]

    # Drag on the velocity relative to the rotating atmosphere
    rad = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    vrel = [v[0] + units.omega * r[1], v[1] - units.omega * r[0], v[2]]
    speed = np.sqrt(
        vrel[0] * vrel[0] + vrel[1] * vrel[1] + vrel[2] * vrel[2] + units.speed_floor
    )
    drag = (
        units.drag
        * np.exp(-(rad - units.re) / units.scale_height)
        * speed
        / (2.0 * mass)
    )

    # Point mass gravity
    grav = units.mu / (rad * rad * rad)

    rates = np.zeros(states.shape)
    for i in range(3):
        rates[:, i] = v[i]
        rates[:, 3 + i] = -grav * r[i] + (thrust / mass) * u[i] - drag * vrel[i]
    rates[:, 6] = -flow

    return rates


def set_initial_guess(model, num_intervals=20, guess="literature", canonical=False):
    """Initial point of one guess family with the rates of the dynamics in every phase"""
    units = Units(canonical)

    # Normalized times of the nodes and the interval midpoints
    tau = np.linspace(0.0, 1.0, num_intervals + 1)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])

    # The final time families fly the linear profile over a scaled last phase
    last = common.family_final_time(guess, units.last_guess)
    profile = guess
    if guess == "tf_half" or guess == "tf_double":
        profile = "linear"

    # The thrust points along the x axis at every node and midpoint
    controls = np.tile([1.0, 0.0, 0.0], (num_intervals + 1, 1))
    mid_controls = np.tile([1.0, 0.0, 0.0], (num_intervals, 1))

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["orbit.tf[0]"] = last

    for p in range(4):
        thrust = units.thrust[p]
        flow = units.flow[p]

        states = phase_states(units, profile, p, last, tau)
        mid_states = phase_states(units, profile, p, last, tau_mid)

        x[f"dyn{p}.x"] = states
        x[f"dynm{p}.x"] = mid_states

        x[f"dyn{p}.xdot"] = vehicle_rates(states, controls, units, thrust, flow)
        x[f"dynm{p}.xdot"] = vehicle_rates(
            mid_states, mid_controls, units, thrust, flow
        )

        x[f"dyn{p}.u"] = controls
        x[f"dynm{p}.u"] = mid_controls

    return


def get_objective(x, canonical=False):
    """Objective of a solution with the final mass in kilograms to maximize"""
    units = Units(canonical)

    return -(x["dyn3.x"][-1, 6] * units.mass)


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

    model = create_delta3_ascent_model(num_intervals=num_intervals)
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

    tf = DURATIONS[0] + DURATIONS[1] + DURATIONS[2] + x["orbit.tf"][0]

    print(f"Objective:                  {get_objective(x):.6f}")
    print(f"Final mass:                 {x['dyn3.x'][-1, 6]:.3f} kg")
    print(f"Flight time:                {tf:.3f} s")
