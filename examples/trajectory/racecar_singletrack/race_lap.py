"""
Race car lap
------------

- Minimum lap time of a single track vehicle on a periodic lap
"""

import argparse
import json
import amigo as am
import numpy as np
from tracks import load_track

# Race on oval_dymos, berlin_2018, modena_2019, handling_track, or rounded_rectangle
track_name = "oval_dymos"

num_intervals = 300

# Vehicle parameters in SI units
M = 800.0
g = 9.8
a_wb = 1.8
b_wb = 1.6
L_wb = a_wb + b_wb
tw = 0.73
Iz = 450.0
rho = 1.2
CdA = 2.0
ClA = 4.0
CoP = 1.6
h_cg = 0.3
chi = 0.5
beta_brake = 0.62
k_lambda = 44.0
mu0 = 1.68
tau_x = 0.2
tau_y = 0.2
P_max = 960000.0

# Smoothing of the split of the thrust into throttle and brake
EPS_THRUST = 0.01

# Weight of the steering smoothness penalty
EPS_DELTA = 1e-3

# Scale factors of the states and the steering
scaling = {
    "t": 60.0,
    "n": 5.0,
    "V": 40.0,
    "alpha": 0.15,
    "lam": 0.05,
    "omega": 0.5,
    "ax": 8.0,
    "ay": 8.0,
    "delta": 0.1,
}


# Trapezoidal rule for the vehicle dynamics over one arc length interval
class RacecarCollocation(am.Component):
    def __init__(self, scaling):
        super().__init__()

        self.scaling = scaling

        self.add_data("ds")
        self.add_data("kappa1", value=0.0)
        self.add_data("kappa2", value=0.0)

        self.add_input("q1", shape=8)
        self.add_input("q2", shape=8)

        self.add_input("u1", shape=2)
        self.add_input("u2", shape=2)

        self.add_constraint("res", shape=8, lower=0.0, upper=0.0)

        return

    def rates(self, q, u, kappa):
        """Rates of the scaled states with respect to the arc length"""
        V = self.scaling["V"] * q[2]
        n = self.scaling["n"] * q[1]
        alpha = self.scaling["alpha"] * q[3]
        lam = self.scaling["lam"] * q[4]
        omega = self.scaling["omega"] * q[5]
        ax_phys = self.scaling["ax"] * q[6]
        ay_phys = self.scaling["ay"] * q[7]
        delta = self.scaling["delta"] * u[0]
        thrust = u[1]

        # Aerodynamic downforce split between the axles
        downforce = 0.5 * rho * ClA * V * V
        df_front = downforce * (1.0 - CoP / L_wb)
        df_rear = downforce * (CoP / L_wb)

        # Normal load on each wheel with the load transfer
        N_fl = (
            (M * g / 2) * (b_wb / L_wb)
            + (M / 4) * (-(ax_phys * h_cg) / L_wb + ay_phys * chi * h_cg / tw)
            + df_front / 2
        )
        N_fr = (
            (M * g / 2) * (b_wb / L_wb)
            + (M / 4) * (-(ax_phys * h_cg) / L_wb - ay_phys * chi * h_cg / tw)
            + df_front / 2
        )
        N_rl = (
            (M * g / 2) * (a_wb / L_wb)
            + (M / 4) * ((ax_phys * h_cg) / L_wb + ay_phys * (1 - chi) * h_cg / tw)
            + df_rear / 2
        )
        N_rr = (
            (M * g / 2) * (a_wb / L_wb)
            + (M / 4) * ((ax_phys * h_cg) / L_wb - ay_phys * (1 - chi) * h_cg / tw)
            + df_rear / 2
        )

        # Smooth split of the thrust into throttle and brake
        smooth_abs = am.sqrt(thrust * thrust + EPS_THRUST * EPS_THRUST)
        throttle = 0.5 * (thrust + smooth_abs)
        brake = 0.5 * (-thrust + smooth_abs)

        # Longitudinal force on each wheel
        S_fl = -(M * g / 2) * brake * beta_brake
        S_fr = -(M * g / 2) * brake * beta_brake
        S_rl = (M * g / 2) * (throttle - brake * (1 - beta_brake))
        S_rr = (M * g / 2) * (throttle - brake * (1 - beta_brake))

        # Lateral force on each wheel from its slip
        F_rr = N_rr * k_lambda * (lam + omega * (b_wb + lam * tw) / V)
        F_rl = N_rl * k_lambda * (lam + omega * (b_wb - lam * tw) / V)
        F_fr = N_fr * k_lambda * (lam + delta - omega * (a_wb - lam * tw) / V)
        F_fl = N_fl * k_lambda * (lam + delta - omega * (a_wb + lam * tw) / V)

        # Total forces and the aerodynamic drag
        F_all = F_fl + F_fr + F_rl + F_rr
        S_all = S_fl + S_fr + S_rl + S_rr
        F_front = F_fl + F_fr
        S_front = S_fl + S_fr
        drag = 0.5 * rho * CdA * V * V

        cos_al = am.cos(alpha - lam)
        sin_al = am.sin(alpha - lam)

        # Rates with respect to time in curvilinear coordinates
        sdot = V * cos_al / (1.0 - n * kappa)
        ndot = V * sin_al
        alphadot = omega - kappa * V * cos_al / (1.0 - n * kappa)
        Vdot = S_all / M - delta * F_front / M - drag / M - omega * V * lam
        lambdadot = omega - Vdot * lam / V - delta * S_front / (M * V) - F_all / (M * V)
        omegadot = (
            (a_wb * F_front) / Iz
            - (b_wb * (F_rr + F_rl)) / Iz
            + (tw * (-S_rr + S_rl - S_fr + S_fl)) / Iz
        )
        axdot = (Vdot + omega * V * lam - ax_phys) / tau_x
        aydot = (omega * V - (V * lambdadot + Vdot * lam) - ay_phys) / tau_y

        # Divide by the speed along the centerline and the scale of each state
        dqds = 8 * [None]
        dqds[0] = 1.0 / (sdot * self.scaling["t"])
        dqds[1] = ndot / (sdot * self.scaling["n"])
        dqds[2] = Vdot / (sdot * self.scaling["V"])
        dqds[3] = alphadot / (sdot * self.scaling["alpha"])
        dqds[4] = lambdadot / (sdot * self.scaling["lam"])
        dqds[5] = omegadot / (sdot * self.scaling["omega"])
        dqds[6] = axdot / (sdot * self.scaling["ax"])
        dqds[7] = aydot / (sdot * self.scaling["ay"])

        return dqds

    def compute(self):
        ds = self.data["ds"]
        kappa1 = self.data["kappa1"]
        kappa2 = self.data["kappa2"]

        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        u1 = self.inputs["u1"]
        u2 = self.inputs["u2"]

        # Rates at both ends of the interval
        f1 = self.rates(q1, u1, kappa1)
        f2 = self.rates(q2, u2, kappa2)

        # Trapezoidal rule over the interval
        res = 8 * [None]

        for i in range(8):
            res[i] = q2[i] - q1[i] - 0.5 * ds * (f1[i] + f2[i])

        self.constraints["res"] = res

        return


# Friction circle of each tire and the power limit at one node
class NodeConstraints(am.Component):
    def __init__(self, scaling):
        super().__init__()

        self.scaling = scaling

        # Start with every state unbounded
        q_min = [-am.inf] * 8
        q_max = [am.inf] * 8

        # Lower bound on the speed
        v_min = 1.0 / scaling["V"]
        q_min[2] = v_min

        # Bounds on the heading relative to the centerline
        alpha_max = (np.pi / 3) / scaling["alpha"]
        q_min[3] = -alpha_max
        q_max[3] = alpha_max

        # Bounds on the slip angle
        lam_max = 0.25 / scaling["lam"]
        q_min[4] = -lam_max
        q_max[4] = lam_max

        # Bounds on the steering angle
        delta_max = 0.5 / scaling["delta"]
        u_min = [-delta_max, -am.inf]
        u_max = [delta_max, am.inf]

        self.add_input("q", shape=8, lower=q_min, upper=q_max)
        self.add_input("u", shape=2, lower=u_min, upper=u_max)

        self.add_constraint("c", shape=5, lower=-am.inf, upper=1.0)

        return

    def compute(self):
        q = self.inputs["q"]
        u = self.inputs["u"]

        V = self.scaling["V"] * q[2]
        lam = self.scaling["lam"] * q[4]
        omega = self.scaling["omega"] * q[5]
        ax_phys = self.scaling["ax"] * q[6]
        ay_phys = self.scaling["ay"] * q[7]
        delta = self.scaling["delta"] * u[0]
        thrust = u[1]

        # Aerodynamic downforce split between the axles
        downforce = 0.5 * rho * ClA * V * V
        df_front = downforce * (1.0 - CoP / L_wb)
        df_rear = downforce * (CoP / L_wb)

        # Normal load on each wheel with the load transfer
        N_fl = (
            (M * g / 2) * (b_wb / L_wb)
            + (M / 4) * (-(ax_phys * h_cg) / L_wb + ay_phys * chi * h_cg / tw)
            + df_front / 2
        )
        N_fr = (
            (M * g / 2) * (b_wb / L_wb)
            + (M / 4) * (-(ax_phys * h_cg) / L_wb - ay_phys * chi * h_cg / tw)
            + df_front / 2
        )
        N_rl = (
            (M * g / 2) * (a_wb / L_wb)
            + (M / 4) * ((ax_phys * h_cg) / L_wb + ay_phys * (1 - chi) * h_cg / tw)
            + df_rear / 2
        )
        N_rr = (
            (M * g / 2) * (a_wb / L_wb)
            + (M / 4) * ((ax_phys * h_cg) / L_wb - ay_phys * (1 - chi) * h_cg / tw)
            + df_rear / 2
        )

        # Smooth split of the thrust into throttle and brake
        smooth_abs = am.sqrt(thrust * thrust + EPS_THRUST * EPS_THRUST)
        throttle = 0.5 * (thrust + smooth_abs)
        brake = 0.5 * (-thrust + smooth_abs)

        # Longitudinal force on each wheel
        S_fl = -(M * g / 2) * brake * beta_brake
        S_fr = -(M * g / 2) * brake * beta_brake
        S_rl = (M * g / 2) * (throttle - brake * (1 - beta_brake))
        S_rr = (M * g / 2) * (throttle - brake * (1 - beta_brake))

        # Lateral force on each wheel per unit normal load
        fn_rr = k_lambda * (lam + omega * (b_wb + lam * tw) / V)
        fn_rl = k_lambda * (lam + omega * (b_wb - lam * tw) / V)
        fn_fr = k_lambda * (lam + delta - omega * (a_wb - lam * tw) / V)
        fn_fl = k_lambda * (lam + delta - omega * (a_wb + lam * tw) / V)

        S_all = S_fl + S_fr + S_rl + S_rr

        # Friction circle of each wheel and the power usage
        c_fl = (S_fl / (N_fl * mu0)) ** 2 + (fn_fl / mu0) ** 2
        c_fr = (S_fr / (N_fr * mu0)) ** 2 + (fn_fr / mu0) ** 2
        c_rl = (S_rl / (N_rl * mu0)) ** 2 + (fn_rl / mu0) ** 2
        c_rr = (S_rr / (N_rr * mu0)) ** 2 + (fn_rr / mu0) ** 2
        power = V * S_all / P_max

        self.constraints["c"] = [c_fl, c_fr, c_rl, c_rr, power]

        return


# The lap starts at zero time
class InitialTime(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("t", lower=-am.inf, upper=am.inf)

        self.add_constraint("res", shape=1, lower=0.0, upper=0.0)

        return

    def compute(self):
        t = self.inputs["t"]

        self.constraints["res"] = [t]

        return


class LapTimeObjective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("t_final", lower=-am.inf, upper=am.inf)

        self.add_objective("obj")

        return

    def compute(self):
        t_final = self.inputs["t_final"]

        self.objective["obj"] = t_final

        return


# Penalty on the change of the steering over one interval
class DeltaSmoothing(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("delta_left")
        self.add_input("delta_right")

        self.add_objective("reg")

        return

    def compute(self):
        delta_left = self.inputs["delta_left"]
        delta_right = self.inputs["delta_right"]

        diff = delta_right - delta_left

        self.objective["reg"] = EPS_DELTA * diff * diff

        return


def create_racecar_model(track, module_name="racecar_mod", num_intervals=300):
    colloc = RacecarCollocation(scaling)
    nc = NodeConstraints(scaling)
    ic = InitialTime()
    obj = LapTimeObjective()
    smooth = DeltaSmoothing()

    model = am.Model(module_name)

    model.add_component("colloc", num_intervals, colloc)
    model.add_component("nc", num_intervals + 1, nc)
    model.add_component("ic", 1, ic)
    model.add_component("obj", 1, obj)
    model.add_component("smooth", num_intervals, smooth)

    # Link the states and the controls that neighboring intervals share
    for i in range(8):
        model.link(f"colloc.q2[:{num_intervals-1}, {i}]", f"colloc.q1[1:, {i}]")
    for i in range(2):
        model.link(f"colloc.u2[:{num_intervals-1}, {i}]", f"colloc.u1[1:, {i}]")

    # Link the end of the lap to its start except for the time
    for i in range(1, 8):
        model.link(f"colloc.q2[{num_intervals-1}, {i}]", f"colloc.q1[0, {i}]")
    for i in range(2):
        model.link(f"colloc.u2[{num_intervals-1}, {i}]", f"colloc.u1[0, {i}]")

    # Link the time at the start and at the end of the lap
    model.link("colloc.q1[0, 0]", "ic.t[0]")
    model.link(f"colloc.q2[{num_intervals-1}, 0]", "obj.t_final[0]")

    # Link the steering at both ends of every interval to the smoothness penalty
    model.link("colloc.u1[:, 0]", "smooth.delta_left")
    model.link("colloc.u2[:, 0]", "smooth.delta_right")

    # Link the states and the controls at every node to the node constraints
    for i in range(8):
        model.link(f"colloc.q1[:, {i}]", f"nc.q[:{num_intervals}, {i}]")
        model.link(f"colloc.q2[{num_intervals-1}, {i}]", f"nc.q[{num_intervals}, {i}]")
    for i in range(2):
        model.link(f"colloc.u1[:, {i}]", f"nc.u[:{num_intervals}, {i}]")
        model.link(f"colloc.u2[{num_intervals-1}, {i}]", f"nc.u[{num_intervals}, {i}]")

    # Bound the lateral offset by the track width and the curvature singularity
    kappa_margin = 0.9
    n_lower = -track.w_right / scaling["n"]
    n_upper = track.w_left / scaling["n"]

    for i in range(num_intervals + 1):
        kappa = track.kappa[i]

        if abs(kappa) > 1e-6:
            n_sing = kappa_margin / (kappa * scaling["n"])

            if kappa > 0:
                n_upper[i] = min(n_upper[i], n_sing)
            else:
                n_lower[i] = max(n_lower[i], n_sing)

    model.set_meta("lower", "nc.q[:, 1]", n_lower)
    model.set_meta("upper", "nc.q[:, 1]", n_upper)

    # Set the curvature at both ends of every interval
    model.set_data("colloc.kappa1", track.kappa[:-1])
    model.set_data("colloc.kappa2", track.kappa[1:])

    # Set the arc length of every interval
    model.set_data("colloc.ds", track.s_total / num_intervals)

    return model


def set_initial_guess(model, track, num_intervals=300):
    """Initial point at a constant speed along the centerline"""
    V_init = 20.0

    # Time at every node at the constant speed
    t_est = track.s_total / V_init
    t_init = np.linspace(0, t_est, num_intervals + 1)

    # Set the initial point by using a view of the initial values
    x = model.get_meta_view("value")

    x["colloc.q1[:, 0]"] = t_init[:-1] / scaling["t"]
    x["colloc.q2[:, 0]"] = t_init[1:] / scaling["t"]

    # The speed is constant over the lap
    x["colloc.q1[:, 2]"] = V_init / scaling["V"]
    x["colloc.q2[:, 2]"] = V_init / scaling["V"]

    # The thrust starts slightly positive
    x["colloc.u1[:, 1]"] = 0.1
    x["colloc.u2[:, 1]"] = 0.1

    return


def node_constraint_values(q, u):
    """Friction circle of each wheel and the power usage at every node"""
    V = scaling["V"] * q[:, 2]
    lam = scaling["lam"] * q[:, 4]
    omega = scaling["omega"] * q[:, 5]
    ax_phys = scaling["ax"] * q[:, 6]
    ay_phys = scaling["ay"] * q[:, 7]
    delta = scaling["delta"] * u[:, 0]
    thrust = u[:, 1]

    # Aerodynamic downforce split between the axles
    downforce = 0.5 * rho * ClA * V * V
    df_front = downforce * (1.0 - CoP / L_wb)
    df_rear = downforce * (CoP / L_wb)

    # Normal load on each wheel with the load transfer
    dN_ax = (M / 4) * (ax_phys * h_cg) / L_wb
    dN_f = (M / 4) * ay_phys * chi * h_cg / tw
    dN_r = (M / 4) * ay_phys * (1 - chi) * h_cg / tw
    N_fl = (M * g / 2) * (b_wb / L_wb) - dN_ax + dN_f + df_front / 2
    N_fr = (M * g / 2) * (b_wb / L_wb) - dN_ax - dN_f + df_front / 2
    N_rl = (M * g / 2) * (a_wb / L_wb) + dN_ax + dN_r + df_rear / 2
    N_rr = (M * g / 2) * (a_wb / L_wb) + dN_ax - dN_r + df_rear / 2

    # Smooth split of the thrust into throttle and brake
    smooth_abs = np.sqrt(thrust * thrust + EPS_THRUST * EPS_THRUST)
    throttle = 0.5 * (thrust + smooth_abs)
    brake = 0.5 * (-thrust + smooth_abs)

    # Longitudinal force on each wheel
    S_fl = -(M * g / 2) * brake * beta_brake
    S_fr = -(M * g / 2) * brake * beta_brake
    S_rl = (M * g / 2) * (throttle - brake * (1 - beta_brake))
    S_rr = (M * g / 2) * (throttle - brake * (1 - beta_brake))

    # Lateral force on each wheel per unit normal load
    fn_rr = k_lambda * (lam + omega * (b_wb + lam * tw) / V)
    fn_rl = k_lambda * (lam + omega * (b_wb - lam * tw) / V)
    fn_fr = k_lambda * (lam + delta - omega * (a_wb - lam * tw) / V)
    fn_fl = k_lambda * (lam + delta - omega * (a_wb + lam * tw) / V)

    # Friction circle of each wheel and the power usage
    c_fl = (S_fl / (N_fl * mu0)) ** 2 + (fn_fl / mu0) ** 2
    c_fr = (S_fr / (N_fr * mu0)) ** 2 + (fn_fr / mu0) ** 2
    c_rl = (S_rl / (N_rl * mu0)) ** 2 + (fn_rl / mu0) ** 2
    c_rr = (S_rr / (N_rr * mu0)) ** 2 + (fn_rr / mu0) ** 2
    S_all = S_fl + S_fr + S_rl + S_rr
    power = V * S_all / P_max

    return np.column_stack([c_fl, c_fr, c_rl, c_rr, power])


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

track = load_track(track_name, num_intervals + 1)

model = create_racecar_model(track, num_intervals=num_intervals)
set_initial_guess(model, track, num_intervals=num_intervals)

if args.build:
    model.build_module()

model.initialize()

print(f"Track: {track.name}")
print(
    f"Track length: {track.s_total:.2f} m, intervals: {num_intervals}, "
    f"ds: {track.s_total / num_intervals:.2f} m"
)
print(f"Curvature: [{track.kappa.min():.4f}, {track.kappa.max():.4f}] 1/m")
print(f"Variables:   {model.num_variables}")
print(f"Constraints: {model.num_constraints}")

opt_options = {
    "max_iterations": 200,
    "max_line_search_iterations": 30,
    "convergence_tolerance": 1e-10,
    "quality_function_balancing_term": "cubic",
    "adaptive_mu_safeguard_factor": 1e-1,
    "second_order_correction": False,
}

# Get the design variables
x = model.create_vector()

# Set up the optimizer and optimize
opt = am.Optimizer(model, x=x)
opt_data = opt.optimize(opt_options)

# Copy the solution from the device to the host
x.copy_device_to_host()

print(f"\nConverged: {opt_data['converged']}")
print(f"Iterations: {len(opt_data['iterations'])}")

q = x["nc.q"]
u = x["nc.u"]

t = q[:, 0] * scaling["t"]
n = q[:, 1] * scaling["n"]
V = q[:, 2] * scaling["V"]
alpha = q[:, 3] * scaling["alpha"]
omega = q[:, 5] * scaling["omega"]
delta = u[:, 0] * scaling["delta"]
thrust = u[:, 1]

print(f"\nLap time: {t[-1]:.4f} s")
print(f"Speed: [{V.min():.2f}, {V.max():.2f}] m/s")
print(f"Lateral offset: [{n.min():.3f}, {n.max():.3f}] m")
print(f"Steering: [{np.degrees(delta.min()):.2f}, {np.degrees(delta.max()):.2f}] deg")
print(f"Thrust: [{thrust.min():.3f}, {thrust.max():.3f}]")

print("\nPeriodicity (start vs end):")
print(f"  V: {V[0]:.6f} vs {V[-1]:.6f}")
print(f"  n: {n[0]:.6f} vs {n[-1]:.6f}")
print(f"  alpha: {alpha[0]:.6f} vs {alpha[-1]:.6f}")
print(f"  omega: {omega[0]:.6f} vs {omega[-1]:.6f}")

c = node_constraint_values(q, u)

print("\nConstraint max (should be <= 1):")
print(f"  FL: {c[:, 0].max():.4f}")
print(f"  FR: {c[:, 1].max():.4f}")
print(f"  RL: {c[:, 2].max():.4f}")
print(f"  RR: {c[:, 3].max():.4f}")
print(f"  Pow: {c[:, 4].max():.4f}")

with open("racecar_opt_data.json", "w") as fp:
    json.dump(opt_data, fp, indent=2)

# Save the lap in physical units for the plot script
with open("racecar_trajectory.json", "w") as fp:
    json.dump(
        {
            "track": track.name,
            "s": track.s.tolist(),
            "t": t.tolist(),
            "n": n.tolist(),
            "V": V.tolist(),
            "delta": delta.tolist(),
            "thrust": thrust.tolist(),
            "c": c.tolist(),
        },
        fp,
    )
