"""
Racecar Minimum Lap Time -- CasADi / IPOPT version.

Same physics and discretization as race_lap.py (amigo version),
re-implemented with CasADi symbolic expressions and solved with IPOPT.
Plots include KKT residual history.
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracks import berlin_2018

# ── Vehicle parameters (identical to race_lap.py) ──────────────────────
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
EPS_THRUST = 0.00001

# Scaling
sc = {
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

# Discretization
num_intervals = 300
num_nodes = num_intervals + 1
EPS_DELTA = 1e-3

track = berlin_2018(num_nodes)
s_final = track.s_total
s_nodes = track.s
kappa_nodes = track.kappa
ds = s_final / num_intervals

print(f"Track: {track.name}")
print(f"Track length: {s_final:.2f} m, Intervals: {num_intervals}, ds: {ds:.2f} m")
print(f"Curvature range: [{kappa_nodes.min():.4f}, {kappa_nodes.max():.4f}] 1/m")


# ── Dynamics function (symbolic) ───────────────────────────────────────
def dynamics(q, u, kappa):
    """Scaled d/ds rates for 8 ODE states. q scaled, u = [delta, thrust]."""
    V = sc["V"] * q[2]
    n = sc["n"] * q[1]
    alpha = sc["alpha"] * q[3]
    lam = sc["lam"] * q[4]
    omega = sc["omega"] * q[5]
    ax_phys = sc["ax"] * q[6]
    ay_phys = sc["ay"] * q[7]
    delta = sc["delta"] * u[0]
    thrust = u[1]

    # Normal forces
    downforce = 0.5 * rho * ClA * V * V
    df_front = downforce * (1.0 - CoP / L_wb)
    df_rear = downforce * (CoP / L_wb)

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

    smooth_abs = ca.sqrt(thrust * thrust + EPS_THRUST * EPS_THRUST)
    throttle = 0.5 * (thrust + smooth_abs)
    brake = 0.5 * (-thrust + smooth_abs)

    S_fl = -(M * g / 2) * brake * beta_brake
    S_fr = -(M * g / 2) * brake * beta_brake
    S_rl = (M * g / 2) * (throttle - brake * (1 - beta_brake))
    S_rr = (M * g / 2) * (throttle - brake * (1 - beta_brake))

    F_rr = N_rr * k_lambda * (lam + omega * (b_wb + lam * tw) / V)
    F_rl = N_rl * k_lambda * (lam + omega * (b_wb - lam * tw) / V)
    F_fr = N_fr * k_lambda * (lam + delta - omega * (a_wb - lam * tw) / V)
    F_fl = N_fl * k_lambda * (lam + delta - omega * (a_wb + lam * tw) / V)

    F_all = F_fl + F_fr + F_rl + F_rr
    S_all = S_fl + S_fr + S_rl + S_rr
    F_front = F_fl + F_fr
    S_front = S_fl + S_fr
    drag = 0.5 * rho * CdA * V * V

    cos_al = ca.cos(alpha - lam)
    sin_al = ca.sin(alpha - lam)

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

    return ca.vertcat(
        1.0 / (sdot * sc["t"]),
        ndot / (sdot * sc["n"]),
        Vdot / (sdot * sc["V"]),
        alphadot / (sdot * sc["alpha"]),
        lambdadot / (sdot * sc["lam"]),
        omegadot / (sdot * sc["omega"]),
        axdot / (sdot * sc["ax"]),
        aydot / (sdot * sc["ay"]),
    )


def node_constraints(q, u):
    """Friction circle (4 wheels) + power. Returns 5-vector, each <= 0 at feasibility."""
    V = sc["V"] * q[2]
    lam = sc["lam"] * q[4]
    omega = sc["omega"] * q[5]
    ax_phys = sc["ax"] * q[6]
    ay_phys = sc["ay"] * q[7]
    delta = sc["delta"] * u[0]
    thrust = u[1]

    downforce = 0.5 * rho * ClA * V * V
    df_front = downforce * (1.0 - CoP / L_wb)
    df_rear = downforce * (CoP / L_wb)

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

    smooth_abs = ca.sqrt(thrust * thrust + EPS_THRUST * EPS_THRUST)
    throttle = 0.5 * (thrust + smooth_abs)
    brake = 0.5 * (-thrust + smooth_abs)

    S_fl = -(M * g / 2) * brake * beta_brake
    S_fr = -(M * g / 2) * brake * beta_brake
    S_rl = (M * g / 2) * (throttle - brake * (1 - beta_brake))
    S_rr = (M * g / 2) * (throttle - brake * (1 - beta_brake))

    F_rr = N_rr * k_lambda * (lam + omega * (b_wb + lam * tw) / V)
    F_rl = N_rl * k_lambda * (lam + omega * (b_wb - lam * tw) / V)
    F_fr = N_fr * k_lambda * (lam + delta - omega * (a_wb - lam * tw) / V)
    F_fl = N_fl * k_lambda * (lam + delta - omega * (a_wb + lam * tw) / V)

    S_all = S_fl + S_fr + S_rl + S_rr

    c_fl = (S_fl / (N_fl * mu0)) ** 2 + (F_fl / (N_fl * mu0)) ** 2
    c_fr = (S_fr / (N_fr * mu0)) ** 2 + (F_fr / (N_fr * mu0)) ** 2
    c_rl = (S_rl / (N_rl * mu0)) ** 2 + (F_rl / (N_rl * mu0)) ** 2
    c_rr = (S_rr / (N_rr * mu0)) ** 2 + (F_rr / (N_rr * mu0)) ** 2

    return ca.vertcat(
        c_fl - 1.0, c_fr - 1.0, c_rl - 1.0, c_rr - 1.0, V * S_all / P_max - 1.0
    )


# ── Build NLP ──────────────────────────────────────────────────────────
print("Building CasADi NLP...")

# Decision variables: q1[k] (8), u1[k] (2) for k=0..N-1, plus q2[N-1] (8), u2[N-1] (2)
# But simpler: define node variables q[k] (8) and u[k] (2) for k=0..N
# Then collocation links q[k] and q[k+1].

n_states = 8
n_controls = 2

# Node-based variables
Q = []  # q[k] for k = 0..num_nodes-1
U = []  # u[k] for k = 0..num_nodes-1
for k in range(num_nodes):
    Q.append(ca.MX.sym(f"q_{k}", n_states))
    U.append(ca.MX.sym(f"u_{k}", n_controls))

w = ca.vertcat(*Q, *U)
n_vars = w.shape[0]
print(f"Num variables: {n_vars}")

# Constraints and bounds
g_list = []
lbg_list = []
ubg_list = []

# 1) Collocation constraints: q[k+1] - q[k] - 0.5*ds*(f(q[k],u[k],kappa[k]) + f(q[k+1],u[k+1],kappa[k+1])) = 0
for k in range(num_intervals):
    f1 = dynamics(Q[k], U[k], kappa_nodes[k])
    f2 = dynamics(Q[k + 1], U[k + 1], kappa_nodes[k + 1])
    res = Q[k + 1] - Q[k] - 0.5 * ds * (f1 + f2)
    g_list.append(res)
    lbg_list.append(np.zeros(n_states))
    ubg_list.append(np.zeros(n_states))

# 2) Cyclic BCs: q[N] = q[0] for states 1..7 (not time)
for i in [1, 2, 3, 4, 5, 6, 7]:
    g_list.append(Q[num_nodes - 1][i] - Q[0][i])
    lbg_list.append([0.0])
    ubg_list.append([0.0])

# Cyclic BCs for controls
for i in range(n_controls):
    g_list.append(U[num_nodes - 1][i] - U[0][i])
    lbg_list.append([0.0])
    ubg_list.append([0.0])

# 3) Initial time: t(0) = 0
g_list.append(Q[0][0])
lbg_list.append([0.0])
ubg_list.append([0.0])

# 4) Node constraints (friction circles + power) at all nodes
for k in range(num_nodes):
    nc = node_constraints(Q[k], U[k])
    g_list.append(nc)
    lbg_list.append(-np.inf * np.ones(5))
    ubg_list.append(np.zeros(5))

g_all = ca.vertcat(*g_list)
lbg = np.concatenate(lbg_list)
ubg = np.concatenate(ubg_list)

n_constraints = g_all.shape[0]
print(f"Num constraints: {n_constraints}")

# Objective: minimize lap time + delta smoothing
obj = Q[num_nodes - 1][0]  # scaled final time
for k in range(num_intervals):
    diff = U[k + 1][0] - U[k][0]
    obj = obj + EPS_DELTA * diff * diff
# Wrap-around delta smoothing (last to first, already linked by cyclic BC but add cost)
diff_wrap = U[0][0] - U[num_nodes - 1][0]
obj = obj + EPS_DELTA * diff_wrap * diff_wrap

# Variable bounds
lbw = -np.inf * np.ones(n_vars)
ubw = np.inf * np.ones(n_vars)

for k in range(num_nodes):
    # q indices in w: k * n_states .. (k+1) * n_states - 1
    idx_q = k * n_states
    # V > 0 (q[k][2] >= 1/sc["V"])
    lbw[idx_q + 2] = 1.0 / sc["V"]
    # Track width bounds on n (q[k][1])
    lbw[idx_q + 1] = -track.w_right[k] / sc["n"]
    ubw[idx_q + 1] = track.w_left[k] / sc["n"]

# u bounds: no explicit bounds (all -inf to inf)
# u indices start at num_nodes * n_states

# Initial guess
w0 = np.zeros(n_vars)
V_init = 20.0
t_est = s_final / V_init

for k in range(num_nodes):
    idx_q = k * n_states
    idx_u = num_nodes * n_states + k * n_controls
    w0[idx_q + 0] = (t_est * k / num_intervals) / sc["t"]  # time
    w0[idx_q + 2] = V_init / sc["V"]  # speed
    w0[idx_u + 1] = 0.1  # thrust

print(f"Initial guess: V = {V_init:.1f} m/s, estimated lap time = {t_est:.1f} s")

# ── Solve with IPOPT ───────────────────────────────────────────────────
nlp = {"x": w, "f": obj, "g": g_all}

import os

output_file = os.path.join(str(Path(__file__).resolve().parent), "ipopt_output.txt")

opts = {
    "ipopt.max_iter": 2000,
    "ipopt.tol": 1e-7,
    "ipopt.mu_init": 0.1,
    "ipopt.print_level": 5,
    "ipopt.linear_solver": "mumps",
    "print_time": True,
    # Match amigo solver settings
    "ipopt.mu_strategy": "adaptive",
    "ipopt.tau_min": 0.995,  # fraction_to_boundary
    "ipopt.max_soc": 10,  # second_order_correction
    "ipopt.accept_every_trial_step": "no",
    "ipopt.output_file": output_file,
}

solver = ca.nlpsol("solver", "ipopt", nlp, opts)

print("\nOptimizing...")
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# ── Extract solution ──────────────────────────────────────────────────
w_sol = np.array(sol["x"]).flatten()

q_sol = np.zeros((num_nodes, n_states))
u_sol = np.zeros((num_nodes, n_controls))
for k in range(num_nodes):
    q_sol[k, :] = w_sol[k * n_states : (k + 1) * n_states]
    u_sol[k, :] = w_sol[
        num_nodes * n_states
        + k * n_controls : num_nodes * n_states
        + (k + 1) * n_controls
    ]

t_sol = q_sol[:, 0] * sc["t"]
n_sol = q_sol[:, 1] * sc["n"]
V_sol = q_sol[:, 2] * sc["V"]
alpha_sol = q_sol[:, 3] * sc["alpha"]
lam_sol = q_sol[:, 4] * sc["lam"]
omega_sol = q_sol[:, 5] * sc["omega"]
ax_sol = q_sol[:, 6] * sc["ax"]
ay_sol = q_sol[:, 7] * sc["ay"]
delta_sol = u_sol[:, 0] * sc["delta"]
thrust_sol = u_sol[:, 1]

print(f"\nLap time: {t_sol[-1]:.4f} s")
print(f"Speed range: [{V_sol.min():.2f}, {V_sol.max():.2f}] m/s")
print(f"Lateral offset range: [{n_sol.min():.3f}, {n_sol.max():.3f}] m")
print(
    f"Steering range: [{np.degrees(delta_sol.min()):.2f}, {np.degrees(delta_sol.max()):.2f}] deg"
)
print(f"Thrust range: [{thrust_sol.min():.3f}, {thrust_sol.max():.3f}]")

print(f"\nPeriodicity check (start vs end):")
print(f"  V:     {V_sol[0]:.4f} vs {V_sol[-1]:.4f} m/s")
print(f"  n:     {n_sol[0]:.4f} vs {n_sol[-1]:.4f} m")
print(f"  alpha: {alpha_sol[0]:.6f} vs {alpha_sol[-1]:.6f} rad")
print(f"  omega: {omega_sol[0]:.6f} vs {omega_sol[-1]:.6f} rad/s")

# ── Parse IPOPT output for KKT residuals ──────────────────────────────
# IPOPT prints iteration info to stdout. We'll get stats from the solver.
stats = solver.stats()
print(f"\nIPOPT return status: {stats['return_status']}")
print(f"IPOPT iterations: {stats['iter_count']}")

# ── Parse IPOPT output file for iteration data ────────────────────────
import re

iterations = []
try:
    with open(output_file, "r") as f:
        lines = f.readlines()

    # IPOPT iteration lines look like:
    # iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
    #    0  1.5443709e+00 2.17e+00 5.70e-01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
    header_found = False
    for line in lines:
        line = line.strip()
        if line.startswith("iter") and "objective" in line:
            header_found = True
            continue
        if header_found and line and line[0].isdigit():
            # Parse iteration line
            parts = line.split()
            if len(parts) >= 10:
                try:
                    it_num = int(parts[0])
                    obj_val = float(parts[1])
                    inf_pr = float(parts[2])
                    inf_du = float(parts[3])
                    lg_mu = float(parts[4])
                    d_norm = float(parts[5])
                    # lg_rg can be '-'
                    alpha_du = float(parts[7])
                    alpha_pr = float(parts[8])
                    iterations.append(
                        {
                            "iter": it_num,
                            "objective": obj_val,
                            "inf_pr": inf_pr,
                            "inf_du": inf_du,
                            "lg_mu": lg_mu,
                            "mu": 10**lg_mu if lg_mu > -30 else 0,
                            "d_norm": d_norm,
                            "alpha_du": alpha_du,
                            "alpha_pr": alpha_pr,
                        }
                    )
                except (ValueError, IndexError):
                    continue
        # Reset on blank lines between iteration blocks
        if header_found and line == "":
            pass  # keep going, there may be restarts

    print(f"Parsed {len(iterations)} iterations from IPOPT output")
    os.remove(output_file)
except Exception as e:
    print(f"Could not parse IPOPT output: {e}")

# ── Plotting ──────────────────────────────────────────────────────────
print("\nPlotting...")


def build_track_edges(x_c, y_c, s_c, w_right, w_left):
    dx = np.gradient(x_c, s_c)
    dy = np.gradient(y_c, s_c)
    mag = np.sqrt(dx**2 + dy**2)
    nx = -dy / mag
    ny = dx / mag
    return x_c + w_left * nx, y_c + w_left * ny, x_c - w_right * nx, y_c - w_right * ny


def racing_line_xy(s_sol, n_sol, s_c, x_c, y_c):
    x_center = np.interp(s_sol, s_c, x_c)
    y_center = np.interp(s_sol, s_c, y_c)
    dx = np.gradient(x_center, s_sol)
    dy = np.gradient(y_center, s_sol)
    mag = np.sqrt(dx**2 + dy**2)
    return x_center + n_sol * (-dy / mag), y_center + n_sol * (dx / mag)


def plot_track_colored(
    ax,
    x_left,
    y_left,
    x_right,
    y_right,
    x_race,
    y_race,
    color_values,
    cmap,
    label,
    title,
):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    ax.fill(
        np.append(x_left, x_right[::-1]),
        np.append(y_left, y_right[::-1]),
        color="0.92",
        zorder=0,
    )
    ax.plot(x_left, y_left, "k-", linewidth=1, alpha=0.7)
    ax.plot(x_right, y_right, "k-", linewidth=1, alpha=0.7)

    points = np.column_stack([x_race, y_race]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=np.min(color_values), vmax=np.max(color_values))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, zorder=5)
    lc.set_array(color_values)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label=label, shrink=0.8)
    ax.set(xlabel="x (m)", ylabel="y (m)", title=title, aspect="equal")
    ax.autoscale_view()
    ax.grid(True, alpha=0.2)


try:
    x_left, y_left, x_right, y_right = build_track_edges(
        track.raw_x, track.raw_y, track.raw_s, track.raw_w_right, track.raw_w_left
    )

    s_fine = np.linspace(0, s_final, 500)
    n_fine = np.interp(s_fine, s_nodes, n_sol)
    V_fine = np.interp(s_fine, s_nodes, V_sol)
    thrust_fine = np.interp(s_fine, s_nodes, thrust_sol)
    delta_fine = np.interp(s_fine, s_nodes, delta_sol)
    x_race, y_race = racing_line_xy(
        s_fine, n_fine, track.raw_s, track.raw_x, track.raw_y
    )

    # Track plots
    for color_val, cmap, label, fname in [
        (V_fine, "viridis", "V (m/s)", "casadi_track_velocity.png"),
        (thrust_fine, "RdYlGn", "thrust", "casadi_track_thrust.png"),
        (delta_fine, "coolwarm", "delta (rad)", "casadi_track_steering.png"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        plot_track_colored(
            ax,
            x_left,
            y_left,
            x_right,
            y_right,
            x_race,
            y_race,
            color_val,
            cmap,
            label,
            f"CasADi/IPOPT - {track.name} ({t_sol[-1]:.2f} s)",
        )
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        print(f"Saved {fname}")

    # Telemetry
    # Recompute constraint utilizations
    c_fl_val = np.zeros(num_nodes)
    c_fr_val = np.zeros(num_nodes)
    c_rl_val = np.zeros(num_nodes)
    c_rr_val = np.zeros(num_nodes)
    power_val = np.zeros(num_nodes)

    nc_sym_q = ca.MX.sym("q_nc", 8)
    nc_sym_u = ca.MX.sym("u_nc", 2)
    nc_expr = node_constraints(nc_sym_q, nc_sym_u)
    nc_func = ca.Function("nc_func", [nc_sym_q, nc_sym_u], [nc_expr])

    for k in range(num_nodes):
        nc_val = np.array(nc_func(q_sol[k], u_sol[k])).flatten()
        c_fl_val[k] = nc_val[0] + 1.0  # undo the -1
        c_fr_val[k] = nc_val[1] + 1.0
        c_rl_val[k] = nc_val[2] + 1.0
        c_rr_val[k] = nc_val[3] + 1.0
        power_val[k] = nc_val[4] + 1.0

    print(f"\nInequality constraint margins (c <= 1, margin = 1 - c, should be >= 0):")
    print(f"  Tire FL: {(1 - c_fl_val).min():.4f}")
    print(f"  Tire FR: {(1 - c_fr_val).min():.4f}")
    print(f"  Tire RL: {(1 - c_rl_val).min():.4f}")
    print(f"  Tire RR: {(1 - c_rr_val).min():.4f}")
    print(f"  Power:   {(1 - power_val).min():.4f}")

    fig4, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    axes[0].plot(s_nodes, V_sol, "b-", lw=1.5)
    axes[0].set_ylabel("V (m/s)")
    axes[0].set_title(f"CasADi/IPOPT - {track.name} - Telemetry")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(s_nodes, n_sol, "r-", lw=1.5)
    axes[1].plot(s_nodes, track.w_left, "k--", alpha=0.3, label="track edge")
    axes[1].plot(s_nodes, -track.w_right, "k--", alpha=0.3)
    axes[1].set_ylabel("n (m)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(s_nodes, thrust_sol, "g-", lw=1.5)
    axes[2].axhline(y=0, color="k", alpha=0.2)
    axes[2].set_ylabel("thrust")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(s_nodes, delta_sol, "m-", lw=1.5)
    axes[3].set_ylabel("delta (rad)")
    axes[3].grid(True, alpha=0.3)
    axes[4].plot(s_nodes, c_fl_val, label="c_fl", alpha=0.7)
    axes[4].plot(s_nodes, c_fr_val, label="c_fr", alpha=0.7)
    axes[4].plot(s_nodes, c_rl_val, label="c_rl", alpha=0.7)
    axes[4].plot(s_nodes, c_rr_val, label="c_rr", alpha=0.7)
    axes[4].plot(s_nodes, power_val, "k-", label="power/Pmax", lw=1.5, alpha=0.8)
    axes[4].axhline(y=1, color="r", ls="--", lw=1.5, label="Limit")
    axes[4].set_ylabel("Constraint")
    axes[4].set_xlabel("s (m)")
    axes[4].legend(loc="upper right", fontsize=8, ncol=3)
    axes[4].grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig("casadi_telemetry.png", dpi=150)
    print("Saved casadi_telemetry.png")

    # Convergence / KKT residual plot
    if iterations:
        iters = [it["iter"] for it in iterations]
        inf_pr = [it["inf_pr"] for it in iterations]
        inf_du = [it["inf_du"] for it in iterations]
        mus = [it["mu"] for it in iterations]
        alpha_pr = [it["alpha_pr"] for it in iterations]

        # Combined KKT residual: max(inf_pr, inf_du)
        kkt_res = [max(p, d) for p, d in zip(inf_pr, inf_du)]

        fig2, (ax_kkt, ax_components, ax_mu, ax_alpha) = plt.subplots(
            4, 1, figsize=(12, 14), sharex=True
        )

        ax_kkt.semilogy(
            iters,
            kkt_res,
            "k-",
            marker="o",
            markersize=2,
            lw=1.5,
            label="max(inf_pr, inf_du)",
        )
        ax_kkt.set_ylabel("KKT residual")
        ax_kkt.set_title("CasADi/IPOPT - Convergence History (KKT Residual)")
        ax_kkt.legend(fontsize=8)
        ax_kkt.grid(True, alpha=0.3)

        ax_components.semilogy(
            iters,
            inf_pr,
            "b-",
            marker=".",
            markersize=2,
            lw=1,
            label="Primal infeasibility",
        )
        ax_components.semilogy(
            iters,
            inf_du,
            "r-",
            marker=".",
            markersize=2,
            lw=1,
            label="Dual infeasibility",
        )
        ax_components.set_ylabel("Infeasibility")
        ax_components.legend(fontsize=8)
        ax_components.grid(True, alpha=0.3)

        ax_mu.semilogy(iters, mus, "r-", marker="s", markersize=2, lw=1.5)
        ax_mu.set_ylabel("Barrier param (mu)")
        ax_mu.grid(True, alpha=0.3)

        ax_alpha.plot(iters, alpha_pr, "g-", marker=".", markersize=2, lw=1)
        ax_alpha.set_ylabel("alpha_pr (step size)")
        ax_alpha.set_xlabel("Iteration")
        ax_alpha.set_ylim(-0.05, 1.05)
        ax_alpha.grid(True, alpha=0.3)

        fig2.tight_layout()
        fig2.savefig("casadi_convergence.png", dpi=300, bbox_inches="tight")
        print("Saved casadi_convergence.png")
    else:
        print("No iteration data to plot convergence.")

    plt.show()

except Exception as e:
    import traceback

    print(f"Could not create plots: {e}")
    traceback.print_exc()
