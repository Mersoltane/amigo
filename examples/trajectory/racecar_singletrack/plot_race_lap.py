"""
Plots of the race car lap
-------------------------
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from tracks import load_track

with open("racecar_trajectory.json", "r") as fp:
    lap = json.load(fp)

with open("racecar_opt_data.json", "r") as fp:
    opt_data = json.load(fp)

s = np.array(lap["s"])
t = np.array(lap["t"])
n = np.array(lap["n"])
V = np.array(lap["V"])
delta = np.array(lap["delta"])
thrust = np.array(lap["thrust"])
c = np.array(lap["c"])

# The track at the nodes of the solve
track = load_track(lap["track"], len(s))

# Normal of the recorded centerline that points to the left
dx = np.gradient(track.raw_x, track.raw_s)
dy = np.gradient(track.raw_y, track.raw_s)
mag = np.sqrt(dx**2 + dy**2)
nx = -dy / mag
ny = dx / mag

# Edges of the track on both sides of the centerline
x_left = track.raw_x + track.raw_w_left * nx
y_left = track.raw_y + track.raw_w_left * ny
x_right = track.raw_x - track.raw_w_right * nx
y_right = track.raw_y - track.raw_w_right * ny

# Solution on a fine grid along the track
s_fine = np.linspace(0, track.s_total, 500)
n_fine = np.interp(s_fine, s, n)
V_fine = np.interp(s_fine, s, V)
thrust_fine = np.interp(s_fine, s, thrust)
delta_fine = np.interp(s_fine, s, delta)

# Racing line offset from the centerline along its normal
xc = np.interp(s_fine, track.raw_s, track.raw_x)
yc = np.interp(s_fine, track.raw_s, track.raw_y)
dxc = np.gradient(xc, s_fine)
dyc = np.gradient(yc, s_fine)
mag_c = np.sqrt(dxc**2 + dyc**2)
x_race = xc + n_fine * (-dyc / mag_c)
y_race = yc + n_fine * (dxc / mag_c)


def plot_track_map(values, cmap, label, filename):
    """Racing line on the track colored by one quantity"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.subplots(1, 1)

    # Track surface and its edges
    ax.fill(
        np.append(x_left, x_right[::-1]),
        np.append(y_left, y_right[::-1]),
        color="0.92",
        zorder=0,
    )
    ax.plot(x_left, y_left, "k-", lw=1, alpha=0.7)
    ax.plot(x_right, y_right, "k-", lw=1, alpha=0.7)

    # Racing line in segments colored by the values
    pts = np.column_stack([x_race, y_race]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(
        segs,
        cmap=cmap,
        norm=Normalize(values.min(), values.max()),
        lw=3,
        zorder=5,
    )
    lc.set_array(values)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label=label, shrink=0.8)

    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        title=f"{track.name} ({t[-1]:.2f} s)",
        aspect="equal",
    )
    ax.autoscale_view()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved {filename}")

    return


plot_track_map(V_fine, "viridis", "V (m/s)", "racecar_track_velocity.png")
plot_track_map(thrust_fine, "RdYlGn", "thrust", "racecar_track_thrust.png")
plot_track_map(delta_fine, "coolwarm", "delta (rad)", "racecar_track_steering.png")

# Telemetry along the track
fig = plt.figure(figsize=(14, 14))
axes = fig.subplots(5, 1, sharex=True)

axes[0].plot(s, V, "b-", lw=1.5)
axes[0].set_ylabel("V (m/s)")
axes[0].set_title(f"{track.name} - Telemetry")

axes[1].plot(s, n, "r-", lw=1.5)
axes[1].plot(s, track.w_left, "k--", alpha=0.3)
axes[1].plot(s, -track.w_right, "k--", alpha=0.3)
axes[1].set_ylabel("n (m)")

axes[2].plot(s, thrust, "g-", lw=1.5)
axes[2].axhline(0, color="k", alpha=0.2)
axes[2].set_ylabel("thrust")

axes[3].plot(s, delta, "m-", lw=1.5)
axes[3].set_ylabel("delta (rad)")

axes[4].plot(s, c[:, 0], label="FL", alpha=0.7)
axes[4].plot(s, c[:, 1], label="FR", alpha=0.7)
axes[4].plot(s, c[:, 2], label="RL", alpha=0.7)
axes[4].plot(s, c[:, 3], label="RR", alpha=0.7)
axes[4].plot(s, c[:, 4], "k-", label="power", lw=1.5, alpha=0.8)
axes[4].axhline(1, color="r", ls="--", lw=1.5, label="limit")
axes[4].set_ylabel("constraint")
axes[4].set_xlabel("s (m)")
axes[4].legend(loc="upper right", fontsize=8, ncol=3)

for ax in axes:
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("racecar_telemetry.png", dpi=150)
print("Saved racecar_telemetry.png")

# Convergence history of the solve
residuals = []

for iteration in opt_data["iterations"]:
    residuals.append(iteration["residual"])

fig = plt.figure()
ax = fig.subplots(1, 1)
ax.semilogy(residuals, marker="o", clip_on=False, lw=2.0)
ax.set(xlabel="Iteration", ylabel="KKT residual norm")
ax.grid(True)
fig.savefig("racecar_convergence.png", dpi=300, bbox_inches="tight")

plt.show()
