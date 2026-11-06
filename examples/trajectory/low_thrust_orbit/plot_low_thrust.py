"""
Plots of the low thrust transfer
--------------------------------
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import niceplots

here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(here, "low_thrust_trajectory.json")

with open(data_path, "r") as fp:
    data = json.load(fp)

t = np.asarray(data["t"])  # seconds
q = np.asarray(data["q"])  # (N, 7): p [ft], f, g, h, k, L [rad], w [lb]
u = np.asarray(data["u"])  # (N, 3): u_r, u_theta, u_h


# Journal-quality settings (AIAA two-column convention; serif + soft black + inward ticks)
SOFT_BLACK = "#222222"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.labelweight": "semibold",
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "text.color": SOFT_BLACK,
        "axes.edgecolor": SOFT_BLACK,
        "axes.labelcolor": SOFT_BLACK,
        "xtick.color": SOFT_BLACK,
        "ytick.color": SOFT_BLACK,
        "patch.edgecolor": SOFT_BLACK,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.unicode_minus": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
# Muted academic palette (desaturated, print-friendly)
colors = [
    "#2E5A87",  # navy blue
    "#B23A48",  # brick red
    "#3F7A47",  # forest green
    "#C26F2D",  # burnt sienna
    "#7E5BA8",  # dusty purple
    "#5C6975",  # slate gray
]


def save(fig, basename):
    png = os.path.join(here, f"{basename}.png")
    pdf = os.path.join(here, f"{basename}.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote: {png}")
    print(f"Wrote: {pdf}")


# Convert to convenient display units
th = t / 3600.0  # time in hours
p_Mft = q[:, 0] / 1e6  # semi-parameter in millions of feet
f, g = q[:, 1], q[:, 2]
h, k = q[:, 3], q[:, 4]
revs = (q[:, 5] - q[0, 5]) / (2.0 * np.pi)  # true longitude in revolutions
w = q[:, 6]
ecc = np.sqrt(f**2 + g**2)
ur, ut, uh = u[:, 0], u[:, 1], u[:, 2]


# -----------------------------------------------------------------------------
# Summary plot (compact 4-panel): semi-parameter, eccentricity, weight, revolutions
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True)

axes[0, 0].plot(th, p_Mft, color=colors[0])
axes[0, 0].set_ylabel(r"Semi-parameter $p$ ($10^6$ ft)")

axes[0, 1].plot(th, ecc, color=colors[1])
axes[0, 1].set_ylabel("Eccentricity")

axes[1, 0].plot(th, w, color=colors[2])
axes[1, 0].set_ylabel("Weight (lb)")
axes[1, 0].set_xlabel("Time (h)")

axes[1, 1].plot(th, revs, color=colors[3])
axes[1, 1].set_ylabel(r"True longitude $L$ (rev)")
axes[1, 1].set_xlabel("Time (h)")

fig.tight_layout()
save(fig, "low_thrust_summary")
plt.close(fig)


# -----------------------------------------------------------------------------
# Full states plot (7-panel)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(7.0, 8.4), sharex=True)

axes[0, 0].plot(th, p_Mft, color=colors[0])
axes[0, 0].set_ylabel(r"$p$ ($10^6$ ft)")

axes[0, 1].plot(th, f, color=colors[1])
axes[0, 1].set_ylabel(r"$f$")

axes[1, 0].plot(th, g, color=colors[2])
axes[1, 0].set_ylabel(r"$g$")

axes[1, 1].plot(th, h, color=colors[3])
axes[1, 1].set_ylabel(r"$h$")

axes[2, 0].plot(th, k, color=colors[4])
axes[2, 0].set_ylabel(r"$k$")

axes[2, 1].plot(th, revs, color=colors[5])
axes[2, 1].set_ylabel(r"$L$ (rev)")
axes[2, 1].set_xlabel("Time (h)")

axes[3, 0].plot(th, w, color=colors[0])
axes[3, 0].set_ylabel("Weight (lb)")
axes[3, 0].set_xlabel("Time (h)")

axes[3, 1].axis("off")

fig.tight_layout()
save(fig, "low_thrust_states")
plt.close(fig)


# -----------------------------------------------------------------------------
# Controls plot (3-panel)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharex=True)

axes[0].plot(th, ur, color=colors[2])
axes[0].set_ylabel(r"$u_r$")
axes[0].set_xlabel("Time (h)")

axes[1].plot(th, ut, color=colors[3])
axes[1].set_ylabel(r"$u_\theta$")
axes[1].set_xlabel("Time (h)")

axes[2].plot(th, uh, color=colors[4])
axes[2].set_ylabel(r"$u_h$")
axes[2].set_xlabel("Time (h)")

fig.tight_layout()
save(fig, "low_thrust_controls")
plt.close(fig)

print(f"\nDone. Trajectory has {len(t)} time steps, final time {th[-1]:.2f} h.")
print(f"Final weight: {w[-1]:.6f} lb, revolutions: {revs[-1]:.2f}.")
