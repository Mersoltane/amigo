"""
Plots of the free flying robot solution
---------------------------------------
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Colors of the states and the controls
blue_color = "#0072BD"
purple_color = "#8242A2"


def plot_panel(ax, x, y, xlabel, ylabel, title, color):
    """One panel of the solution figure"""
    ax.plot(x, y, color=color, linewidth=2.0)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=10)

    return


with open("free_flying_robot_solution.json", "r") as fp:
    solution = json.load(fp)

t = np.array(solution["t"])
q = np.array(solution["q"])
u = np.array(solution["u"])

# Net thrust of each thruster
T1 = u[:, 0] - u[:, 1]
T2 = u[:, 2] - u[:, 3]

fig = plt.figure(figsize=(15, 9))
axes = fig.subplots(3, 3)

# The positions and the angle
plot_panel(axes[0, 0], t, q[:, 0], "Time (s)", "x (m)", "Horizontal Position", blue_color)
plot_panel(axes[0, 1], t, q[:, 1], "Time (s)", "y (m)", "Vertical Position", blue_color)
plot_panel(axes[0, 2], t, q[:, 2], "Time (s)", "θ (rad)", "Orientation", blue_color)

# The velocities
plot_panel(axes[1, 0], t, q[:, 3], "Time (s)", "vx (m/s)", "Horizontal Velocity", blue_color)
plot_panel(axes[1, 1], t, q[:, 4], "Time (s)", "vy (m/s)", "Vertical Velocity", blue_color)
plot_panel(axes[1, 2], t, q[:, 5], "Time (s)", "ω (rad/s)", "Angular Velocity", blue_color)

# The path and the thrusts
plot_panel(axes[2, 0], q[:, 0], q[:, 1], "x (m)", "y (m)", "Trajectory", blue_color)
plot_panel(axes[2, 1], t, T1, "Time (s)", "T1 (N)", "Thruster 1", purple_color)
plot_panel(axes[2, 2], t, T2, "Time (s)", "T2 (N)", "Thruster 2", purple_color)

axes[2, 0].axis("equal")

# Use Arial when it is installed
fontname = "DejaVu Sans"
for font in font_manager.fontManager.ttflist:
    if font.name == "Arial":
        fontname = "Arial"

for ax in axes.flat:
    ax.xaxis.label.set_fontname(fontname)
    ax.yaxis.label.set_fontname(fontname)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(fontname)

plt.tight_layout()
plt.savefig("free_flying_robot_solution.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
