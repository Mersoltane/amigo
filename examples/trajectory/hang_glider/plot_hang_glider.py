"""
Plots of the hang glider solution
---------------------------------
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Colors of the states and the control
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


with open("hang_glider_solution.json", "r") as fp:
    solution = json.load(fp)

t = np.array(solution["t"])
q = np.array(solution["q"])
CL = np.array(solution["CL"])

fig = plt.figure(figsize=(12, 9))
axes = fig.subplots(3, 2)

# The horizontal distance and the altitude
plot_panel(axes[0, 0], t, q[:, 0], "Time (s)", "$x$ (m)", "Horizontal Position", blue_color)
plot_panel(axes[0, 1], t, q[:, 1], "Time (s)", "$y$ (m)", "Altitude", blue_color)

# The velocities
plot_panel(axes[1, 0], t, q[:, 2], "Time (s)", "$v_x$ (m/s)", "Horizontal Velocity", blue_color)
plot_panel(axes[1, 1], t, q[:, 3], "Time (s)", "$v_y$ (m/s)", "Vertical Velocity", blue_color)

# The path and the lift coefficient
plot_panel(axes[2, 0], q[:, 0], q[:, 1], "$x$ (m)", "$y$ (m)", "Trajectory", blue_color)
plot_panel(axes[2, 1], t, CL, "Time (s)", "$C_L$", "Control", purple_color)

# Use Arial for the axis labels and the ticks
fontname = "Arial"

for ax in axes.flat:
    ax.xaxis.label.set_fontname(fontname)
    ax.yaxis.label.set_fontname(fontname)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(fontname)

plt.tight_layout()
plt.savefig("hang_glider_solution.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
