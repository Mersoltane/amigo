"""
Plots of the cannon solution
----------------------------
"""

import json
import matplotlib.pyplot as plt
import niceplots
import numpy as np


def label_panel(ax, xlabel, ylabel, title):
    """Labels and grid of one panel of the solution figure"""
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)

    return


with open("cannon_solution.json", "r") as fp:
    solution = json.load(fp)

t = np.array(solution["t"])
q = np.array(solution["q"])

x_pos = q[:, 0]
y_pos = q[:, 1]
vx = q[:, 2]
vy = q[:, 3]

with plt.style.context(niceplots.get_style()):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # The flight path from the start to the target
    axes[0, 0].plot(x_pos, y_pos, "b-", linewidth=2, label="Trajectory")
    axes[0, 0].plot(x_pos[0], y_pos[0], "go", markersize=8, label="Start")
    axes[0, 0].plot(x_pos[-1], y_pos[-1], "ro", markersize=8, label="End")
    axes[0, 0].plot(6.0, 0.0, "r*", markersize=12, label="Target")
    label_panel(axes[0, 0], "X Position", "Y Position", "Trajectory")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].axis("equal")

    # The positions over time
    axes[0, 1].plot(t, x_pos, "b-", linewidth=2)
    label_panel(axes[0, 1], "Time (s)", "X Position", "X Position vs Time")

    axes[1, 0].plot(t, y_pos, "r-", linewidth=2)
    label_panel(axes[1, 0], "Time (s)", "Y Position", "Y Position vs Time")

    # The velocities over time
    axes[1, 1].plot(t, vx, "b-", linewidth=2, label="V_x")
    axes[1, 1].plot(t, vy, "r-", linewidth=2, label="V_y")
    label_panel(axes[1, 1], "Time (s)", "Velocity", "Velocities vs Time")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig("cannon_trajectory.png", dpi=300, bbox_inches="tight")
    fig.savefig("cannon_trajectory.svg", bbox_inches="tight")
    plt.show()
