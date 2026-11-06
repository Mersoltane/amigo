"""
Plots of the brachistochrone solution
-------------------------------------
"""

import json
import matplotlib.pyplot as plt
import niceplots
import numpy as np


def plot_panel(ax, x, y, xlabel, ylabel):
    """One panel of the solution figure"""
    ax.plot(x, y)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    return


with open("brachistochrone_solution.json", "r") as fp:
    solution = json.load(fp)

t = np.array(solution["t"])
q = np.array(solution["q"])
theta = np.array(solution["theta"])

with plt.style.context(niceplots.get_style()):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # The path and the positions over time
    plot_panel(axes[0, 0], q[:, 0], q[:, 1], "x Position (m)", "y Position (m)")
    plot_panel(axes[0, 1], t, q[:, 1], "Time (s)", "Position (m)")
    axes[0, 1].plot(t, q[:, 0])

    # The speed and the angle over time
    plot_panel(axes[1, 0], t, q[:, 2], "Time (s)", "Velocity (m/s)")
    plot_panel(axes[1, 1], t, theta, "Time (s)", "Theta (rad)")

    plt.tight_layout()
    plt.savefig("brachistochrone.png", dpi=300, bbox_inches="tight")
    plt.show()
