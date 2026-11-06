"""
Plots of the time to climb solution
-----------------------------------
"""

import json
import matplotlib.pyplot as plt
import niceplots
import numpy as np


def plot_panel(ax, t, y, ylabel):
    """One panel of the solution figure"""
    ax.plot(t, y)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    ax.grid(True)

    return


with open("time_to_climb_solution.json", "r") as fp:
    solution = json.load(fp)

t = np.array(solution["t"])
q = np.array(solution["q"])
alpha = np.array(solution["alpha"])

with plt.style.context(niceplots.get_style()):
    fig = plt.figure(figsize=(12, 10))
    axes = fig.subplots(3, 2)

    # The velocity and the flight path angle
    plot_panel(axes[0, 0], t, 100 * q[:, 0], "Velocity (m/s)")
    plot_panel(axes[0, 1], t, q[:, 1], "Flight path angle (deg)")

    # The altitude and the range
    plot_panel(axes[1, 0], t, q[:, 2], "Altitude (km)")
    plot_panel(axes[1, 1], t, q[:, 3], "Range (km)")

    # The mass and the angle of attack
    plot_panel(axes[2, 0], t, 1000 * q[:, 4], "Mass (kg)")
    plot_panel(axes[2, 1], t, alpha, "Angle of attack (deg)")

    plt.tight_layout()
    plt.savefig("time_to_climb_results.png", dpi=300, bbox_inches="tight")
    plt.show()
