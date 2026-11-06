"""
Plots of the cart pole solution
-------------------------------
"""

import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import niceplots
import numpy as np
from matplotlib import font_manager
from matplotlib.text import Text

# Length of the pole
L = 0.5

# Colors of the states and the control
blue_color = "#0072BD"
purple_color = "#8242A2"

# Mark every fifth knot in the documentation figure
knot_step = 5

# Use Helvetica when it is installed
fontname = "Helvetica"
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
if fontname not in available_fonts:
    fontname = "DejaVu Sans"


def set_axis_font(ax):
    """Set the font of every text in one axis"""
    for text in ax.findobj(match=Text):
        text.set_fontname(fontname)

    return


def plot_panel(ax, t, y, ylabel, color):
    """One panel of the documentation figure"""
    ax.plot(t, y, color=color, linewidth=2.5)
    ax.plot(
        t[::knot_step],
        y[::knot_step],
        "o",
        color="black",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1,
    )

    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=10)

    return


with open("cart_pole_solution.json", "r") as fp:
    solution = json.load(fp)

t = np.array(solution["t"])
q = np.array(solution["q"])
u = np.array(solution["u"])
residual = np.array(solution["residual"])

# Position of the cart and angle of the pole
d = q[:, 0]
theta = q[:, 1]

# The position, the angle, and the force stacked over time
with plt.style.context(niceplots.get_style()):
    data = {}
    data["Cart pos."] = d
    data["Pole angle"] = (180 / np.pi) * theta
    data["Control force"] = u

    fig, axes = niceplots.stacked_plots(
        "Time (s)",
        t,
        [data],
        lines_only=True,
        figsize=(10, 6),
        line_scaler=0.5,
    )

    for ax in axes:
        set_axis_font(ax)

    fig.savefig("cart_stacked.png")

# The residual norm at every iteration
with plt.style.context(niceplots.get_style()):
    fig, ax = plt.subplots(1, 1)

    ax.semilogy(residual, marker="o", clip_on=False, lw=2.0)
    ax.set_ylabel("KKT Residual Norm")
    ax.set_xlabel("Iteration")

    niceplots.adjust_spines(ax)
    set_axis_font(ax)

    fig.savefig("cart_residual_norm.png")

# A time lapse of the cart and the pole
with plt.style.context(niceplots.get_style()):
    fig, ax = plt.subplots(1, figsize=(10, 4.5))
    ax.axis("equal")
    ax.axis("off")

    values = np.linspace(0.1, 0.9, d.shape[0])
    cmap = plt.get_cmap("viridis")

    # Size of the cart
    hx = 0.03
    hy = 0.03

    for i in range(d.shape[0]):
        color = cmap(values[i])

        x1 = d[i]
        y1 = 0.0
        x2 = d[i] + L * np.sin(theta[i])
        y2 = -L * np.cos(theta[i])

        # Draw the pole and the cart at every other knot
        if i % 2 == 0:
            ax.plot([x1, x2], [y1, y2], linewidth=2, color=color)
            ax.fill(
                [x1 - hx, x1 + hx, x1 + hx, x1 - hx, x1 - hx],
                [y1, y1, y1 + hy, y1 + hy, y1],
                alpha=0.5,
                linewidth=2,
                color=color,
            )

        # Mark the tip of the pole at every knot
        ax.plot([x2], [y2], color=color, marker="o")

    # An invisible box that sets the extent of the view
    x = 0
    y = -L
    height = 2 * L
    width = 2 + 0.6 * L

    rect = patches.Rectangle(
        (x, y), width, height, linewidth=2, edgecolor="none", facecolor="none"
    )
    ax.add_patch(rect)

    fig.savefig("cart_pole_history.png")

# The states and the control with every fifth knot marked
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

plot_panel(axes[0], t, d, "Position (m)", blue_color)
plot_panel(axes[1], t, theta, "Angle (rad)", blue_color)
plot_panel(axes[2], t, u, "Force (N)", purple_color)

axes[0].set_title("State", fontsize=18, fontweight="bold", pad=10)
axes[2].set_title("Control", fontsize=18, fontweight="bold", pad=10)
axes[2].set_xlabel("Time (s)", fontsize=18)

for ax in axes:
    set_axis_font(ax)

plt.tight_layout()
fig.savefig("cart_pole_solution.png", dpi=300, bbox_inches="tight", facecolor="white")
