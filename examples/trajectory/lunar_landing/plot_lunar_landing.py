"""Figure 1 of Liu, Hager, and Rao redrawn from the results of lunar_landing.py."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COMPUTED = "#2a78d6"
EXACT = "#eb6834"

C = 1.5
U_MAX = 3.0


def exact_solution(t, ts):
    """Altitude, velocity, and thrust of the coast and burn arcs."""
    t = np.asarray(t, dtype=float)
    coast = t <= ts
    y1s = 10.0 - 2.0 * ts - 0.5 * C * ts**2
    y2s = -2.0 - C * ts
    d = t - ts
    y1 = np.where(coast, 10.0 - 2.0 * t - 0.5 * C * t**2, y1s + y2s * d + 0.5 * (U_MAX - C) * d**2)
    y2 = np.where(coast, -2.0 - C * t, y2s + (U_MAX - C) * d)
    u = np.where(coast, 0.0, U_MAX)
    return y1, y2, u


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel("t", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default=os.path.join("results", "lunar_landing.json"))
    parser.add_argument("--out", default=None, help="Image path, next to the results by default")
    args = parser.parse_args()

    with open(args.results) as fp:
        data = json.load(fp)
    ts = data["exact"]["ts"]
    tf = data["exact"]["tf"]
    meshes = data["meshes"]
    final = meshes[-1]

    t_fine = np.linspace(0.0, tf, 2001)
    y1, y2, u = exact_solution(t_fine, ts)
    t_states = np.array(final["t_states"])
    states = np.array(final["states"])
    t_coll = np.array(final["t_collocation"])
    controls = np.array(final["controls"])[:, 0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE)
    marker = dict(marker="o", markersize=4, linestyle="none", color=COMPUTED, markeredgewidth=0.0,
                  label="hp solution", zorder=3)
    line = dict(color=EXACT, linewidth=1.5, label="exact", zorder=2)

    panels = [
        (axes[0, 0], "(a) y1(t)", "y1", y1, t_states, states[:, 0]),
        (axes[0, 1], "(b) y2(t)", "y2", y2, t_states, states[:, 1]),
        (axes[1, 0], "(c) u(t)", "u", u, t_coll, controls),
    ]
    for ax, title, label, exact, t_points, values in panels:
        ax.plot(t_fine, exact, **line)
        ax.plot(t_points, values, **marker)
        style(ax, title, label)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    # Every mesh is a row of its collocation times
    ax = axes[1, 1]
    for i, mesh in enumerate(meshes):
        t = np.array(mesh["t_collocation"])
        ax.plot(t, np.full(len(t), i), marker="o", markersize=3, linestyle="none", color=COMPUTED,
                markeredgewidth=0.0, zorder=3)
    ax.axvline(ts, color=EXACT, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_ylim(-0.5, len(meshes) - 0.2)
    ax.text(ts, len(meshes) - 0.35, " exact switch time", color=INK, fontsize=8, va="center")
    ax.set_yticks(range(len(meshes)))
    style(ax, "(d) collocation point history", "mesh refinement iteration")

    fig.suptitle(
        f"Soft lunar landing with hp mesh refinement, tolerance 1e-6, {len(meshes)} meshes",
        color=INK, fontsize=11,
    )
    fig.tight_layout()
    out = args.out or os.path.splitext(args.results)[0] + ".png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
