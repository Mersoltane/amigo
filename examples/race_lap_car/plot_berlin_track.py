"""
Berlin 2018 track visualization — broadcast style.
Grey filled track, dark border, white s-markers, no legend.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.race_lap_car.tracks import berlin_2018

# ── Load track ───────────────────────────────────────────────────────────────
track = berlin_2018(num_nodes=2000, smooth_sigma=8.0)

x = track.x
y = track.y
s = track.s

# Unit left-normal along centerline (smoothed to avoid kinks at corners)
from scipy.ndimage import gaussian_filter1d

dx = gaussian_filter1d(np.gradient(x, s), sigma=10, mode="wrap")
dy = gaussian_filter1d(np.gradient(y, s), sigma=10, mode="wrap")
spd = np.hypot(dx, dy)
nx = -dy / spd
ny = dx / spd

# Boundary polylines
wr = track.w_right
wl = track.w_left
x_right = x - wr * nx
y_right = y - wr * ny
x_left = x + wl * nx
y_left = y + wl * ny

# Closed loops
xr = np.append(x_right, x_right[0])
yr = np.append(y_right, y_right[0])
xl = np.append(x_left, x_left[0])
yl = np.append(y_left, y_left[0])

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 13))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Filled track surface
left_pts = np.c_[x_left, y_left]
right_pts = np.c_[x_right, y_right][::-1]
poly = np.vstack([left_pts, right_pts, left_pts[:1]])
ax.fill(poly[:, 0], poly[:, 1], color="#e0e0e0", zorder=1, linewidth=0)

# Centerline
ax.plot(
    np.append(x, x[0]),
    np.append(y, y[0]),
    color="#1f77b4",
    lw=1.4,
    zorder=2,
    linestyle="--",
    dashes=(8, 5),
)

# Outer / inner borders (black outline)
border_kw = dict(color="black", lw=2.0, zorder=3, solid_capstyle="round")
ax.plot(xl, yl, **border_kw)
ax.plot(xr, yr, **border_kw)

# White s-markers: full-width lines across the track every 500 m (no labels)
for s_mark in np.arange(500.0, track.s_total, 500.0):
    idx = np.argmin(np.abs(s - s_mark))
    p0 = np.array([x_right[idx], y_right[idx]])
    p1 = np.array([x_left[idx], y_left[idx]])
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color="white",
        lw=4.5,
        zorder=4,
        solid_capstyle="butt",
    )

# Checkered finish line at s=0
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

idx0 = 0
p_r = np.array([x_right[idx0], y_right[idx0]])
p_l = np.array([x_left[idx0], y_left[idx0]])
width = np.linalg.norm(p_l - p_r)  # track width at start
along = np.array([dx[idx0], dy[idx0]]) / spd[idx0]  # tangent direction
perp = np.array([nx[idx0], ny[idx0]])  # left normal

n_cols = 10  # squares across the track
n_rows = 6  # squares along the track
sq_w = width / n_cols  # square size across
sq_h = sq_w * 1.5  # taller squares for more visible span

# origin corner: right boundary, slightly behind (−along)
origin = p_r - along * (sq_h * n_rows / 2)

for row in range(n_rows):
    for col in range(n_cols):
        if (row + col) % 2 == 0:
            color = "black"
        else:
            color = "white"
        corner = origin + perp * (col * sq_w) + along * (row * sq_h)
        # Build a rotated rectangle via a polygon
        pts = np.array(
            [
                corner,
                corner + perp * sq_w,
                corner + perp * sq_w + along * sq_h,
                corner + along * sq_h,
            ]
        )
        ax.fill(pts[:, 0], pts[:, 1], color=color, zorder=5, linewidth=0)

# thin border around the whole chequered band
band_corners = np.array(
    [
        origin,
        origin + perp * width,
        origin + perp * width + along * sq_h * n_rows,
        origin + along * sq_h * n_rows,
        origin,
    ]
)
ax.plot(band_corners[:, 0], band_corners[:, 1], color="#333", lw=0.8, zorder=6)

ax.set_aspect("equal")
ax.axis("off")
ax.margins(0.02)  # minimal whitespace around the data
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

out = Path(__file__).parent / "berlin_2018_track.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print(f"Saved: {out}")
plt.show()
