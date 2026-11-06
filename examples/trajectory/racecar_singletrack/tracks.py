"""
Race tracks
-----------
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy import interpolate

TRACKS_DIR = Path(__file__).resolve().parent / "tracks"


# Closed track at the nodes of the solve and as recorded
@dataclass
class Track:
    name: str
    s_total: float

    # Arc length, curvature, centerline, and half widths at the nodes
    s: np.ndarray
    kappa: np.ndarray
    x: np.ndarray
    y: np.ndarray
    w_right: np.ndarray
    w_left: np.ndarray

    # Centerline, arc length, and half widths as recorded
    raw_x: np.ndarray
    raw_y: np.ndarray
    raw_s: np.ndarray
    raw_w_right: np.ndarray
    raw_w_left: np.ndarray


def oval_dymos(num_nodes=301):
    """Oval of the Dymos race car example with the curvature of its spline"""
    # Type, length or sweep, radius, and side of each straight and quarter turn
    segments = [
        [0, 150.0, 0.0, -1],
        [1, np.pi / 2, 50.0, 0],
        [0, 100.0, 0.0, -1],
        [1, np.pi / 2, 90.0, 0],
        [0, 300.0, 0.0, -1],
        [1, np.pi / 2, 50.0, 0],
        [0, 100.0, 0.0, -1],
        [1, np.pi / 2, 90.0, 0],
        [0, 155.0, 0.0, -1],
    ]

    pos = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])
    points = [[0.0, 0.0]]

    # Points every 5 m along the straights and every tenth point along the turns
    for seg in segments:
        seg_type = seg[0]
        length = seg[1]
        radius = seg[2]
        side = seg[3]

        if seg_type == 0:
            for j in range(1, int(length) - 1):
                if j % 5 == 0:
                    points.append(list(pos + direction * j))

            pos = pos + direction * length
        else:
            if side == 0:
                normal = np.array([-direction[1], direction[0]])
            else:
                normal = np.array([direction[1], -direction[0]])

            xc = pos[0] + radius * normal[0]
            yc = pos[1] + radius * normal[1]
            theta_line = np.arctan2(direction[1], direction[0])
            theta_0 = np.arctan2(pos[1] - yc, pos[0] - xc)

            if side == 0:
                theta_end = theta_0 + length
                direction = np.array(
                    [np.cos(theta_line + length), np.sin(theta_line + length)]
                )
            else:
                theta_end = theta_0 - length
                direction = np.array(
                    [np.cos(theta_line - length), np.sin(theta_line - length)]
                )

            theta_vec = np.linspace(theta_0, theta_end, 100)
            xarc = xc + radius * np.cos(theta_vec)
            yarc = yc + radius * np.sin(theta_vec)

            for j in range(len(xarc)):
                if j % 10 == 0:
                    points.append([xarc[j], yarc[j]])

            pos = np.array([xarc[-1], yarc[-1]])

    pts = np.array(points)

    # Quintic spline through the points on a fine parameter grid
    tck = interpolate.splprep(pts.T, s=0.0, k=5)[0]
    interval = 0.0001
    u_fine = np.arange(0, 1.0, interval)
    n_fine = len(u_fine)

    xy_fine = interpolate.splev(u_fine, tck, der=0)
    dxy_fine = interpolate.splev(u_fine, tck, der=1)
    ddxy_fine = interpolate.splev(u_fine, tck, der=2)

    x_fine = np.asarray(xy_fine[0])
    y_fine = np.asarray(xy_fine[1])
    dx = np.asarray(dxy_fine[0])
    dy = np.asarray(dxy_fine[1])
    ddx = np.asarray(ddxy_fine[0])
    ddy = np.asarray(ddxy_fine[1])

    # Curvature of the spline
    speed = np.sqrt(dx**2 + dy**2)
    curv = (dx * ddy - dy * ddx) / speed**3

    # Arc length of the spline by the trapezoidal rule over its parameter
    du = u_fine[1] - u_fine[0]
    s_from_u = np.zeros(n_fine)
    s_from_u[1:] = np.cumsum(0.5 * (speed[:-1] + speed[1:]) * du)
    s_total_spline = s_from_u[-1]

    # Nominal length of the straights and the turns
    lengths = []

    for seg in segments:
        if seg[0] == 0:
            lengths.append(seg[1])
        else:
            lengths.append(seg[1] * seg[2])

    s_total_nominal = sum(lengths)

    # Fine point at the share of the nominal length of each node
    s_nodes = np.linspace(0, s_total_nominal, num_nodes)
    indices = np.floor(s_nodes / s_total_nominal * n_fine).astype(int)
    indices = np.minimum(indices, n_fine - 1)
    kappa_nodes = curv[indices]
    x_nodes = x_fine[indices]
    y_nodes = y_fine[indices]

    # Arc length of the fine points scaled to the nominal length
    s_fine_al = s_from_u * (s_total_nominal / s_total_spline)

    # Uniform half width of 4 m on both sides
    w_fine = 4.0 * np.ones(n_fine)

    return Track(
        name="oval_dymos",
        s_total=s_total_nominal,
        s=s_nodes,
        kappa=kappa_nodes,
        x=x_nodes,
        y=y_nodes,
        w_right=4.0 * np.ones(num_nodes),
        w_left=4.0 * np.ones(num_nodes),
        raw_x=x_fine,
        raw_y=y_fine,
        raw_s=s_fine_al,
        raw_w_right=w_fine,
        raw_w_left=w_fine,
    )


def list_tracks():
    """Names of the recorded tracks in the tracks folder"""
    names = []

    for path in TRACKS_DIR.glob("*.csv"):
        names.append(path.stem)

    return sorted(names)


def load_csv(csv_path, num_nodes, smooth_sigma=3.0):
    """Recorded track with a smoothed curvature at evenly spaced nodes"""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x_raw = data[:, 0]
    y_raw = data[:, 1]
    w_right_raw = data[:, 2]
    w_left_raw = data[:, 3]

    # Close the loop with the first points
    n_wrap = min(5, len(x_raw))
    x_closed = np.concatenate([x_raw, x_raw[:n_wrap]])
    y_closed = np.concatenate([y_raw, y_raw[:n_wrap]])

    # Arc length along the closed loop
    dx = np.diff(x_closed)
    dy = np.diff(y_closed)
    ds_raw = np.sqrt(dx**2 + dy**2)
    s_closed = np.concatenate([[0], np.cumsum(ds_raw)])

    s_total = s_closed[len(x_raw)]
    s_raw = s_closed[: len(x_raw)]

    # Curvature from the first and second derivatives along the arc length
    dxds = np.gradient(x_closed, s_closed)
    dyds = np.gradient(y_closed, s_closed)
    d2xds2 = np.gradient(dxds, s_closed)
    d2yds2 = np.gradient(dyds, s_closed)
    denom = (dxds**2 + dyds**2) ** 1.5
    kappa_closed = (dxds * d2yds2 - dyds * d2xds2) / denom
    kappa_raw = kappa_closed[: len(x_raw)]

    # Gaussian smoothing of the curvature that wraps around the loop
    if smooth_sigma > 0:
        ds_avg = s_total / len(s_raw)
        sigma_pts = max(1, int(smooth_sigma / ds_avg))
        kernel_size = 6 * sigma_pts + 1
        kernel_x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (kernel_x / sigma_pts) ** 2)
        kernel /= kernel.sum()
        pad = kernel_size
        kappa_padded = np.concatenate([kappa_raw[-pad:], kappa_raw, kappa_raw[:pad]])
        kappa_smooth = np.convolve(kappa_padded, kernel, mode="same")[pad:-pad]
    else:
        kappa_smooth = kappa_raw

    # Resample at evenly spaced nodes
    s_nodes = np.linspace(0, s_total, num_nodes)
    kappa_nodes = np.interp(s_nodes, s_raw, kappa_smooth)
    x_nodes = np.interp(s_nodes, s_raw, x_raw)
    y_nodes = np.interp(s_nodes, s_raw, y_raw)
    w_right_nodes = np.interp(s_nodes, s_raw, w_right_raw)
    w_left_nodes = np.interp(s_nodes, s_raw, w_left_raw)

    return Track(
        name=csv_path.stem,
        s_total=s_total,
        s=s_nodes,
        kappa=kappa_nodes,
        x=x_nodes,
        y=y_nodes,
        w_right=w_right_nodes,
        w_left=w_left_nodes,
        raw_x=x_raw,
        raw_y=y_raw,
        raw_s=s_raw,
        raw_w_right=w_right_raw,
        raw_w_left=w_left_raw,
    )


def load_track(name, num_nodes=301, smooth_sigma=3.0):
    """Track by name at evenly spaced nodes"""
    if name == "oval_dymos":
        return oval_dymos(num_nodes)

    csv_path = TRACKS_DIR / f"{name}.csv"

    if not csv_path.exists():
        available = list_tracks()
        raise FileNotFoundError(
            f"Track '{name}' not found. Available tracks: {available}"
        )

    return load_csv(csv_path, num_nodes, smooth_sigma)
