"""Legendre-Gauss-Radau points and the hp mesh of intervals they are collocated on."""

import numpy as np
from numpy.polynomial import legendre


def radau_points(n, flipped=False):
    """The n Legendre-Gauss-Radau points on [-1, 1] in increasing order.

    The standard points are the roots of P_{n-1} + P_n and include -1. The
    flipped points are their mirror image and include +1.
    """
    if n < 1:
        raise ValueError(f"a Radau set needs at least one point, not {n}")
    c = np.zeros(n + 1)
    c[n - 1] = 1.0
    c[n] = 1.0
    x = np.sort(legendre.legroots(c).real)
    x[0] = -1.0
    return -x[::-1] if flipped else x


def radau_weights(n, flipped=False):
    """Quadrature weights of the n Radau points, exact for polynomials of degree 2n - 2."""
    x = radau_points(n)
    c = np.zeros(n)
    c[n - 1] = 1.0
    p = legendre.legval(x, c)
    w = (1.0 - x) / (n * n * p * p)
    w[0] = 2.0 / (n * n)
    return w[::-1] if flipped else w


def support_points(n, flipped=False):
    """The n Radau points with the uncollocated end of [-1, 1] added, where the state is stored."""
    x = radau_points(n, flipped)
    return np.insert(x, 0, -1.0) if flipped else np.append(x, 1.0)


def _barycentric_weights(nodes):
    """Weights of the barycentric Lagrange formula on distinct nodes."""
    diff = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(diff, 1.0)
    return 1.0 / diff.prod(axis=1)


def lagrange_matrix(nodes, x):
    """Values of the Lagrange basis on the nodes at the points x, one row per point."""
    nodes = np.asarray(nodes, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    lam = _barycentric_weights(nodes)
    diff = x[:, None] - nodes[None, :]
    exact = diff == 0.0
    diff[exact] = 1.0
    terms = lam / diff
    basis = terms / terms.sum(axis=1, keepdims=True)

    # A point on a node takes that node's value exactly
    on_node = exact.any(axis=1)
    basis[on_node] = exact[on_node]
    return basis


def differentiation_matrix(n, flipped=False):
    """Derivatives of the state basis at the n collocation points, an n by n + 1 matrix."""
    z = support_points(n, flipped)
    lam = _barycentric_weights(z)
    diff = z[:, None] - z[None, :]
    np.fill_diagonal(diff, 1.0)
    full = (lam[None, :] / lam[:, None]) / diff
    np.fill_diagonal(full, 0.0)
    np.fill_diagonal(full, -full.sum(axis=1))
    return full[1:] if flipped else full[:-1]


def integration_matrix(n, flipped=False):
    """Radau integration matrix, an n by n matrix.

    Row i integrates the Lagrange polynomial of the rates at the collocation
    points from -1 to support point i + 1, so the state rise from the interval
    start to that point is the row times the rates.
    """
    coll = radau_points(n, flipped)
    ends = support_points(n, flipped)[1:]
    g, gw = legendre.leggauss(n + 1)
    rows = []
    for z in ends:
        s = -1.0 + 0.5 * (z + 1.0) * (g + 1.0)
        rows.append(0.5 * (z + 1.0) * gw @ lagrange_matrix(coll, s))
    return np.array(rows)


class RadauMesh:
    """Mesh intervals on [-1, 1] with the number of Radau collocation points in each.

    The number of collocation points in an interval is also the degree of the
    state polynomial there. The state is stored on the support points of every
    interval, and two neighbors share the support point at their common mesh
    point, so interval k holds states offsets[k] to offsets[k + 1] and controls
    offsets[k] to offsets[k + 1] - 1.
    """

    def __init__(self, points, degrees, flipped=False):
        self.points = np.array(points, dtype=float)
        self.degrees = np.array(degrees, dtype=int)
        self.flipped = bool(flipped)

        if self.points.ndim != 1 or len(self.points) != len(self.degrees) + 1:
            raise ValueError("a mesh needs one more point than it has intervals")
        if self.points[0] != -1.0 or self.points[-1] != 1.0:
            raise ValueError("the mesh points must run from -1 to 1")
        if np.any(np.diff(self.points) <= 0.0):
            raise ValueError("the mesh points must increase")
        if np.any(self.degrees < 1):
            raise ValueError("every interval needs at least one collocation point")

        self.offsets = np.concatenate([[0], np.cumsum(self.degrees)])

    @classmethod
    def uniform(cls, num_intervals, degree, flipped=False):
        """Equal intervals with the same number of collocation points."""
        points = np.linspace(-1.0, 1.0, num_intervals + 1)
        return cls(points, np.full(num_intervals, degree), flipped)

    def __repr__(self):
        return f"RadauMesh(points={self.points.tolist()}, degrees={self.degrees.tolist()}, flipped={self.flipped})"

    @property
    def num_intervals(self):
        return len(self.degrees)

    @property
    def num_collocation(self):
        return int(self.offsets[-1])

    @property
    def num_states(self):
        return self.num_collocation + 1

    @property
    def half_widths(self):
        return 0.5 * np.diff(self.points)

    @property
    def midpoints(self):
        return 0.5 * (self.points[:-1] + self.points[1:])

    def quadrature_weights(self):
        """Radau weight of every collocation point times the half width of its interval."""
        return np.concatenate(
            [h * radau_weights(n, self.flipped) for h, n in zip(self.half_widths, self.degrees)]
        )

    def state_indices(self, k):
        """Positions of the state values of interval k in the state array."""
        return np.arange(self.offsets[k], self.offsets[k + 1] + 1)

    def collocation_indices(self, k):
        """Positions of the control and rate values of interval k in their arrays."""
        return np.arange(self.offsets[k], self.offsets[k + 1])

    def collocated_states(self):
        """Position in the state array of the state at every collocation point."""
        return np.arange(self.num_collocation) + (1 if self.flipped else 0)

    def to_mesh(self, k, s):
        """Mesh coordinate of the local coordinate s of interval k."""
        return self.midpoints[k] + self.half_widths[k] * np.asarray(s, dtype=float)

    def state_points(self):
        """Mesh coordinates of all state values."""
        tau = np.empty(self.num_states)
        for k, n in enumerate(self.degrees):
            tau[self.state_indices(k)] = self.to_mesh(k, support_points(n, self.flipped))
        return tau

    def collocation_points(self):
        """Mesh coordinates of all collocation points."""
        return np.concatenate(
            [self.to_mesh(k, radau_points(n, self.flipped)) for k, n in enumerate(self.degrees)]
        )

    def interval_of(self, tau):
        """Interval of every mesh coordinate, giving a mesh point to the interval that collocates it."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        side = "left" if self.flipped else "right"
        k = np.searchsorted(self.points, tau, side=side) - 1
        return np.clip(k, 0, self.num_intervals - 1)

    def interpolate_states(self, states, tau):
        """Values of the piecewise state polynomial at the mesh coordinates tau."""
        states = np.asarray(states, dtype=float)
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        out = np.empty((len(tau),) + states.shape[1:])
        intervals = self.interval_of(tau)
        for k in np.unique(intervals):
            rows = intervals == k
            s = (tau[rows] - self.midpoints[k]) / self.half_widths[k]
            basis = lagrange_matrix(support_points(self.degrees[k], self.flipped), s)
            out[rows] = np.tensordot(basis, states[self.state_indices(k)], axes=1)
        return out

    def interpolate_controls(self, controls, tau):
        """Values of the piecewise polynomial through the collocation values at the mesh coordinates tau."""
        controls = np.asarray(controls, dtype=float)
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        out = np.empty((len(tau),) + controls.shape[1:])
        intervals = self.interval_of(tau)
        for k in np.unique(intervals):
            rows = intervals == k
            s = (tau[rows] - self.midpoints[k]) / self.half_widths[k]
            basis = lagrange_matrix(radau_points(self.degrees[k], self.flipped), s)
            out[rows] = np.tensordot(basis, controls[self.collocation_indices(k)], axes=1)
        return out


def mesh_time(tau, t0, tf):
    """Time of the mesh coordinate tau on a phase from t0 to tf."""
    return t0 + 0.5 * (np.asarray(tau, dtype=float) + 1.0) * (tf - t0)


def carry_densities(old_mesh, new_mesh, values, old_weights, new_weights):
    """Values at the collocation points of one mesh moved to another as densities over the given weights.

    Each value is divided by the weight of its point, the quotients are carried
    by the polynomial through the collocation points of each interval of the
    old mesh, and the results at the new collocation points are multiplied by
    their weights. Multipliers of collocated constraints behave like a weight
    times a smooth density, so this moves them between meshes for a warm start.
    """
    values = np.asarray(values, dtype=float)
    shape = (-1,) + (1,) * (values.ndim - 1)
    density = values / np.asarray(old_weights, dtype=float).reshape(shape)
    carried = old_mesh.interpolate_controls(density, new_mesh.collocation_points())
    return carried * np.asarray(new_weights, dtype=float).reshape(shape)
