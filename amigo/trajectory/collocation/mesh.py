"""
Radau mesh
----------

- Legendre-Gauss-Radau points, weights, and differentiation and integration matrices
- The hp mesh of intervals the points are collocated on
"""

import numpy as np
from numpy.polynomial import legendre


def radau_points(n, flipped=False):
    """The n Legendre-Gauss-Radau points on [-1, 1] in increasing order"""
    if n < 1:
        raise ValueError(f"a Radau set needs at least one point, not {n}")

    # The standard points are the roots of P_{n-1} + P_n and include -1
    c = np.zeros(n + 1)
    c[n - 1] = 1.0
    c[n] = 1.0

    x = np.sort(legendre.legroots(c).real)
    x[0] = -1.0

    # The flipped points are their mirror image and include +1
    if flipped:
        return -x[::-1]

    return x


def radau_weights(n, flipped=False):
    """Quadrature weights of the n Radau points, exact up to degree 2n - 2"""
    x = radau_points(n)

    # Legendre polynomial P_{n-1} at the points
    c = np.zeros(n)
    c[n - 1] = 1.0
    p = legendre.legval(x, c)

    w = (1.0 - x) / (n * n * p * p)
    w[0] = 2.0 / (n * n)

    if flipped:
        return w[::-1]

    return w


def support_points(n, flipped=False):
    """The n Radau points with the end of [-1, 1] that is not collocated"""
    x = radau_points(n, flipped)

    # The state is also stored at the end that is not collocated
    if flipped:
        return np.insert(x, 0, -1.0)

    return np.append(x, 1.0)


def _barycentric_weights(nodes):
    """Weights of the barycentric Lagrange formula on distinct nodes"""
    diff = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(diff, 1.0)

    return 1.0 / diff.prod(axis=1)


def lagrange_matrix(nodes, x):
    """Values of the Lagrange basis on the nodes at the points x, one row per point"""
    nodes = np.asarray(nodes, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))

    lam = _barycentric_weights(nodes)

    # Distance from every point to every node
    diff = x[:, None] - nodes[None, :]

    # A point on a node would divide by zero
    exact = diff == 0.0
    diff[exact] = 1.0

    # Barycentric formula
    terms = lam / diff
    basis = terms / terms.sum(axis=1, keepdims=True)

    # A point on a node takes that node's value exactly
    on_node = exact.any(axis=1)
    basis[on_node] = exact[on_node]

    return basis


def differentiation_matrix(n, flipped=False):
    """Derivatives of the state basis at the n collocation points, an n by n + 1 matrix"""
    z = support_points(n, flipped)
    lam = _barycentric_weights(z)

    # Off diagonal entries from the barycentric weights
    diff = z[:, None] - z[None, :]
    np.fill_diagonal(diff, 1.0)

    full = (lam[None, :] / lam[:, None]) / diff

    # Each diagonal entry makes its row sum to zero
    np.fill_diagonal(full, 0.0)
    np.fill_diagonal(full, -full.sum(axis=1))

    # Keep the rows of the collocation points
    if flipped:
        return full[1:]

    return full[:-1]


def integration_matrix(n, flipped=False):
    """Radau integration matrix, an n by n matrix"""
    coll = radau_points(n, flipped)
    ends = support_points(n, flipped)[1:]

    # Gauss points and weights on [-1, 1]
    gauss = legendre.leggauss(n + 1)
    g = gauss[0]
    gw = gauss[1]

    # Row i integrates the rate polynomial from -1 to support point i + 1
    rows = []

    for z in ends:
        half = 0.5 * (z + 1.0)
        s = -1.0 + half * (g + 1.0)

        L = lagrange_matrix(coll, s)
        rows.append(half * gw @ L)

    return np.array(rows)


class RadauMesh:
    """Mesh intervals on [-1, 1] with the number of Radau collocation points in each"""

    def __init__(self, points, degrees, flipped=False):
        self.points = np.array(points, dtype=float)
        self.degrees = np.array(degrees, dtype=int)
        self.flipped = bool(flipped)

        # Check the mesh
        if self.points.ndim != 1 or len(self.points) != len(self.degrees) + 1:
            raise ValueError("a mesh needs one more point than it has intervals")

        if self.points[0] != -1.0 or self.points[-1] != 1.0:
            raise ValueError("the mesh points must run from -1 to 1")

        if np.any(np.diff(self.points) <= 0.0):
            raise ValueError("the mesh points must increase")

        if np.any(self.degrees < 1):
            raise ValueError("every interval needs at least one collocation point")

        # Interval k holds the states offsets[k] to offsets[k + 1] and shares its end states with its neighbors
        self.offsets = np.concatenate([[0], np.cumsum(self.degrees)])

    @classmethod
    def uniform(cls, num_intervals, degree, flipped=False):
        """Equal intervals with the same number of collocation points"""
        points = np.linspace(-1.0, 1.0, num_intervals + 1)
        degrees = np.full(num_intervals, degree)

        return cls(points, degrees, flipped)

    def __repr__(self):
        points = self.points.tolist()
        degrees = self.degrees.tolist()

        return f"RadauMesh(points={points}, degrees={degrees}, flipped={self.flipped})"

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
        """Radau weight of every collocation point times the half width of its interval"""
        half_widths = self.half_widths
        weights = []

        for k in range(self.num_intervals):
            n = self.degrees[k]
            weights.append(half_widths[k] * radau_weights(n, self.flipped))

        return np.concatenate(weights)

    def state_indices(self, k):
        """Positions of the state values of interval k in the state array"""
        return np.arange(self.offsets[k], self.offsets[k + 1] + 1)

    def collocation_indices(self, k):
        """Positions of the control and rate values of interval k in their arrays"""
        return np.arange(self.offsets[k], self.offsets[k + 1])

    def collocated_states(self):
        """Position in the state array of the state at every collocation point"""
        positions = np.arange(self.num_collocation)

        # The flipped points skip the first state
        if self.flipped:
            return positions + 1

        return positions

    def to_mesh(self, k, s):
        """Mesh coordinate of the local coordinate s of interval k"""
        s = np.asarray(s, dtype=float)

        return self.midpoints[k] + self.half_widths[k] * s

    def state_points(self):
        """Mesh coordinates of all state values"""
        tau = np.empty(self.num_states)

        for k in range(self.num_intervals):
            n = self.degrees[k]
            s = support_points(n, self.flipped)

            tau[self.state_indices(k)] = self.to_mesh(k, s)

        return tau

    def collocation_points(self):
        """Mesh coordinates of all collocation points"""
        tau = []

        for k in range(self.num_intervals):
            n = self.degrees[k]
            s = radau_points(n, self.flipped)

            tau.append(self.to_mesh(k, s))

        return np.concatenate(tau)

    def interval_of(self, tau):
        """Interval of every mesh coordinate, where a mesh point goes to the interval that collocates it"""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))

        # The standard intervals collocate their left end and the flipped ones their right end
        side = "right"
        if self.flipped:
            side = "left"

        k = np.searchsorted(self.points, tau, side=side) - 1

        return np.clip(k, 0, self.num_intervals - 1)

    def interpolate_states(self, states, tau):
        """Values of the piecewise state polynomial at the mesh coordinates tau"""
        states = np.asarray(states, dtype=float)
        tau = np.atleast_1d(np.asarray(tau, dtype=float))

        out = np.empty((len(tau),) + states.shape[1:])
        intervals = self.interval_of(tau)

        # Evaluate the polynomial of each interval at the points it holds
        for k in np.unique(intervals):
            rows = intervals == k
            s = (tau[rows] - self.midpoints[k]) / self.half_widths[k]

            nodes = support_points(self.degrees[k], self.flipped)
            basis = lagrange_matrix(nodes, s)

            out[rows] = np.tensordot(basis, states[self.state_indices(k)], axes=1)

        return out

    def interpolate_controls(self, controls, tau):
        """Values of the piecewise polynomial through the collocation values at the mesh coordinates tau"""
        controls = np.asarray(controls, dtype=float)
        tau = np.atleast_1d(np.asarray(tau, dtype=float))

        out = np.empty((len(tau),) + controls.shape[1:])
        intervals = self.interval_of(tau)

        # Evaluate the polynomial of each interval at the points it holds
        for k in np.unique(intervals):
            rows = intervals == k
            s = (tau[rows] - self.midpoints[k]) / self.half_widths[k]

            nodes = radau_points(self.degrees[k], self.flipped)
            basis = lagrange_matrix(nodes, s)

            out[rows] = np.tensordot(basis, controls[self.collocation_indices(k)], axes=1)

        return out


def mesh_time(tau, t0, tf):
    """Time of the mesh coordinate tau on a phase from t0 to tf"""
    tau = np.asarray(tau, dtype=float)

    return t0 + 0.5 * (tau + 1.0) * (tf - t0)


def carry_densities(old_mesh, new_mesh, values, old_weights, new_weights):
    """Values at the collocation points of one mesh moved to another as densities over the weights"""
    values = np.asarray(values, dtype=float)
    old_weights = np.asarray(old_weights, dtype=float)
    new_weights = np.asarray(new_weights, dtype=float)

    # The weights broadcast over the trailing axes of the values
    shape = (-1,) + (1,) * (values.ndim - 1)

    # A multiplier is a weight times a smooth density, so the density is what moves
    density = values / old_weights.reshape(shape)
    carried = old_mesh.interpolate_controls(density, new_mesh.collocation_points())

    # Multiply by the weights of the new points
    return carried * new_weights.reshape(shape)
