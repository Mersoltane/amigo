"""
hp mesh refinement
------------------

- The hp mesh refinement of Liu, Hager, and Rao, IEEE Conference on Decision and Control, 2014
- The error estimate of Patterson, Hager, and Rao, Optimal Control Applications and Methods, 2015
- A first mesh takes q as N, so one refinement can grow the mesh by orders of magnitude
"""

import math
from dataclasses import dataclass, field

import numpy as np

from ..collocation.mesh import (
    RadauMesh,
    integration_matrix,
    lagrange_matrix,
    mesh_time,
    radau_points,
    support_points,
)
from .base import (
    Assessment,
    MeshIteration,
    MeshRefinement,
    MeshSolution,
    best_iteration,
)


@dataclass(frozen=True)
class RefinementOptions:
    """Settings of the hp mesh refinement"""

    # Smoothness threshold Q above which an interval counts as smooth
    smoothness_threshold: float

    # Relative error tolerance that every interval must meet
    tolerance: float = 1e-6

    # Most subintervals that one h refinement creates
    max_subintervals: int = 10

    # Most collocation points in one interval
    max_points: int = 10

    # Fewest collocation points that a degree reduction leaves
    min_points: int = 1

    # Most meshes solved before the refinement stops
    max_iterations: int = 10

    def __post_init__(self):
        if self.smoothness_threshold < 2.5:
            raise ValueError("Eq. 24 needs a smoothness threshold of at least 5/2")

        if self.tolerance <= 0.0:
            raise ValueError("the tolerance must be positive")

        if self.max_subintervals < 2:
            raise ValueError("an h refinement must create at least two subintervals")

        if not 1 <= self.min_points <= self.max_points:
            raise ValueError(
                "the point limits must satisfy 1 <= min_points <= max_points"
            )

        if self.max_iterations < 1:
            raise ValueError("at least one mesh must be solved")


def integrated_states(solution, k, dynamics):
    """States of interval k integrated to the support points of N_k + 1 Radau points"""
    mesh = solution.mesh
    n = int(mesh.degrees[k])
    m = n + 1

    states = solution.states[mesh.state_indices(k)]
    controls = solution.controls[mesh.collocation_indices(k)]

    # The state and control polynomials at the m Radau points
    s_hat = radau_points(m, mesh.flipped)

    state_basis = lagrange_matrix(support_points(n, mesh.flipped), s_hat)
    control_basis = lagrange_matrix(radau_points(n, mesh.flipped), s_hat)

    y = state_basis @ states
    u = control_basis @ controls

    # The dynamics at those points
    t = mesh_time(mesh.to_mesh(k, s_hat), solution.t0, solution.tf)
    rates = np.asarray(dynamics(y, u, t), dtype=float).reshape(m, -1)

    # Integrate from the interval start to each later point
    scale = 0.5 * (solution.tf - solution.t0) * mesh.half_widths[k]
    integration = integration_matrix(m, mesh.flipped)

    out = np.empty((m + 1, states.shape[1]))
    out[0] = states[0]
    out[1:] = states[0] + scale * integration @ rates

    return out


def relative_errors(solution, dynamics):
    """Largest relative state error on every interval"""
    mesh = solution.mesh
    errors = np.empty(mesh.num_intervals)

    for k in range(mesh.num_intervals):
        n = mesh.degrees[k]

        # The state polynomial on the support points of n + 1 Radau points
        s_hat = support_points(n + 1, mesh.flipped)
        basis = lagrange_matrix(support_points(n, mesh.flipped), s_hat)
        y = basis @ solution.states[mesh.state_indices(k)]

        # The integrated state on the same points
        y_int = integrated_states(solution, k, dynamics)

        # Each component is divided by one plus its largest magnitude
        scale = 1.0 + np.abs(y).max(axis=0)
        errors[k] = (np.abs(y_int - y) / scale).max()

    return errors


def smoothness(previous, current):
    """Number of continuous derivatives q from one interval on two meshes"""
    # The width, the points, and the error on the previous mesh and of an interval inside it now
    h0 = previous[0]
    n0 = previous[1]
    e0 = previous[2]

    h1 = current[0]
    n1 = current[1]
    e1 = current[2]

    # A zero width or error leaves q undetermined
    if min(h0, h1, e0, e1) <= 0.0:
        return None

    # Logarithms of the two error equalities eliminate c and leave q
    denominator = math.log(h0 / h1) - math.log(n0 / n1)

    if denominator == 0.0:
        return None

    return (math.log(e0 / e1) - 2.5 * math.log(n0 / n1)) / denominator


def smoothness_estimates(mesh, errors, parents=None, previous=None):
    """Smoothness q of every interval, taken as its collocation points where q is not determined"""
    q = mesh.degrees.astype(float)

    if previous is None:
        return q

    # The previous mesh with its errors
    old_mesh = previous[0]
    old_errors = previous[1]

    # The parent is the interval of the previous mesh that holds each interval, or -1 after a merge
    for k in range(len(parents)):
        parent = parents[k]

        if parent < 0:
            continue

        old = (
            2.0 * old_mesh.half_widths[parent],
            old_mesh.degrees[parent],
            old_errors[parent],
        )
        new = (2.0 * mesh.half_widths[k], mesh.degrees[k], errors[k])

        estimate = smoothness(old, new)

        if estimate is not None:
            q[k] = estimate

    return q


def h_refinement(error, q, options):
    """Number of equal subintervals that replace an interval with q <= Q"""
    if q <= 0.0:
        return options.max_subintervals

    # Subintervals that bring the error down to the tolerance at the rate q
    log_ratio = math.log(error / options.tolerance) / q

    if log_ratio >= math.log(options.max_subintervals):
        return options.max_subintervals

    return min(math.ceil(math.exp(log_ratio)), options.max_subintervals)


def p_refinement(n, error, q, options):
    """Collocation points and number of equal subintervals for an interval with q > Q"""
    # Points that bring the error down to the tolerance, which are many when q is just above Q
    points = math.ceil(n * math.exp(math.log(error / options.tolerance) / (q - 2.5)))

    # Too many points split the interval instead, with no cap on the subintervals
    if points > options.max_points:
        return n, math.ceil(points / n)

    return points, 1


def power_coefficients(nodes, center=0.0):
    """Coefficients a with the Lagrange basis ell_j(s) the sum over i of a[i, j] (s - center)**i"""
    nodes = np.asarray(nodes, dtype=float)
    a = np.empty((len(nodes), len(nodes)))

    # The numerator of ell_j is the product of s minus every other node
    for j in range(len(nodes)):
        others = np.delete(nodes, j)
        a[:, j] = np.poly(others - center)[::-1] / np.prod(nodes[j] - others)

    return a


def series(values, flipped):
    """Power series coefficients of a state polynomial about its interval midpoint"""
    values = np.asarray(values, dtype=float).reshape(len(values), -1)
    nodes = support_points(len(values) - 1, flipped)

    return power_coefficients(nodes) @ values


def reduced_degree(values, flipped, options):
    """Collocation points left after dropping the highest powers with negligible coefficients"""
    b = series(values, flipped)

    # Keep the powers up to the highest one that exceeds the tolerance in some component
    kept = np.nonzero((np.abs(b) > options.tolerance).any(axis=1))[0]

    degree = 0
    if len(kept) > 0:
        degree = int(kept[-1])

    return max(degree, options.min_points)


def truncated_values(values, flipped, n):
    """Values on the support points of degree n of the state series cut after the power n"""
    b = series(values, flipped)[: n + 1]
    s = support_points(n, flipped)

    # Powers of the support points times the coefficients
    powers = s[:, None] ** np.arange(len(b))[None, :]

    return powers @ b


def merge_bound(left, right, h_left, h_right, flipped):
    """Bound on the gap between two neighboring state polynomials over the smaller interval"""
    # Values on the support points of the two intervals
    left = np.asarray(left, dtype=float).reshape(len(left), -1)
    right = np.asarray(right, dtype=float).reshape(len(right), -1)

    n = len(left) - 1
    nodes = support_points(n, flipped)

    # Expand each polynomial about the shared mesh point
    c_minus = power_coefficients(nodes, 1.0) @ left
    c_plus = power_coefficients(nodes, -1.0) @ right

    # Rescale both to the smaller half width
    hbar = min(h_left, h_right)
    j = np.arange(n + 1)[:, None]

    b_minus = c_minus * (hbar / h_left) ** j
    b_plus = c_plus * (hbar / h_right) ** j

    # Sum 2**j times the gap of the power j coefficients and take the largest component
    gaps = (2.0**j) * np.abs(b_minus - b_plus)

    return float(gaps.sum(axis=0).max())


@dataclass
class _Piece:
    """An interval of the next mesh while it is being assembled"""

    left: float
    right: float
    points: int
    parent: int
    values: np.ndarray = field(default=None, repr=False)


def can_merge(a, b, flipped, options):
    """Whether two neighboring pieces merge into one"""
    if b is None:
        return False

    # Only the intervals within the tolerance carry values
    if a.values is None or b.values is None:
        return False

    if a.points != b.points:
        return False

    # The two polynomials must agree to within the tolerance
    h_a = 0.5 * (a.right - a.left)
    h_b = 0.5 * (b.right - b.left)

    return merge_bound(a.values, b.values, h_a, h_b, flipped) < options.tolerance


def refine(solution, errors, q, options):
    """The next mesh, the parent of each of its intervals, and the action on each current interval"""
    mesh = solution.mesh

    pieces = []
    actions = []

    for k in range(mesh.num_intervals):
        left = mesh.points[k]
        right = mesh.points[k + 1]
        n = int(mesh.degrees[k])

        # An interval above the tolerance splits when q <= Q and gets more points when q > Q
        if errors[k] > options.tolerance:
            if q[k] <= options.smoothness_threshold:
                count = h_refinement(errors[k], q[k], options)
                points = n
            else:
                refinement = p_refinement(n, errors[k], q[k], options)
                points = refinement[0]
                count = refinement[1]

            edges = np.linspace(left, right, count + 1)

            for i in range(count):
                pieces.append(_Piece(edges[i], edges[i + 1], points, k))

            if count > 1:
                actions.append("split")
            else:
                actions.append("raise")

        # An interval within the tolerance drops its negligible powers
        else:
            values = solution.states[mesh.state_indices(k)]
            points = reduced_degree(values, mesh.flipped, options)
            truncated = truncated_values(values, mesh.flipped, points)

            pieces.append(_Piece(left, right, points, k, truncated))

            if points < n:
                actions.append("lower")
            elif points > n:
                actions.append("raise")
            else:
                actions.append("keep")

    # Neighbors within the tolerance merge at most once on each refinement, with the parent -1
    merged = []
    i = 0

    while i < len(pieces):
        a = pieces[i]

        b = None
        if i + 1 < len(pieces):
            b = pieces[i + 1]

        if can_merge(a, b, mesh.flipped, options):
            merged.append(_Piece(a.left, b.right, a.points, -1))

            actions[a.parent] = "merge"
            actions[b.parent] = "merge"

            i += 2
        else:
            merged.append(a)
            i += 1

    # The next mesh from the merged pieces
    points = []
    degrees = []
    parents = []

    for piece in merged:
        points.append(piece.left)
        degrees.append(piece.points)
        parents.append(piece.parent)

    points.append(merged[-1].right)

    next_mesh = RadauMesh(points, degrees, mesh.flipped)

    return next_mesh, np.array(parents), actions


class HPRefinement(MeshRefinement):
    """The hp mesh refinement of Liu, Hager, and Rao in the loop of MeshRefinement"""

    def __init__(self, options, dynamics):
        super().__init__(options)

        # dynamics(states, controls, t) returns the time rates of the states at a set of points
        self.dynamics = dynamics

    def start(self, mesh):
        """The parent of every interval and the previous mesh with its errors, unknown on a first mesh"""
        return {"parents": None, "previous": None}

    def assess(self, solution, state):
        """Relative errors and smoothness of every interval against the previous mesh"""
        errors = relative_errors(solution, self.dynamics)
        q = smoothness_estimates(
            solution.mesh, errors, state["parents"], state["previous"]
        )

        state["previous"] = (solution.mesh, errors)

        return Assessment(errors, smoothness=q)

    def decide(self, solution, assessment, state):
        """The next mesh with the action on every interval"""
        result = refine(
            solution, assessment.errors, assessment.smoothness, self.options
        )

        mesh = result[0]
        state["parents"] = result[1]
        actions = result[2]

        return mesh, actions

    def stay(self, solution, state):
        """Every interval of a mesh that stands is its own parent"""
        state["parents"] = np.arange(solution.mesh.num_intervals)
