"""The hp mesh refinement of Liu, Hager, and Rao for Radau collocation.

The reference is Liu, Hager, and Rao, 53rd IEEE Conference on Decision and Control, 2014.
Equation and section numbers below refer to that paper unless they name Ref. 8.
Ref. 8 is Patterson, Hager, and Rao, Optimal Control Applications and Methods 36, 2015, the source of its error estimate.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .mesh import (
    RadauMesh,
    integration_matrix,
    lagrange_matrix,
    mesh_time,
    radau_points,
    support_points,
)
from .mesh_refinement import Assessment, MeshIteration, MeshRefinement, MeshSolution, best_iteration


@dataclass(frozen=True)
class RefinementOptions:
    """Settings of the hp mesh refinement."""

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
            raise ValueError("the point limits must satisfy 1 <= min_points <= max_points")
        if self.max_iterations < 1:
            raise ValueError("at least one mesh must be solved")


def integrated_states(solution, k, dynamics):
    """States of interval k integrated to the support points of N_k + 1 Radau points, Eq. 28 of Ref. 8.

    The polynomials and the dynamics are evaluated at the M_k = N_k + 1 Radau points.
    The Radau integration matrix carries the state from the interval start to each later point.
    The first value is the state at the interval start.
    """
    mesh = solution.mesh
    n = int(mesh.degrees[k])
    m = n + 1
    states = solution.states[mesh.state_indices(k)]
    controls = solution.controls[mesh.collocation_indices(k)]

    s_hat = radau_points(m, mesh.flipped)
    y = lagrange_matrix(support_points(n, mesh.flipped), s_hat) @ states
    u = lagrange_matrix(radau_points(n, mesh.flipped), s_hat) @ controls
    t = mesh_time(mesh.to_mesh(k, s_hat), solution.t0, solution.tf)
    rates = np.asarray(dynamics(y, u, t), dtype=float).reshape(m, -1)

    scale = 0.5 * (solution.tf - solution.t0) * mesh.half_widths[k]
    out = np.empty((m + 1, states.shape[1]))
    out[0] = states[0]
    out[1:] = states[0] + scale * integration_matrix(m, mesh.flipped) @ rates
    return out


def relative_errors(solution, dynamics):
    """Largest relative state error on every interval, Eqs. 14 and 15.

    The state polynomial and the integrated state are compared on the support points of N_k + 1 Radau points.
    Each component of the difference is divided by one plus its largest magnitude there.
    """
    mesh = solution.mesh
    errors = np.empty(mesh.num_intervals)
    for k, n in enumerate(mesh.degrees):
        s_hat = support_points(n + 1, mesh.flipped)
        basis = lagrange_matrix(support_points(n, mesh.flipped), s_hat)
        y = basis @ solution.states[mesh.state_indices(k)]
        y_int = integrated_states(solution, k, dynamics)
        scale = 1.0 + np.abs(y).max(axis=0)
        errors[k] = (np.abs(y_int - y) / scale).max()
    return errors


def smoothness(previous, current):
    """Number of continuous derivatives q from one interval on two meshes, Eqs. 17 and 18.

    previous holds the width, the points, and the error on mesh j and current the same for an interval inside it.
    Logarithms of the two equalities eliminate c and leave q.
    The result is None when the two meshes do not determine q.
    """
    (h0, n0, e0), (h1, n1, e1) = previous, current
    if min(h0, h1, e0, e1) <= 0.0:
        return None
    denominator = math.log(h0 / h1) - math.log(n0 / n1)
    if denominator == 0.0:
        return None
    return (math.log(e0 / e1) - 2.5 * math.log(n0 / n1)) / denominator


def smoothness_estimates(mesh, errors, parents=None, previous=None):
    """The smoothness q of every interval, taken as its collocation points where q is not determined.

    parents gives the interval of the previous mesh that holds each interval,
    or -1 for an interval made by merging, and previous is the previous mesh
    with its errors.
    """
    q = mesh.degrees.astype(float)
    if previous is None:
        return q
    old_mesh, old_errors = previous
    for k, parent in enumerate(parents):
        if parent < 0:
            continue
        estimate = smoothness(
            (2.0 * old_mesh.half_widths[parent], old_mesh.degrees[parent], old_errors[parent]),
            (2.0 * mesh.half_widths[k], mesh.degrees[k], errors[k]),
        )
        if estimate is not None:
            q[k] = estimate
    return q


def h_refinement(error, q, options):
    """Number of equal subintervals that replace an interval with q <= Q, Eqs. 21 and 22."""
    if q <= 0.0:
        return options.max_subintervals
    log_ratio = math.log(error / options.tolerance) / q
    if log_ratio >= math.log(options.max_subintervals):
        return options.max_subintervals
    return min(math.ceil(math.exp(log_ratio)), options.max_subintervals)


def p_refinement(n, error, q, options):
    """Collocation points and number of equal subintervals for an interval with q > Q, Eqs. 24 and 25.

    Eq. 24 raises the error ratio to the power 1/(q - 5/2) and asks for many points just above Q.
    Eq. 25 splits the interval into ceil(N_new/N) subintervals with no cap.
    A first mesh takes q as N and one refinement can then grow the mesh by orders of magnitude.
    """
    points = math.ceil(n * math.exp(math.log(error / options.tolerance) / (q - 2.5)))
    if points > options.max_points:
        return n, math.ceil(points / n)
    return points, 1


def power_coefficients(nodes, center=0.0):
    """Coefficients a with ell_j(s) the sum over i of a[i, j] (s - center)**i, Eqs. 27 and 29.

    ell_j is the Lagrange basis polynomial of node j. Its numerator is the
    product of s minus every other node, expanded in powers of s - center.
    """
    nodes = np.asarray(nodes, dtype=float)
    a = np.empty((len(nodes), len(nodes)))
    for j in range(len(nodes)):
        others = np.delete(nodes, j)
        a[:, j] = np.poly(others - center)[::-1] / np.prod(nodes[j] - others)
    return a


def series(values, flipped):
    """Power series coefficients of a state polynomial about its interval midpoint, Eq. 28."""
    values = np.asarray(values, dtype=float).reshape(len(values), -1)
    return power_coefficients(support_points(len(values) - 1, flipped)) @ values


def reduced_degree(values, flipped, options):
    """Collocation points left after dropping the highest powers with negligible coefficients, Sec. IV-C.3.

    Powers are dropped from the highest down until a coefficient exceeds the
    tolerance in some state component.
    """
    b = series(values, flipped)
    kept = np.nonzero((np.abs(b) > options.tolerance).any(axis=1))[0]
    degree = int(kept[-1]) if len(kept) else 0
    return max(degree, options.min_points)


def truncated_values(values, flipped, n):
    """Values on the support points of degree n of the state series cut after the power n."""
    b = series(values, flipped)[: n + 1]
    s = support_points(n, flipped)
    return (s[:, None] ** np.arange(len(b))[None, :]) @ b


def merge_bound(left, right, h_left, h_right, flipped):
    """Bound on the gap between two neighbor state polynomials over the smaller interval, Sec. IV-C.4.

    left and right hold values on the support points and h_left and h_right the half widths.
    Each polynomial expands about the shared mesh point and rescales to the smaller half width.
    The bound sums 2**j times the gap of the power j coefficients and takes the largest component.
    """
    left = np.asarray(left, dtype=float).reshape(len(left), -1)
    right = np.asarray(right, dtype=float).reshape(len(right), -1)
    n = len(left) - 1
    nodes = support_points(n, flipped)
    c_minus = power_coefficients(nodes, 1.0) @ left
    c_plus = power_coefficients(nodes, -1.0) @ right
    hbar = min(h_left, h_right)
    j = np.arange(n + 1)[:, None]
    b_minus = c_minus * (hbar / h_left) ** j
    b_plus = c_plus * (hbar / h_right) ** j
    return float(((2.0**j) * np.abs(b_minus - b_plus)).sum(axis=0).max())


@dataclass
class _Piece:
    """An interval of the next mesh while it is being assembled."""

    left: float
    right: float
    points: int
    parent: int
    values: np.ndarray = field(default=None, repr=False)


def refine(solution, errors, q, options):
    """The next mesh, the parent of each of its intervals, and the action on each current interval.

    An interval above the tolerance splits when q <= Q and gets more points when q > Q.
    An interval within the tolerance drops its negligible powers.
    Neighbors within the tolerance merge when their polynomials agree, with the parent -1.
    """
    mesh = solution.mesh
    pieces = []
    actions = []
    for k in range(mesh.num_intervals):
        left, right = mesh.points[k], mesh.points[k + 1]
        n = int(mesh.degrees[k])
        if errors[k] > options.tolerance:
            if q[k] <= options.smoothness_threshold:
                count, points = h_refinement(errors[k], q[k], options), n
            else:
                points, count = p_refinement(n, errors[k], q[k], options)
            edges = np.linspace(left, right, count + 1)
            pieces += [_Piece(edges[i], edges[i + 1], points, k) for i in range(count)]
            actions.append("split" if count > 1 else "raise")
        else:
            values = solution.states[mesh.state_indices(k)]
            points = reduced_degree(values, mesh.flipped, options)
            pieces.append(_Piece(left, right, points, k, truncated_values(values, mesh.flipped, points)))
            actions.append("lower" if points < n else "raise" if points > n else "keep")

    # Neighbors within the tolerance merge at most once on each refinement
    merged = []
    i = 0
    while i < len(pieces):
        a = pieces[i]
        b = pieces[i + 1] if i + 1 < len(pieces) else None
        if (
            b is not None
            and a.values is not None
            and b.values is not None
            and a.points == b.points
            and merge_bound(
                a.values, b.values, 0.5 * (a.right - a.left), 0.5 * (b.right - b.left), mesh.flipped
            )
            < options.tolerance
        ):
            merged.append(_Piece(a.left, b.right, a.points, -1))
            actions[a.parent] = actions[b.parent] = "merge"
            i += 2
        else:
            merged.append(a)
            i += 1

    points = [p.left for p in merged] + [merged[-1].right]
    next_mesh = RadauMesh(points, [p.points for p in merged], mesh.flipped)
    return next_mesh, np.array([p.parent for p in merged]), actions


class HPRefinement(MeshRefinement):
    """The hp mesh refinement of Liu, Hager, and Rao in the loop of MeshRefinement.

    dynamics(states, controls, t) returns the rates of the states with respect
    to time at a set of points.
    """

    def __init__(self, options, dynamics):
        super().__init__(options)
        self.dynamics = dynamics

    def start(self, mesh):
        """The parent of every interval and the previous mesh with its errors, both unknown on a first mesh."""
        return {"parents": None, "previous": None}

    def assess(self, solution, state):
        """The relative errors of Eqs. 14 and 15 and the smoothness of Eqs. 17 and 18 against the previous mesh."""
        errors = relative_errors(solution, self.dynamics)
        q = smoothness_estimates(solution.mesh, errors, state["parents"], state["previous"])
        state["previous"] = (solution.mesh, errors)
        return Assessment(errors, smoothness=q)

    def decide(self, solution, assessment, state):
        """The next mesh of Sec. IV with the action on every interval."""
        mesh, state["parents"], actions = refine(solution, assessment.errors, assessment.smoothness, self.options)
        return mesh, actions

    def stay(self, solution, state):
        """Every interval of a mesh that stands is its own parent."""
        state["parents"] = np.arange(solution.mesh.num_intervals)
