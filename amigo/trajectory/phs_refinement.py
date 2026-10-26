"""The phs mesh refinement of Haman and Rao for Radau collocation.

G. V. Haman III and A. V. Rao, Adaptive mesh refinement and error estimation
method for optimal control using direct collocation, arXiv 2410.07488, 2024.
Equation numbers and lines below refer to that paper and its Algorithm 1. The
paper collocates at the standard Radau points, and the flipped points use the
same definitions.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .mesh import RadauMesh, lagrange_matrix, mesh_time, radau_points, support_points
from .mesh_refinement import Assessment, MeshRefinement

DIRECTIONS = ("forward", "backward")


@dataclass(frozen=True)
class PHSOptions:
    """Settings of the phs mesh refinement."""

    # Relative error tolerance that every interval must meet
    tolerance: float = 1e-6

    # Fewest collocation points in an interval and the points of every new subinterval
    min_points: int = 2

    # Most collocation points in one interval
    max_points: int = 10

    # Most meshes solved before the refinement stops
    max_iterations: int = 40

    # Integration method of scipy solve_ivp for the flights
    ode_method: str = "DOP853"

    # Relative and absolute tolerance of the flights
    ode_tolerance: float = 1e-10

    # Evenly spaced points per interval compared along with the solver steps
    even_points: int = 21

    # Flights that enter the error estimate
    directions: tuple = DIRECTIONS

    def __post_init__(self):
        if self.tolerance <= 0.0:
            raise ValueError("the tolerance must be positive")
        if not 2 <= self.min_points <= self.max_points:
            raise ValueError("the point limits must satisfy 2 <= min_points <= max_points")
        if self.max_iterations < 1:
            raise ValueError("at least one mesh must be solved")
        if self.ode_tolerance <= 0.0:
            raise ValueError("the ODE tolerance must be positive")
        if self.even_points < 0:
            raise ValueError("the number of evenly spaced points cannot be negative")
        if not self.directions or set(self.directions) - set(DIRECTIONS):
            raise ValueError(f"the directions must be a nonempty subset of {DIRECTIONS}")


class FlightError(RuntimeError):
    """A flight that the ODE solver could not finish."""


def scale_factors(solution):
    """Factor of every state component, one over one plus its largest magnitude on the mesh, Eq. 25."""
    return 1.0 / (1.0 + np.abs(solution.states).max(axis=0))


def _polynomial(solution, first, last):
    """Function of time that evaluates the state polynomial of the interval holding each time, Eqs. 13 and 33."""
    mesh = solution.mesh
    t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)
    pieces = [
        (t_mesh[j], t_mesh[j + 1], support_points(int(mesh.degrees[j]), mesh.flipped), solution.states[mesh.state_indices(j)])
        for j in range(first, last + 1)
    ]

    def evaluate(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        owner = np.searchsorted(t_mesh[first + 1 : last + 1], t, side="left")
        out = np.empty((len(t), solution.states.shape[1]))
        for i, (a, b, nodes, values) in enumerate(pieces):
            rows = owner == i
            if rows.any():
                out[rows] = lagrange_matrix(nodes, 2.0 * (t[rows] - a) / (b - a) - 1.0) @ values
        return out

    return evaluate


def _gap(polynomial, t, y, gamma):
    """Largest scaled difference of flown states from the state polynomial, Eqs. 24 and 33."""
    return float((gamma * np.abs(y - polynomial(t))).max())


def _fly(solution, k, dynamics, start, span, options, direction, limit=None):
    """Times and states of a flight under the control polynomial of interval k, Eqs. 21 to 23, 31, and 32.

    The control is the Lagrange polynomial through the collocation controls of
    interval k and is carried past the interval when the span leaves it. The
    states are taken at the solver steps and at evenly spaced points of every
    interval that the span crosses. A limit holds the state polynomial and the
    tolerance of a merge test, and the flight then stops at the first solver
    step whose scaled gap exceeds the tolerance and returns None.
    """
    mesh = solution.mesh
    t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)
    a, b = t_mesh[k], t_mesh[k + 1]
    nodes = radau_points(int(mesh.degrees[k]), mesh.flipped)
    controls = solution.controls[mesh.collocation_indices(k)]

    def rates(t, y):
        u = lagrange_matrix(nodes, [2.0 * (t - a) / (b - a) - 1.0]) @ controls
        return np.asarray(dynamics(y[None, :], u, np.array([t])), dtype=float).reshape(-1)

    events = None
    if limit is not None:
        polynomial, gamma, tolerance = limit

        def margin(t, y):
            return tolerance - _gap(polynomial, t, y[None, :], gamma)

        margin.terminal = True
        margin.direction = -1
        events = [margin]

    sol = solve_ivp(
        rates,
        span,
        np.asarray(start, dtype=float),
        method=options.ode_method,
        rtol=options.ode_tolerance,
        atol=options.ode_tolerance,
        dense_output=options.even_points > 0,
        events=events,
    )
    if sol.status == 1:
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise FlightError(f"the {direction} flight of interval {k} failed: {sol.message}")

    if options.even_points == 0:
        return sol.t, sol.y.T
    lo, hi = min(span), max(span)
    crossed = [j for j in range(mesh.num_intervals) if t_mesh[j] < hi and t_mesh[j + 1] > lo]
    even = np.concatenate([np.linspace(t_mesh[j], t_mesh[j + 1], options.even_points) for j in crossed])
    t = np.union1d(sol.t, even)
    return t, sol.sol(t).T


def _flights(solution, first, last, dynamics, options, limit=None):
    """Flights across intervals first to last in the directions the options select, or None once one stops.

    The forward flight starts from the state at the start of interval first
    under its control polynomial, and the backward flight starts from the state
    at the end of interval last under its control polynomial.
    """
    mesh = solution.mesh
    t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)
    flights = []
    if "forward" in options.directions:
        start = solution.states[mesh.state_indices(first)[0]]
        span = (t_mesh[first], t_mesh[last + 1])
        flights.append(_fly(solution, first, dynamics, start, span, options, "forward", limit))
        if flights[-1] is None:
            return None
    if "backward" in options.directions:
        start = solution.states[mesh.state_indices(last)[-1]]
        span = (t_mesh[last + 1], t_mesh[first])
        flights.append(_fly(solution, last, dynamics, start, span, options, "backward", limit))
        if flights[-1] is None:
            return None
    return flights


def simulated_errors(solution, dynamics, options):
    """Largest relative state error of every interval against its forward and backward flights, Eqs. 21 to 26.

    A flight that the ODE solver cannot finish stops the estimate with an
    error that names the interval and the direction.
    """
    gamma = scale_factors(solution)
    errors = np.empty(solution.mesh.num_intervals)
    for k in range(len(errors)):
        polynomial = _polynomial(solution, k, k)
        flights = _flights(solution, k, k, dynamics, options)
        errors[k] = max(_gap(polynomial, t, y, gamma) for t, y in flights)
    return errors


def path_violations(solution, constraints, options):
    """Largest violation of the path constraints between the points of every interval, where it lies, and whether one is active.

    constraints(states, controls, t) returns values that are feasible at or
    below zero and scaled so that the tolerance applies to them. The NLP holds
    them only at the collocation points, so they are evaluated on the state and
    control polynomials of each interval at its support points and at evenly
    spaced points. The place of the largest violation is a mesh coordinate, or
    nan where nothing is violated. A constraint is active on an interval when
    it lies within the tolerance of its bound at one of its collocation points.
    These tests go beyond Algorithm 1.
    """
    mesh = solution.mesh
    violations = np.zeros(mesh.num_intervals)
    places = np.full(mesh.num_intervals, np.nan)
    active = np.zeros(mesh.num_intervals, dtype=bool)
    for k in range(mesh.num_intervals):
        tau, _, values = constraint_samples(solution, k, constraints, options)
        worst = values.max(axis=1)
        if worst.max() > 0.0:
            violations[k] = float(worst.max())
            places[k] = float(mesh.to_mesh(k, tau[int(worst.argmax())]))
        nodes = radau_points(int(mesh.degrees[k]), mesh.flipped)
        active[k] = bool((worst[np.isin(tau, nodes)] >= -options.tolerance).any())
    return violations, places, active


def constraint_samples(solution, k, constraints, options):
    """Local coordinates, times, and path constraint values of interval k at its support points and evenly spaced points."""
    mesh = solution.mesh
    t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)
    n = int(mesh.degrees[k])
    tau = support_points(n, mesh.flipped)
    if options.even_points > 0:
        tau = np.union1d(tau, np.linspace(-1.0, 1.0, options.even_points))
    states = lagrange_matrix(support_points(n, mesh.flipped), tau) @ solution.states[mesh.state_indices(k)]
    controls = lagrange_matrix(radau_points(n, mesh.flipped), tau) @ solution.controls[mesh.collocation_indices(k)]
    t = mesh_time(tau, t_mesh[k], t_mesh[k + 1])
    values = np.asarray(constraints(states, controls, t), dtype=float).reshape(len(tau), -1)
    return tau, t, values


def touch_points(solution, constraints, options, margin=None):
    """Times inside the phase where a path constraint reaches a local maximum within a margin of its bound, with its column.

    These are the touch points of Jacobson, Lele, and Speyer, where a state
    constraint meets its boundary at a single time instead of along an arc.
    The constraints are sampled as in path_violations, so a touch is found to
    the spacing of the samples, and the margin is the tolerance unless given.
    """
    margin = options.tolerance if margin is None else margin
    times, values = [], []
    for k in range(solution.mesh.num_intervals):
        _, t, v = constraint_samples(solution, k, constraints, options)

        # Neighboring intervals share their end point, which is sampled once
        first = 0 if k == 0 else 1
        times.append(t[first:])
        values.append(v[first:])
    t, v = np.concatenate(times), np.vstack(values)
    touches = []
    for column in range(v.shape[1]):
        g = v[:, column]
        for i in range(1, len(g) - 1):
            if g[i] >= -margin and g[i] >= g[i - 1] and g[i] > g[i + 1]:
                touches.append((float(t[i]), column))
    return sorted(touches)


def pair_error(solution, k, dynamics, options, gamma):
    """Largest relative error of the flights across intervals k and k + 1, Eqs. 30 to 34.

    The forward flight carries the control polynomial of interval k into
    interval k + 1 and the backward flight carries that of interval k + 1 back
    into interval k. Only an error within the tolerance matters to the merge,
    so a flight stops at the first solver step past the tolerance. The error
    is then infinite, as it is when the ODE solver cannot finish a flight.
    """
    polynomial = _polynomial(solution, k, k + 1)
    try:
        flights = _flights(solution, k, k + 1, dynamics, options, (polynomial, gamma, options.tolerance))
    except FlightError:
        return math.inf
    if flights is None:
        return math.inf
    return max(_gap(polynomial, t, y, gamma) for t, y in flights)


def merge_error(solution, k, options, gamma):
    """Largest relative error of the one polynomial that would replace intervals k and k + 1, a test beyond Eq. 34.

    The polynomial has the points of Eq. 35 and interpolates the state
    polynomials of both intervals at its support points. It is compared with
    them at their own support points and at evenly spaced points of each.
    Eq. 34 carries only the control across the pair, while this test asks
    whether one polynomial can follow the state over both.
    """
    mesh = solution.mesh
    t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)
    a, b = t_mesh[k], t_mesh[k + 2]
    polynomial = _polynomial(solution, k, k + 1)
    nodes = support_points(max(int(mesh.degrees[k]), int(mesh.degrees[k + 1])), mesh.flipped)
    values = polynomial(mesh_time(nodes, a, b))
    checks = []
    for j in (k, k + 1):
        checks.append(mesh_time(support_points(int(mesh.degrees[j]), mesh.flipped), t_mesh[j], t_mesh[j + 1]))
        if options.even_points > 0:
            checks.append(np.linspace(t_mesh[j], t_mesh[j + 1], options.even_points))
    t = np.concatenate(checks)
    merged = lagrange_matrix(nodes, 2.0 * (t - a) / (b - a) - 1.0) @ values
    return float((gamma * np.abs(merged - polynomial(t))).max())


def p_refinement(n, error, options):
    """Collocation points that an interval above the tolerance asks for, Eq. 27."""
    return n + math.ceil(math.log10(error / options.tolerance))


def h_refinement(points, options):
    """Number of equal subintervals of the fewest points that replace an interval, Eq. 28."""
    return max(2, math.ceil(points / options.min_points))


def p_reduction(n, error, options):
    """Collocation points left on an interval within the tolerance, Eq. 36.

    One point is dropped for every delta = Nmin + Nmax - n orders of magnitude
    that the error lies below the tolerance.
    """
    if error <= 0.0:
        return options.min_points
    delta = options.min_points + options.max_points - n
    drop = math.floor(math.log10(options.tolerance / error) / delta)
    return max(options.min_points, n - drop)


def refine(solution, errors, dynamics, options, reduce=True, keep=None, cuts=None):
    """The next mesh and the action on every interval, lines 19 to 36 of Algorithm 1.

    An interval above the tolerance gets the points of Eq. 27, or splits into
    the equal subintervals of Eq. 28 with the fewest points when that passes
    the most points. Neighbors within the tolerance merge when their flights
    across both meet it and one polynomial would follow their state within it,
    in order of increasing error, each interval at most once, and keep the
    larger point count by Eq. 35. Every other interval within the tolerance
    drops points by Eq. 36. Without reduce, intervals within the tolerance keep
    their points and no neighbors merge. An interval marked in keep, where a
    path constraint is active, neither merges nor drops points. An interval
    with a cut inside it, where a path constraint slips most between its
    points, splits there into two halves with its points, so the NLP holds the
    constraint at that place on the next mesh.
    """
    mesh = solution.mesh
    within = errors <= options.tolerance
    gamma = scale_factors(solution)
    keep = np.zeros(mesh.num_intervals, dtype=bool) if keep is None else keep
    cuts = np.full(mesh.num_intervals, np.nan) if cuts is None else cuts

    # Neighbors within the tolerance merge in order of increasing error
    candidates = []
    for k in range(mesh.num_intervals - 1 if reduce else 0):
        if within[k] and within[k + 1] and not (keep[k] or keep[k + 1]):
            # The one polynomial of the merge must follow the state before the flights test the control
            r = merge_error(solution, k, options, gamma)
            if r <= options.tolerance:
                e = pair_error(solution, k, dynamics, options, gamma)
                if e <= options.tolerance:
                    candidates.append((max(errors[k], errors[k + 1], e, r), k))
    taken = np.zeros(mesh.num_intervals, dtype=bool)
    lefts = set()
    for _, k in sorted(candidates):
        if not taken[k] and not taken[k + 1]:
            taken[k] = taken[k + 1] = True
            lefts.add(k)

    points = [mesh.points[0]]
    degrees = []
    actions = []
    k = 0
    while k < mesh.num_intervals:
        n = int(mesh.degrees[k])
        if k in lefts:
            points.append(mesh.points[k + 2])
            degrees.append(max(n, int(mesh.degrees[k + 1])))
            actions += ["merge", "merge"]
            k += 2
            continue
        if within[k]:
            points.append(mesh.points[k + 1])
            degrees.append(p_reduction(n, errors[k], options) if reduce and not keep[k] else n)
            actions.append("lower" if degrees[-1] < n else "keep")
        elif mesh.points[k] < cuts[k] < mesh.points[k + 1]:
            points.extend([cuts[k], mesh.points[k + 1]])
            degrees.extend([n, n])
            actions.append("cut")
        else:
            wanted = p_refinement(n, errors[k], options)
            if wanted <= options.max_points:
                points.append(mesh.points[k + 1])
                degrees.append(wanted)
                actions.append("raise")
            else:
                count = h_refinement(wanted, options)
                points.extend(np.linspace(mesh.points[k], mesh.points[k + 1], count + 1)[1:])
                degrees.extend([options.min_points] * count)
                actions.append("split")
        k += 1
    return RadauMesh(points, degrees, mesh.flipped), actions


def same_mesh(a, b):
    """Whether two meshes have the same points and the same number of collocation points in every interval."""
    return (
        a.num_intervals == b.num_intervals
        and np.array_equal(a.degrees, b.degrees)
        and np.allclose(a.points, b.points, rtol=0.0, atol=1e-12)
    )


def next_mesh(solution, errors, dynamics, options, solved, growing, keep=None, cuts=None):
    """The next mesh of one phase, the action on every interval, and whether it only gains points from then on, a rule beyond Algorithm 1."""
    mesh, actions = refine(solution, errors, dynamics, options, reduce=not growing, keep=keep, cuts=cuts)

    # A phase that gets back a mesh it already solved is refined without reductions from then on
    if not growing and any(same_mesh(mesh, old) for old in solved):
        mesh, actions = refine(solution, errors, dynamics, options, reduce=False, keep=keep, cuts=cuts)
        return mesh, actions, True
    return mesh, actions, growing


class PHSRefinement(MeshRefinement):
    """The phs mesh refinement of Haman and Rao in the loop of MeshRefinement.

    dynamics(states, controls, t) returns the rates of the states with respect
    to time at a set of points, and the optional constraints(states, controls,
    t) the path constraints of path_violations. Beyond Algorithm 1, a phase
    that gets back a mesh it already solved or violates a path constraint
    between its points only gains points from then on, an interval with a
    violation splits where it is largest, and an interval where a constraint
    is active is never coarsened.
    """

    def __init__(self, options, dynamics, constraints=None):
        super().__init__(options)
        self.dynamics = dynamics
        self.constraints = constraints

    def start(self, mesh):
        """The meshes a phase has solved and whether it only gains points."""
        if mesh.degrees.max() > self.options.max_points:
            raise ValueError("every interval of the initial mesh needs at most max_points collocation points")
        return {"solved": [], "growing": False}

    def estimate(self, solution):
        """The Assessment of every interval of one solved mesh, with no state of the refinement.

        The error of an interval is the largest of its flights and of any path
        constraint violation between its points. Its cut is the place of that
        violation when it passes the tolerance, and nan otherwise.
        """
        errors = simulated_errors(solution, self.dynamics, self.options)
        if self.constraints is None:
            return Assessment(errors, keep=np.zeros(len(errors), dtype=bool), cuts=np.full(len(errors), np.nan))
        violations, places, active = path_violations(solution, self.constraints, self.options)
        cuts = np.where(violations > self.options.tolerance, places, np.nan)
        return Assessment(np.maximum(errors, violations), keep=active, cuts=cuts)

    def assess(self, solution, state):
        """The estimate of a solved mesh, which the phase also records as solved."""
        state["solved"].append(solution.mesh)
        return self.estimate(solution)

    def decide(self, solution, assessment, state):
        """The next mesh of lines 19 to 36 of Algorithm 1 with the rules beyond it."""
        # A violation moves with the switch times of the solve, so the mesh keeps every point it gains
        state["growing"] = state["growing"] or bool(np.isfinite(assessment.cuts).any())
        mesh, actions, state["growing"] = next_mesh(
            solution, assessment.errors, self.dynamics, self.options, state["solved"], state["growing"],
            assessment.keep, assessment.cuts,
        )
        return mesh, actions
