"""
phs mesh refinement
-------------------

- The phs mesh refinement of Haman and Rao, arXiv 2410.07488, 2024
- The flipped points use the same definitions as its standard Radau points
- Beyond the paper, a violated interval splits at the violation and an active one never coarsens
- A repeated mesh or a violation makes a phase only gain points from then on
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..collocation.mesh import RadauMesh, lagrange_matrix, mesh_time, radau_points, support_points
from .base import Assessment, MeshRefinement

DIRECTIONS = ("forward", "backward")


@dataclass(frozen=True)
class PHSOptions:
    """Settings of the phs mesh refinement"""

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
    """A flight that the ODE solver could not finish"""


def scale_factors(solution):
    """Factor of every state component, one over one plus its largest magnitude on the mesh"""
    return 1.0 / (1.0 + np.abs(solution.states).max(axis=0))


class StatePolynomial:
    """State polynomials of the intervals first to last as one function of time"""

    def __init__(self, solution, first, last):
        mesh = solution.mesh

        self.t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)
        self.first = first
        self.last = last
        self.num_states = solution.states.shape[1]

        # The time span, the support points, and the state values of every interval
        self.starts = []
        self.ends = []
        self.nodes = []
        self.values = []

        for j in range(first, last + 1):
            self.starts.append(self.t_mesh[j])
            self.ends.append(self.t_mesh[j + 1])
            self.nodes.append(support_points(int(mesh.degrees[j]), mesh.flipped))
            self.values.append(solution.states[mesh.state_indices(j)])

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))

        # The interval that holds each time
        inner = self.t_mesh[self.first + 1 : self.last + 1]
        owner = np.searchsorted(inner, t, side="left")

        out = np.empty((len(t), self.num_states))

        for i in range(len(self.starts)):
            rows = owner == i

            if rows.any():
                a = self.starts[i]
                b = self.ends[i]

                s = 2.0 * (t[rows] - a) / (b - a) - 1.0
                out[rows] = lagrange_matrix(self.nodes[i], s) @ self.values[i]

        return out


def _gap(polynomial, t, y, gamma):
    """Largest scaled difference of flown states from the state polynomial"""
    return float((gamma * np.abs(y - polynomial(t))).max())


class ControlledRates:
    """Rates of the states under the control polynomial of interval k"""

    def __init__(self, solution, k, dynamics):
        mesh = solution.mesh
        t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)

        self.a = t_mesh[k]
        self.b = t_mesh[k + 1]
        self.nodes = radau_points(int(mesh.degrees[k]), mesh.flipped)
        self.controls = solution.controls[mesh.collocation_indices(k)]
        self.dynamics = dynamics

    def __call__(self, t, y):
        # The control polynomial carries past the interval when the flight leaves it
        s = 2.0 * (t - self.a) / (self.b - self.a) - 1.0
        u = lagrange_matrix(self.nodes, [s]) @ self.controls

        rates = self.dynamics(y[None, :], u, np.array([t]))

        return np.asarray(rates, dtype=float).reshape(-1)


class MarginEvent:
    """Event that stops a flight at the first step past the tolerance of a merge test"""

    terminal = True
    direction = -1

    def __init__(self, polynomial, gamma, tolerance):
        self.polynomial = polynomial
        self.gamma = gamma
        self.tolerance = tolerance

    def __call__(self, t, y):
        return self.tolerance - _gap(self.polynomial, t, y[None, :], self.gamma)


@dataclass
class Flight:
    """Times and states of one flight, or why it ended early"""

    t: np.ndarray = None
    y: np.ndarray = None

    # The flight stopped at the first step past the tolerance of a merge test
    stopped: bool = False

    # Why the ODE solver could not finish the flight
    failure: str = None


def _fly(solution, first, last, dynamics, options, direction, event=None):
    """Times and states flown across the intervals first to last under the control of an end interval"""
    mesh = solution.mesh
    t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)

    # The forward flight starts at the start of interval first under its control polynomial
    if direction == "forward":
        k = first
        start = solution.states[mesh.state_indices(first)[0]]
        span = (t_mesh[first], t_mesh[last + 1])

    # The backward flight starts at the end of interval last under its control polynomial
    else:
        k = last
        start = solution.states[mesh.state_indices(last)[-1]]
        span = (t_mesh[last + 1], t_mesh[first])

    rates = ControlledRates(solution, k, dynamics)

    events = None
    if event is not None:
        events = [event]

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
        return Flight(stopped=True)

    if not sol.success or not np.all(np.isfinite(sol.y)):
        return Flight(failure=f"the {direction} flight of interval {k} failed: {sol.message}")

    # The states at the solver steps
    if options.even_points == 0:
        return Flight(sol.t, sol.y.T)

    # The states also at evenly spaced points of every interval the flight crosses
    lo = min(span)
    hi = max(span)

    even = []

    for j in range(mesh.num_intervals):
        if t_mesh[j] < hi and t_mesh[j + 1] > lo:
            even.append(np.linspace(t_mesh[j], t_mesh[j + 1], options.even_points))

    t = np.union1d(sol.t, np.concatenate(even))

    return Flight(t, sol.sol(t).T)


def _flights(solution, first, last, dynamics, options, event=None):
    """Flights across the intervals first to last in the directions the options select"""
    flights = []

    for direction in DIRECTIONS:
        if direction not in options.directions:
            continue

        flight = _fly(solution, first, last, dynamics, options, direction, event)
        flights.append(flight)

        # A flight that ended early ends the others too
        if flight.stopped or flight.failure is not None:
            break

    return flights


def simulated_errors(solution, dynamics, options):
    """Largest relative state error of every interval against its forward and backward flights"""
    gamma = scale_factors(solution)
    errors = np.empty(solution.mesh.num_intervals)

    for k in range(len(errors)):
        polynomial = StatePolynomial(solution, k, k)
        flights = _flights(solution, k, k, dynamics, options)

        # A flight that the ODE solver cannot finish stops the estimate
        gaps = []

        for flight in flights:
            if flight.failure is not None:
                raise FlightError(flight.failure)

            gaps.append(_gap(polynomial, flight.t, flight.y, gamma))

        errors[k] = max(gaps)

    return errors


def constraint_samples(solution, k, constraints, options):
    """Local coordinates, times, and path constraint values of interval k at its sample points"""
    mesh = solution.mesh
    t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)
    n = int(mesh.degrees[k])

    # The support points and the evenly spaced points of the interval
    tau = support_points(n, mesh.flipped)

    if options.even_points > 0:
        tau = np.union1d(tau, np.linspace(-1.0, 1.0, options.even_points))

    # The states and the controls at those points
    state_basis = lagrange_matrix(support_points(n, mesh.flipped), tau)
    control_basis = lagrange_matrix(radau_points(n, mesh.flipped), tau)

    states = state_basis @ solution.states[mesh.state_indices(k)]
    controls = control_basis @ solution.controls[mesh.collocation_indices(k)]

    # constraints(states, controls, t) returns scaled values that are feasible at or below zero
    t = mesh_time(tau, t_mesh[k], t_mesh[k + 1])
    values = np.asarray(constraints(states, controls, t), dtype=float).reshape(len(tau), -1)

    return tau, t, values


def path_violations(solution, constraints, options):
    """Largest path violation between the points of every interval, its place, and whether one is active"""
    mesh = solution.mesh

    violations = np.zeros(mesh.num_intervals)
    places = np.full(mesh.num_intervals, np.nan)
    active = np.zeros(mesh.num_intervals, dtype=bool)

    for k in range(mesh.num_intervals):
        samples = constraint_samples(solution, k, constraints, options)
        tau = samples[0]
        values = samples[2]

        # The largest constraint value at every sample
        worst = values.max(axis=1)

        # The place of the largest violation as a mesh coordinate, and nan where nothing is violated
        if worst.max() > 0.0:
            violations[k] = float(worst.max())
            places[k] = float(mesh.to_mesh(k, tau[int(worst.argmax())]))

        # A constraint is active within the tolerance of its bound at a collocation point
        nodes = radau_points(int(mesh.degrees[k]), mesh.flipped)
        at_nodes = worst[np.isin(tau, nodes)]

        active[k] = bool((at_nodes >= -options.tolerance).any())

    return violations, places, active


def touch_points(solution, constraints, options, margin=None):
    """Times inside the phase where a path constraint peaks within a margin of its bound, with its column"""
    # A touch point of Jacobson, Lele, and Speyer meets the bound at one time instead of along an arc
    if margin is None:
        margin = options.tolerance

    # Sample the constraints over the whole phase
    times = []
    values = []

    for k in range(solution.mesh.num_intervals):
        samples = constraint_samples(solution, k, constraints, options)
        t = samples[1]
        v = samples[2]

        # Neighboring intervals share their end point, which is sampled once
        first = 1
        if k == 0:
            first = 0

        times.append(t[first:])
        values.append(v[first:])

    t = np.concatenate(times)
    v = np.vstack(values)

    # A touch is a peak within the margin, found to the spacing of the samples
    touches = []

    for column in range(v.shape[1]):
        g = v[:, column]

        for i in range(1, len(g) - 1):
            if g[i] >= -margin and g[i] >= g[i - 1] and g[i] > g[i + 1]:
                touches.append((float(t[i]), column))

    return sorted(touches)


def pair_error(solution, k, dynamics, options, gamma):
    """Largest relative error of the flights across intervals k and k + 1"""
    polynomial = StatePolynomial(solution, k, k + 1)

    # Each flight carries the control of the interval it starts in and stops at the first step past the tolerance
    event = MarginEvent(polynomial, gamma, options.tolerance)
    flights = _flights(solution, k, k + 1, dynamics, options, event)

    # The error is infinite when a flight stops or fails
    gaps = []

    for flight in flights:
        if flight.stopped or flight.failure is not None:
            return math.inf

        gaps.append(_gap(polynomial, flight.t, flight.y, gamma))

    return max(gaps)


def merge_error(solution, k, options, gamma):
    """Largest relative error of one polynomial over intervals k and k + 1"""
    mesh = solution.mesh
    t_mesh = mesh_time(mesh.points, solution.t0, solution.tf)

    a = t_mesh[k]
    b = t_mesh[k + 2]

    # The one polynomial interpolates the state polynomials of both intervals
    polynomial = StatePolynomial(solution, k, k + 1)

    n = max(int(mesh.degrees[k]), int(mesh.degrees[k + 1]))
    nodes = support_points(n, mesh.flipped)
    values = polynomial(mesh_time(nodes, a, b))

    # Compare at the support points and the evenly spaced points of both intervals
    checks = []

    for j in (k, k + 1):
        interval_nodes = support_points(int(mesh.degrees[j]), mesh.flipped)
        checks.append(mesh_time(interval_nodes, t_mesh[j], t_mesh[j + 1]))

        if options.even_points > 0:
            checks.append(np.linspace(t_mesh[j], t_mesh[j + 1], options.even_points))

    t = np.concatenate(checks)

    # This test follows the state over both intervals, while the flights carry only the control
    merged = lagrange_matrix(nodes, 2.0 * (t - a) / (b - a) - 1.0) @ values

    return float((gamma * np.abs(merged - polynomial(t))).max())


def p_refinement(n, error, options):
    """Collocation points that an interval above the tolerance asks for"""
    return n + math.ceil(math.log10(error / options.tolerance))


def h_refinement(points, options):
    """Number of equal subintervals of the fewest points that replace an interval"""
    return max(2, math.ceil(points / options.min_points))


def p_reduction(n, error, options):
    """Collocation points left on an interval within the tolerance"""
    if error <= 0.0:
        return options.min_points

    # One point is dropped for every delta orders of magnitude the error lies below the tolerance
    delta = options.min_points + options.max_points - n
    drop = math.floor(math.log10(options.tolerance / error) / delta)

    return max(options.min_points, n - drop)


def refine(solution, errors, dynamics, options, reduce=True, keep=None, cuts=None):
    """The next mesh and the action on every interval"""
    mesh = solution.mesh
    within = errors <= options.tolerance
    gamma = scale_factors(solution)

    # An interval in keep never merges or drops points, and an interval with a cut splits there
    if keep is None:
        keep = np.zeros(mesh.num_intervals, dtype=bool)

    if cuts is None:
        cuts = np.full(mesh.num_intervals, np.nan)

    # Neighbors within the tolerance that may merge, when reductions are allowed
    num_pairs = 0
    if reduce:
        num_pairs = mesh.num_intervals - 1

    candidates = []

    for k in range(num_pairs):
        if within[k] and within[k + 1] and not (keep[k] or keep[k + 1]):
            # The one polynomial of the merge must follow the state before the flights test the control
            r = merge_error(solution, k, options, gamma)

            if r <= options.tolerance:
                e = pair_error(solution, k, dynamics, options, gamma)

                if e <= options.tolerance:
                    candidates.append((max(errors[k], errors[k + 1], e, r), k))

    # Merge in order of increasing error, each interval at most once
    taken = np.zeros(mesh.num_intervals, dtype=bool)
    lefts = set()

    for candidate in sorted(candidates):
        k = candidate[1]

        if not taken[k] and not taken[k + 1]:
            taken[k] = True
            taken[k + 1] = True
            lefts.add(k)

    # Build the next mesh one interval at a time
    points = [mesh.points[0]]
    degrees = []
    actions = []

    k = 0

    while k < mesh.num_intervals:
        n = int(mesh.degrees[k])

        # A merged pair keeps the larger number of points
        if k in lefts:
            points.append(mesh.points[k + 2])
            degrees.append(max(n, int(mesh.degrees[k + 1])))

            actions.append("merge")
            actions.append("merge")

            k += 2
            continue

        # An interval within the tolerance drops points when reductions are allowed
        if within[k]:
            points.append(mesh.points[k + 1])

            if reduce and not keep[k]:
                degrees.append(p_reduction(n, errors[k], options))
            else:
                degrees.append(n)

            if degrees[-1] < n:
                actions.append("lower")
            else:
                actions.append("keep")

        # A violated interval splits at the violation
        elif mesh.points[k] < cuts[k] < mesh.points[k + 1]:
            points.append(cuts[k])
            points.append(mesh.points[k + 1])

            degrees.append(n)
            degrees.append(n)

            actions.append("cut")

        # An interval above the tolerance gets more points, or splits when it would need too many
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

    new_mesh = RadauMesh(points, degrees, mesh.flipped)

    return new_mesh, actions


def same_mesh(a, b):
    """Whether two meshes have the same points and the same number of collocation points in every interval"""
    if a.num_intervals != b.num_intervals:
        return False

    if not np.array_equal(a.degrees, b.degrees):
        return False

    return np.allclose(a.points, b.points, rtol=0.0, atol=1e-12)


def next_mesh(solution, errors, dynamics, options, solved, growing, keep=None, cuts=None):
    """The next mesh of one phase, the action on every interval, and whether it only gains points from then on"""
    result = refine(solution, errors, dynamics, options, reduce=not growing, keep=keep, cuts=cuts)

    mesh = result[0]
    actions = result[1]

    if growing:
        return mesh, actions, growing

    # A phase that gets back a mesh it already solved is refined without reductions from then on
    repeated = False

    for old in solved:
        if same_mesh(mesh, old):
            repeated = True
            break

    if repeated:
        result = refine(solution, errors, dynamics, options, reduce=False, keep=keep, cuts=cuts)

        return result[0], result[1], True

    return mesh, actions, growing


class PHSRefinement(MeshRefinement):
    """The phs mesh refinement of Haman and Rao in the loop of MeshRefinement"""

    def __init__(self, options, dynamics, constraints=None):
        super().__init__(options)

        # dynamics(states, controls, t) returns the state rates and constraints the path constraints
        self.dynamics = dynamics
        self.constraints = constraints

    def start(self, mesh):
        """The meshes a phase has solved and whether it only gains points"""
        if mesh.degrees.max() > self.options.max_points:
            raise ValueError("every interval of the initial mesh needs at most max_points collocation points")

        return {"solved": [], "growing": False}

    def estimate(self, solution):
        """Assessment of every interval of one solved mesh, with no state of the refinement"""
        errors = simulated_errors(solution, self.dynamics, self.options)

        if self.constraints is None:
            keep = np.zeros(len(errors), dtype=bool)
            cuts = np.full(len(errors), np.nan)

            return Assessment(errors, keep=keep, cuts=cuts)

        # The error of an interval is the largest of its flights and of any path violation between its points
        result = path_violations(solution, self.constraints, self.options)
        violations = result[0]
        places = result[1]
        active = result[2]

        # The cut is the place of a violation that passes the tolerance
        cuts = np.where(violations > self.options.tolerance, places, np.nan)

        return Assessment(np.maximum(errors, violations), keep=active, cuts=cuts)

    def assess(self, solution, state):
        """Estimate of a solved mesh, which the phase also records as solved"""
        state["solved"].append(solution.mesh)

        return self.estimate(solution)

    def decide(self, solution, assessment, state):
        """The next mesh of Algorithm 1 with the rules beyond it"""
        # A violation moves with the switch times of the solve, so the mesh keeps every point it gains
        state["growing"] = state["growing"] or bool(np.isfinite(assessment.cuts).any())

        result = next_mesh(
            solution,
            assessment.errors,
            self.dynamics,
            self.options,
            state["solved"],
            state["growing"],
            assessment.keep,
            assessment.cuts,
        )

        state["growing"] = result[2]

        return result[0], result[1]
