"""
phs mesh refinement
-------------------

- The error estimate from forward and backward flights of Haman and Rao
- The point and interval decisions, the merges, and the path constraint checks
- The refinement loop with the estimate or the decisions replaced where only the loop is tested
"""

import numpy as np
import pytest

import amigo.trajectory.refinement.phs as phs
from amigo.trajectory.collocation.mesh import (
    RadauMesh,
    integration_matrix,
    lagrange_matrix,
    radau_points,
    support_points,
)
from amigo.trajectory.refinement.base import MeshSolution


def double_integrator(states, controls, t):
    """Rates of a position and a velocity under an acceleration control"""
    return np.column_stack([states[:, 1], controls[:, 0]])


def blowup(states, controls, t):
    """Rates that blow up in finite time"""
    return states**2


def piecewise_solution(flipped, u_left, u_right):
    """The double integrator on two intervals with a constant control on each, flown from x = (0, 1)"""
    mesh = RadauMesh([-1.0, 0.0, 1.0], [3, 3], flipped)
    t = mesh.state_points() + 1.0

    # Time spent in each interval
    late = np.maximum(t - 1.0, 0.0)
    early = np.minimum(t, 1.0)

    # The velocity at the end of the first interval
    v1 = 1.0 + u_left

    position = early + 0.5 * u_left * early**2 + v1 * late + 0.5 * u_right * late**2
    velocity = 1.0 + u_left * early + u_right * late

    states = np.column_stack([position, velocity])
    controls = np.concatenate([np.full(3, u_left), np.full(3, u_right)])[:, None]

    return MeshSolution(mesh, states, controls, 0.0, 2.0)


@pytest.mark.parametrize("even_points", [0, 21])
@pytest.mark.parametrize("flipped", [False, True])
def test_error_estimate_vanishes_when_the_polynomials_solve_the_dynamics(
    flipped, even_points
):
    mesh = RadauMesh([-1.0, -0.2, 0.5, 1.0], [2, 3, 4], flipped)

    t0 = 0.0
    tf = 3.0
    t = 0.5 * (mesh.state_points() + 1.0) * (tf - t0)

    # A constant acceleration gives a quadratic position and a linear velocity
    states = np.column_stack([1.0 + 2.0 * t + 0.75 * t**2, 2.0 + 1.5 * t])
    controls = np.full((mesh.num_collocation, 1), 1.5)
    solution = MeshSolution(mesh, states, controls, t0, tf)

    errors = phs.simulated_errors(
        solution, double_integrator, phs.PHSOptions(even_points=even_points)
    )

    assert np.all(errors < 1e-10)


def quadrature_solution(n, flipped):
    """The Radau solution of dy/dt = pi cos(pi t) from y(-1) = 0 on one interval"""
    rates = np.pi * np.cos(np.pi * radau_points(n, flipped))
    states = np.concatenate([[0.0], integration_matrix(n, flipped) @ rates])

    mesh = RadauMesh([-1.0, 1.0], [n], flipped)

    return MeshSolution(mesh, states[:, None], np.zeros((n, 1)), -1.0, 1.0)


def sine_rates(states, controls, t):
    """Rate of the sine solution"""
    return np.pi * np.cos(np.pi * np.atleast_1d(t))[:, None]


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("n", [6, 8, 10])
def test_error_estimate_is_the_gap_to_the_flights_from_both_ends(n, flipped):
    solution = quadrature_solution(n, flipped)

    estimate = phs.simulated_errors(solution, sine_rates, phs.PHSOptions())[0]

    # The forward flight is sin(pi t) and the backward flight adds the end value of the polynomial
    t = np.linspace(-1.0, 1.0, 4001)
    y = lagrange_matrix(support_points(n, flipped), t) @ solution.states[:, 0]
    gamma = 1.0 / (1.0 + np.abs(solution.states).max())

    forward = np.abs(np.sin(np.pi * t) - y).max()
    backward = np.abs(np.sin(np.pi * t) + solution.states[-1, 0] - y).max()
    true = gamma * max(forward, backward)

    # The flights carry the error of the ODE tolerance on top of the true gap
    assert 0.8 * true < estimate <= true + 1e-9


def test_p_refinement_adds_one_point_per_order_of_magnitude():
    opts = phs.PHSOptions(tolerance=1e-6)

    assert phs.p_refinement(4, 2e-6, opts) == 5
    assert phs.p_refinement(4, 5e-4, opts) == 7


def test_h_refinement_keeps_the_points_asked_for_in_subintervals_of_the_fewest_points():
    opts = phs.PHSOptions(min_points=3)

    assert phs.h_refinement(13, opts) == 5
    assert phs.h_refinement(4, opts) == 2
    assert phs.h_refinement(3, opts) == 2


def test_p_reduction_drops_one_point_per_delta_orders_of_magnitude():
    opts = phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=10)

    assert phs.p_reduction(10, 2e-12, opts) == 8
    assert phs.p_reduction(10, 1e-6, opts) == 10
    assert phs.p_reduction(10, 0.0, opts) == 2
    assert phs.p_reduction(3, 2e-15, opts) == 3
    assert phs.p_reduction(3, 3e-26, opts) == 2


def zero_solution(mesh):
    """A solution with zero states and controls on a mesh"""
    states = np.zeros((mesh.num_states, 1))
    controls = np.zeros((mesh.num_collocation, 1))

    return MeshSolution(mesh, states, controls, 0.0, 1.0)


def test_refine_raises_points_or_splits_above_the_tolerance_and_reduces_below_it():
    solution = zero_solution(RadauMesh.uniform(4, 4))

    errors = np.array([5e-5, 5e-4, 2e-9, 5e-6])
    opts = phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=6)

    result = phs.refine(solution, errors, None, opts)
    new_mesh = result[0]
    actions = result[1]

    # Interval 1 asks for seven points and splits into four subintervals of two
    assert new_mesh.degrees.tolist() == [6, 2, 2, 2, 2, 4, 5]
    assert np.allclose(
        new_mesh.points, [-1.0, -0.5, -0.375, -0.25, -0.125, 0.0, 0.5, 1.0]
    )
    assert actions == ["raise", "split", "keep", "raise"]


class TablePairError:
    """pair_error replaced by a table of the error of every pair"""

    def __init__(self, table):
        self.table = table

    def __call__(self, solution, k, dynamics, options, gamma):
        return self.table[k]


class ConstantPairError:
    """pair_error replaced by one error for every pair"""

    def __init__(self, value):
        self.value = value

    def __call__(self, solution, k, dynamics, options, gamma):
        return self.value


def test_neighbors_merge_in_order_of_increasing_error_once_per_pass(monkeypatch):
    solution = zero_solution(RadauMesh.uniform(4, 3))
    errors = np.full(4, 1e-9)

    monkeypatch.setattr(phs, "pair_error", TablePairError({0: 5e-7, 1: 1e-8, 2: 3e-7}))

    result = phs.refine(solution, errors, None, phs.PHSOptions(tolerance=1e-6))
    new_mesh = result[0]
    actions = result[1]

    # The middle pair has the smallest error and blocks both of its neighbors
    assert np.allclose(new_mesh.points, [-1.0, -0.5, 0.5, 1.0])
    assert new_mesh.degrees.tolist() == [3, 3, 3]
    assert actions == ["keep", "merge", "merge", "keep"]


def test_a_merged_interval_keeps_the_larger_point_count_and_a_failed_pair_does_not_merge(
    monkeypatch,
):
    solution = zero_solution(RadauMesh([-1.0, 0.0, 1.0], [3, 5]))
    errors = np.full(2, 1e-9)

    # A pair within the tolerance merges
    monkeypatch.setattr(phs, "pair_error", ConstantPairError(1e-8))

    result = phs.refine(solution, errors, None, phs.PHSOptions(tolerance=1e-6))
    merged = result[0]
    actions = result[1]

    assert merged.points.tolist() == [-1.0, 1.0] and merged.degrees.tolist() == [5]
    assert actions == ["merge", "merge"]

    # A pair above the tolerance does not
    monkeypatch.setattr(phs, "pair_error", ConstantPairError(2e-6))

    kept = phs.refine(solution, errors, None, phs.PHSOptions(tolerance=1e-6))[0]

    assert kept.points.tolist() == [-1.0, 0.0, 1.0]


@pytest.mark.parametrize("flipped", [False, True])
def test_flights_across_a_control_jump_forbid_the_merge(flipped):
    opts = phs.PHSOptions(tolerance=1e-6)

    # One control on both sides merges
    smooth = piecewise_solution(flipped, 0.5, 0.5)
    gamma = phs.scale_factors(smooth)

    assert phs.pair_error(smooth, 0, double_integrator, opts, gamma) < 1e-10

    errors = phs.simulated_errors(smooth, double_integrator, opts)
    merged = phs.refine(smooth, errors, double_integrator, opts)[0]

    assert merged.points.tolist() == [-1.0, 1.0] and merged.degrees.tolist() == [3]

    # Each side alone is exact but the carried control misses the other side
    jump = piecewise_solution(flipped, 0.0, 1.0)
    errors = phs.simulated_errors(jump, double_integrator, opts)

    assert np.all(errors < 1e-10)
    assert (
        phs.pair_error(jump, 0, double_integrator, opts, phs.scale_factors(jump)) > 0.1
    )
    assert phs.refine(jump, errors, double_integrator, opts)[0].points.tolist() == [
        -1.0,
        0.0,
        1.0,
    ]


def sine_solution(points, degrees, flipped):
    """One period of a sine with no control, exact at every state point of the mesh"""
    mesh = RadauMesh(points, degrees, flipped)
    t = mesh.state_points()

    states = np.sin(np.pi * t)[:, None]
    controls = np.zeros((mesh.num_collocation, 1))

    return MeshSolution(mesh, states, controls, -1.0, 1.0)


@pytest.mark.parametrize("flipped", [False, True])
def test_one_polynomial_that_holds_the_state_of_both_intervals_has_no_merge_error(
    flipped,
):
    smooth = piecewise_solution(flipped, 0.5, 0.5)

    assert (
        phs.merge_error(smooth, 0, phs.PHSOptions(), phs.scale_factors(smooth)) < 1e-12
    )


@pytest.mark.parametrize("flipped", [False, True])
def test_a_state_that_one_polynomial_cannot_follow_does_not_merge(flipped):
    opts = phs.PHSOptions(tolerance=1e-6)

    halves = sine_solution([-1.0, 0.0, 1.0], [8, 8], flipped)
    errors = phs.simulated_errors(halves, sine_rates, opts)
    gamma = phs.scale_factors(halves)

    # Each half and the flights across both meet the tolerance, so the flights alone would merge them
    assert np.all(errors < 1e-7)
    assert phs.pair_error(halves, 0, sine_rates, opts, gamma) < 1e-7

    # One polynomial of eight points over the period misses it by as much as the merged mesh does
    predicted = phs.merge_error(halves, 0, opts, gamma)
    actual = phs.simulated_errors(
        sine_solution([-1.0, 1.0], [8], flipped), sine_rates, opts
    )[0]

    assert predicted > 1e-4 and actual > 1e-4
    assert predicted == pytest.approx(actual, rel=0.1)

    # The halves stay as they are
    result = phs.refine(halves, errors, sine_rates, opts)
    kept = result[0]
    actions = result[1]

    assert kept.points.tolist() == [-1.0, 0.0, 1.0] and kept.degrees.tolist() == [8, 8]
    assert actions == ["keep", "keep"]


def blowup_solution():
    """Two intervals whose forward flight from y = 2 blows up halfway through the first"""
    mesh = RadauMesh([-1.0, 0.0, 1.0], [2, 2])

    states = np.array([2.0, 1.5, 1.0, 0.7, 0.5])[:, None]
    controls = np.zeros((mesh.num_collocation, 1))

    return MeshSolution(mesh, states, controls, 0.0, 2.0)


def test_a_failed_flight_stops_the_estimate_and_names_the_interval_and_direction():
    solution = blowup_solution()

    with pytest.raises(phs.FlightError, match="forward flight of interval 0"):
        phs.simulated_errors(solution, blowup, phs.PHSOptions())

    # The backward flights alone finish
    errors = phs.simulated_errors(
        solution, blowup, phs.PHSOptions(directions=("backward",))
    )

    assert np.all(np.isfinite(errors))


def test_a_pair_whose_flight_fails_cannot_merge():
    solution = blowup_solution()
    gamma = phs.scale_factors(solution)

    assert phs.pair_error(solution, 0, blowup, phs.PHSOptions(), gamma) == np.inf


class CountedDecay:
    """Decay rates under the control that fail the test when a flight takes too many steps"""

    def __init__(self, c):
        self.c = c
        self.calls = []

    def __call__(self, states, controls, t):
        self.calls.append(t)

        if len(self.calls) > 100000:
            raise AssertionError("the merge flight kept going past the tolerance")

        return -self.c * controls * states


def test_a_merge_flight_stops_at_the_first_step_past_the_tolerance():
    # A tenth degree control carried back across a neighbor nine times wider makes the decay stiff
    mesh = RadauMesh([-1.0, 0.8, 1.0], [3, 10])
    c = 1e-3

    right = np.exp(c * 0.1 * (1.0 - support_points(10) ** 9) / 9.0)
    states = np.concatenate([np.full(3, right[0]), right])[:, None]
    controls = np.concatenate([np.zeros(3), radau_points(10) ** 8])[:, None]
    solution = MeshSolution(mesh, states, controls, 0.0, 2.0)

    decay = CountedDecay(c)

    # The flight matches the narrow interval and leaves the tolerance soon after it
    opts = phs.PHSOptions(directions=("backward",))

    assert (
        phs.pair_error(solution, 0, decay, opts, phs.scale_factors(solution)) == np.inf
    )
    assert phs.simulated_errors(solution, decay, opts)[1] < 1e-9


def line_solution(mesh, t0, tf, parameters=None, converged=True):
    """A solution whose state is the line through the state points of a mesh"""
    tau = mesh.state_points()

    states = np.column_stack([tau])
    controls = np.zeros((mesh.num_collocation, 1))

    if parameters is None:
        return MeshSolution(mesh, states, controls, t0, tf, converged=converged)

    return MeshSolution(mesh, states, controls, t0, tf, parameters, converged=converged)


class LineSolve:
    """Solve of one phase that returns the line through every mesh and records its intervals"""

    def __init__(self):
        self.calls = []

    def __call__(self, mesh, previous):
        self.calls.append(mesh.num_intervals)

        return line_solution(mesh, 0.0, 1.0)


def wide_interval_errors(solution, dynamics, options):
    """Errors above the tolerance on the intervals wider than 0.6 only"""
    return np.where(solution.mesh.half_widths > 0.3, 1e-3, 1e-9)


def test_run_stops_when_every_interval_meets_the_tolerance(monkeypatch):
    solve = LineSolve()

    monkeypatch.setattr(phs, "simulated_errors", wide_interval_errors)

    loop = phs.PHSRefinement(
        phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=4), dynamics=None
    )
    history = loop.run(RadauMesh.uniform(2, 3), solve)

    assert solve.calls == [2, 6]
    assert history[-1].errors.max() <= 1e-6
    assert history[-1].smoothness is None


class PhaseLineSolve:
    """Solve of several phases that returns the line through every mesh and records their points"""

    def __init__(self):
        self.calls = []

    def __call__(self, meshes, previous):
        degrees = []
        solutions = []

        for p in range(len(meshes)):
            mesh = meshes[p]

            degrees.append(mesh.degrees.tolist())
            solutions.append(line_solution(mesh, float(p), p + 1.0))

        self.calls.append(degrees)

        return solutions


def second_phase_errors(solution, dynamics, options):
    """Errors above the tolerance on the wide intervals of the second phase only"""
    if solution.t0 == 0.0:
        return np.full(solution.mesh.num_intervals, 1e-9)

    return np.where(solution.mesh.half_widths > 0.3, 1e-3, 1e-9)


def test_run_on_phases_refines_only_the_phases_above_the_tolerance(monkeypatch):
    solve = PhaseLineSolve()

    # Only the second phase has intervals above the tolerance on its first mesh
    monkeypatch.setattr(phs, "simulated_errors", second_phase_errors)
    monkeypatch.setattr(phs, "pair_error", ConstantPairError(np.inf))

    loop = phs.PHSRefinement(
        phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=4), dynamics=None
    )
    history = loop.run([RadauMesh.uniform(2, 3), RadauMesh.uniform(2, 3)], solve)

    # The first phase keeps its points instead of dropping to the fewest
    assert solve.calls == [[[3, 3], [3, 3]], [[3, 3], [2] * 6]]
    assert len(history) == 2 and len(history[-1]) == 2

    # Every phase of the last mesh meets the tolerance
    largest = []

    for step in history[-1]:
        largest.append(step.errors.max())

    assert max(largest) <= 1e-6


class Forbidden:
    """A replacement that fails the test when it is called"""

    def __init__(self, message):
        self.message = message

    def __call__(self, *args):
        raise AssertionError(self.message)


def failed_solve(mesh, previous):
    """Solve of one phase that never converges"""
    return line_solution(mesh, 0.0, 1.0, {"s": 0.5}, converged=False)


def test_run_stops_at_a_failed_solve_without_flying_it(monkeypatch):
    monkeypatch.setattr(
        phs, "simulated_errors", Forbidden("a failed solve must not be flown")
    )

    history = phs.PHSRefinement(phs.PHSOptions(), dynamics=None).run(
        RadauMesh.uniform(2, 3), failed_solve
    )

    assert len(history) == 1
    assert np.all(np.isnan(history[0].errors))
    assert history[0].solution.parameters == {"s": 0.5}


def test_refine_without_reductions_keeps_the_points_of_the_intervals_within_the_tolerance(
    monkeypatch,
):
    solution = zero_solution(RadauMesh.uniform(4, 6))

    errors = np.array([5e-4, 1e-15, 1e-15, 5e-6])
    opts = phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=8)

    # With reductions the two quiet intervals merge
    monkeypatch.setattr(phs, "pair_error", ConstantPairError(1e-9))

    result = phs.refine(solution, errors, None, opts)
    reduced = result[0]
    actions = result[1]

    assert reduced.degrees.tolist() == [2, 2, 2, 2, 2, 6, 7]
    assert actions == ["split", "merge", "merge", "raise"]

    # Without them the two quiet intervals neither merge nor drop points, while the others refine as before
    monkeypatch.setattr(
        phs, "pair_error", Forbidden("no merge is tried without reductions")
    )

    result = phs.refine(solution, errors, None, opts, reduce=False)
    kept = result[0]
    actions = result[1]

    assert kept.degrees.tolist() == [2, 2, 2, 2, 2, 6, 6, 7]
    assert actions == ["split", "keep", "keep", "raise"]
    assert np.allclose(kept.points[-4:], [-0.5, 0.0, 0.5, 1.0])


class CyclingRefine:
    """A refinement that swaps between two meshes with reductions and gives a finer mesh without them"""

    def __init__(self, first, second, finer):
        self.first = first
        self.second = second
        self.finer = finer

    def __call__(
        self, solution, errors, dynamics, options, reduce=True, keep=None, cuts=None
    ):
        actions = ["raise"] * solution.mesh.num_intervals

        if not reduce:
            return self.finer, actions

        if phs.same_mesh(solution.mesh, self.first):
            return self.second, actions

        return self.first, actions


def zero_phase_solution(mesh, t0, tf):
    """A solution with zero states and controls between two times"""
    states = np.zeros((mesh.num_states, 1))
    controls = np.zeros((mesh.num_collocation, 1))

    return MeshSolution(mesh, states, controls, t0, tf)


class ZeroSolve:
    """Solve of one phase that returns zero states and records the intervals of every mesh"""

    def __init__(self):
        self.calls = []

    def __call__(self, mesh, previous):
        self.calls.append(mesh.num_intervals)

        return zero_phase_solution(mesh, 0.0, 1.0)


def quiet_on_four_intervals(solution, dynamics, options):
    """Errors within the tolerance on a mesh of four intervals only"""
    error = 1e-3
    if solution.mesh.num_intervals == 4:
        error = 1e-9

    return np.full(solution.mesh.num_intervals, error)


def test_run_refines_without_reductions_once_a_mesh_comes_back(monkeypatch):
    first = RadauMesh.uniform(2, 3)
    second = RadauMesh.uniform(3, 3)
    finer = RadauMesh.uniform(4, 4)

    solve = ZeroSolve()

    # Only the finer mesh meets the tolerance
    monkeypatch.setattr(phs, "simulated_errors", quiet_on_four_intervals)
    monkeypatch.setattr(phs, "refine", CyclingRefine(first, second, finer))

    loop = phs.PHSRefinement(
        phs.PHSOptions(tolerance=1e-6, max_iterations=10), dynamics=None
    )
    history = loop.run(first, solve)

    assert solve.calls == [2, 3, 4]
    assert history[-1].errors.max() <= 1e-6


class PhaseZeroSolve:
    """Solve of several phases that returns zero states and records the meshes it was given"""

    def __init__(self):
        self.calls = []
        self.given = []

    def __call__(self, meshes, previous):
        counts = []
        solutions = []

        for p in range(len(meshes)):
            mesh = meshes[p]

            counts.append(mesh.num_intervals)
            solutions.append(zero_phase_solution(mesh, float(p), p + 1.0))

        self.calls.append(counts)
        self.given.append(meshes)

        return solutions


def quiet_first_phase_or_four_intervals(solution, dynamics, options):
    """Errors within the tolerance on the first phase and on a mesh of four intervals"""
    error = 1e-3
    if solution.t0 == 0.0 or solution.mesh.num_intervals == 4:
        error = 1e-9

    return np.full(solution.mesh.num_intervals, error)


def test_run_on_phases_refines_without_reductions_only_the_phase_whose_mesh_comes_back(
    monkeypatch,
):
    first = RadauMesh.uniform(2, 3)
    second = RadauMesh.uniform(3, 3)
    finer = RadauMesh.uniform(4, 4)

    solve = PhaseZeroSolve()

    # The first phase meets the tolerance at once, while the second cycles until the finer mesh
    monkeypatch.setattr(phs, "simulated_errors", quiet_first_phase_or_four_intervals)
    monkeypatch.setattr(phs, "refine", CyclingRefine(first, second, finer))

    loop = phs.PHSRefinement(
        phs.PHSOptions(tolerance=1e-6, max_iterations=10), dynamics=None
    )
    meshes = [RadauMesh.uniform(5, 3), first]
    loop.run(meshes, solve)

    assert solve.calls == [[5, 2], [5, 3], [5, 4]]

    # A solve may keep the list it was given, so no list changes after its solve
    kept_counts = []

    for kept in solve.given:
        counts = []

        for mesh in kept:
            counts.append(mesh.num_intervals)

        kept_counts.append(counts)

    assert kept_counts == solve.calls


def parabola_solution(n, flipped):
    """The state t squared on one interval from -1 to 1, lowest between its points"""
    mesh = RadauMesh([-1.0, 1.0], [n], flipped)
    t = mesh.state_points()

    return MeshSolution(mesh, (t**2)[:, None], np.zeros((n, 1)), -1.0, 1.0)


class AtLeast:
    """A path constraint that keeps the first state at or above a bound"""

    def __init__(self, bound):
        self.bound = bound

    def __call__(self, states, controls, t):
        return self.bound - states[:, :1]


@pytest.mark.parametrize("flipped", [False, True])
def test_path_violations_find_a_dip_between_the_collocation_points(flipped):
    solution = parabola_solution(2, flipped)
    lowest = (radau_points(2, flipped) ** 2).min()
    opts = phs.PHSOptions(tolerance=1e-6)

    # A bound just under the lowest collocation point holds there but not between the points
    result = phs.path_violations(solution, AtLeast(0.99 * lowest), opts)
    violations = result[0]
    places = result[1]
    active = result[2]

    assert violations[0] == pytest.approx(0.99 * lowest)
    assert places[0] == pytest.approx(0.0)
    assert not active[0]

    # A bound on the lowest collocation point is active there
    active = phs.path_violations(solution, AtLeast(lowest), opts)[2]

    assert active[0]


def test_refine_neither_merges_nor_reduces_an_interval_where_a_constraint_is_active(
    monkeypatch,
):
    solution = zero_solution(RadauMesh.uniform(4, 4))

    errors = np.full(4, 1e-30)
    opts = phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=10)

    monkeypatch.setattr(phs, "pair_error", ConstantPairError(1e-8))

    assert phs.refine(solution, errors, None, opts)[0].points.tolist() == [
        -1.0,
        0.0,
        1.0,
    ]

    # The first interval keeps its four points and its neighbor merges with the next one instead
    result = phs.refine(
        solution, errors, None, opts, keep=np.array([True, False, False, False])
    )
    kept = result[0]
    actions = result[1]

    assert kept.points.tolist() == [-1.0, -0.5, 0.5, 1.0]
    assert kept.degrees.tolist() == [4, 4, 2]
    assert actions == ["keep", "merge", "merge", "lower"]


def test_refine_splits_an_interval_at_its_cut_into_halves_with_its_points():
    solution = zero_solution(RadauMesh.uniform(2, 4))

    errors = np.array([1e-3, 1e-3])
    opts = phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=10)

    result = phs.refine(solution, errors, None, opts, cuts=np.array([-0.3, np.nan]))
    new_mesh = result[0]
    actions = result[1]

    # The second interval has no cut and gets more points instead
    assert np.allclose(new_mesh.points, [-1.0, -0.3, 0.0, 1.0])
    assert new_mesh.degrees.tolist() == [4, 4, 7]
    assert actions == ["cut", "raise"]


class DipSolve:
    """Solve of one phase whose state dips below the bound in the middle of a single interval only"""

    def __init__(self):
        self.meshes = []

    def __call__(self, mesh, previous):
        self.meshes.append(mesh)

        tau = mesh.state_points()

        if mesh.num_intervals == 1:
            states = tau**2
        else:
            states = np.ones_like(tau)

        return MeshSolution(
            mesh, states[:, None], np.zeros((mesh.num_collocation, 1)), -1.0, 1.0
        )


def no_flight_errors(solution, dynamics, options):
    """No error from the flights on any interval"""
    return np.zeros(solution.mesh.num_intervals)


def test_run_splits_an_interval_where_its_constraint_slips_most(monkeypatch):
    solve = DipSolve()

    monkeypatch.setattr(phs, "simulated_errors", no_flight_errors)

    loop = phs.PHSRefinement(
        phs.PHSOptions(tolerance=1e-6), dynamics=None, constraints=AtLeast(0.5)
    )
    history = loop.run(RadauMesh([-1.0, 1.0], [2]), solve)

    assert solve.meshes[1].points.tolist() == [-1.0, 0.0, 1.0] and solve.meshes[
        1
    ].degrees.tolist() == [2, 2]
    assert history[0].errors[0] == pytest.approx(0.5)
    assert history[-1].errors.max() <= 1e-6


def bowl_solution(points, center, flipped=False):
    """The state (t - center) squared plus one half on a mesh of three point intervals over -1 to 1"""
    mesh = RadauMesh(points, [3] * (len(points) - 1), flipped)
    t = mesh.state_points()

    states = ((t - center) ** 2 + 0.5)[:, None]
    controls = np.zeros((mesh.num_collocation, 1))

    return MeshSolution(mesh, states, controls, -1.0, 1.0)


@pytest.mark.parametrize("flipped", [False, True])
def test_touch_points_are_interior_local_maxima_of_a_constraint_at_its_bound(flipped):
    opts = phs.PHSOptions(tolerance=1e-6)

    # The state comes down to the bound of one half at 0.3 and leaves it again
    solution = bowl_solution([-1.0, 1.0], 0.3, flipped)

    assert phs.touch_points(solution, AtLeast(0.5), opts) == [(pytest.approx(0.3), 0)]

    # A bound well below the lowest state is never touched
    assert phs.touch_points(solution, AtLeast(0.4), opts) == []

    # The lowest state at the start of the phase is no interior touch
    assert (
        phs.touch_points(bowl_solution([-1.0, 1.0], -1.0, flipped), AtLeast(0.5), opts)
        == []
    )


def test_a_touch_on_the_point_two_intervals_share_counts_once():
    solution = bowl_solution([-1.0, 0.3, 1.0], 0.3)

    assert phs.touch_points(solution, AtLeast(0.5), phs.PHSOptions(tolerance=1e-6)) == [
        (pytest.approx(0.3), 0)
    ]


class RecordingRefine:
    """A refinement that records whether it may reduce and gives one more interval each time"""

    def __init__(self):
        self.reduces = []

    def __call__(
        self, solution, errors, dynamics, options, reduce=True, keep=None, cuts=None
    ):
        self.reduces.append(reduce)

        new_mesh = RadauMesh.uniform(solution.mesh.num_intervals + 1, 3)
        actions = ["raise"] * solution.mesh.num_intervals

        return new_mesh, actions


def dipping_solve(mesh, previous):
    """A fake solve whose state dips below a bound of one half only on two intervals"""
    tau = mesh.state_points()

    if mesh.num_intervals == 2:
        states = tau**2
    else:
        states = np.ones_like(tau)

    return MeshSolution(
        mesh, states[:, None], np.zeros((mesh.num_collocation, 1)), -1.0, 1.0
    )


def test_run_only_gains_points_once_a_constraint_slips_between_the_points(monkeypatch):
    refine = RecordingRefine()

    monkeypatch.setattr(phs, "refine", refine)
    monkeypatch.setattr(phs, "simulated_errors", quiet_on_four_intervals)

    options = phs.PHSOptions(tolerance=1e-6, max_iterations=10)
    loop = phs.PHSRefinement(options, dynamics=None, constraints=AtLeast(0.5))
    loop.run(RadauMesh.uniform(1, 3), dipping_solve)

    # The first mesh may reduce, and every mesh after the dip on the second only gains points
    assert refine.reduces == [True, False, False]


def quiet_late_phase_or_four_intervals(solution, dynamics, options):
    """Errors within the tolerance on the phase that starts at 5 and on a mesh of four intervals"""
    error = 1e-3
    if solution.t0 == 5.0 or solution.mesh.num_intervals == 4:
        error = 1e-9

    return np.full(solution.mesh.num_intervals, error)


def flat_then_dipping_solve(meshes, previous):
    """Solve of two phases where the first holds its constraint and the second dips below it"""
    mesh = meshes[0]

    states = np.ones((mesh.num_states, 1))
    controls = np.zeros((mesh.num_collocation, 1))
    first = MeshSolution(mesh, states, controls, 5.0, 6.0)

    return [first, dipping_solve(meshes[1], None)]


def test_run_on_phases_only_gains_points_in_the_phase_whose_constraint_slips(
    monkeypatch,
):
    refine = RecordingRefine()

    monkeypatch.setattr(phs, "refine", refine)

    # The first phase holds its constraint and meets the tolerance, so only the second is refined
    monkeypatch.setattr(phs, "simulated_errors", quiet_late_phase_or_four_intervals)

    options = phs.PHSOptions(tolerance=1e-6, max_iterations=10)
    loop = phs.PHSRefinement(options, dynamics=None, constraints=AtLeast(0.5))
    loop.run(
        [RadauMesh.uniform(5, 3), RadauMesh.uniform(1, 3)], flat_then_dipping_solve
    )

    assert refine.reduces == [True, False, False]


def test_run_rejects_an_initial_mesh_above_the_most_points():
    solve = Forbidden("the mesh must be rejected before the first solve")

    loop = phs.PHSRefinement(phs.PHSOptions(max_points=4), dynamics=None)

    with pytest.raises(ValueError):
        loop.run(RadauMesh.uniform(2, 5), solve)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_points": 1},
        {"min_points": 5, "max_points": 4},
        {"tolerance": 0.0},
        {"max_iterations": 0},
        {"ode_tolerance": 0.0},
        {"even_points": -1},
        {"directions": ()},
        {"directions": ("sideways",)},
    ],
)
def test_options_reject_values_the_method_cannot_use(kwargs):
    with pytest.raises(ValueError):
        phs.PHSOptions(**kwargs)
