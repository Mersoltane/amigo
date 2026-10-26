"""Tests of the phs mesh refinement against the equations of Haman and Rao."""

import numpy as np
import pytest

import amigo.trajectory.phs_refinement as phs
from amigo.trajectory.mesh import RadauMesh, integration_matrix, lagrange_matrix, radau_points, support_points
from amigo.trajectory.refinement import MeshSolution


def double_integrator(states, controls, t):
    return np.column_stack([states[:, 1], controls[:, 0]])


def blowup(states, controls, t):
    return states**2


def piecewise_solution(flipped, u_left, u_right):
    """The double integrator on two intervals with a constant control on each, flown from x = (0, 1)."""
    mesh = RadauMesh([-1.0, 0.0, 1.0], [3, 3], flipped)
    t = mesh.state_points() + 1.0
    late = np.maximum(t - 1.0, 0.0)
    early = np.minimum(t, 1.0)
    v1 = 1.0 + u_left
    states = np.column_stack(
        [
            early + 0.5 * u_left * early**2 + v1 * late + 0.5 * u_right * late**2,
            1.0 + u_left * early + u_right * late,
        ]
    )
    controls = np.concatenate([np.full(3, u_left), np.full(3, u_right)])[:, None]
    return MeshSolution(mesh, states, controls, 0.0, 2.0)


@pytest.mark.parametrize("even_points", [0, 21])
@pytest.mark.parametrize("flipped", [False, True])
def test_error_estimate_vanishes_when_the_polynomials_solve_the_dynamics(flipped, even_points):
    mesh = RadauMesh([-1.0, -0.2, 0.5, 1.0], [2, 3, 4], flipped)
    t0, tf = 0.0, 3.0
    t = 0.5 * (mesh.state_points() + 1.0) * (tf - t0)
    states = np.column_stack([1.0 + 2.0 * t + 0.75 * t**2, 2.0 + 1.5 * t])
    controls = np.full((mesh.num_collocation, 1), 1.5)
    solution = MeshSolution(mesh, states, controls, t0, tf)
    errors = phs.simulated_errors(solution, double_integrator, phs.PHSOptions(even_points=even_points))
    assert np.all(errors < 1e-10)


def quadrature_solution(n, flipped):
    """The Radau solution of dy/dt = pi cos(pi t) from y(-1) = 0 on one interval."""
    rates = np.pi * np.cos(np.pi * radau_points(n, flipped))
    states = np.concatenate([[0.0], integration_matrix(n, flipped) @ rates])
    return MeshSolution(RadauMesh([-1.0, 1.0], [n], flipped), states[:, None], np.zeros((n, 1)), -1.0, 1.0)


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("n", [6, 8, 10])
def test_error_estimate_is_the_gap_to_the_flights_from_both_ends(n, flipped):
    solution = quadrature_solution(n, flipped)

    def rates(states, controls, t):
        return np.pi * np.cos(np.pi * np.atleast_1d(t))[:, None]

    estimate = phs.simulated_errors(solution, rates, phs.PHSOptions())[0]

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


def test_refine_raises_points_or_splits_above_the_tolerance_and_reduces_below_it():
    mesh = RadauMesh.uniform(4, 4)
    solution = MeshSolution(mesh, np.zeros((mesh.num_states, 1)), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)
    errors = np.array([5e-5, 5e-4, 2e-9, 5e-6])
    opts = phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=6)
    new_mesh, actions = phs.refine(solution, errors, None, opts)

    # Interval 1 asks for seven points and splits into four subintervals of two
    assert new_mesh.degrees.tolist() == [6, 2, 2, 2, 2, 4, 5]
    assert np.allclose(new_mesh.points, [-1.0, -0.5, -0.375, -0.25, -0.125, 0.0, 0.5, 1.0])
    assert actions == ["raise", "split", "keep", "raise"]


def test_neighbors_merge_in_order_of_increasing_error_once_per_pass(monkeypatch):
    mesh = RadauMesh.uniform(4, 3)
    solution = MeshSolution(mesh, np.zeros((mesh.num_states, 1)), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)
    errors = np.full(4, 1e-9)
    pair = {0: 5e-7, 1: 1e-8, 2: 3e-7}
    monkeypatch.setattr(phs, "pair_error", lambda solution, k, dynamics, options, gamma: pair[k])
    new_mesh, actions = phs.refine(solution, errors, None, phs.PHSOptions(tolerance=1e-6))

    # The middle pair has the smallest error and blocks both of its neighbors
    assert np.allclose(new_mesh.points, [-1.0, -0.5, 0.5, 1.0])
    assert new_mesh.degrees.tolist() == [3, 3, 3]
    assert actions == ["keep", "merge", "merge", "keep"]


def test_a_merged_interval_keeps_the_larger_point_count_and_a_failed_pair_does_not_merge(monkeypatch):
    mesh = RadauMesh([-1.0, 0.0, 1.0], [3, 5])
    solution = MeshSolution(mesh, np.zeros((mesh.num_states, 1)), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)
    errors = np.full(2, 1e-9)
    monkeypatch.setattr(phs, "pair_error", lambda *args: 1e-8)
    merged, actions = phs.refine(solution, errors, None, phs.PHSOptions(tolerance=1e-6))
    assert merged.points.tolist() == [-1.0, 1.0] and merged.degrees.tolist() == [5]
    assert actions == ["merge", "merge"]

    monkeypatch.setattr(phs, "pair_error", lambda *args: 2e-6)
    kept, _ = phs.refine(solution, errors, None, phs.PHSOptions(tolerance=1e-6))
    assert kept.points.tolist() == [-1.0, 0.0, 1.0]


@pytest.mark.parametrize("flipped", [False, True])
def test_flights_across_a_control_jump_forbid_the_merge(flipped):
    opts = phs.PHSOptions(tolerance=1e-6)
    smooth = piecewise_solution(flipped, 0.5, 0.5)
    gamma = phs.scale_factors(smooth)
    assert phs.pair_error(smooth, 0, double_integrator, opts, gamma) < 1e-10
    errors = phs.simulated_errors(smooth, double_integrator, opts)
    merged, _ = phs.refine(smooth, errors, double_integrator, opts)
    assert merged.points.tolist() == [-1.0, 1.0] and merged.degrees.tolist() == [3]

    # Each side alone is exact but the carried control misses the other side
    jump = piecewise_solution(flipped, 0.0, 1.0)
    errors = phs.simulated_errors(jump, double_integrator, opts)
    assert np.all(errors < 1e-10)
    assert phs.pair_error(jump, 0, double_integrator, opts, phs.scale_factors(jump)) > 0.1
    assert phs.refine(jump, errors, double_integrator, opts)[0].points.tolist() == [-1.0, 0.0, 1.0]


def sine_rates(states, controls, t):
    return np.pi * np.cos(np.pi * np.atleast_1d(t))[:, None]


def sine_solution(points, degrees, flipped):
    """One period of a sine with no control, exact at every state point of the mesh."""
    mesh = RadauMesh(points, degrees, flipped)
    t = mesh.state_points()
    return MeshSolution(mesh, np.sin(np.pi * t)[:, None], np.zeros((mesh.num_collocation, 1)), -1.0, 1.0)


@pytest.mark.parametrize("flipped", [False, True])
def test_one_polynomial_that_holds_the_state_of_both_intervals_has_no_merge_error(flipped):
    smooth = piecewise_solution(flipped, 0.5, 0.5)
    assert phs.merge_error(smooth, 0, phs.PHSOptions(), phs.scale_factors(smooth)) < 1e-12


@pytest.mark.parametrize("flipped", [False, True])
def test_a_state_that_one_polynomial_cannot_follow_does_not_merge(flipped):
    opts = phs.PHSOptions(tolerance=1e-6)
    halves = sine_solution([-1.0, 0.0, 1.0], [8, 8], flipped)
    errors = phs.simulated_errors(halves, sine_rates, opts)
    gamma = phs.scale_factors(halves)

    # Each half and the flights across both meet the tolerance, so Eq. 34 alone would merge them
    assert np.all(errors < 1e-7)
    assert phs.pair_error(halves, 0, sine_rates, opts, gamma) < 1e-7

    # One polynomial of eight points over the period misses it by as much as the merged mesh does
    predicted = phs.merge_error(halves, 0, opts, gamma)
    actual = phs.simulated_errors(sine_solution([-1.0, 1.0], [8], flipped), sine_rates, opts)[0]
    assert predicted > 1e-4 and actual > 1e-4
    assert predicted == pytest.approx(actual, rel=0.1)
    kept, actions = phs.refine(halves, errors, sine_rates, opts)
    assert kept.points.tolist() == [-1.0, 0.0, 1.0] and kept.degrees.tolist() == [8, 8]
    assert actions == ["keep", "keep"]


def blowup_solution():
    """Two intervals whose forward flight from y = 2 blows up halfway through the first."""
    mesh = RadauMesh([-1.0, 0.0, 1.0], [2, 2])
    states = np.array([2.0, 1.5, 1.0, 0.7, 0.5])[:, None]
    return MeshSolution(mesh, states, np.zeros((mesh.num_collocation, 1)), 0.0, 2.0)


def test_a_failed_flight_stops_the_estimate_and_names_the_interval_and_direction():
    solution = blowup_solution()
    with pytest.raises(phs.FlightError, match="forward flight of interval 0"):
        phs.simulated_errors(solution, blowup, phs.PHSOptions())

    # The backward flights alone finish
    errors = phs.simulated_errors(solution, blowup, phs.PHSOptions(directions=("backward",)))
    assert np.all(np.isfinite(errors))


def test_a_pair_whose_flight_fails_cannot_merge():
    solution = blowup_solution()
    gamma = phs.scale_factors(solution)
    assert phs.pair_error(solution, 0, blowup, phs.PHSOptions(), gamma) == np.inf


def test_a_merge_flight_stops_at_the_first_step_past_the_tolerance():
    # A tenth degree control carried back across a neighbor nine times wider makes the decay stiff
    mesh = RadauMesh([-1.0, 0.8, 1.0], [3, 10])
    c = 1e-3
    right = np.exp(c * 0.1 * (1.0 - support_points(10) ** 9) / 9.0)
    states = np.concatenate([np.full(3, right[0]), right])[:, None]
    controls = np.concatenate([np.zeros(3), radau_points(10) ** 8])[:, None]
    solution = MeshSolution(mesh, states, controls, 0.0, 2.0)
    calls = []

    def decay(states, controls, t):
        calls.append(t)
        if len(calls) > 100000:
            raise AssertionError("the merge flight kept going past the tolerance")
        return -c * controls * states

    # The flight matches the narrow interval and leaves the tolerance soon after it
    opts = phs.PHSOptions(directions=("backward",))
    assert phs.pair_error(solution, 0, decay, opts, phs.scale_factors(solution)) == np.inf
    assert phs.simulated_errors(solution, decay, opts)[1] < 1e-9


def test_run_stops_when_every_interval_meets_the_tolerance(monkeypatch):
    calls = []

    def solve(mesh, previous):
        calls.append(mesh.num_intervals)
        tau = mesh.state_points()
        return MeshSolution(mesh, np.column_stack([tau]), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)

    def fake_errors(solution, dynamics, options):
        return np.where(solution.mesh.half_widths > 0.3, 1e-3, 1e-9)

    monkeypatch.setattr(phs, "simulated_errors", fake_errors)
    loop = phs.PHSRefinement(phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=4), dynamics=None)
    history = loop.run(RadauMesh.uniform(2, 3), solve)
    assert calls == [2, 6]
    assert history[-1].errors.max() <= 1e-6
    assert history[-1].smoothness is None


def test_run_on_phases_refines_only_the_phases_above_the_tolerance(monkeypatch):
    calls = []

    def solve(meshes, previous):
        calls.append([m.degrees.tolist() for m in meshes])
        solutions = []
        for p, mesh in enumerate(meshes):
            tau = mesh.state_points()
            solutions.append(
                MeshSolution(mesh, np.column_stack([tau]), np.zeros((mesh.num_collocation, 1)), float(p), p + 1.0)
            )
        return solutions

    # Only the second phase has intervals above the tolerance on its first mesh
    def fake_errors(solution, dynamics, options):
        if solution.t0 == 0.0:
            return np.full(solution.mesh.num_intervals, 1e-9)
        return np.where(solution.mesh.half_widths > 0.3, 1e-3, 1e-9)

    monkeypatch.setattr(phs, "simulated_errors", fake_errors)
    monkeypatch.setattr(phs, "pair_error", lambda *args: np.inf)
    loop = phs.PHSRefinement(phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=4), dynamics=None)
    history = loop.run([RadauMesh.uniform(2, 3), RadauMesh.uniform(2, 3)], solve)

    # The first phase keeps its points instead of dropping to the fewest
    assert calls == [[[3, 3], [3, 3]], [[3, 3], [2] * 6]]
    assert len(history) == 2 and len(history[-1]) == 2
    assert max(step.errors.max() for step in history[-1]) <= 1e-6


def test_run_stops_at_a_failed_solve_without_flying_it(monkeypatch):
    def solve(mesh, previous):
        tau = mesh.state_points()
        return MeshSolution(
            mesh, np.column_stack([tau]), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0, {"s": 0.5}, converged=False
        )

    def unexpected(*args):
        raise AssertionError("a failed solve must not be flown")

    monkeypatch.setattr(phs, "simulated_errors", unexpected)
    history = phs.PHSRefinement(phs.PHSOptions(), dynamics=None).run(RadauMesh.uniform(2, 3), solve)
    assert len(history) == 1
    assert np.all(np.isnan(history[0].errors))
    assert history[0].solution.parameters == {"s": 0.5}


def test_refine_without_reductions_keeps_the_points_of_the_intervals_within_the_tolerance(monkeypatch):
    mesh = RadauMesh.uniform(4, 6)
    solution = MeshSolution(mesh, np.zeros((mesh.num_states, 1)), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)
    errors = np.array([5e-4, 1e-15, 1e-15, 5e-6])
    opts = phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=8)
    monkeypatch.setattr(phs, "pair_error", lambda *args: 1e-9)
    reduced, actions = phs.refine(solution, errors, None, opts)
    assert reduced.degrees.tolist() == [2, 2, 2, 2, 2, 6, 7]
    assert actions == ["split", "merge", "merge", "raise"]

    # The two quiet intervals neither merge nor drop points, while the others refine as before
    def unexpected(*args):
        raise AssertionError("no merge is tried without reductions")

    monkeypatch.setattr(phs, "pair_error", unexpected)
    kept, actions = phs.refine(solution, errors, None, opts, reduce=False)
    assert kept.degrees.tolist() == [2, 2, 2, 2, 2, 6, 6, 7]
    assert actions == ["split", "keep", "keep", "raise"]
    assert np.allclose(kept.points[-4:], [-0.5, 0.0, 0.5, 1.0])


def cycling_refine(first, second, finer):
    """A refinement that swaps between two meshes with reductions and gives a finer mesh without them."""

    def refine(solution, errors, dynamics, options, reduce=True, keep=None, cuts=None):
        actions = ["raise"] * solution.mesh.num_intervals
        if not reduce:
            return finer, actions
        return (second if phs.same_mesh(solution.mesh, first) else first), actions

    return refine


def test_run_refines_without_reductions_once_a_mesh_comes_back(monkeypatch):
    first, second, finer = RadauMesh.uniform(2, 3), RadauMesh.uniform(3, 3), RadauMesh.uniform(4, 4)
    calls = []

    def solve(mesh, previous):
        calls.append(mesh.num_intervals)
        return MeshSolution(mesh, np.zeros((mesh.num_states, 1)), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)

    # Only the finer mesh meets the tolerance
    def fake_errors(solution, dynamics, options):
        return np.full(solution.mesh.num_intervals, 1e-9 if solution.mesh.num_intervals == 4 else 1e-3)

    monkeypatch.setattr(phs, "simulated_errors", fake_errors)
    monkeypatch.setattr(phs, "refine", cycling_refine(first, second, finer))
    loop = phs.PHSRefinement(phs.PHSOptions(tolerance=1e-6, max_iterations=10), dynamics=None)
    history = loop.run(first, solve)
    assert calls == [2, 3, 4]
    assert history[-1].errors.max() <= 1e-6


def test_run_on_phases_refines_without_reductions_only_the_phase_whose_mesh_comes_back(monkeypatch):
    first, second, finer = RadauMesh.uniform(2, 3), RadauMesh.uniform(3, 3), RadauMesh.uniform(4, 4)
    calls = []

    given = []

    def solve(meshes, previous):
        calls.append([m.num_intervals for m in meshes])
        given.append(meshes)
        return [
            MeshSolution(m, np.zeros((m.num_states, 1)), np.zeros((m.num_collocation, 1)), float(p), p + 1.0)
            for p, m in enumerate(meshes)
        ]

    # The first phase meets the tolerance at once, while the second cycles until the finer mesh
    def fake_errors(solution, dynamics, options):
        quiet = solution.t0 == 0.0 or solution.mesh.num_intervals == 4
        return np.full(solution.mesh.num_intervals, 1e-9 if quiet else 1e-3)

    monkeypatch.setattr(phs, "simulated_errors", fake_errors)
    monkeypatch.setattr(phs, "refine", cycling_refine(first, second, finer))
    loop = phs.PHSRefinement(phs.PHSOptions(tolerance=1e-6, max_iterations=10), dynamics=None)
    meshes = [RadauMesh.uniform(5, 3), first]
    loop.run(meshes, solve)
    assert calls == [[5, 2], [5, 3], [5, 4]]

    # A solve may keep the list it was given, so no list changes after its solve
    assert [[m.num_intervals for m in kept] for kept in given] == calls


def parabola_solution(n, flipped):
    """The state t squared on one interval from -1 to 1, lowest between its points."""
    mesh = RadauMesh([-1.0, 1.0], [n], flipped)
    t = mesh.state_points()
    return MeshSolution(mesh, (t**2)[:, None], np.zeros((n, 1)), -1.0, 1.0)


def at_least(bound):
    """A path constraint that keeps the first state at or above a bound."""
    return lambda states, controls, t: bound - states[:, :1]


@pytest.mark.parametrize("flipped", [False, True])
def test_path_violations_find_a_dip_between_the_collocation_points(flipped):
    solution = parabola_solution(2, flipped)
    lowest = (radau_points(2, flipped) ** 2).min()
    opts = phs.PHSOptions(tolerance=1e-6)

    # A bound just under the lowest collocation point holds there but not between the points
    violations, places, active = phs.path_violations(solution, at_least(0.99 * lowest), opts)
    assert violations[0] == pytest.approx(0.99 * lowest)
    assert places[0] == pytest.approx(0.0)
    assert not active[0]

    # A bound on the lowest collocation point is active there
    _, _, active = phs.path_violations(solution, at_least(lowest), opts)
    assert active[0]


def test_refine_neither_merges_nor_reduces_an_interval_where_a_constraint_is_active(monkeypatch):
    mesh = RadauMesh.uniform(4, 4)
    solution = MeshSolution(mesh, np.zeros((mesh.num_states, 1)), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)
    errors = np.full(4, 1e-30)
    opts = phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=10)
    monkeypatch.setattr(phs, "pair_error", lambda *args: 1e-8)
    assert phs.refine(solution, errors, None, opts)[0].points.tolist() == [-1.0, 0.0, 1.0]

    # The first interval keeps its four points and its neighbor merges with the next one instead
    kept, actions = phs.refine(solution, errors, None, opts, keep=np.array([True, False, False, False]))
    assert kept.points.tolist() == [-1.0, -0.5, 0.5, 1.0]
    assert kept.degrees.tolist() == [4, 4, 2]
    assert actions == ["keep", "merge", "merge", "lower"]


def test_refine_splits_an_interval_at_its_cut_into_halves_with_its_points():
    mesh = RadauMesh.uniform(2, 4)
    solution = MeshSolution(mesh, np.zeros((mesh.num_states, 1)), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)
    errors = np.array([1e-3, 1e-3])
    opts = phs.PHSOptions(tolerance=1e-6, min_points=2, max_points=10)
    new_mesh, actions = phs.refine(solution, errors, None, opts, cuts=np.array([-0.3, np.nan]))

    # The second interval has no cut and gets the points of Eq. 27
    assert np.allclose(new_mesh.points, [-1.0, -0.3, 0.0, 1.0])
    assert new_mesh.degrees.tolist() == [4, 4, 7]
    assert actions == ["cut", "raise"]


def test_run_splits_an_interval_where_its_constraint_slips_most(monkeypatch):
    meshes = []

    # The fake solve dips below the bound in the middle of a single interval only
    def solve(mesh, previous):
        meshes.append(mesh)
        tau = mesh.state_points()
        states = tau**2 if mesh.num_intervals == 1 else np.ones_like(tau)
        return MeshSolution(mesh, states[:, None], np.zeros((mesh.num_collocation, 1)), -1.0, 1.0)

    def no_flight_errors(solution, dynamics, options):
        return np.zeros(solution.mesh.num_intervals)

    monkeypatch.setattr(phs, "simulated_errors", no_flight_errors)
    loop = phs.PHSRefinement(phs.PHSOptions(tolerance=1e-6), dynamics=None, constraints=at_least(0.5))
    history = loop.run(RadauMesh([-1.0, 1.0], [2]), solve)
    assert meshes[1].points.tolist() == [-1.0, 0.0, 1.0] and meshes[1].degrees.tolist() == [2, 2]
    assert history[0].errors[0] == pytest.approx(0.5)
    assert history[-1].errors.max() <= 1e-6


def bowl_solution(points, center, flipped=False):
    """The state (t - center) squared plus one half on a mesh of three point intervals over -1 to 1."""
    mesh = RadauMesh(points, [3] * (len(points) - 1), flipped)
    t = mesh.state_points()
    return MeshSolution(mesh, ((t - center) ** 2 + 0.5)[:, None], np.zeros((mesh.num_collocation, 1)), -1.0, 1.0)


@pytest.mark.parametrize("flipped", [False, True])
def test_touch_points_are_interior_local_maxima_of_a_constraint_at_its_bound(flipped):
    opts = phs.PHSOptions(tolerance=1e-6)

    # The state comes down to the bound of one half at 0.3 and leaves it again
    solution = bowl_solution([-1.0, 1.0], 0.3, flipped)
    assert phs.touch_points(solution, at_least(0.5), opts) == [(pytest.approx(0.3), 0)]

    # A bound well below the lowest state is never touched
    assert phs.touch_points(solution, at_least(0.4), opts) == []

    # The lowest state at the start of the phase is no interior touch
    assert phs.touch_points(bowl_solution([-1.0, 1.0], -1.0, flipped), at_least(0.5), opts) == []


def test_a_touch_on_the_point_two_intervals_share_counts_once():
    solution = bowl_solution([-1.0, 0.3, 1.0], 0.3)
    assert phs.touch_points(solution, at_least(0.5), phs.PHSOptions(tolerance=1e-6)) == [(pytest.approx(0.3), 0)]


def recording_refine(reduces):
    """A refinement that records whether it may reduce and gives one more interval each time."""

    def refine(solution, errors, dynamics, options, reduce=True, keep=None, cuts=None):
        reduces.append(reduce)
        return RadauMesh.uniform(solution.mesh.num_intervals + 1, 3), ["raise"] * solution.mesh.num_intervals

    return refine


def dipping_solve(mesh, previous):
    """A fake solve whose state dips below a bound of one half only on two intervals."""
    tau = mesh.state_points()
    states = tau**2 if mesh.num_intervals == 2 else np.ones_like(tau)
    return MeshSolution(mesh, states[:, None], np.zeros((mesh.num_collocation, 1)), -1.0, 1.0)


def test_run_only_gains_points_once_a_constraint_slips_between_the_points(monkeypatch):
    reduces = []
    monkeypatch.setattr(phs, "refine", recording_refine(reduces))

    def fake_errors(solution, dynamics, options):
        return np.full(solution.mesh.num_intervals, 1e-9 if solution.mesh.num_intervals == 4 else 1e-3)

    monkeypatch.setattr(phs, "simulated_errors", fake_errors)
    loop = phs.PHSRefinement(phs.PHSOptions(tolerance=1e-6, max_iterations=10), dynamics=None, constraints=at_least(0.5))
    loop.run(RadauMesh.uniform(1, 3), dipping_solve)

    # The first mesh may reduce, and every mesh after the dip on the second only gains points
    assert reduces == [True, False, False]


def test_run_on_phases_only_gains_points_in_the_phase_whose_constraint_slips(monkeypatch):
    reduces = []
    monkeypatch.setattr(phs, "refine", recording_refine(reduces))

    def fake_errors(solution, dynamics, options):
        quiet = solution.t0 == 5.0 or solution.mesh.num_intervals == 4
        return np.full(solution.mesh.num_intervals, 1e-9 if quiet else 1e-3)

    # The first phase holds its constraint and meets the tolerance, so only the second is refined
    def solve(meshes, previous):
        mesh = meshes[0]
        first = MeshSolution(mesh, np.ones((mesh.num_states, 1)), np.zeros((mesh.num_collocation, 1)), 5.0, 6.0)
        return [first, dipping_solve(meshes[1], None)]

    monkeypatch.setattr(phs, "simulated_errors", fake_errors)
    loop = phs.PHSRefinement(phs.PHSOptions(tolerance=1e-6, max_iterations=10), dynamics=None, constraints=at_least(0.5))
    loop.run([RadauMesh.uniform(5, 3), RadauMesh.uniform(1, 3)], solve)
    assert reduces == [True, False, False]


def test_run_rejects_an_initial_mesh_above_the_most_points():
    def solve(mesh, previous):
        raise AssertionError("the mesh must be rejected before the first solve")

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
