"""Tests of the hp mesh refinement against the equations of Liu, Hager, and Rao."""

import math

import numpy as np
import pytest

import amigo.trajectory.refinement as hp
from amigo.trajectory.mesh import (
    RadauMesh,
    integration_matrix,
    lagrange_matrix,
    radau_points,
    support_points,
)


def options(**kwargs):
    kwargs.setdefault("smoothness_threshold", 3.0)
    return hp.RefinementOptions(**kwargs)


def bound_error(c, q, h, n):
    """Error of Eq. 16 used as an equality."""
    return c * h**q / n ** (q - 2.5)


def double_integrator(states, controls, t):
    return np.column_stack([states[:, 1], controls[:, 0]])


@pytest.mark.parametrize("flipped", [False, True])
def test_error_estimate_vanishes_when_the_polynomials_solve_the_dynamics(flipped):
    mesh = RadauMesh([-1.0, -0.2, 0.5, 1.0], [2, 3, 4], flipped)
    t0, tf = 0.0, 3.0
    t = 0.5 * (mesh.state_points() + 1.0) * (tf - t0)
    states = np.column_stack([1.0 + 2.0 * t + 0.75 * t**2, 2.0 + 1.5 * t])
    controls = np.full((mesh.num_collocation, 1), 1.5)
    solution = hp.MeshSolution(mesh, states, controls, t0, tf)
    assert np.all(hp.relative_errors(solution, double_integrator) < 1e-13)


def quadrature_solution(n, flipped):
    """The Radau solution of dy/dtau = pi cos(pi tau) from y(-1) = 0 on one interval."""
    rates = np.pi * np.cos(np.pi * radau_points(n, flipped))
    states = np.concatenate([[0.0], integration_matrix(n, flipped) @ rates])
    return hp.MeshSolution(RadauMesh([-1.0, 1.0], [n], flipped), states[:, None], np.zeros((n, 1)), -1.0, 1.0)


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("n", [6, 8, 10])
def test_error_estimate_tracks_the_true_error_of_a_smooth_solution(n, flipped):
    solution = quadrature_solution(n, flipped)

    def rates(states, controls, t):
        return np.pi * np.cos(np.pi * t)

    estimate = hp.relative_errors(solution, rates)[0]

    # The true error at the same points with the same scaling as Eq. 14
    s_hat = support_points(n + 1, flipped)
    y = lagrange_matrix(support_points(n, flipped), s_hat) @ solution.states[:, 0]
    true = np.abs(y - np.sin(np.pi * s_hat)).max() / (1.0 + np.abs(y).max())
    assert 0.5 < estimate / true < 2.0


@pytest.mark.parametrize("q", [1.0, 2.0, 4.5])
def test_smoothness_recovers_q_after_an_h_refinement(q):
    previous = (0.4, 3, bound_error(2.0, q, 0.4, 3))
    current = (0.1, 3, bound_error(2.0, q, 0.1, 3))
    assert hp.smoothness(previous, current) == pytest.approx(q, rel=1e-12)


@pytest.mark.parametrize("q", [3.5, 6.0, 9.0])
def test_smoothness_recovers_q_after_a_p_refinement(q):
    previous = (0.4, 3, bound_error(2.0, q, 0.4, 3))
    current = (0.4, 7, bound_error(2.0, q, 0.4, 7))
    assert hp.smoothness(previous, current) == pytest.approx(q, rel=1e-12)


def test_smoothness_is_undetermined_on_an_unchanged_interval_or_a_zero_error():
    assert hp.smoothness((0.4, 3, 1e-3), (0.4, 3, 1e-4)) is None
    assert hp.smoothness((0.4, 3, 0.0), (0.1, 3, 1e-4)) is None


def test_smoothness_estimates_take_the_collocation_points_where_q_is_undetermined():
    old = RadauMesh([-1.0, 0.0, 1.0], [3, 3])
    new = RadauMesh([-1.0, -0.5, 0.0, 1.0], [3, 3, 3])
    old_errors = np.array([bound_error(1.0, 2.0, 1.0, 3), 1e-9])
    errors = np.array([bound_error(1.0, 2.0, 0.5, 3), bound_error(1.0, 2.0, 0.5, 3), 1e-9])
    q = hp.smoothness_estimates(new, errors, np.array([0, 0, -1]), (old, old_errors))
    assert q[0] == pytest.approx(2.0) and q[1] == pytest.approx(2.0)
    assert q[2] == 3.0
    assert np.array_equal(hp.smoothness_estimates(new, errors), [3.0, 3.0, 3.0])


def test_h_refinement_creates_ceil_r_subintervals_up_to_rmax():
    opts = options(tolerance=1e-6, max_subintervals=10)
    assert hp.h_refinement(50e-6, 3.0, opts) == math.ceil(50.0 ** (1.0 / 3.0))
    assert hp.h_refinement(1.5e-6, 3.0, opts) == 2
    assert hp.h_refinement(1.0, 1.0, opts) == 10
    assert hp.h_refinement(1e-3, -0.5, opts) == 10


def test_p_refinement_raises_the_points_or_splits_past_nmax():
    opts = options(tolerance=1e-6, max_points=10)
    assert hp.p_refinement(4, 2e-6, 4.5, opts) == (math.ceil(4 * 2.0 ** 0.5), 1)
    points = math.ceil(8 * 10.0 ** 0.5)
    assert hp.p_refinement(8, 1e-5, 4.5, opts) == (8, math.ceil(points / 8))


@pytest.mark.parametrize("center", [0.0, 1.0, -1.0])
def test_power_coefficients_reproduce_the_lagrange_basis(center):
    nodes = support_points(5)
    a = hp.power_coefficients(nodes, center)
    s = np.linspace(-1.0, 1.0, 23)
    powers = (s[:, None] - center) ** np.arange(len(nodes))[None, :]
    assert np.allclose(powers @ a, lagrange_matrix(nodes, s), atol=1e-11)


@pytest.mark.parametrize("flipped", [False, True])
def test_degree_reduction_drops_negligible_powers(flipped):
    s = support_points(6, flipped)
    quadratic = 1.0 - 2.0 * s + 0.5 * s**2
    linear = 3.0 + 0.25 * s
    values = np.column_stack([quadratic, linear])
    assert hp.reduced_degree(values, flipped, options(tolerance=1e-8)) == 2

    # A cubic term below the tolerance is dropped and one above it is kept
    small = quadratic + 1e-9 * s**3
    assert hp.reduced_degree(small, flipped, options(tolerance=1e-8)) == 2
    assert hp.reduced_degree(quadratic + 1e-7 * s**3, flipped, options(tolerance=1e-8)) == 3
    assert hp.reduced_degree(np.full(7, 2.0), flipped, options(tolerance=1e-8, min_points=1)) == 1


@pytest.mark.parametrize("flipped", [False, True])
def test_merge_bound_vanishes_for_one_polynomial_and_bounds_the_gap_otherwise(flipped):
    n = 3
    s = support_points(n, flipped)
    left_points = -0.6 + 0.4 * s
    right_points = 0.1 + 0.3 * s

    def cubic(tau):
        return 0.3 - tau + 0.7 * tau**2 + 0.2 * tau**3

    bound = hp.merge_bound(cubic(left_points), cubic(right_points), 0.4, 0.3, flipped)
    assert bound < 1e-12

    def other(tau):
        return cubic(tau) + 0.05 * (tau + 0.2) ** 2

    left = cubic(left_points)
    right = other(right_points)
    bound = hp.merge_bound(left, right, 0.4, 0.3, flipped)

    # The two polynomials meet at -0.2 and their gap over the smaller interval stays below the bound
    tau = np.linspace(-0.2, 0.4, 201)
    a = lagrange_matrix(left_points, tau) @ left
    b = lagrange_matrix(right_points, tau) @ right
    assert np.abs(a - b).max() <= bound + 1e-12
    assert bound > 1e-3


def test_refine_splits_reduces_and_merges():
    mesh = RadauMesh.uniform(4, 3)
    tau = mesh.state_points()
    states = np.column_stack([1.0 + tau - 0.5 * tau**2, 2.0 - tau])
    solution = hp.MeshSolution(mesh, states, np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)
    errors = np.array([1e-9, 5e-5, 1e-9, 1e-9])
    q = np.full(4, 3.0)
    opts = options(tolerance=1e-6, max_subintervals=10)

    new_mesh, parents, actions = hp.refine(solution, errors, q, opts)
    assert actions == ["lower", "split", "merge", "merge"]

    # Interval 1 splits into ceil of 50 to the one third equal parts
    count = math.ceil(50.0 ** (1.0 / 3.0))
    assert np.allclose(new_mesh.points[1 : 2 + count], np.linspace(-0.5, 0.0, count + 1))
    assert list(parents[1 : 1 + count]) == [1] * count

    # Interval 0 keeps its parent and the last two intervals merge with two points
    assert new_mesh.num_intervals == 1 + count + 1
    assert parents[0] == 0 and new_mesh.degrees[0] == 2
    assert parents[-1] == -1 and new_mesh.degrees[-1] == 2
    assert new_mesh.points[-2] == 0.0


def test_refine_uses_p_refinement_when_q_exceeds_q_threshold():
    mesh = RadauMesh.uniform(2, 4)
    states = np.zeros((mesh.num_states, 1))
    solution = hp.MeshSolution(mesh, states, np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)
    errors = np.array([2e-6, 0.0])
    q = np.array([4.5, 4.5])
    new_mesh, parents, actions = hp.refine(solution, errors, q, options(tolerance=1e-6))
    assert new_mesh.degrees[0] == math.ceil(4 * 2.0 ** 0.5)
    assert parents[0] == 0 and actions[0] == "raise"


def test_run_stops_when_every_interval_meets_the_tolerance(monkeypatch):
    calls = []

    def solve(mesh, previous):
        calls.append(mesh.num_intervals)
        tau = mesh.state_points()
        return hp.MeshSolution(mesh, np.column_stack([tau]), np.zeros((mesh.num_collocation, 1)), 0.0, 1.0)

    def fake_errors(solution, dynamics):
        mesh = solution.mesh
        return np.where(mesh.half_widths > 0.3, 1e-3, 1e-9)

    monkeypatch.setattr(hp, "relative_errors", fake_errors)
    loop = hp.HPRefinement(options(tolerance=1e-6, max_subintervals=2), dynamics=None)
    history = loop.run(RadauMesh.uniform(2, 3), solve)
    assert history[-1].errors.max() <= 1e-6
    assert calls[0] == 2 and len(history) == len(calls)


def test_run_stops_at_a_mesh_whose_solve_failed(monkeypatch):
    calls = []

    def solve(mesh, previous):
        calls.append(mesh.num_intervals)
        tau = mesh.state_points()
        states = np.column_stack([tau])
        return hp.MeshSolution(
            mesh, states, np.zeros((mesh.num_collocation, 1)), 0.0, 1.0, {"s": 0.5}, converged=False
        )

    monkeypatch.setattr(hp, "relative_errors", lambda solution, dynamics: np.full(solution.mesh.num_intervals, 1.0))
    history = hp.HPRefinement(options(), dynamics=None).run(RadauMesh.uniform(2, 3), solve)
    assert len(history) == 1 and calls == [2]
    assert not history[0].solution.converged
    assert history[0].solution.parameters == {"s": 0.5}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoothness_threshold": 2.0},
        {"smoothness_threshold": 3.0, "max_subintervals": 1},
        {"smoothness_threshold": 3.0, "min_points": 0},
        {"smoothness_threshold": 3.0, "min_points": 11, "max_points": 10},
        {"smoothness_threshold": 3.0, "tolerance": 0.0},
    ],
)
def test_options_reject_values_the_equations_cannot_use(kwargs):
    with pytest.raises(ValueError):
        hp.RefinementOptions(**kwargs)
