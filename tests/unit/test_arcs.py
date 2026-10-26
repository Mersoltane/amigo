"""Tests of solutions held as consecutive arcs that meet at switches."""

import numpy as np
import pytest

from amigo.trajectory import (
    MeshIteration,
    MeshSolution,
    RadauMesh,
    best_iteration,
    combine_arcs,
    insert_arcs,
    merge_short_arcs,
    mesh_time,
    sign_agreement,
    split_arcs,
    stretch_arcs,
    sub_mesh,
    switch_brackets,
    switch_estimates,
    switching_arcs,
    uniform_arc_meshes,
)


def solution_on(mesh, t0=0.5, tf=3.5, seed=0):
    """A solution with distinct states and controls at every point of a mesh."""
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((mesh.num_states, 2))
    controls = rng.standard_normal((mesh.num_collocation, 1))
    return MeshSolution(mesh, states, controls, t0, tf, {"p": 1.0})


@pytest.mark.parametrize("flipped", [False, True])
def test_splitting_at_mesh_points_and_combining_gives_the_solution_back(flipped):
    mesh = RadauMesh([-1.0, -0.6, -0.1, 0.3, 1.0], [3, 2, 4, 3], flipped)
    solution = solution_on(mesh)
    times = mesh_time(mesh.points, solution.t0, solution.tf)
    pieces = split_arcs(solution, [times[0], times[1], times[3], times[4]])

    # The arcs hold one, two, and one of the intervals with their own end times
    assert [p.mesh.num_intervals for p in pieces] == [1, 2, 1]
    assert [p.t0 for p in pieces] == pytest.approx([times[0], times[1], times[3]])
    assert pieces[1].mesh.points == pytest.approx([-1.0, 2.0 * (times[2] - times[1]) / (times[3] - times[1]) - 1.0, 1.0])

    back = combine_arcs(pieces)
    assert back.mesh.flipped == flipped
    assert back.mesh.points == pytest.approx(mesh.points, abs=1e-15)
    assert back.mesh.degrees.tolist() == mesh.degrees.tolist()
    assert np.array_equal(back.states, solution.states)
    assert np.array_equal(back.controls, solution.controls)
    assert (back.t0, back.tf) == (solution.t0, solution.tf)
    assert back.parameters == solution.parameters


def test_combining_arcs_and_splitting_at_their_ends_gives_the_arcs_back():
    arcs = [
        solution_on(RadauMesh([-1.0, 0.2, 1.0], [3, 4], True), 0.0, 1.0, 1),
        solution_on(RadauMesh([-1.0, 1.0], [5], True), 1.0, 1.4, 2),
        solution_on(RadauMesh([-1.0, -0.5, 0.0, 1.0], [2, 3, 2], True), 1.4, 3.0, 3),
    ]

    # Consecutive arcs share the state at their switch
    for a, b in zip(arcs[:-1], arcs[1:]):
        b.states[0] = a.states[-1]
    pieces = split_arcs(combine_arcs(arcs), [0.0, 1.0, 1.4, 3.0])
    for piece, arc in zip(pieces, arcs):
        assert piece.mesh.points == pytest.approx(arc.mesh.points, abs=1e-14)
        assert np.array_equal(piece.states, arc.states)
        assert np.array_equal(piece.controls, arc.controls)
        assert (piece.t0, piece.tf) == pytest.approx((arc.t0, arc.tf))


def test_splitting_needs_increasing_edges_at_mesh_points():
    solution = solution_on(RadauMesh([-1.0, 0.0, 1.0], [3, 3], True), 0.0, 2.0)
    with pytest.raises(ValueError):
        split_arcs(solution, [0.0, 0.5, 2.0])
    with pytest.raises(ValueError):
        split_arcs(solution, [0.0, 2.0, 1.0])


def test_stretching_scales_every_time_about_the_first_start():
    pieces = split_arcs(solution_on(RadauMesh([-1.0, 0.0, 1.0], [3, 3], True), 1.0, 3.0), [1.0, 2.0, 3.0])
    stretched = stretch_arcs(pieces, 5.0)
    assert [(s.t0, s.tf) for s in stretched] == pytest.approx([(1.0, 3.0), (3.0, 5.0)])
    for s, p in zip(stretched, pieces):
        assert s.states is p.states and s.controls is p.controls and s.mesh is p.mesh


@pytest.mark.parametrize("flipped", [False, True])
def test_the_arcs_of_a_switching_function_are_its_switches_in_time(flipped):
    mesh = RadauMesh([-1.0, -0.2, 0.6, 1.0], [4, 4, 4], flipped)
    solution = solution_on(mesh, 2.0, 6.0)
    tau = mesh.collocation_points()
    sigma = np.where((tau > -0.5) & (tau < 0.4), -1.0, 1.0) * (0.1 + np.abs(tau))
    control = (sigma < 0.0).astype(float)
    edges, signs = switching_arcs(solution, sigma, control)
    switches, expected = switch_estimates(mesh, sigma, control)
    assert edges == pytest.approx(mesh_time(np.concatenate([[-1.0], switches, [1.0]]), 2.0, 6.0))
    assert list(signs) == list(expected) == [1.0, -1.0, 1.0]


def test_switch_brackets_reach_halfway_to_the_next_switches_and_open_at_the_ends():
    assert switch_brackets([0.0, 4.0]) == []
    assert switch_brackets([0.0, 1.0, 4.0]) == [(-np.inf, np.inf)]
    assert switch_brackets([0.0, 1.0, 3.0, 4.0]) == [(-np.inf, 2.0), (2.0, np.inf)]
    assert switch_brackets([0.0, 1.0, 3.0, 4.0, 8.0]) == [(-np.inf, 2.0), (2.0, 3.5), (3.5, np.inf)]


def test_uniform_arc_meshes_share_the_intervals_by_duration_with_a_floor():
    meshes = uniform_arc_meshes([0.0, 5.0, 5.1, 10.0], intervals=20, degree=4, min_intervals=2)
    assert [m.num_intervals for m in meshes] == [10, 2, 10]
    assert all(m.degrees.tolist() == [4] * m.num_intervals and m.flipped for m in meshes)


def test_sign_agreement_counts_the_points_on_the_wrong_side_and_scales_the_worst():
    arcs = split_arcs(solution_on(RadauMesh([-1.0, 0.0, 1.0], [3, 3], True), 0.0, 2.0), [0.0, 1.0, 2.0])
    sigma = np.array([-2.0, -4.0, -1.0, 3.0, 8.0, 5.0])
    assert sign_agreement([-1.0, 1.0], arcs, sigma) == (1.0, 0.0)
    sigma[2] = 0.4
    agreement, worst = sign_agreement([-1.0, 1.0], arcs, sigma)
    assert agreement == pytest.approx(5.0 / 6.0)
    assert worst == pytest.approx(0.4 / 8.0)
    with pytest.raises(ValueError):
        sign_agreement([-1.0, 1.0], arcs, sigma[:5])


def test_a_sub_mesh_keeps_the_points_inside_and_the_degree_of_each_interval_it_cuts():
    mesh = RadauMesh([-1.0, -0.5, 0.0, 1.0], [4, 6, 8], True)
    part = sub_mesh(mesh, 0.0, 4.0, 1.5, 4.0)
    assert part.points == pytest.approx([-1.0, -0.6, 1.0])
    assert part.degrees.tolist() == [6, 8] and part.flipped

    # A point closer than the minimum width to the end drops out with the sliver it would leave
    part = sub_mesh(mesh, 0.0, 4.0, 0.0, 2.05, min_width=0.1)
    assert part.points == pytest.approx([-1.0, 2.0 / 2.05 - 1.0, 1.0])
    assert part.degrees.tolist() == [4, 6]


def test_a_short_arc_joins_its_neighbors_of_the_other_kind_or_goes_to_one_of_its_own():
    kinds = ["burn", "coast", "burn", "coast", "coast", "burn"]
    assert merge_short_arcs(kinds, [0.0, 1.0, 1.001, 2.0, 3.0, 4.0, 5.0], 1e-3) == (
        ["burn", "coast", "coast", "burn"], [0.0, 2.0, 3.0, 4.0, 5.0])

    # A short arc of a contact event goes to the arc of its kind, and short arcs at the ends go inward
    assert merge_short_arcs(["burn", "coast", "coast", "burn"], [0.0, 1.0, 2.0, 2.001, 3.0], 1e-3) == (
        ["burn", "coast", "burn"], [0.0, 1.0, 2.001, 3.0])
    assert merge_short_arcs(["coast", "burn"], [0.0, 0.001, 1.0], 1e-3) == (["burn"], [0.0, 1.0])
    assert merge_short_arcs(["burn", "coast"], [0.0, 1.0, 1.001], 1e-3) == (["burn"], [0.0, 1.001])


def solved_arcs(kinds, edges, intervals=8):
    """Solutions of consecutive arcs on uniform flipped meshes, which only their times matter for."""
    mesh = RadauMesh.uniform(intervals, 3, True)
    return [solution_on(mesh, a, b) for a, b in zip(edges[:-1], edges[1:])]


def switching(arcs, function):
    """A switching function at the collocation points of consecutive arcs."""
    return np.concatenate([function(mesh_time(s.mesh.collocation_points(), s.t0, s.tf)) for s in arcs])


def other(kind):
    return "coast" if kind == "burn" else "burn"


def test_arcs_of_the_other_kind_go_where_the_switching_function_has_the_wrong_sign():
    arcs = solved_arcs(["burn", "coast"], [0.0, 2.0, 4.0])

    # A hump above zero inside the burn asks for a coast, and a dip below zero at the start of the coast moves its switch
    def sigma(t):
        return np.where(t <= 2.0, np.exp(-((t - 1.0) / 0.3) ** 2) - 0.5, t - 2.5)

    kinds, edges = insert_arcs(["burn", "coast"], arcs, switching(arcs, sigma), [-1.0, 1.0], other, floor=1e-3)
    zero = 0.3 * np.sqrt(np.log(2.0))
    assert kinds == ["burn", "coast", "burn", "coast"]
    assert edges[0] == 0.0 and edges[-1] == 4.0
    assert edges[1:4] == pytest.approx([1.0 - zero, 1.0 + zero, 2.5], abs=0.02)


def test_arcs_the_switching_function_agrees_with_come_back_as_they_are_and_contact_events_stay():
    edges = [0.0, 1.0, 2.0, 3.0]
    arcs = solved_arcs(["burn", "coast", "coast"], edges)

    def sigma(t):
        return np.where(t <= 1.0, -1.0, 1.0)

    assert insert_arcs(["burn", "coast", "coast"], arcs, switching(arcs, sigma), [-1.0, 1.0, 1.0], other) == (
        ["burn", "coast", "coast"], edges)

    # A region within the noise or shorter than twice the floor asks for nothing
    def bumpy(t):
        return sigma(t) + np.where(np.abs(t - 0.5) < 0.05, 1.0 + 1e-6, 0.0)

    noisy = switching(arcs, lambda t: np.where(np.abs(t - 0.5) < 0.05, 1e-6, sigma(t)))
    assert insert_arcs(["burn", "coast", "coast"], arcs, noisy, [-1.0, 1.0, 1.0], other, noise=1e-4)[0] == [
        "burn", "coast", "coast"]
    assert insert_arcs(["burn", "coast", "coast"], arcs, switching(arcs, bumpy), [-1.0, 1.0, 1.0], other,
                       floor=0.5)[0] == ["burn", "coast", "coast"]

    # A coast region inside the burn leaves the contact event between the two coasts in place
    kinds, new = insert_arcs(["burn", "coast", "coast"], arcs, switching(arcs, bumpy), [-1.0, 1.0, 1.0], other)
    assert kinds == ["burn", "coast", "burn", "coast", "coast"]
    assert new[3:] == [1.0, 2.0, 3.0]


def iteration(error, converged=True):
    mesh = RadauMesh([-1.0, 1.0], [2], True)
    solution = MeshSolution(mesh, np.zeros((3, 1)), np.zeros((2, 1)), 0.0, 1.0, converged=converged)
    return MeshIteration(solution, np.array([error]))


def test_the_last_entry_stands_when_it_meets_the_tolerance_or_else_the_least_error():
    assert best_iteration([iteration(1e-3), iteration(1e-9)], 1e-8) == 1
    assert best_iteration([iteration(1e-3), iteration(2e-8), iteration(5e-8)], 1e-8) == 1
    assert best_iteration([iteration(1e-3), iteration(np.nan, converged=False)], 1e-8) == 0
    assert best_iteration([iteration(np.nan, converged=False)], 1e-8) == 0


def test_an_entry_of_phases_stands_by_its_largest_error():
    history = [[iteration(1e-9), iteration(3e-8)], [iteration(2e-8), iteration(2e-8)], [iteration(1e-7), iteration(1e-9)]]
    assert best_iteration(history, 1e-8) == 1
