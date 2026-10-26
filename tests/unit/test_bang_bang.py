"""Tests of the switch estimates of Agamawi, Hager, and Rao."""

import numpy as np
import pytest

from amigo.trajectory import RadauMesh, split_at_switches, switch_estimates


@pytest.mark.parametrize("flipped", [False, True])
def test_a_sign_change_inside_an_interval_averages_the_two_midpoints(flipped):
    mesh = RadauMesh([-1.0, 0.0, 1.0], [4, 4], flipped)
    tau = mesh.collocation_points()

    # Sigma changes sign between points 1 and 2 while the control jumps most between points 2 and 3
    sigma = np.array([1.0, 0.5, -0.5, -1.0, -1.0, -1.0, -1.0, -1.0])
    control = np.array([0.0, 0.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0])
    switches, signs = switch_estimates(mesh, sigma, control)
    expected = 0.5 * (0.5 * (tau[1] + tau[2]) + 0.5 * (tau[2] + tau[3]))
    assert switches == pytest.approx([expected])
    assert list(signs) == [1.0, -1.0]


@pytest.mark.parametrize("flipped", [False, True])
def test_a_sign_change_between_intervals_switches_at_their_common_mesh_point(flipped):
    mesh = RadauMesh([-1.0, 0.25, 1.0], [3, 3], flipped)
    sigma = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    control = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    switches, signs = switch_estimates(mesh, sigma, control)
    assert switches == pytest.approx([0.25])
    assert list(signs) == [-1.0, 1.0]


def test_two_switches_in_one_interval_share_its_largest_control_jump():
    mesh = RadauMesh([-1.0, 1.0], [6], flipped=True)
    tau = mesh.collocation_points()
    sigma = np.array([1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    control = np.array([0.0, 1.0, 1.0, 0.2, 0.0, 0.0])
    switches, signs = switch_estimates(mesh, sigma, control)
    t_u = 0.5 * (tau[0] + tau[1])
    assert switches == pytest.approx([t_u, 0.5 * (0.5 * (tau[2] + tau[3]) + t_u)])
    assert list(signs) == [1.0, -1.0, 1.0]


def test_a_sign_change_where_the_control_does_not_jump_switches_at_the_midpoint_of_sigma():
    mesh = RadauMesh([-1.0, 1.0], [5], flipped=True)
    tau = mesh.collocation_points()

    # A coast whose switching function turns negative calls for a burn that its constant control does not show
    switches, signs = switch_estimates(mesh, [1.0, 0.5, -0.5, -1.0, 0.5], np.zeros(5))
    assert switches == pytest.approx([0.5 * (tau[1] + tau[2]), 0.5 * (tau[3] + tau[4])])
    assert list(signs) == [1.0, -1.0, 1.0]


def test_values_within_the_noise_count_as_zero_and_keep_the_sign_before_them():
    mesh = RadauMesh.uniform(2, 3, flipped=True)
    tau = mesh.collocation_points()
    sigma = np.array([1.0, -1e-6, 1.0, 2.0, -0.5, -1.0])

    # The flip of the second point is within the noise while the one in the second interval is not
    switches, signs = switch_estimates(mesh, sigma, np.zeros(6), noise=1e-4)
    assert switches == pytest.approx([0.5 * (tau[3] + tau[4])])
    assert list(signs) == [1.0, -1.0]
    assert len(switch_estimates(mesh, sigma, np.zeros(6))[0]) == 3


def test_a_zero_of_sigma_keeps_the_sign_before_it():
    mesh = RadauMesh.uniform(2, 3, flipped=True)
    switches, signs = switch_estimates(mesh, [0.0, 1.0, 0.0, 2.0, 1.0, 3.0], np.zeros(6))
    assert len(switches) == 0
    assert list(signs) == [1.0]


def test_only_the_intervals_that_hold_a_sign_change_split_into_equal_parts():
    mesh = RadauMesh([-1.0, -0.5, 0.5, 1.0], [3, 3, 3], flipped=True)

    # Sigma changes sign inside the second interval only
    split = split_at_switches(mesh, [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0], parts=4)
    assert split.points.tolist() == pytest.approx([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    assert split.degrees.tolist() == [3, 3, 3, 3, 3, 3]


def test_a_sign_change_across_a_mesh_point_splits_both_intervals():
    mesh = RadauMesh([-1.0, 0.0, 1.0], [2, 2], flipped=True)
    split = split_at_switches(mesh, [1.0, 1.0, -1.0, -1.0], parts=2)
    assert split.points.tolist() == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])


def test_a_mesh_without_sign_changes_beyond_the_noise_comes_back_as_it_is():
    mesh = RadauMesh.uniform(2, 3, flipped=True)
    assert split_at_switches(mesh, [1.0, -1e-6, 1.0, 2.0, 1.0, 3.0], noise=1e-4) is mesh


def test_a_vanishing_switching_function_or_a_wrong_size_is_rejected():
    mesh = RadauMesh.uniform(2, 3, flipped=True)
    with pytest.raises(ValueError):
        switch_estimates(mesh, np.zeros(6), np.zeros(6))
    with pytest.raises(ValueError):
        switch_estimates(mesh, np.ones(5), np.zeros(6))
