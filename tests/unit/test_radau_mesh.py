"""Tests for the Radau points, their matrices, and the hp mesh layout."""

import numpy as np
import pytest

from amigo.trajectory.mesh import (
    RadauMesh,
    carry_densities,
    differentiation_matrix,
    integration_matrix,
    lagrange_matrix,
    mesh_time,
    radau_points,
    radau_weights,
    support_points,
)

DEGREES = range(1, 11)
CONVENTIONS = [False, True]


def poly(c, x):
    """Values and derivatives of the polynomial with increasing coefficients c."""
    p = np.polynomial.Polynomial(c)
    return p(x), p.deriv()(x)


@pytest.mark.parametrize("n", DEGREES)
def test_standard_points_hold_minus_one_and_flipped_points_mirror_them(n):
    x = radau_points(n)
    assert len(x) == n and x[0] == -1.0
    assert np.all(np.diff(x) > 0.0) and x[-1] < 1.0
    assert np.allclose(radau_points(n, flipped=True), -x[::-1], atol=0.0, rtol=0.0)


def test_points_match_closed_forms():
    assert np.allclose(radau_points(2), [-1.0, 1.0 / 3.0], atol=1e-15)
    r6 = np.sqrt(6.0)
    assert np.allclose(radau_points(3), [-1.0, (1.0 - r6) / 5.0, (1.0 + r6) / 5.0], atol=1e-15)


@pytest.mark.parametrize("flipped", CONVENTIONS)
@pytest.mark.parametrize("n", DEGREES)
def test_weights_integrate_polynomials_of_degree_two_n_minus_two(n, flipped):
    x = radau_points(n, flipped)
    w = radau_weights(n, flipped)
    for k in range(2 * n - 1):
        exact = (1.0 - (-1.0) ** (k + 1)) / (k + 1)
        assert abs(w @ x**k - exact) < 1e-13


@pytest.mark.parametrize("flipped", CONVENTIONS)
@pytest.mark.parametrize("n", DEGREES)
def test_differentiation_matrix_is_exact_for_the_state_degree(n, flipped):
    z = support_points(n, flipped)
    x = radau_points(n, flipped)
    D = differentiation_matrix(n, flipped)
    assert D.shape == (n, n + 1)
    rng = np.random.default_rng(n)
    c = rng.standard_normal(n + 1)
    values, _ = poly(c, z)
    _, rates = poly(c, x)
    assert np.allclose(D @ values, rates, atol=1e-11)


@pytest.mark.parametrize("flipped", CONVENTIONS)
@pytest.mark.parametrize("n", DEGREES)
def test_integration_matrix_inverts_the_differentiation_matrix(n, flipped):
    I = integration_matrix(n, flipped)
    D = differentiation_matrix(n, flipped)
    assert np.allclose(I @ D[:, 1:], np.eye(n), atol=1e-11)

    # The state rise from the interval start follows from the rates
    z = support_points(n, flipped)
    x = radau_points(n, flipped)
    c = np.random.default_rng(n + 20).standard_normal(n + 1)
    values, _ = poly(c, z)
    _, rates = poly(c, x)
    assert np.allclose(I @ rates, values[1:] - values[0], atol=1e-12)


def test_lagrange_matrix_reproduces_polynomials_and_nodes():
    nodes = support_points(6)
    x = np.linspace(-1.0, 1.0, 37)
    c = np.random.default_rng(3).standard_normal(7)
    values, _ = poly(c, nodes)
    exact, _ = poly(c, x)
    assert np.allclose(lagrange_matrix(nodes, x) @ values, exact, atol=1e-12)
    assert np.array_equal(lagrange_matrix(nodes, nodes), np.eye(len(nodes)))


@pytest.mark.parametrize("flipped", CONVENTIONS)
def test_mesh_layout_shares_the_state_at_mesh_points(flipped):
    mesh = RadauMesh([-1.0, -0.4, 0.1, 1.0], [3, 5, 2], flipped)
    assert mesh.num_collocation == 10 and mesh.num_states == 11

    tau = mesh.state_points()
    assert tau[0] == -1.0 and tau[-1] == 1.0
    assert np.all(np.diff(tau) > 0.0)
    for k in range(mesh.num_intervals):
        ends = tau[mesh.state_indices(k)][[0, -1]]
        assert np.allclose(ends, mesh.points[k : k + 2], atol=1e-15)

    coll = mesh.collocation_points()
    assert np.allclose(tau[mesh.collocated_states()], coll, atol=1e-15)


@pytest.mark.parametrize("flipped", CONVENTIONS)
def test_interpolation_reproduces_piecewise_polynomials(flipped):
    mesh = RadauMesh([-1.0, -0.3, 0.5, 1.0], [4, 3, 5], flipped)
    c = np.array([0.4, -1.1, 0.7, 0.3])
    states, _ = poly(c, mesh.state_points())
    tau = np.linspace(-1.0, 1.0, 101)
    exact, _ = poly(c, tau)
    assert np.allclose(mesh.interpolate_states(states[:, None], tau)[:, 0], exact, atol=1e-12)

    # The control polynomial has one degree less than the state polynomial
    c = c[:2]
    controls, _ = poly(c, mesh.collocation_points())
    exact, _ = poly(c, tau)
    assert np.allclose(mesh.interpolate_controls(controls, tau), exact, atol=1e-12)


def test_mesh_time_maps_the_ends_to_the_phase_times():
    assert np.allclose(mesh_time([-1.0, 0.0, 1.0], 2.0, 6.0), [2.0, 4.0, 6.0])


@pytest.mark.parametrize("flipped", CONVENTIONS)
def test_quadrature_weights_integrate_over_the_whole_mesh(flipped):
    mesh = RadauMesh([-1.0, -0.3, 0.5, 1.0], [4, 3, 5], flipped)
    w = mesh.quadrature_weights()
    tau = mesh.collocation_points()
    assert w.sum() == pytest.approx(2.0, rel=1e-13)
    assert w @ tau**3 == pytest.approx(0.0, abs=1e-13)


@pytest.mark.parametrize("flipped", CONVENTIONS)
def test_carried_densities_are_exact_for_a_polynomial_density(flipped):
    old = RadauMesh([-1.0, -0.3, 0.5, 1.0], [4, 3, 5], flipped)
    new = RadauMesh([-1.0, -0.6, 0.1, 0.4, 1.0], [6, 3, 4, 3], flipped)
    c = np.array([0.4, -1.1, 0.7])
    old_density, _ = poly(c, old.collocation_points())
    new_density, _ = poly(c, new.collocation_points())

    # Two columns of multipliers that scale with the quadrature weights
    values = old.quadrature_weights()[:, None] * np.column_stack([old_density, 2.0 * old_density])
    carried = carry_densities(old, new, values, old.quadrature_weights(), new.quadrature_weights())
    expected = new.quadrature_weights()[:, None] * np.column_stack([new_density, 2.0 * new_density])
    assert np.allclose(carried, expected, atol=1e-12)


@pytest.mark.parametrize(
    "points, degrees",
    [
        ([-1.0, 0.0, 1.0], [3]),
        ([-1.0, 0.0, 0.9], [3, 3]),
        ([-1.0, 0.2, 0.1, 1.0], [3, 3, 3]),
        ([-1.0, 0.0, 1.0], [3, 0]),
    ],
)
def test_mesh_rejects_inconsistent_layouts(points, degrees):
    with pytest.raises(ValueError):
        RadauMesh(points, degrees)
