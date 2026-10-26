"""Radau collocation of the soft lunar landing of Liu, Hager, and Rao against its exact solution."""

import os
import sys
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from amigo.trajectory import (
    HPRefinement,
    PHSOptions,
    PHSRefinement,
    RadauMesh,
    RefinementOptions,
    lagrange_matrix,
    mesh_time,
    radau_points,
)

EXAMPLE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "trajectory", "lunar_landing")
)
if EXAMPLE not in sys.path:
    sys.path.insert(0, EXAMPLE)
import lunar_landing as ll  # noqa: E402


@pytest.fixture(scope="module", params=[False, True], ids=["standard", "flipped"])
def solver(request, tmp_path_factory):
    flipped = request.param
    module = "lunar_landing_test_flipped" if flipped else "lunar_landing_test"
    catalog, _ = ll.build_model(ll.catalog_mesh(1, 10, flipped), module)

    # Build in a temporary directory so the repository stays untouched
    tmp_path = tmp_path_factory.mktemp(module)
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        catalog.build_module()
    finally:
        os.chdir(orig_dir)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    options = {"convergence_tolerance": 1e-10, "max_iterations": 500}
    return ll.LandingSolver(module, 5.0, options), flipped


def test_fine_uniform_mesh_matches_the_exact_solution(solver):
    solve, flipped = solver
    sol = solve(RadauMesh.uniform(40, 3, flipped), None)
    assert solve.reports[-1]["converged"]
    assert abs(sol.tf - ll.T_FINAL) < 1e-3
    assert abs(ll.cost(sol) - ll.COST) < 1e-3
    assert np.allclose(sol.states[0], ll.Y0, atol=1e-9)
    assert np.allclose(sol.states[-1], ll.YF, atol=1e-9)


def test_intervals_with_different_points_solve_together(solver):
    solve, flipped = solver
    sol = solve(RadauMesh([-1.0, -0.5, 0.2, 1.0], [1, 7, 10], flipped), None)
    assert solve.reports[-1]["converged"]
    assert np.allclose(sol.states[-1], ll.YF, atol=1e-9)
    assert np.all(sol.controls >= -1e-9) and np.all(sol.controls <= ll.U_MAX + 1e-9)


def test_a_new_mesh_reuses_the_module_built_for_the_catalog(solver):
    solve, flipped = solver
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve(RadauMesh.uniform(4, 2, flipped), None)


def test_a_solution_warm_starts_the_next_mesh(solver):
    solve, flipped = solver
    coarse = solve(RadauMesh.uniform(10, 3, flipped), None)
    fine = solve(RadauMesh.uniform(20, 4, flipped), coarse)
    assert solve.reports[-1]["converged"]
    assert abs(fine.tf - ll.T_FINAL) < abs(coarse.tf - ll.T_FINAL) + 1e-6


def test_hp_refinement_meets_the_tolerance_of_the_paper(solver):
    solve, flipped = solver

    # The settings of section V of the paper with three points as the floor
    options = RefinementOptions(
        smoothness_threshold=3.0, tolerance=1e-6, max_subintervals=10, max_points=10, min_points=3
    )
    history = HPRefinement(options, ll.rates).run(RadauMesh.uniform(10, 3, flipped), solve)
    final = history[-1]
    assert final.errors.max() <= 1e-6
    assert abs(final.solution.tf - ll.T_FINAL) < 1e-5
    assert abs(ll.cost(final.solution) - ll.COST) < 1e-5

    # The interval that holds the switch is far narrower than the initial intervals
    mesh = final.solution.mesh
    t = mesh_time(mesh.points, 0.0, final.solution.tf)
    k = int(np.searchsorted(t, ll.T_SWITCH)) - 1
    assert t[k + 1] - t[k] < 0.1 * t[-1] / 10


def flown_end(solution):
    """State at the final time flown from the start under the control polynomial of every interval."""
    mesh = solution.mesh
    t = mesh_time(mesh.points, solution.t0, solution.tf)
    y = np.array(solution.states[0], dtype=float)
    for k, n in enumerate(mesh.degrees):
        nodes = radau_points(n, mesh.flipped)
        u = solution.controls[mesh.collocation_indices(k), 0]
        a, b = t[k], t[k + 1]

        def rates(s_t, s, a=a, b=b, nodes=nodes, u=u):
            return [s[1], lagrange_matrix(nodes, [2.0 * (s_t - a) / (b - a) - 1.0])[0] @ u - ll.C]

        y = solve_ivp(rates, [a, b], y, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
    return y


def test_phs_refinement_meets_the_tolerance_with_flights_that_reach_the_target(solver):
    solve, flipped = solver
    options = PHSOptions(tolerance=1e-6, min_points=3, max_points=10)
    history = PHSRefinement(options, ll.rates).run(RadauMesh.uniform(10, 3, flipped), solve)
    final = history[-1]
    assert solve.reports[-1]["converged"]
    assert final.errors.max() <= 1e-6
    assert np.allclose(flown_end(final.solution), ll.YF, atol=1e-8)

    # The flights check the solved control and not its distance from the optimum
    assert abs(final.solution.tf - ll.T_FINAL) < 1e-3
    assert abs(ll.cost(final.solution) - ll.COST) < 1e-3
