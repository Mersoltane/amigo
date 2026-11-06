"""
Radau collocation of the lunar landing
--------------------------------------

- The soft lunar landing of Liu, Hager, and Rao against its exact solution
- Uniform meshes, mixed meshes, warm starts, and the hp and phs refinements
"""

import contextlib
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

# The lunar landing example supplies the model and the exact solution
EXAMPLE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "examples",
        "trajectory",
        "lunar_landing",
    )
)

if EXAMPLE not in sys.path:
    sys.path.insert(0, EXAMPLE)

import lunar_landing as ll  # noqa: E402


@pytest.fixture(scope="module", params=[False, True], ids=["standard", "flipped"])
def flipped(request):
    """Whether the landing uses the flipped Radau points"""
    return request.param


@pytest.fixture(scope="module")
def solve(flipped, tmp_path_factory):
    """Landing solver on the standard or the flipped Radau points"""
    module = "lunar_landing_test"
    if flipped:
        module = "lunar_landing_test_flipped"

    catalog = ll.build_model(ll.catalog_mesh(1, 10, flipped), module)[0]

    # Build in a temporary directory so the repository stays untouched
    tmp_path = tmp_path_factory.mktemp(module)

    with contextlib.chdir(tmp_path):
        catalog.build_module()

    # The compiled module is imported from the build directory
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    options = {"convergence_tolerance": 1e-10, "max_iterations": 500}

    return ll.LandingSolver(module, 5.0, options)


def test_fine_uniform_mesh_matches_the_exact_solution(solve, flipped):
    sol = solve(RadauMesh.uniform(40, 3, flipped), None)

    assert solve.reports[-1]["converged"]

    # The final time and the cost match the exact solution
    assert abs(sol.tf - ll.T_FINAL) < 1e-3
    assert abs(ll.cost(sol) - ll.COST) < 1e-3

    # The trajectory starts and ends at the prescribed states
    assert np.allclose(sol.states[0], ll.Y0, atol=1e-9)
    assert np.allclose(sol.states[-1], ll.YF, atol=1e-9)


def test_intervals_with_different_points_solve_together(solve, flipped):
    sol = solve(RadauMesh([-1.0, -0.5, 0.2, 1.0], [1, 7, 10], flipped), None)

    assert solve.reports[-1]["converged"]

    # The landing reaches the target with the control inside its bounds
    assert np.allclose(sol.states[-1], ll.YF, atol=1e-9)
    assert np.all(sol.controls >= -1e-9)
    assert np.all(sol.controls <= ll.U_MAX + 1e-9)


def test_a_new_mesh_reuses_the_module_built_for_the_catalog(solve, flipped):
    # Any warning fails the test
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve(RadauMesh.uniform(4, 2, flipped), None)


def test_a_solution_warm_starts_the_next_mesh(solve, flipped):
    coarse = solve(RadauMesh.uniform(10, 3, flipped), None)
    fine = solve(RadauMesh.uniform(20, 4, flipped), coarse)

    assert solve.reports[-1]["converged"]

    # The finer mesh moves the final time closer to the exact one
    assert abs(fine.tf - ll.T_FINAL) < abs(coarse.tf - ll.T_FINAL) + 1e-6


def test_hp_refinement_meets_the_tolerance_of_the_paper(solve, flipped):
    # The settings of section V of the paper with three points as the floor
    options = RefinementOptions(
        smoothness_threshold=3.0,
        tolerance=1e-6,
        max_subintervals=10,
        max_points=10,
        min_points=3,
    )

    refinement = HPRefinement(options, ll.rates)
    history = refinement.run(RadauMesh.uniform(10, 3, flipped), solve)

    # The last mesh meets the tolerance and the exact solution
    final = history[-1]

    assert final.errors.max() <= 1e-6
    assert abs(final.solution.tf - ll.T_FINAL) < 1e-5
    assert abs(ll.cost(final.solution) - ll.COST) < 1e-5

    # The interval that holds the switch is far narrower than the initial intervals
    mesh = final.solution.mesh
    t = mesh_time(mesh.points, 0.0, final.solution.tf)
    k = int(np.searchsorted(t, ll.T_SWITCH)) - 1

    assert t[k + 1] - t[k] < 0.1 * t[-1] / 10


class IntervalRates:
    """Rates of the landing under the control polynomial of one interval"""

    def __init__(self, a, b, nodes, u):
        self.a = a
        self.b = b
        self.nodes = nodes
        self.u = u

    def __call__(self, s_t, s):
        # The control of the interval is the polynomial through its collocation values
        tau = 2.0 * (s_t - self.a) / (self.b - self.a) - 1.0
        control = lagrange_matrix(self.nodes, [tau])[0] @ self.u

        return [s[1], control - ll.C]


def flown_end(solution):
    """State at the final time flown from the start under the control polynomial of every interval"""
    mesh = solution.mesh
    t = mesh_time(mesh.points, solution.t0, solution.tf)
    y = np.array(solution.states[0], dtype=float)

    for k in range(mesh.num_intervals):
        a = t[k]
        b = t[k + 1]

        nodes = radau_points(mesh.degrees[k], mesh.flipped)
        u = solution.controls[mesh.collocation_indices(k), 0]

        # Each interval starts from the state the previous one reached
        rates = IntervalRates(a, b, nodes, u)
        flight = solve_ivp(rates, [a, b], y, method="DOP853", rtol=1e-12, atol=1e-12)
        y = flight.y[:, -1]

    return y


def test_phs_refinement_meets_the_tolerance_with_flights_that_reach_the_target(
    solve, flipped
):
    options = PHSOptions(tolerance=1e-6, min_points=3, max_points=10)
    refinement = PHSRefinement(options, ll.rates)
    history = refinement.run(RadauMesh.uniform(10, 3, flipped), solve)

    # The last mesh meets the tolerance and its flown control lands on the target
    final = history[-1]

    assert solve.reports[-1]["converged"]
    assert final.errors.max() <= 1e-6
    assert np.allclose(flown_end(final.solution), ll.YF, atol=1e-8)

    # The flights check the solved control and not its distance from the optimum
    assert abs(final.solution.tf - ll.T_FINAL) < 1e-3
    assert abs(ll.cost(final.solution) - ll.COST) < 1e-3
