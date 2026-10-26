"""Tests of the refinement loop that every method runs, with a toy method in place of a real one."""

from types import SimpleNamespace

import numpy as np

from amigo.trajectory import Assessment, MeshRefinement, MeshSolution, RadauMesh, RefinementHistory


class Halving(MeshRefinement):
    """A toy method whose error halves with every interval it adds, and that counts the meshes it keeps."""

    def __init__(self, tolerance, max_iterations):
        super().__init__(SimpleNamespace(tolerance=tolerance, max_iterations=max_iterations))
        self.stays = 0

    def assess(self, solution, state):
        n = solution.mesh.num_intervals
        return Assessment(np.full(n, 1e-3 / 2.0**n))

    def decide(self, solution, assessment, state):
        n = solution.mesh.num_intervals
        return RadauMesh.uniform(n + 1, 3), ["split"] + ["keep"] * (n - 1)

    def stay(self, solution, state):
        self.stays += 1


def solved(mesh, converged=True):
    """A solution on a mesh that the toy method only counts the intervals of."""
    zeros = np.zeros((mesh.num_states, 1)), np.zeros((mesh.num_collocation, 1))
    return MeshSolution(mesh, *zeros, 0.0, 1.0, converged=converged)


def test_one_mesh_gives_one_iteration_per_mesh_with_its_actions():
    history = Halving(1.3e-4, 10).run(RadauMesh.uniform(1, 3), lambda mesh, previous: solved(mesh))
    assert isinstance(history, RefinementHistory)
    assert [entry.solution.mesh.num_intervals for entry in history] == [1, 2, 3]

    # The last mesh meets the tolerance, so it stands and nothing was done to it
    assert [entry.actions for entry in history] == [["split"], ["split", "keep"], None]
    assert history.best_index == 2 and history.best is history[-1]


def test_phases_give_a_list_per_mesh_and_a_phase_within_the_tolerance_keeps_its_mesh():
    method = Halving(1.3e-4, 10)
    meshes = [RadauMesh.uniform(1, 3), RadauMesh.uniform(4, 3)]
    history = method.run(meshes, lambda meshes, previous: [solved(m) for m in meshes])
    assert [[step.solution.mesh.num_intervals for step in entry] for entry in history] == [[1, 4], [2, 4], [3, 4]]

    # The second phase starts within the tolerance and keeps its mesh on both refinements
    assert [entry[1].actions for entry in history] == [None, None, None]
    assert method.stays == 2


def test_a_failed_solve_ends_the_run_unassessed_and_the_best_converged_mesh_stands():
    history = Halving(1e-12, 10).run(RadauMesh.uniform(1, 3), lambda mesh, previous: solved(mesh, mesh.num_intervals < 3))
    assert [entry.solution.mesh.num_intervals for entry in history] == [1, 2, 3]
    assert np.isnan(history[-1].errors).all()
    assert history.best_index == 1


def test_each_solve_starts_from_the_solution_of_the_mesh_before():
    seen = []

    def solve(mesh, previous):
        seen.append(None if previous is None else previous.mesh.num_intervals)
        return solved(mesh)

    Halving(1.3e-4, 10).run(RadauMesh.uniform(1, 3), solve)
    assert seen == [None, 1, 2]


def test_a_solution_given_for_the_first_mesh_is_assessed_without_solving_it_again():
    seen = []

    def solve(mesh, previous):
        seen.append(previous.mesh.num_intervals)
        return solved(mesh)

    mesh = RadauMesh.uniform(1, 3)
    first = solved(mesh)
    history = Halving(1.3e-4, 10).run(mesh, solve, first=first)
    assert history[0].solution is first
    assert seen == [1, 2]


def test_a_stop_ends_the_run_after_the_mesh_it_names_and_that_mesh_stands_by_its_error():
    history = Halving(1e-12, 10).run(RadauMesh.uniform(1, 3), lambda mesh, previous: solved(mesh),
                                     stop=lambda solution: solution.mesh.num_intervals == 2)
    assert [entry.solution.mesh.num_intervals for entry in history] == [1, 2]
    assert history.best_index == 1
