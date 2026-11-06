"""
Mesh refinement loop
--------------------

- The loop that every refinement method runs, with a toy method in place of a real one
"""

from types import SimpleNamespace

import numpy as np

from amigo.trajectory import (
    Assessment,
    MeshRefinement,
    MeshSolution,
    RadauMesh,
    RefinementHistory,
)


class Halving(MeshRefinement):
    """A toy method whose error halves with every interval it adds, and that counts the meshes it keeps"""

    def __init__(self, tolerance, max_iterations):
        super().__init__(
            SimpleNamespace(tolerance=tolerance, max_iterations=max_iterations)
        )

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
    """A solution on a mesh that the toy method only counts the intervals of"""
    states = np.zeros((mesh.num_states, 1))
    controls = np.zeros((mesh.num_collocation, 1))

    return MeshSolution(mesh, states, controls, 0.0, 1.0, converged=converged)


def solve_one_phase(mesh, previous):
    """Solve of one phase that always converges"""
    return solved(mesh)


def solve_two_phases(meshes, previous):
    """Solve of several phases that always converge"""
    solutions = []

    for mesh in meshes:
        solutions.append(solved(mesh))

    return solutions


def solve_until_three_intervals(mesh, previous):
    """Solve of one phase that fails once the mesh has three intervals"""
    return solved(mesh, mesh.num_intervals < 3)


def stop_at_two_intervals(solution):
    """Stop once the mesh has two intervals"""
    return solution.mesh.num_intervals == 2


class PreviousRecorder:
    """Solve of one phase that records the intervals of the solution it starts from"""

    def __init__(self):
        self.seen = []

    def __call__(self, mesh, previous):
        if previous is None:
            self.seen.append(None)
        else:
            self.seen.append(previous.mesh.num_intervals)

        return solved(mesh)


def interval_counts(history):
    """Intervals of the mesh of every entry of a one phase history"""
    counts = []

    for entry in history:
        counts.append(entry.solution.mesh.num_intervals)

    return counts


def test_one_mesh_gives_one_iteration_per_mesh_with_its_actions():
    history = Halving(1.3e-4, 10).run(RadauMesh.uniform(1, 3), solve_one_phase)

    assert isinstance(history, RefinementHistory)
    assert interval_counts(history) == [1, 2, 3]

    # The last mesh meets the tolerance, so it stands and nothing was done to it
    actions = []

    for entry in history:
        actions.append(entry.actions)

    assert actions == [["split"], ["split", "keep"], None]
    assert history.best_index == 2 and history.best is history[-1]


def test_phases_give_a_list_per_mesh_and_a_phase_within_the_tolerance_keeps_its_mesh():
    method = Halving(1.3e-4, 10)
    meshes = [RadauMesh.uniform(1, 3), RadauMesh.uniform(4, 3)]

    history = method.run(meshes, solve_two_phases)

    # The intervals of every phase on every mesh
    counts = []

    for entry in history:
        phase_counts = []

        for step in entry:
            phase_counts.append(step.solution.mesh.num_intervals)

        counts.append(phase_counts)

    assert counts == [[1, 4], [2, 4], [3, 4]]

    # The second phase starts within the tolerance and keeps its mesh on both refinements
    second_actions = []

    for entry in history:
        second_actions.append(entry[1].actions)

    assert second_actions == [None, None, None]
    assert method.stays == 2


def test_a_failed_solve_ends_the_run_unassessed_and_the_best_converged_mesh_stands():
    history = Halving(1e-12, 10).run(
        RadauMesh.uniform(1, 3), solve_until_three_intervals
    )

    assert interval_counts(history) == [1, 2, 3]
    assert np.isnan(history[-1].errors).all()
    assert history.best_index == 1


def test_each_solve_starts_from_the_solution_of_the_mesh_before():
    solve = PreviousRecorder()

    Halving(1.3e-4, 10).run(RadauMesh.uniform(1, 3), solve)

    assert solve.seen == [None, 1, 2]


def test_a_solution_given_for_the_first_mesh_is_assessed_without_solving_it_again():
    solve = PreviousRecorder()

    mesh = RadauMesh.uniform(1, 3)
    first = solved(mesh)

    history = Halving(1.3e-4, 10).run(mesh, solve, first=first)

    assert history[0].solution is first
    assert solve.seen == [1, 2]


def test_a_stop_ends_the_run_after_the_mesh_it_names_and_that_mesh_stands_by_its_error():
    history = Halving(1e-12, 10).run(
        RadauMesh.uniform(1, 3), solve_one_phase, stop=stop_at_two_intervals
    )

    assert interval_counts(history) == [1, 2]
    assert history.best_index == 1
