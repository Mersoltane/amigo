"""The mesh refinement loop that every refinement method runs, and the records it keeps.

Each pass solves the NLP on all phases and assesses every interval.
Only a phase above the tolerance gets a new mesh, as in Agamawi, Hager, and Rao.
A method supplies the assessment and the decision.
"""

from dataclasses import dataclass, field

import numpy as np

from .mesh import RadauMesh


@dataclass
class MeshSolution:
    """A solution on one mesh with the states on the support points and the controls on the collocation points."""

    mesh: RadauMesh
    states: np.ndarray
    controls: np.ndarray
    t0: float
    tf: float

    # Static parameters of the problem, carried to the next mesh
    parameters: dict = field(default_factory=dict)

    # Whether the NLP on this mesh converged
    converged: bool = True


@dataclass
class MeshIteration:
    """One solved mesh of one phase with its error estimate and what the refinement did with it."""

    solution: MeshSolution
    errors: np.ndarray

    # Smoothness of every interval when the method estimates it
    smoothness: np.ndarray = None

    # What the refinement did to every interval to make the next mesh, or None where the mesh stands
    actions: list = None


@dataclass
class Assessment:
    """What a method learns about every interval of one solved mesh of one phase."""

    # Largest error of every interval, which the tolerance applies to
    errors: np.ndarray

    # Smoothness of every interval when the method estimates it
    smoothness: np.ndarray = None

    # Intervals that the next mesh may not coarsen
    keep: np.ndarray = None

    # Mesh coordinate where the next mesh splits each interval, or nan where it need not
    cuts: np.ndarray = None


def best_iteration(history, tolerance):
    """Position of the entry of a refinement history that stands.

    The last entry stands when it meets the tolerance.
    Otherwise the converged entry of least error stands.
    """
    # An entry of a history of phases holds one iteration per phase
    entries = [entry if isinstance(entry, list) else [entry] for entry in history]
    errors = [max(float(np.max(step.errors)) for step in entry) for entry in entries]
    converged = [all(step.solution.converged for step in entry) for entry in entries]
    if converged[-1] and errors[-1] <= tolerance:
        return len(entries) - 1
    candidates = [k for k in range(len(entries)) if converged[k]]
    return min(candidates, key=lambda k: errors[k]) if candidates else len(entries) - 1


class RefinementHistory(list):
    """The solved meshes in order, each an iteration of one phase or a list of iterations of several."""

    def __init__(self, tolerance, entries=()):
        super().__init__(entries)
        self.tolerance = tolerance

    @property
    def best_index(self):
        """Position of the entry that stands."""
        return best_iteration(self, self.tolerance)

    @property
    def best(self):
        """The entry that stands, the last when it meets the tolerance or else the converged one of least error."""
        return self[self.best_index]


class MeshRefinement:
    """The refinement loop around a user supplied solve, with the assessment and the decision of a method.

    A method overrides start, assess, and decide, and stay when it must note
    that a phase keeps its mesh.
    """

    def __init__(self, options):
        self.options = options

    def start(self, mesh):
        """The state the method keeps for one phase from its first mesh on."""
        return {}

    def assess(self, solution, state):
        """The Assessment of every interval of a solved mesh of one phase."""
        raise NotImplementedError

    def decide(self, solution, assessment, state):
        """The next mesh of one phase above the tolerance and the action taken on each of its intervals."""
        raise NotImplementedError

    def stay(self, solution, state):
        """Note that a phase within the tolerance keeps its mesh."""

    def run(self, meshes, solve, first=None, stop=None):
        """Solve on a sequence of meshes until every interval of every phase meets the tolerance.

        solve maps the meshes and the previous solutions to one solution per phase.
        first holds solutions already solved on the first meshes.
        The loop ends when stop returns True or a solve fails to converge.
        """
        single = isinstance(meshes, RadauMesh)
        meshes = [meshes] if single else list(meshes)
        states = [self.start(mesh) for mesh in meshes]
        history = RefinementHistory(self.options.tolerance)
        previous = None
        for k in range(self.options.max_iterations):
            if k == 0 and first is not None:
                solutions = [first] if single else list(first)
            elif single:
                solutions = [solve(meshes[0], None if previous is None else previous[0])]
            else:
                solutions = solve(meshes, previous)
            if not all(s.converged for s in solutions):
                history.append([MeshIteration(s, np.full(s.mesh.num_intervals, np.nan)) for s in solutions])
                break
            assessments = [self.assess(s, state) for s, state in zip(solutions, states)]
            entry = [MeshIteration(s, a.errors, a.smoothness) for s, a in zip(solutions, assessments)]
            history.append(entry)
            if stop is not None and stop(solutions[0] if single else solutions):
                break
            if max(float(a.errors.max()) for a in assessments) <= self.options.tolerance:
                break

            # Every solve gets a list of its own, since a solve may keep the list it was given
            refined = []
            for s, a, state, record in zip(solutions, assessments, states, entry):
                if a.errors.max() > self.options.tolerance:
                    mesh, record.actions = self.decide(s, a, state)
                else:
                    mesh = s.mesh
                    self.stay(s, state)
                refined.append(mesh)
            meshes, previous = refined, solutions
        if single:
            return RefinementHistory(history.tolerance, [entry[0] for entry in history])
        return history
