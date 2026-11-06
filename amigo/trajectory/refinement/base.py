"""
Mesh refinement loop
--------------------

- Each pass solves the NLP on all phases and assesses every interval
- Only a phase above the tolerance gets a new mesh, as in Agamawi, Hager, and Rao
- A method overrides start, assess, and decide, and stay when a phase keeps its mesh
"""

from dataclasses import dataclass, field

import numpy as np

from ..collocation.mesh import RadauMesh


@dataclass
class MeshSolution:
    """Solution of one phase on one mesh"""

    mesh: RadauMesh

    # States on the support points and controls on the collocation points
    states: np.ndarray
    controls: np.ndarray

    # Start and end times of the phase
    t0: float
    tf: float

    # Static parameters of the problem, carried to the next mesh
    parameters: dict = field(default_factory=dict)

    # Whether the NLP on this mesh converged
    converged: bool = True


@dataclass
class MeshIteration:
    """One solved mesh of one phase with its error estimate"""

    solution: MeshSolution
    errors: np.ndarray

    # Smoothness of every interval when the method estimates it
    smoothness: np.ndarray = None

    # What the refinement did to every interval, or None where the mesh stands
    actions: list = None


@dataclass
class Assessment:
    """What a method learns about every interval of one solved mesh of one phase"""

    # Largest error of every interval, which the tolerance applies to
    errors: np.ndarray

    # Smoothness of every interval when the method estimates it
    smoothness: np.ndarray = None

    # Intervals that the next mesh may not coarsen
    keep: np.ndarray = None

    # Mesh coordinate where the next mesh splits each interval, or nan where it need not
    cuts: np.ndarray = None


def best_iteration(history, tolerance):
    """Position of the entry of a refinement history that stands"""
    # An entry of a history of phases holds one iteration per phase
    entries = []

    for entry in history:
        if isinstance(entry, list):
            entries.append(entry)
        else:
            entries.append([entry])

    # The largest error of every entry
    errors = []

    for entry in entries:
        step_errors = []

        for step in entry:
            step_errors.append(float(np.max(step.errors)))

        errors.append(max(step_errors))

    # Whether every phase of an entry converged
    converged = []

    for entry in entries:
        entry_converged = True

        for step in entry:
            if not step.solution.converged:
                entry_converged = False
                break

        converged.append(entry_converged)

    # The last entry stands when it meets the tolerance
    last = len(entries) - 1

    if converged[last] and errors[last] <= tolerance:
        return last

    # Otherwise the converged entry of least error stands
    best = None

    for k in range(len(entries)):
        if converged[k]:
            if best is None or errors[k] < errors[best]:
                best = k

    # The last entry stands when none converged
    if best is None:
        return last

    return best


class RefinementHistory(list):
    """The solved meshes in order, each an iteration of one phase or a list of iterations of several"""

    def __init__(self, tolerance, entries=()):
        super().__init__(entries)
        self.tolerance = tolerance

    @property
    def best_index(self):
        """Position of the entry that stands"""
        return best_iteration(self, self.tolerance)

    @property
    def best(self):
        """The entry that stands"""
        return self[self.best_index]


class MeshRefinement:
    """Refinement loop around a user supplied solve"""

    def __init__(self, options):
        self.options = options

    def start(self, mesh):
        """State the method keeps for one phase from its first mesh on"""
        return {}

    def assess(self, solution, state):
        """Assessment of every interval of a solved mesh of one phase"""
        raise NotImplementedError

    def decide(self, solution, assessment, state):
        """Next mesh of a phase above the tolerance and the action taken on each of its intervals"""
        raise NotImplementedError

    def stay(self, solution, state):
        """Note that a phase within the tolerance keeps its mesh"""

    def run(self, meshes, solve, first=None, stop=None):
        """Solve on a sequence of meshes until every interval of every phase meets the tolerance"""
        # A single mesh is a problem of one phase
        single = isinstance(meshes, RadauMesh)

        if single:
            meshes = [meshes]
        else:
            meshes = list(meshes)

        # The state the method keeps for every phase
        states = []

        for mesh in meshes:
            states.append(self.start(mesh))

        history = RefinementHistory(self.options.tolerance)
        previous = None

        for k in range(self.options.max_iterations):
            # The first solutions may be given, and the solve maps the meshes and the previous solutions to new ones
            if k == 0 and first is not None:
                if single:
                    solutions = [first]
                else:
                    solutions = list(first)

            elif single:
                last = None
                if previous is not None:
                    last = previous[0]

                solutions = [solve(meshes[0], last)]

            else:
                solutions = solve(meshes, previous)

            # Stop at a solve that failed to converge
            failed = False

            for solution in solutions:
                if not solution.converged:
                    failed = True
                    break

            if failed:
                entry = []

                for solution in solutions:
                    errors = np.full(solution.mesh.num_intervals, np.nan)
                    entry.append(MeshIteration(solution, errors))

                history.append(entry)
                break

            # Assess every interval of every phase
            assessments = []

            for i in range(len(solutions)):
                assessments.append(self.assess(solutions[i], states[i]))

            # Record the solved meshes with their errors
            entry = []

            for i in range(len(assessments)):
                solution = solutions[i]
                assessment = assessments[i]

                entry.append(MeshIteration(solution, assessment.errors, assessment.smoothness))

            history.append(entry)

            # Stop when the caller asks to
            if stop is not None:
                if single:
                    done = stop(solutions[0])
                else:
                    done = stop(solutions)

                if done:
                    break

            # Stop when every interval of every phase meets the tolerance
            largest = []

            for assessment in assessments:
                largest.append(float(assessment.errors.max()))

            if max(largest) <= self.options.tolerance:
                break

            # Refine the phases above the tolerance into a new list, since a solve may keep the one it was given
            refined = []

            for i in range(len(assessments)):
                solution = solutions[i]
                assessment = assessments[i]
                state = states[i]
                record = entry[i]

                if assessment.errors.max() > self.options.tolerance:
                    decision = self.decide(solution, assessment, state)
                    mesh = decision[0]
                    record.actions = decision[1]
                else:
                    mesh = solution.mesh
                    self.stay(solution, state)

                refined.append(mesh)

            meshes = refined
            previous = solutions

        # A single phase gets its iterations back without the phase lists
        if single:
            iterations = []

            for entry in history:
                iterations.append(entry[0])

            return RefinementHistory(history.tolerance, iterations)

        return history
