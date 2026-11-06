"""
Bang-bang problem
-----------------
"""

from .arcs import arc_edges, uniform_arc_meshes


class BangBangProblem:
    """What a bang-bang problem supplies to BangBangSolver, its parameter, its single phase, and its arcs"""

    # The kind of arc at the upper limit of the control and the kind at its lower limit
    kinds = ("burn", "coast")

    def __init__(self):
        # The solves add their NLP iterations here, and the stages of a step record them
        self.iterations = 0

    def set_value(self, value):
        """Set the parameter of the family to a value"""
        raise NotImplementedError

    def single(self, mesh, guess, blend, warm="same"):
        """The single phase solved on a mesh from a guess"""
        # blend is None for energy, a smoothing value, or 0 for the objective, and warm is "same", "always", or "never"
        raise NotImplementedError

    def single_costate(self):
        """Costate at the collocation points of the last converged single phase solve"""
        raise NotImplementedError

    def single_error(self, solution):
        """Largest mesh error of a single phase solution, or nan when it did not converge"""
        raise NotImplementedError

    def reset_single(self):
        """Forget the warm start of the single phase, so its next solve starts from its guess alone"""

    def switching(self, solution, costate):
        """Switching function at the collocation points, negative where the control sits at its upper limit"""
        raise NotImplementedError

    def sign(self, kind):
        """Sign of the switching function on an arc of a kind, negative on an arc at the upper limit"""
        if kind == self.kinds[0]:
            return -1.0

        return 1.0

    def other(self, kind):
        """The kind of arc at the other limit of the control"""
        if kind == self.kinds[0]:
            return self.kinds[1]

        return self.kinds[0]

    def read(self, source, sigma, noise):
        """Arcs read from a switching function, with the shift and the parameters of their start"""
        # Each arc is a kind with its start and end times
        raise NotImplementedError

    def arc_meshes(self, arcs):
        """Uniform meshes of five points per interval with about forty intervals over all the arcs"""
        return uniform_arc_meshes(arc_edges(arcs), intervals=40, degree=5, min_intervals=2)

    def arc_start(self, source, arcs, meshes, shift, parameters):
        """Solutions of the arcs on their meshes from a source solution"""
        raise NotImplementedError

    def arc_solver(self, kinds, start, brackets, record, budget):
        """A solve of the arcs that takes their meshes and previous solutions, with costate(record) and records"""
        # The first meshes start warm from the multipliers of record, or cold when it is None
        raise NotImplementedError

    def primer(self, solution, costate):
        """The solution with the control of the minimum principle wherever it rests, for arcs found again from it"""
        return solution

    def touches(self, kinds, solutions):
        """The arcs split where they touch a path constraint, as kinds, arcs, meshes, and start, or None"""
        return None

    def refinement(self, tolerance, max_iterations=None):
        """The mesh refinement of the single phase or of the arcs to a tolerance"""
        raise NotImplementedError

    def carried(self, carry, value):
        """The arcs a step carries to another value of the parameter"""
        return carry

    def objective(self, solution):
        """The objective of a solution, lower is better"""
        raise NotImplementedError
