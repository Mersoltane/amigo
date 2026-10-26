import numpy as np
from .fem_space import SolutionSpace
from .cell_types import CellType


class Quadrature:
    def get_args(self):
        return []


class TriangleQuadrature(Quadrature):
    def __init__(self, order):
        self.weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        self.points = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        self.args = [{"n": 0}, {"n": 1}, {"n": 2}]

        if order == 1:
            # 1-point (exact for degree 1)
            self.xi = np.array([1 / 3])
            self.eta = np.array([1 / 3])
            self.weights = np.array([1.0 / 2.0])

        self.xi, self.eta, self.weights = self._duffy_quadrature((order - 1) ** 2)

        self.args = []
        for n in range(len(self.weights)):
            self.args.append({"n": n})

        return

    def _duffy_quadrature(self, degree):
        """
        Quadrature on the reference triangle

            (0,0), (1,0), (0,1)

        exact for complete polynomials of total degree <= degree.

        Returns
        -------
        pts : (nq, 2) ndarray
            Quadrature points [x, y].
        weights : (nq,) ndarray
            Quadrature weights.
        """

        # Under the Duffy transformation, a degree-p polynomial
        # times the Jacobian can have degree p+1 in r.
        n = (degree + 2) // 2

        # Gauss-Legendre rule on [-1, 1]
        pts, wi = np.polynomial.legendre.leggauss(n)

        # Transform to [0, 1]
        q = 0.5 * (pts + 1.0)
        w = 0.5 * wi

        weights = []
        xi = []
        eta = []

        for i, r in enumerate(q):
            for j, s in enumerate(q):
                xi.append(r)
                eta.append((1.0 - r) * s)

                weights.append(w[i] * w[j] * (1.0 - r))

        return xi, eta, weights

    def get_args(self):
        return self.args

    def get_point(self, n=0):
        return self.weights[n], [self.xi[n], self.eta[n]]


class QuadQuadrature(Quadrature):
    def __init__(self, npts):
        self.points, self.weights = np.polynomial.legendre.leggauss(npts)
        self.args = []

        for m in range(npts):
            for n in range(npts):
                self.args.append({"n": n, "m": m})

    def get_args(self):
        return self.args

    def get_point(self, n=0, m=0):
        wt = self.weights[n] * self.weights[m]
        pt = [self.points[n], self.points[m]]
        return wt, pt


class ReducedQuadQuadrature(Quadrature):
    def __init__(self):
        self.args = [{"n": 0, "m": 0}]
        self.points = np.array([0.0])
        self.weights = np.array([4.0])  # full area of biunit square

    def get_args(self):
        return self.args

    def get_point(self, n=0, m=0):
        return self.weights[0], [0.0, 0.0]


class LineQuadrature(Quadrature):
    def __init__(self, npts):
        pts, wts = np.polynomial.legendre.leggauss(npts)
        self.xi = pts
        self.weights = wts
        self.args = [{"n": n} for n in range(npts)]

    def get_args(self):
        return self.args

    def get_point(self, n=0):
        return self.weights[n], [self.xi[n]]


def make_quadrature(space: SolutionSpace, cell_type: CellType) -> Quadrature:
    max_degree = 0
    for func in space.get_spaces():
        if func.degree > max_degree:
            max_degree = func.degree

    if cell_type == CellType.SEGMENT:
        return LineQuadrature(max_degree + 1)
    elif cell_type == CellType.TRIANGLE:
        return TriangleQuadrature(max_degree + 1)
    elif cell_type == CellType.QUADRILATERAL:
        return QuadQuadrature(max_degree + 1)
    else:
        raise NotImplementedError
