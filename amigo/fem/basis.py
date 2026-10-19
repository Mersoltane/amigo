import amigo as am
import numpy as np
from dataclasses import dataclass
from .fem_space import Space, SolutionSpace
from .cell_types import CellType
from .vandermonde import Vandermonde1D, Vandermonde2D, VecVandermonde2D
from .fem_space import FunctionSpace
from .cell_types import CellType, DofLayout
from ..expressions import Expr


def dot_product(x, y, n=1):
    val = x[0] * y[0]
    for i in range(1, n):
        val = val + x[i] * y[i]
    return val


def triple_product(d, x, y, n=1):
    val = d[0] * x[0] * y[0]
    for i in range(1, n):
        val = val + d[i] * x[i] * y[i]
    return val


def curl_2d(x, y, n=1):
    return x[0] * y[1] - x[1] * y[0]


def mat_vec(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]]


def mat_vec_transpose(A, x, m=1, n=1):
    return [A[0][0] * x[0] + A[1][0] * x[1], A[0][1] * x[0] + A[1][1] * x[1]]


@dataclass
class ConstValue:
    value: Expr | None = None


@dataclass
class H1Value:
    value: Expr | None = None
    grad: list[Expr] | None = None


@dataclass
class HdivValue:
    vec: list[Expr] | None = None
    div: Expr | None = None


class Basis:
    def __init__(self, names: list[str], layout: DofLayout, kind: str = "input"):
        if isinstance(names, (list, tuple)):
            self.names = names
        elif isinstance(names, str):
            self.names = [names]
        self.layout = layout
        self.kind = kind

        if not (
            self.kind == "input"
            or self.kind == "data"
            or self.kind == "multiplier"
            or self.kind == "output"
        ):
            raise ValueError(f"{self.kind} not recognized")

    def add_declarations(self, comp):
        """Add the declarations to the component"""

        ndof = self.layout.ndof
        if self.kind == "input":
            for name in self.names:
                comp.add_input(name, shape=(ndof,))
        elif self.kind == "data":
            for name in self.names:
                comp.add_data(name, shape=(ndof,))
        elif self.kind == "multiplier":
            for name in self.names:
                comp.add_constraint(f"res_{name}", shape=(ndof,))
        elif self.kind == "output":
            for name in self.names:
                for dim in range(ndof):
                    comp.add_output(f"{name}{dim}")


class ConstantBasis(Basis):
    def __init__(self, names, space, kind="input"):

        layout = DofLayout(ref_cell=CellType.POINT, space=space, pts=([0.0]))
        super().__init__(names, layout=layout, kind=kind)

    def transform(self, Jdict, orig):
        soln = {}
        for name in self.names:
            value = orig[name].value
            soln[name] = ConstValue(value=value)
        return soln

    def eval(self, comp, pt):
        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = ConstValue(value=u[0])

        return soln


class LagrangeBasis1D(Basis):
    def __init__(self, names: list[str], space: FunctionSpace, kind="input"):
        if not space.func_space is Space.H1:
            raise ValueError("Space must be H1")
        layout = DofLayout.make_layout(space, CellType.SEGMENT)
        super().__init__(names, layout, kind=kind)

        self.vand = Vandermonde1D(space.degree, layout.pts)

    def transform(self, Jdict, orig):
        soln = {}
        for name in self.names:
            value = orig[name].value
            grad = orig[name].grad
            soln[name] = H1Value(value=value, grad=[Jdict["Jinv"] * grad[0]])

        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi = geo["x"].grad[0]
        y_xi = geo["y"].grad[0]

        detJ = am.sqrt(x_xi**2 + y_xi**2)
        Jinv = 1.0 / detJ
        J = detJ

        return {"detJ": detJ, "J": J, "Jinv": Jinv}

    def eval(self, comp, pt):
        xi = pt[0]

        # Evaluate the monomials
        N = self.vand.eval_basis(xi)
        Nx = self.vand.eval_basis_grad(xi)

        soln = {}
        ndof = self.layout.ndof
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = H1Value(
                value=dot_product(u, N, n=ndof), grad=[dot_product(u, Nx, n=ndof)]
            )

        return soln


class LagrangeBasis2D(Basis):
    def __init__(
        self,
        names: list[str],
        layout: DofLayout,
        exps: tuple[tuple[int]],
        kind: str = "input",
    ):
        super().__init__(names, layout, kind=kind)
        self.vand = Vandermonde2D(layout.ndof, exps, layout.pts)
        return

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        # Evaluate the monomials
        N = self.vand.eval_basis(xi, eta)

        # Evaluate the derivatives of the monomials
        Nxi, Neta = self.vand.eval_basis_grad(xi, eta)

        soln = {}
        ndof = self.layout.ndof
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = H1Value(
                value=dot_product(u, N, n=ndof),
                grad=[dot_product(u, Nxi, n=ndof), dot_product(u, Neta, n=ndof)],
            )

        return soln

    def transform(self, Jdict, orig):
        soln = {}
        for name in self.names:
            value = orig[name].value
            grad = orig[name].grad
            soln[name] = H1Value(
                value=value, grad=mat_vec_transpose(Jdict["Jinv"], grad, n=2, m=2)
            )

        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi, x_eta = geo["x"].grad
        y_xi, y_eta = geo["y"].grad

        detJ = x_xi * y_eta - x_eta * y_xi
        J = [[x_xi, x_eta], [y_xi, y_eta]]
        inv = 1.0 / detJ
        Jinv = [[y_eta * inv, -x_eta * inv], [-y_xi * inv, x_xi * inv]]

        return {"detJ": detJ, "J": J, "Jinv": Jinv}


class TriangleLagrangeBasis(LagrangeBasis2D):
    def __init__(self, names: list[str], space: FunctionSpace, kind: str = "input"):
        if not space.func_space is Space.H1:
            raise ValueError("Space must be H1")
        layout = DofLayout.make_layout(space, CellType.TRIANGLE)
        exps = self._get_monomial_exponents(space.degree)

        super().__init__(names, layout, exps, kind=kind)
        return

    def _get_monomial_exponents(self, p):
        exps = []
        for i in range(p + 1):
            for j in range(p + 1 - i):
                exps.append((i, j))
        return exps


class QuadLagrangeBasis(LagrangeBasis2D):
    def __init__(self, names: list[str], space: FunctionSpace, kind: str = "input"):
        if not space.func_space is Space.H1:
            raise ValueError("Space must be H1")
        layout = DofLayout.make_layout(space, CellType.QUADRILATERAL)
        exps = self._get_monomial_exponents(space.degree)
        super().__init__(names, layout, exps, kind=kind)

        return

    def _get_monomial_exponents(self, p):
        exps = []
        for j in range(p + 1):
            for i in range(p + 1):
                exps.append((i, j))
        return exps


class RTBasis2D(Basis):
    def __init__(
        self,
        names: list[str],
        layout: DofLayout,
        uexps: tuple[tuple[int]],
        vexps: tuple[tuple[int]],
        kind="input",
    ):
        super().__init__(names, layout, kind=kind)
        self.vand = VecVandermonde2D(layout.ndof, uexps, vexps, layout.pts, layout.dirs)
        return

    def add_declarations(self, comp):
        """Add the declarations to the component"""

        ndof = self.layout.ndof
        if self.kind == "input":
            for name in self.names:
                comp.add_input(name, shape=(2, ndof))
        elif self.kind == "data":
            for name in self.names:
                comp.add_data(name, shape=(2, ndof))
        elif self.kind == "multiplier":
            for name in self.names:
                comp.add_constraint(f"res_{name}", shape=(2, ndof))

        comp.add_data("hdiv_signs", shape=(ndof,))

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        # Evaluate the monomials
        N = self.vand.eval_basis(xi, eta)

        # Evaluate the derivatives of the monomials
        Nxi, Neta = self.vand.eval_basis_grad(xi, eta)

        soln = {}
        d = comp.data["hdiv_signs"]
        ndof = self.layout.ndof
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            vx = d[0] * N[0, 0] * u[0, 0]
            vy = d[0] * N[1, 0] * u[1, 0]
            div = d[0] * (Nxi[0, 0] * u[0, 0] + Neta[1, 0] * u[0, 0])

            for i in range(1, ndof):
                vx = vx + d[i] * N[0, i] * u[0, i]
                vy = vy + d[i] * N[1, i] * u[1, i]
                div = div + d[i] * (Nxi[0, i] * u[0, i] + Neta[1, i] * u[0, i])

            soln[name] = HdivValue(vec=[vx, vy], div=div)

        return soln

    def transform(self, Jdict, orig):
        soln = {}
        for name in self.names:
            vec = orig[name].vec
            div = orig[name].div
            detJ = Jdict["detJ"]
            J = Jdict["J"]
            vx = (J[0][0] * vec[0] + J[0][1] * vec[1]) / detJ
            vy = (J[1][0] * vec[0] + J[1][1] * vec[1]) / detJ
            soln[name] = HdivValue(vec=[vx, vy], div=div / detJ)

        return soln

    def compute_transform(self, geo):
        if "x" not in geo or "y" not in geo:
            raise ValueError("Coordinates not defined")

        x_xi, x_eta = geo["x"].grad
        y_xi, y_eta = geo["y"].grad

        detJ = x_xi * y_eta - x_eta * y_xi
        J = [[x_xi, x_eta], [y_xi, y_eta]]
        inv = 1.0 / detJ
        Jinv = [[y_eta * inv, -x_eta * inv], [-y_xi * inv, x_xi * inv]]

        return {"detJ": detJ, "J": J, "Jinv": Jinv}


class QuadRTBasis(RTBasis2D):
    def __init__(self, names: list[str], space: FunctionSpace, kind="input"):
        if not space.func_space is Space.HDIV:
            raise ValueError("Space must be H(div)")
        if space.degree == 1:
            uexps = [(0, 0), (-1, -1), (1, 0), (0, 0)]
            vexps = [(-1, -1), (0, 0), (-1, -1), (0, 1)]
        elif space.degree == 2:
            uexps = []
            vexps = []
        else:
            raise ValueError(f"Degree {space.degree} must be <= 2")

        layout = DofLayout.make_layout(space, CellType.QUADRILATERAL)
        super().__init__(names, layout, uexps, vexps, kind=kind)
        return


class TriangleRTBasis(RTBasis2D):
    def __init__(self, names: list[str], space: FunctionSpace, kind="input"):
        if not space.func_space is Space.HDIV:
            raise ValueError("Space must be H(div)")
        if space.degree == 1:
            uexps = [(0, 0), (-1, -1), (1, 0)]
            vexps = [(-1, -1), (0, 0), (0, 1)]
        elif space.degree == 2:
            uexps = [
                (0, 0),
                (-1, -1),
                (1, 0),
                (-1, -1),
                (0, 1),
                (-1, -1),
                (2, 0),
                (1, 1),
            ]
            vexps = [
                (-1, -1),
                (0, 0),
                (-1, -1),
                (1, 0),
                (-1, -1),
                (0, 1),
                (1, 1),
                (0, 2),
            ]
        else:
            raise ValueError(f"Degree {space.degree} must be <= 2")

        layout = DofLayout.make_layout(space, CellType.TRIANGLE)
        super().__init__(names, layout, uexps, vexps, kind=kind)
        return


class BasisCollection:
    def __init__(self, basis=[]):
        self.basis = basis

    def add_declarations(self, comp):
        for basis in self.basis:
            basis.add_declarations(comp)

    def eval(self, comp, pt):
        soln = {}
        for basis in self.basis:
            soln.update(basis.eval(comp, pt))
        return soln

    def transform(self, Jdict, orig):
        soln = {}
        for basis in self.basis:
            soln.update(basis.transform(Jdict, orig))
        return soln

    def compute_transform(self, geo):
        return self.basis[0].compute_transform(geo)


def make_basis(
    solution_space: SolutionSpace, cell_type: CellType, kind: str = "input"
) -> BasisCollection:
    objs = []

    for space in solution_space.get_spaces():
        names = solution_space.get_names(space)

        obj = None
        if space.func_space == Space.CONST:
            obj = ConstantBasis(names, space, kind=kind)
        elif space.func_space == Space.H1:
            if cell_type == CellType.SEGMENT:
                obj = LagrangeBasis1D(names, space, kind=kind)
            elif cell_type == CellType.TRIANGLE:
                obj = TriangleLagrangeBasis(names, space, kind=kind)
            elif cell_type == CellType.QUADRILATERAL:
                obj = QuadLagrangeBasis(names, space, kind=kind)
            else:
                raise NotImplementedError
        elif space.func_space == Space.HDIV:
            if cell_type == CellType.TRIANGLE:
                obj = TriangleRTBasis(names, space, kind=kind)
            elif cell_type == CellType.QUADRILATERAL:
                obj = QuadRTBasis(names, space, kind=kind)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        objs.append(obj)

    return BasisCollection(objs)
