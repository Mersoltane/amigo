"""Amigo's native LDL direct solver for the KKT system.

Wraps the C++ SparseLDL factorization. Supports inertia queries,
which lets it drive Algorithm IC inertia correction.
"""

from . import LinearSolver
from amigo import SolverType, SparseLDL, OrderingType


class AmigoSolver(LinearSolver):
    def __init__(self, options, state):
        self.options = options
        self.hessian = state.hessian

        ustab = 0.01
        pivot_tol = 1e-14

        self.ldl = SparseLDL(
            self.hessian,
            SolverType.LDL,
            ustab=ustab,
            pivot_tol=pivot_tol,
            order=OrderingType.DEFAULT,
        )

    def factor(self, hessian, diagonal):
        if hessian != self.hessian:
            raise ValueError("Hessian instance must be the same")

        flag = self.ldl.factor(diagonal)
        if flag != 0:
            raise RuntimeError(
                f"{self.solver_name} factorization failed with flag = {flag}"
            )

        return flag

    def solve(self, b, x):
        x.copy(b)
        self.ldl.solve(x)
        return

    def inertia_enabled(self):
        return True

    def get_inertia(self):
        return self.ldl.get_inertia()

    def set_pivot_tolerance(self, pivtol):
        pass
