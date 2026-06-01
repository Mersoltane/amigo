"""Amigo's native LDL direct solver for the KKT system.

Wraps the C++ SparseLDL factorization. Supports inertia queries,
which lets it drive Algorithm IC inertia correction.
"""

import numpy as np

from . import DirectSparseSolver, LinearSolverNew
from amigo import SolverType, SparseLDL, OrderingType


class AmigoSolverNew(LinearSolverNew):
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


class AmigoSolver(DirectSparseSolver):
    """Direct KKT solver using Amigo's native LDL factorization."""

    supports_inertia = True
    solver_name = "am.SparseLDL"

    def __init__(self, problem, ustab=0.1, pivot_tol=1e-14):
        self._init_sparse_structure(problem)
        self.ldl = SparseLDL(
            self.hess,
            SolverType.LDL,
            ustab=ustab,
            pivot_tol=pivot_tol,
            order=OrderingType.DEFAULT,
        )

    def _do_factor(self):
        """Run the LDL numerical factorization on self.hess."""
        flag = self.ldl.factor()
        if flag != 0:
            raise RuntimeError(
                f"{self.solver_name} factorization failed with flag = {flag}"
            )

        if self.verbose:
            npos, nneg = self.ldl.get_inertia()
            print(
                f"  [LDL] factor flag={flag}, npos={npos}, nneg={nneg}, total={npos + nneg}"
            )

    def solve(self, bx, px):
        """Solve K x = bx using the stored LDL factorization."""
        bx.copy_device_to_host()
        b_arr = bx.get_array().copy()
        px.get_array()[:] = bx.get_array()[:]
        self.ldl.solve(px)
        px.copy_host_to_device()

        if self.verbose:
            b_nrm = np.max(np.abs(b_arr))
            x_nrm = np.max(np.abs(px.get_array()))
            print(f"  [LDL solve] ||b||_inf={b_nrm:.3e}, ||x||_inf={x_nrm:.3e}")

    def get_inertia(self):
        """Return (n_positive, n_negative) from the factorization."""
        return self.ldl.get_inertia()
