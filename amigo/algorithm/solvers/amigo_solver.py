"""Amigo's native LDL direct solver for the KKT system.

The C++ SparseLDL factors the KKT matrix after symmetric Ruiz equilibration.
Equilibration lets the pivot tolerances act on entries of order one.
The scaling is a congruence and keeps the inertia of the original matrix.
"""

import numpy as np

from . import LinearSolver
from amigo import SolverType, SparseLDL, OrderingType, Vector

# Largest pivot threshold the escalation may reach
MAX_THRESHOLD = 0.5

# Backward error past which a refined solve raises the pivot threshold
ESCALATE_ERROR = 1e-6

# Factorization flag for a matrix with a NaN or infinite entry
NONFINITE_MATRIX = -1

# Factorization flag of the sparse LDL when delayed pivots overflow a front
FRONT_OVERFLOW = -100


class AmigoSolver(LinearSolver):
    def __init__(self, options, state):
        self.state = state
        self.hessian = state.hessian
        self.pivot_tol = options["amigo_pivot_tol"]
        self.sweeps = options["amigo_scaling_sweeps"]
        self.refinement_steps = options["amigo_refinement_steps"]
        self.residual_tol = options["amigo_residual_tol"]

        # Threshold pivoting parameter, raised when a solve stays inaccurate
        self.threshold = options["amigo_pivot_threshold"]

        # The scaled matrix that is factored, separate from the Hessian in the state
        self.kkt = state.hessian.duplicate()
        n = self.hessian.get_nonzero_structure()[0]
        self.scale = np.ones(n)
        self.scale_vec = Vector(n)
        self.work = Vector(n)
        self.product = Vector(n)

        # Hessian version and unscaled diagonal behind the scaled matrix
        self.factored_version = -1
        self.factored_diag = np.zeros(n)
        self.diag_update = Vector(n)

        self.last_error = 0.0
        self.max_error = 0.0
        self.ldl = SparseLDL(
            self.kkt,
            SolverType.LDL,
            ustab=self.threshold,
            pivot_tol=self.pivot_tol,
            order=OrderingType.DEFAULT,
        )

    def reset(self, options, state):
        """Rebind to a new solve and keep the symbolic analysis, False when the options need a new solver."""
        if options["solver"] != "amigo" or options["amigo_pivot_tol"] != self.pivot_tol:
            return False
        self.state = state
        self.hessian = state.hessian
        self.sweeps = options["amigo_scaling_sweeps"]
        self.refinement_steps = options["amigo_refinement_steps"]
        self.residual_tol = options["amigo_residual_tol"]
        self.threshold = options["amigo_pivot_threshold"]
        self.ldl.set_pivot_threshold(self.threshold)
        self.factored_version = -1
        self.max_error = 0.0
        return True

    def _equilibrate(self):
        """Symmetric Ruiz sweeps that drive every row max-abs to one, False on a non-finite entry."""
        finite = self.kkt.equilibrate(self.sweeps, self.scale_vec)
        self.scale = self.scale_vec.get_array()
        return finite

    def factor(self, hessian, diagonal):
        if hessian != self.hessian:
            raise ValueError("Hessian instance must be the same")

        diagonal.copy_device_to_host()
        diag = diagonal.get_array()

        if self.state.hessian_version != self.factored_version:
            # A new Hessian is copied, given its diagonal, and equilibrated
            self.hessian.copy_data_device_to_host()
            self.kkt.copy(hessian)
            self.kkt.add_diagonal_host(diagonal)
            if not self._equilibrate():
                # No regularization repairs a non-finite matrix
                self.factored_version = -1
                return NONFINITE_MATRIX
            self.factored_version = self.state.hessian_version
        else:
            # Only the diagonal changed, so the scaled diagonal moves by s^2 times the change
            update = self.diag_update.get_array()
            np.subtract(diag, self.factored_diag, out=update)
            update *= self.scale
            update *= self.scale
            self.kkt.add_diagonal_host(self.diag_update)
        self.factored_diag[:] = diag

        return self.ldl.factor()

    def describe_failure(self, flag):
        """Reason for a failed factorization of the scaled KKT matrix."""
        if flag == NONFINITE_MATRIX:
            return "the matrix has a NaN or infinite entry"
        if flag == FRONT_OVERFLOW:
            return "delayed pivots overflowed a front and the pivoting broke down"
        return f"the LDL factorization returned flag {flag}"

    def _solve_scaled(self, rhs):
        work = self.work.get_array()
        work[:] = rhs
        self.ldl.solve(self.work)
        return work.copy()

    def _residual(self, rhs, y):
        """Residual and backward error of the scaled system, whose entries are at most one."""
        self.work.get_array()[:] = y
        self.kkt.mult(self.work, self.product)
        r = rhs - self.product.get_array()
        err = np.abs(r).max() / (np.abs(rhs).max() + np.abs(y).max() + 1e-300)
        return r, err

    def solve(self, b, x):
        b.copy_device_to_host()
        rhs = self.scale * b.get_array()

        y = self._solve_scaled(rhs)
        r, err = self._residual(rhs, y)
        for _ in range(self.refinement_steps):
            if err <= self.residual_tol:
                break
            y_new = y + self._solve_scaled(r)
            r_new, err_new = self._residual(rhs, y_new)
            if err_new >= err:
                break
            y, r, err = y_new, r_new, err_new

        # An inaccurate solve asks for more stable pivoting from the next factorization on
        if err > ESCALATE_ERROR and self.threshold < MAX_THRESHOLD:
            self.threshold = min(10.0 * self.threshold, MAX_THRESHOLD)
            self.ldl.set_pivot_threshold(self.threshold)
            print(
                f"  AmigoSolver: solve backward error {err:.1e}, "
                f"raising the pivot threshold to {self.threshold:.1e}"
            )

        self.last_error = err
        self.max_error = max(self.max_error, err)
        x.get_array()[:] = self.scale * y
        x.copy_host_to_device()

    def inertia_enabled(self):
        return True

    def get_inertia(self):
        return self.ldl.get_inertia()

    def min_pivot(self):
        """Smallest pivot magnitude of the factored scaled system."""
        return self.ldl.get_min_pivot()

    def add_log_info(self, info):
        info["solve_backward_error"] = self.max_error
        self.max_error = 0.0
        info["kkt_min_pivot"] = self.ldl.get_min_pivot()
        info["kkt_delayed_pivots"] = self.ldl.get_num_delayed()
