from abc import ABC, abstractmethod
import numpy as np


class LinearSolver(ABC):
    @abstractmethod
    def factor(self, alpha, x, diag):
        pass

    @abstractmethod
    def solve(self, bx, px):
        pass

    supports_inertia = False

    def get_inertia(self):
        raise NotImplementedError(
            "This solver does not support inertia queries"
        )  # TODO scipy need inertia

    def assemble_hessian(self, alpha, x):
        """Assemble Lagrangian Hessian and return its diagonal.

        Leaves the assembled matrix in self.hess for a subsequent
        add_diagonal_and_factor() call.  Cost: one Hessian evaluation +
        one device-to-host copy.  No factorization.
        """
        self.problem.hessian(alpha, x, self.hess)
        self.hess.copy_data_device_to_host()
        return self.hess.get_data()[self._diag_indices].copy()

    def get_hessian_diagonal(self, alpha, x):
        """Evaluate Hessian and return its diagonal. O(n), no factorization."""
        self.problem.hessian(alpha, x, self.hess)
        self.hess.copy_data_device_to_host()
        return self.hess.get_data()[self._diag_indices]

    @staticmethod
    def _find_diag_indices(rowp, cols, nrows):
        """Find the CSR data-array index of each diagonal entry (row == col)."""
        diag_idx = np.empty(nrows, dtype=np.intp)
        for i in range(nrows):
            start, end = rowp[i], rowp[i + 1]
            row_cols = cols[start:end]
            pos = np.searchsorted(row_cols, i)
            if pos < len(row_cols) and row_cols[pos] == i:
                diag_idx[i] = start + pos
            else:
                diag_idx[i] = start  # fallback (should not happen)
        return diag_idx
