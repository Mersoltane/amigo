from abc import ABC, abstractmethod
import numpy as np


class LinearSolver(ABC):
    @abstractmethod
    def factor(self, hess, diag):
        return

    @abstractmethod
    def solve(self, b, x):
        return

    def inertia_enabled(self):
        return False

    def get_inertia(self):
        return None

    def set_pivot_tolerance(self, pivtol):
        pass

    @staticmethod
    def find_diag_indices(rowp, cols, nrows):
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
