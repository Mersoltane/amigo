from abc import ABC, abstractmethod
import numpy as np

# Factorization flag for a singular matrix that regularization can repair
SINGULAR_MATRIX = 1


class LinearSolver(ABC):
    @abstractmethod
    def factor(self, hess, diag):
        """Factor hess plus diag, returning SINGULAR_MATRIX or another nonzero flag on failure."""
        return

    @abstractmethod
    def solve(self, b, x):
        return

    def describe_failure(self, flag):
        """Reason for a nonzero factorization flag other than SINGULAR_MATRIX."""
        return f"the solver returned flag {flag}"

    def reset(self, options, state):
        """Prepare the solver for a new solve on state, False when it must be rebuilt."""
        return False

    def inertia_enabled(self):
        return False

    def get_inertia(self):
        return None

    def static_pivot_floor(self):
        # Static-pivoting solvers return their perturbation epsilon floor
        return 0.0

    def min_pivot(self):
        """Smallest pivot magnitude of the last factorization, None when the solver cannot say."""
        return None

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
