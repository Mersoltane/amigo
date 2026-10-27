from . import DirectSparseSolver
import numpy as np
from scipy.sparse import csr_matrix


class PardisoSolver(DirectSparseSolver):
    """Sparse LDL^T solver via Intel MKL PARDISO with inertia detection.

    The symmetric indefinite factorization with mtype -2 reports the exact inertia.
    The upper triangle structure is cached once and factor only copies the data.
    Reusing one matrix object lets pypardiso skip the symbolic analysis.
    """

    supports_inertia = True
    solver_name = "PardisoSolver"

    def __init__(self, problem):
        from pypardiso import PyPardisoSolver

        self._init_sparse_structure(problem)

        # mtype=-2: real symmetric indefinite
        self.pardiso = PyPardisoSolver(mtype=-2)

        # Pre-compute the upper-triangle mask, the structure is fixed
        upper_mask = np.empty(self.nnz, dtype=bool)
        for i in range(self.nrows):
            start, end = self.rowp[i], self.rowp[i + 1]
            upper_mask[start:end] = self.cols[start:end] >= i
        self._upper_mask = upper_mask

        # Build upper-triangle CSR structure once (indices/indptr are fixed)
        upper_cols = self.cols[upper_mask]
        upper_indptr = np.zeros(self.nrows + 1, dtype=self.rowp.dtype)
        for i in range(self.nrows):
            start, end = self.rowp[i], self.rowp[i + 1]
            upper_indptr[i + 1] = upper_indptr[i] + int(
                np.sum(self.cols[start:end] >= i)
            )
        upper_data = np.zeros(len(upper_cols))

        # Persistent CSR matrix, the same object goes to pardiso every time
        # so symbolic analysis (phase 11) runs once, then only numerical
        # factorization (phase 22) on subsequent calls
        self._matrix = csr_matrix(
            (upper_data, upper_cols.copy(), upper_indptr.copy()),
            shape=(self.nrows, self.ncols),
        )

    def _do_factor(self):
        """Copy upper triangle into pypardiso's matrix and factorize."""
        self._matrix.data[:] = self.hess.get_data()[self._upper_mask]
        self.pardiso.factorize(self._matrix)

    def get_inertia(self):
        """Return (n_positive, n_negative) eigenvalue counts from LDL^T.

        Uses PARDISO iparm(22) and iparm(23) (1-based Fortran indexing),
        which are iparm[21] and iparm[22] in 0-based Python indexing.
        """
        return int(self.pardiso.iparm[21]), int(self.pardiso.iparm[22])

    def solve(self, bx, px):
        """Solve the KKT system."""
        bx.copy_device_to_host()
        px.get_array()[:] = self.pardiso.solve(self._matrix, bx.get_array())
        px.copy_host_to_device()
