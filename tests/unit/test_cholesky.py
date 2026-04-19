"""
Tests for Sparse Cholesky factorization. This tests both the
SparseCholesky solver and the Cholesky solver in SparseLDL
"""

import amigo as am
from scipy.sparse import csr_matrix
import numpy as np


def get_matrix(index):
    """Get some simple matrices to test"""
    if index == 0:
        entries = [
            [2, 1, 0, 0, 0],
            [1, 2, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 2, 1],
            [0, 0, 0, 1, 2],
        ]
        mat = np.array(entries)
    elif index == 1:
        entries = [
            [2, 0, 1, 0, 0],
            [0, 2, 0, 1, 0],
            [1, 0, 2, 0, 1],
            [0, 1, 0, 2, 1],
            [0, 0, 1, 1, 2],
        ]
        mat = np.array(entries)
    elif index == 2:
        entries = [
            [2, 1, 1, 0, 0],
            [1, 2, 0, 1, 0],
            [1, 0, 2, 0, 1],
            [0, 1, 0, 2, 1],
            [0, 0, 1, 1, 2],
        ]
        mat = np.array(entries)

    # Now create matrix, the solution and the right-hand-side
    nrows, ncols = mat.shape
    x = np.ones(nrows)
    rhs = mat @ x
    mat = csr_matrix(np.array(entries))
    csr = am.CSRMat(nrows, ncols, mat.indptr, mat.indices, mat.data)

    return csr, x, rhs


def test_sparse_ldl():
    for index in range(3):
        csr, xsoln, rhs = get_matrix(index)

        xvec = am.Vector(len(xsoln))
        xvec[:] = rhs

        ldl = am.SparseLDL(csr, solver_type=am.SolverType.CHOLESKY, ustab=0.4)
        ldl.factor()

        x = np.array(xvec[:], dtype=float)
        err = np.linalg.norm(x - xvec[:])

        assert err < 1e-15


def test_sparse_cholesky():
    for index in range(3):
        csr, xsoln, rhs = get_matrix(index)

        xvec = am.Vector(len(xsoln))
        xvec[:] = rhs

        chol = am.SparseCholesky(csr)
        chol.factor()

        x = np.array(xvec[:], dtype=float)
        err = np.linalg.norm(x - xvec[:])

        assert err < 1e-15
