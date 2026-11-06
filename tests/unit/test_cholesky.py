"""
Sparse Cholesky
---------------

- The SparseCholesky solver and the Cholesky solver inside SparseLDL
"""

import numpy as np
from scipy.sparse import csr_matrix

import amigo as am


def get_matrix(index):
    """One of three small symmetric positive definite matrices with a known solution"""
    if index == 0:
        entries = [
            [2, 1, 0, 0, 0],
            [1, 2, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 2, 1],
            [0, 0, 0, 1, 2],
        ]
    elif index == 1:
        entries = [
            [2, 0, 1, 0, 0],
            [0, 2, 0, 1, 0],
            [1, 0, 2, 0, 1],
            [0, 1, 0, 2, 1],
            [0, 0, 1, 1, 2],
        ]
    elif index == 2:
        entries = [
            [2, 1, 1, 0, 0],
            [1, 2, 0, 1, 0],
            [1, 0, 2, 0, 1],
            [0, 1, 0, 2, 1],
            [0, 0, 1, 1, 2],
        ]

    mat = np.array(entries)

    # The solution is all ones
    nrows = mat.shape[0]
    ncols = mat.shape[1]

    x = np.ones(nrows)
    rhs = mat @ x

    # The matrix in the CSR form of amigo
    mat = csr_matrix(np.array(entries))
    csr = am.CSRMat(nrows, ncols, mat.indptr, mat.indices, mat.data)

    return csr, x, rhs


def test_sparse_ldl():
    for index in range(3):
        problem = get_matrix(index)
        csr = problem[0]
        xsoln = problem[1]
        rhs = problem[2]

        xvec = am.Vector(len(xsoln))
        xvec[:] = rhs

        ldl = am.SparseLDL(csr, solver_type=am.SolverType.CHOLESKY, ustab=0.4)
        ldl.factor()

        x = np.array(xvec[:], dtype=float)
        err = np.linalg.norm(x - xvec[:])

        assert err < 1e-15


def test_sparse_cholesky():
    for index in range(3):
        problem = get_matrix(index)
        csr = problem[0]
        xsoln = problem[1]
        rhs = problem[2]

        xvec = am.Vector(len(xsoln))
        xvec[:] = rhs

        chol = am.SparseCholesky(csr)
        chol.factor()

        x = np.array(xvec[:], dtype=float)
        err = np.linalg.norm(x - xvec[:])

        assert err < 1e-15
