"""
Tests for the SparseLDL solver with solver_type = SolverType.LDL
"""

import amigo as am
from scipy.sparse import csr_matrix
import numpy as np


def get_inertia(mat, tol=1e-12):
    """Get the inertia for a small dense matrix. Enforce non-singularity"""
    eig, _ = np.linalg.eigh(mat)

    if np.min(np.absolute(eig)) < tol:
        raise ValueError("Singular matrix")

    return (int(np.sum(eig >= 0)), int(np.sum(eig < 0)))


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
        inertia = (5, 0)
        mat = np.array(entries)
    elif index == 1:
        entries = [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 2, 1],
            [0, 0, 0, 1, 2],
        ]
        inertia = (4, 1)
        mat = np.array(entries)
    elif index == 2:
        entries = [
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 1, 2, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
        ]
        mat = np.array(entries)
        inertia = get_inertia(mat)
    elif index == 3:
        entries = [
            [0, 1, 0, 0, 1],
            [1, 0, 2, 0, 1],
            [0, 2, 0, 0, 1],
            [0, 0, 0, 2, 1],
            [1, 1, 1, 1, 0],
        ]
        mat = np.array(entries)
        inertia = get_inertia(mat)
    elif index == 4:
        entries = [[0, 1], [1, 0]]
        mat = np.array(entries)
        inertia = get_inertia(mat)
    elif index == 5:
        entries = [[1, 1], [1, 0]]
        mat = np.array(entries)
        inertia = get_inertia(mat)
    elif index == 6:
        entries = [[1, 2, 3], [2, 1, 3], [3, 3, 0]]
        mat = np.array(entries)
        inertia = get_inertia(mat)

    # Now create matrix, the solution and the right-hand-side
    nrows, ncols = mat.shape
    x = np.ones(nrows)
    rhs = mat @ x
    mat = csr_matrix(np.array(entries))
    csr = am.CSRMat(nrows, ncols, mat.indptr, mat.indices, mat.data)

    return csr, x, rhs, inertia


def test_sparse_ldl():
    for index in range(7):
        csr, xsoln, rhs, inertia = get_matrix(index)

        xvec = am.Vector(len(xsoln))
        xvec[:] = rhs

        ldl = am.SparseLDL(
            csr, solver_type=am.SolverType.LDL, ustab=0.4, order=am.OrderingType.NATURAL
        )
        ldl.factor()

        ldl_inertia = ldl.get_inertia()
        ldl.solve(xvec)

        x = np.array(xvec[:], dtype=float)
        err = np.linalg.norm(x - xvec[:])

        assert ldl_inertia[0] == inertia[0]
        assert ldl_inertia[1] == inertia[1]
        assert err < 1e-15


def test_sparse_ldl_amd():
    for index in range(7):
        csr, xsoln, rhs, inertia = get_matrix(index)

        xvec = am.Vector(len(xsoln))
        xvec[:] = rhs

        ldl = am.SparseLDL(
            csr, solver_type=am.SolverType.LDL, ustab=0.4, order=am.OrderingType.AMD
        )
        ldl.factor()

        ldl_inertia = ldl.get_inertia()
        ldl.solve(xvec)

        x = np.array(xvec[:], dtype=float)
        err = np.linalg.norm(x - xvec[:])

        assert ldl_inertia[0] == inertia[0]
        assert ldl_inertia[1] == inertia[1]
        assert err < 1e-15
