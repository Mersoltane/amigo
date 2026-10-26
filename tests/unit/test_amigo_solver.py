"""Tests for the equilibrated and refined AmigoSolver on badly scaled KKT matrices."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix

import amigo as am
from amigo.algorithm.default_options import get_default_options
from amigo.algorithm.solvers.amigo_solver import NONFINITE_MATRIX, AmigoSolver


def make_kkt(n=24, m=8, seed=0):
    """KKT matrix [H J^T; J 0] with entries spanning eight orders of magnitude."""
    rng = np.random.default_rng(seed)
    scales = 10.0 ** rng.uniform(-4, 4, n)
    A = rng.standard_normal((n, n))
    H = A @ A.T * np.outer(scales, scales) ** 0.5 + np.diag(scales)
    J = rng.standard_normal((m, n)) * 10.0 ** rng.uniform(-3, 3, (m, 1))
    K = np.zeros((n + m, n + m))
    K[:n, :n] = H
    K[n:, :n] = J
    K[:n, n:] = J.T

    # Keep every diagonal entry in the pattern so the diagonal can be added
    rows, cols = np.nonzero(K)
    rows = np.concatenate([rows, np.arange(n + m)])
    cols = np.concatenate([cols, np.arange(n + m)])
    data = np.concatenate([K[K != 0.0], np.zeros(n + m)])
    csr = coo_matrix((data, (rows, cols)), shape=K.shape).tocsr()
    csr.sort_indices()
    mat = am.CSRMat(n + m, n + m, csr.indptr, csr.indices, csr.data)
    return mat, K


def make_solver(mat, **overrides):
    options = get_default_options(overrides)
    return AmigoSolver(options, SimpleNamespace(hessian=mat, hessian_version=0))


def dense_inertia(K):
    eig = np.linalg.eigvalsh(K)
    return int(np.sum(eig > 0)), int(np.sum(eig < 0))


@pytest.mark.parametrize("sweeps", [0, 3])
def test_solve_matches_dense(sweeps):
    mat, K = make_kkt()
    n = K.shape[0]
    rng = np.random.default_rng(1)
    d = np.zeros(n)
    d[:24] = 10.0 ** rng.uniform(-2, 2, 24)
    diagonal = am.Vector(n)
    diagonal[:] = d

    solver = make_solver(mat, amigo_scaling_sweeps=sweeps)
    solver.factor(mat, diagonal)

    Kd = K + np.diag(d)
    b = rng.standard_normal(n)
    bvec = am.Vector(n)
    bvec[:] = b
    xvec = am.Vector(n)
    solver.solve(bvec, xvec)

    x = np.array(xvec[:], dtype=float)
    error = np.abs(Kd @ x - b).max() / (np.abs(Kd).sum(axis=1).max() * np.abs(x).max())
    assert error < 1e-12
    assert solver.get_inertia() == dense_inertia(Kd)


def test_scaling_keeps_inertia():
    mat, K = make_kkt(seed=3)
    n = K.shape[0]
    diagonal = am.Vector(n)
    diagonal[:] = np.zeros(n)

    inertia = []
    for sweeps in (0, 3):
        solver = make_solver(mat, amigo_scaling_sweeps=sweeps)
        solver.factor(mat, diagonal)
        inertia.append(solver.get_inertia())

    assert inertia[0] == inertia[1] == dense_inertia(K)


@pytest.mark.parametrize("sweeps", [0, 1])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e300])
def test_nonfinite_entry_flags_the_matrix(sweeps, value):
    mat, K = make_kkt(seed=5)
    n = K.shape[0]
    d = np.zeros(n)
    d[3] = value
    diagonal = am.Vector(n)
    diagonal[:] = d

    # Only a NaN or infinite entry marks the matrix as not finite
    solver = make_solver(mat, amigo_scaling_sweeps=sweeps)
    flag = solver.factor(mat, diagonal)
    assert (flag == NONFINITE_MATRIX) == (not np.isfinite(value))


def test_diagonal_refactor_matches_fresh_factor():
    mat, K = make_kkt(seed=7)
    n = K.shape[0]
    rng = np.random.default_rng(4)
    first = np.zeros(n)
    first[:24] = 10.0 ** rng.uniform(-2, 2, 24)
    second = first + 10.0 ** rng.uniform(-3, 1, n)

    # The same Hessian version refactors from the updated scaled diagonal
    solver = make_solver(mat)
    for d in (first, second):
        diagonal = am.Vector(n)
        diagonal[:] = d
        solver.factor(mat, diagonal)

    Kd = K + np.diag(second)
    b = rng.standard_normal(n)
    bvec = am.Vector(n)
    bvec[:] = b
    xvec = am.Vector(n)
    solver.solve(bvec, xvec)
    x = np.array(xvec[:], dtype=float)
    error = np.abs(Kd @ x - b).max() / (np.abs(Kd).sum(axis=1).max() * np.abs(x).max())
    assert error < 1e-12
    assert solver.get_inertia() == dense_inertia(Kd)
