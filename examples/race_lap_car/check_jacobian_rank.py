"""Check the constraint Jacobian rank of the race_lap problem.

Assembles the KKT matrix at the initial point, extracts the J block,
and computes its rank via sparse SVD.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------- Build the same model as race_lap.py ----------
from examples.race_lap_car.race_lap import opt  # noqa: E402 — reuse the built model

# Grab optimizer internals after build
ipo = opt.optimizer
solver = opt.solver
mult_ind = opt.problem.get_multiplier_indicator()

n_total = len(mult_ind)
n_primal = int(np.sum(~mult_ind))
n_dual = int(np.sum(mult_ind))
print(f"KKT size: {n_total} (primal={n_primal}, dual={n_dual})")

# Assemble the KKT matrix at the current point (initial guess)
x = opt.vars.get_solution()
opt._update_gradient(x)
solver.assemble_hessian(1.0, opt.vars, opt.diag)

# Extract the sparse KKT matrix
nr, nc, nnz, rp, cl = solver.hess.get_nonzero_structure()
data = solver.hess.get_data().copy()
K = csr_matrix((data, cl, rp), shape=(nr, nc))

# Extract the J block: rows = dual indices, cols = primal indices
primal_idx = np.where(~mult_ind)[0]
dual_idx = np.where(mult_ind)[0]

# J = K[dual_idx, :][:, primal_idx]  (constraint rows, primal cols)
J = K[np.ix_(dual_idx, primal_idx)]

print(f"J shape: {J.shape} ({n_dual} constraints x {n_primal} primals)")
print(f"J nnz: {J.nnz}")
print(f"Max possible rank: {min(n_dual, n_primal)}")

# Compute rank via SVD of the smaller dimension
k = min(n_primal, n_dual) - 1  # svds needs k < min(m,n)
k = min(k, 500)  # limit for speed
print(f"\nComputing top-{k} singular values of J...")
try:
    U, s, Vt = svds(J.astype(np.float64).tocsc(), k=k, which="SM")
    s_sorted = np.sort(s)
    print(f"Smallest {min(20, len(s_sorted))} singular values:")
    for i, sv in enumerate(s_sorted[:20]):
        print(f"  sigma[{i}] = {sv:.6e}")

    # Count near-zero singular values
    for tol in [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]:
        n_zero = np.sum(s_sorted < tol)
        if n_zero > 0:
            print(f"  {n_zero} singular values < {tol:.0e}")
except Exception as e:
    print(f"  SVD failed: {e}")

# Also check: how many rows of J are all zeros?
row_nnz = np.diff(J.indptr)
n_empty = np.sum(row_nnz == 0)
print(f"\nEmpty rows in J (constraints with no primal dependence): {n_empty}")

# Check J^T*J rank via dense diagonal (cheap)
JtJ_diag = np.array((J.T @ J).diagonal())
print(f"J^T*J diagonal: min={JtJ_diag.min():.6e}, max={JtJ_diag.max():.6e}")
n_zero_diag = np.sum(np.abs(JtJ_diag) < 1e-14)
print(f"Zero diagonal entries in J^T*J: {n_zero_diag}")
