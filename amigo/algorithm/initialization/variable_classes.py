"""Defined and free variables of a model from the structure of its equality constraints.

A generated residual carries an exact unit coefficient on the variable it
defines, so a bipartite matching of the equality rows to the primal columns
through the unit entries of the Jacobian recovers which variables the
equalities determine. Slack rows define their slacks. The primals left
unmatched are free and parameterize the null space of the constraints, and
the free ones with curvature in the Lagrangian Hessian are the rows that
carry the inertia shift.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching

from ...amigo import OptVarType

# Tolerance on a Jacobian entry that counts as an exact unit coefficient
UNIT_TOL = 1e-12

# Largest entries of a row tried in turn when the unit entries leave rows unmatched
WIDENINGS = (1, 2, 4, 8, 16, None)


def _kkt_blocks(model, multipliers, alpha):
    """Scipy form of the KKT matrix at the initial point with the given multipliers."""
    n = model.num_variables
    cons = np.asarray(model.constraint_indices, dtype=int)
    x = model.create_vector()
    x.copy(model.get_initial_point())
    xv = x.get_vector().get_array()
    xv[cons] = multipliers
    x.get_vector().copy_host_to_device()
    hess = model.create_matrix()
    model.eval_hessian(x, hess, alpha=alpha, zero_fixed_rows_and_columns=False)
    _, _, _, rowp, cols = hess.get_nonzero_structure()
    return sp.csr_matrix((np.array(hess.get_data()), np.asarray(cols), np.asarray(rowp)), shape=(n, n))


def _largest_entries(A, keep):
    """Graph of the keep largest entries of each row of A."""
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    order = np.lexsort((-A.data, rows))
    rank = np.empty(order.size, dtype=int)
    rank[order] = np.arange(order.size) - np.searchsorted(rows[order], rows[order])
    mask = rank < keep
    counts = np.bincount(rows[mask], minlength=A.shape[0])
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return sp.csr_matrix((np.ones(int(mask.sum())), A.indices[mask], indptr), shape=A.shape)


def classify_variables(model):
    """Defined, free, and shifted primal indices of the model.

    The shifted ones are the free variables that influence, through the
    defining rows, a variable whose Lagrangian Hessian row is nonzero under
    random multipliers at the initial point. A free variable that only
    feeds linear rows carries no curvature to correct.
    """
    n = model.num_variables
    cons = np.asarray(model.constraint_indices, dtype=int)
    is_con = np.zeros(n, dtype=bool)
    is_con[cons] = True

    # A variable held at equal bounds is no column of the problem and belongs to no class
    types = np.asarray(model.problem.get_var_types().get_array())
    fixed = types == int(OptVarType.FIXED)
    primal = np.flatnonzero(~is_con & ~fixed)
    slacks = np.asarray(model.slack_indices, dtype=int)
    slack_rows = np.asarray(model.ineq_constraint_indices, dtype=int)
    eq_rows = np.setdiff1d(cons, slack_rows)
    columns = np.setdiff1d(primal, slacks)

    K = _kkt_blocks(model, 0.0, 0.0)
    J = K[eq_rows][:, columns].tocsr()
    J.eliminate_zeros()
    A = abs(J)

    # The unit entries first, then the largest entries of each row until every row is matched
    unit = A.copy()
    unit.data = (np.abs(unit.data - 1.0) < UNIT_TOL).astype(float)
    unit.eliminate_zeros()
    match = maximum_bipartite_matching(unit, perm_type="column")
    for keep in WIDENINGS:
        if int((match >= 0).sum()) == eq_rows.size:
            break
        graph = A if keep is None else _largest_entries(A, keep)
        match = maximum_bipartite_matching(graph, perm_type="column")
    if int((match >= 0).sum()) != eq_rows.size:
        raise ValueError("The equality constraints do not define a variable each")

    defined = np.concatenate([columns[match], slacks])
    free = np.setdiff1d(primal, defined)

    # Curvature under random multipliers marks the variables the Lagrangian Hessian touches
    rng = np.random.default_rng(0)
    K = _kkt_blocks(model, rng.standard_normal(cons.size), 1.0)
    H = K[primal][:, primal].tocsr()
    H.eliminate_zeros()
    reached = np.zeros(n, dtype=bool)
    reached[primal[np.diff(H.indptr) > 0]] = True

    # Every variable of a defining row influences the variable the row defines
    G = sp.csr_matrix((np.ones(J.nnz), J.indices, J.indptr), shape=J.shape)
    influence = sp.csr_matrix((G.data, columns[G.indices], G.indptr), shape=(eq_rows.size, n))
    defined_of_row = columns[match]

    # Curvature propagates back from the reached variables to those that influence them
    frontier = reached.copy()
    while True:
        rows = np.flatnonzero(frontier[defined_of_row])
        if rows.size == 0:
            break
        touched = np.zeros(n, dtype=bool)
        touched[np.unique(influence[rows].indices)] = True
        frontier = touched & ~reached
        reached |= touched

    shifted = free[reached[free]]
    return defined, free, shifted


def hessian_perturbation_rows(model):
    """Primal rows that carry delta_x when the free variables of the model do."""
    _, _, shifted = classify_variables(model)
    return shifted
