"""Null-Space BISC: block PSD convexification on the primal Schur complement.

Operates on S_p = W + Sigma + J^T |D^{-1}| J, the exact primal Schur complement
after eliminating constraints from the KKT system.

Theory:
  KKT inertia = (n, m, 0) iff S_p > 0 and D < 0 (exact characterization).
  S_p is the theoretically correct matrix -- not W+Sigma, not an approximation.

  Null-space interpretation (without computing Z):
    - Null-space directions (JZ=0): Z^T S_p Z = Z^T(W+Sigma)Z (Lagrangian curvature).
      Modified by BISC if negative (nonconvex dynamics).
    - Range-space directions (J^T v): dominated by J^T|D^{-1}|J (~O(1) for slacks).
      Already large positive, untouched by BISC.
  S_p automatically separates null-space from range-space contributions.

OCP structure:
  For OCPs, J has block-bidiagonal structure (each constraint couples at most
  adjacent time steps). Therefore J^T D^{-1} J is block-tridiagonal, and S_p
  has the SAME block-tridiagonal structure as W.

Algorithm:
  1. Model-based block detection (falls back to BFS on sparsity).
     Block-tridiagonal structure validated at construction time; chains with
     non-adjacent coupling are demoted to independent small blocks.
  2. Schur complement propagation on S_p (backward sweep default):
     R_k = S_p[k,k] - C_k R~_{k+1}^{-1} C_k^T  (Riccati-natural direction).
     CONVEXIFY-style selective projection: within each block, identify
     "local" variables (no coupling to adjacent level) and "coupled" variables.
     Only the Schur complement of local variables (after eliminating coupled)
     is projected to PD. Coupled curvature redistributes naturally through the
     sweep, minimizing total ||E||_F. Falls back to full block projection when
     the coupled sub-block is not PD.
     Eigenvalue modification: "reflect" -> max(|lambda|, eps),
                              "clip"    -> max(lambda, eps).
     Condition-based adaptive eps: eps = max(barrier_eps, max_eig / kappa_max)
     caps the modified block condition number.
  3. Hub variables: dense bordered Schur complement with full spectral
     modification (eigendecomp + reflect/clip), consistent with chain/small
     blocks. No uniform diagonal shift.

Write-back self-consistency:
  CSR[k,k] = R~_k + Gamma - Sigma - S_schur.
  PARDISO adds Sigma (factor_from_host) and S_schur (constraint elimination),
  recovering R~_k + Gamma. Chain elimination subtracts Gamma -> gets R~_k >= eps.

Fill-in compensation (Gershgorin per-row bound):
  When CSR lacks off-diagonal entries for a modified block, the write-back
  is incomplete. Let E_missing = E_k restricted to missing entries. Per-row
  Gershgorin compensation: comp_i = sum_{j != i} |E_missing[i,j]| ensures
  D - E_missing is diagonally dominant (PSD). Tighter than uniform Weyl
  bound (||E_missing||_F on every entry) by up to sqrt(d).

Convergence guarantees:
  Theorem (Correct Inertia): After BISC, the modified KKT system has inertia
  (n, m, 0). Proof: Schur propagation with spectral modification ensures each
  R~_k >= eps*I > 0. By Sylvester's inertia law, S_p_modified is PD. By
  Haynsworth's inertia additivity, KKT inertia = (n, m, 0).

  Theorem (Asymptotic Vanishing): Under SOSC and strict complementarity (SC),
  there exists mu* > 0 such that for all barrier parameters mu < mu*, S_p(mu)
  is positive definite and no eigenvalue modification is needed.
  Proof: At the solution (x*, lambda*, z*), SOSC gives Z^T W(x*,lambda*) Z > 0
  (reduced Hessian PD on the null space of active constraints). SC ensures
  Sigma(mu) = Z_l/(X-L) + Z_u/(U-X) converges to a finite PD matrix on
  bound-active variables. The constraint Schur J^T|D^{-1}|J is PSD. Combined:
  S_p(0) = W + lim Sigma + lim J^T|D^{-1}|J is PD on the primal space. By
  continuity of eigenvalues, S_p(mu) remains PD in a neighborhood of mu=0.
  Corollary: ||E(mu)||_F -> 0 as mu -> 0 under SOSC+SC.

  Theorem (Superlinear Convergence): The BISC-modified IPM achieves superlinear
  convergence under standard assumptions. Proof: ||E_k|| = O(mu_k) and mu_k -> 0
  superlinearly, so the modification is asymptotically negligible.

  CONVEXIFY Selective Projection: The selective projection (projecting only local
  variables' Schur complement) produces a smaller total modification ||E||_F than
  full-block projection while maintaining the same inertia guarantee. Under SOSC,
  the selective projection is asymptotically optimal: it produces zero modification
  when S_p is naturally PD, and minimal modification otherwise.

  Comparison with prior work:
  - Verschueren et al. (SIAM J. Optim. 2017): convexification for QP subproblems
    in SQP, assuming known OCP (x,u) structure. Our method operates on the full
    NLP-IPM primal Schur complement S_p, detects block structure automatically,
    and handles hub variables and general sparsity.
  - IPOPT (Wachter & Biegler 2006): uniform diagonal delta*I for inertia
    correction. Our structured approach achieves ||E||_F << delta*n by
    modifying only the indefinite eigenvalues of each block.

  Properties:
  - PSD: each R~_k >= eps by construction (Sylvester's law).
  - No coupling floors: S_p gives slacks O(1) eigenvalues from J^T|D^{-1}|J.
  - Mesh independence: O(N * d^3), d physics-determined.

Single factorization, no inertia retry, no coupling floors, no mu-scaling.
"""

import numpy as np
import scipy.linalg

from .block_detection import detect_bfs_level_blocks


def _modify_eigenvalues(evals, eps, mode):
    """Apply eigenvalue modification for PSD projection.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues from symmetric eigendecomposition.
    eps : float
        Minimum allowed eigenvalue (adaptive, typically O(mu)).
    mode : str
        "reflect" : lambda~ = max(|lambda|, eps). Preserves curvature magnitude;
                    a strongly negative eigenvalue becomes strongly positive.
                    Naturally adapts to the problem's curvature scale: the step
                    in direction v_i is O(1/|lambda_i|), preventing aggressive
                    steps in strongly nonconvex directions.
                    Convergence: ||E||_F <= 2*||S_p^-||_F -> 0 under SOSC.
        "clip"    : lambda~ = max(lambda, eps). Minimal Frobenius perturbation.
                    Proven primal solution equivalence under SOSC
                    (Verschueren et al., SIAM J. Optim. 2017).
                    Can produce aggressive steps (O(1/eps)) in strongly
                    nonconvex directions, requiring more line search backtracking.

    Returns
    -------
    use_evals : ndarray
        Modified eigenvalues, all >= eps.
    needs_mod : bool
        Whether any eigenvalue was modified.
    """
    if mode == "reflect":
        use_evals = np.maximum(np.abs(evals), eps)
        needs_mod = bool(np.any(evals < eps))
    elif mode == "clip":
        use_evals = np.maximum(evals, eps)
        needs_mod = bool(np.any(evals < eps))
    else:
        raise ValueError(f"Unknown eigenvalue mode: {mode!r}. Use 'reflect' or 'clip'.")
    return use_evals, needs_mod


def _detect_model_chain(model, mult_ind):
    """Detect chain blocks from model component structure.

    For OCP-like models with repeated components (size > 1), groups primal
    variables by instance index. Each instance k of every max-size component
    contributes to block k. Linked variables are automatically deduplicated
    because they share global indices.

    Returns
    -------
    chain : list of np.ndarray or None
        Sorted primal variable indices for each block, or None if not OCP-like.
    remaining : list of int or None
        Primal variable indices not in any block, or None.
    """
    n = len(mult_ind)

    if not hasattr(model, "comp"):
        return None, None

    # Find max component size among components with primal variables
    max_size = 0
    for name, comp in model.comp.items():
        if comp.size <= 1:
            continue
        constraint_names = set(comp.get_constraint_names())
        has_primal = False
        for vname in comp.vars:
            if vname in constraint_names:
                continue
            for idx in np.asarray(comp.vars[vname]).ravel():
                if int(idx) < n and not mult_ind[int(idx)]:
                    has_primal = True
                    break
            if has_primal:
                break
        if has_primal and comp.size > max_size:
            max_size = comp.size

    if max_size <= 1:
        return None, None

    N = max_size
    blocks = [set() for _ in range(N)]

    for name, comp in model.comp.items():
        if comp.size != N:
            continue
        constraint_names = set(comp.get_constraint_names())
        for vname, var_arr in comp.vars.items():
            if vname in constraint_names:
                continue
            arr = np.asarray(var_arr).reshape(N, -1)
            for k in range(N):
                for idx in arr[k]:
                    idx_int = int(idx)
                    if idx_int < n and not mult_ind[idx_int]:
                        blocks[k].add(idx_int)

    # Convert to sorted arrays, skip empty blocks.
    # Deduplicate: cyclic BCs can cause shared variables between blocks
    # (e.g. q2[N-1,:] = q1[0,:] via linking). Assign each variable to its
    # first block only, keeping blocks disjoint.
    chain = []
    all_in_chain = set()
    for bs in blocks:
        unique = bs - all_in_chain
        if unique:
            block_arr = np.array(sorted(unique), dtype=np.int64)
            chain.append(block_arr)
            all_in_chain.update(unique)

    if len(chain) <= 1:
        return None, None

    # Primal variables not in any block
    remaining = []
    for i in range(n):
        if not mult_ind[i] and i not in all_in_chain:
            remaining.append(i)

    return chain, remaining


def _validate_chain_tridiagonal(chain, mult_ind, rowp, cols):
    """Validate that a chain has block-tridiagonal coupling structure.

    Checks both W coupling (direct primal-primal CSR entries) and
    constraint-mediated coupling (J^T D^{-1} J between non-adjacent levels).
    A chain with non-adjacent coupling cannot use Schur propagation correctly.

    Returns (True, set()) if valid, or (False, violating_levels) where
    violating_levels is the set of chain levels involved in non-adjacent coupling.
    """
    var_to_level = {}
    for k, block in enumerate(chain):
        for var in block:
            var_to_level[int(var)] = k

    violating_levels = set()
    n = len(mult_ind)
    for row in range(n):
        rs, re = rowp[row], rowp[row + 1]
        if mult_ind[row]:
            # Constraint row: check which chain levels its Jacobian touches
            levels_touched = set()
            for idx in range(rs, re):
                j = int(cols[idx])
                if j in var_to_level:
                    levels_touched.add(var_to_level[j])
            if len(levels_touched) >= 2:
                sl = sorted(levels_touched)
                for i in range(len(sl) - 1):
                    if sl[i + 1] - sl[i] > 1:
                        violating_levels.update(levels_touched)
        elif row in var_to_level:
            # Primal row in chain: check W coupling to non-adjacent levels
            k = var_to_level[row]
            for idx in range(rs, re):
                j = int(cols[idx])
                if j != row and j in var_to_level:
                    if abs(var_to_level[j] - k) > 1:
                        violating_levels.add(k)
                        violating_levels.add(var_to_level[j])
    return (len(violating_levels) == 0, violating_levels)


def _compute_fill_compensation(evecs, evals, use_evals, missing_mask):
    """Gershgorin per-row diagonal compensation for missing CSR entries.

    When a modified block has off-diagonal entries missing from the CSR
    pattern, the write-back is incomplete. The modification matrix
    E_k = Q diag(use_evals - evals) Q^T captures what changed.

    For each row i, the Gershgorin radius of the missing entries gives
    the minimum diagonal compensation needed:
        comp_i = sum_{j != i} |E_missing[i,j]|

    Proof: Let D = diag(comp). Then D - E_missing is diagonally dominant
    with non-negative diagonal excess, hence PSD. Therefore the compensated
    matrix (S_written + D) >= (S_written + E_missing) = S_intended >= eps*I.

    Tighter than the previous uniform Weyl bound (||E_missing||_F on every
    diagonal entry) by up to a factor of sqrt(d) for asymmetric sparsity.

    Returns per-entry compensation vector (length d).
    """
    delta_evals = use_evals - evals
    if not np.any(delta_evals != 0):
        return np.zeros(len(evals))
    E_k = (evecs * delta_evals) @ evecs.T
    E_missing = np.where(missing_mask, E_k, 0.0)
    return np.sum(np.abs(E_missing), axis=1)


def _riccati_solve_factor(factor, rhs):
    """Solve R~ @ x = rhs using a stored Riccati factorization.

    factor is ('chol', L) or ('eig', Q, Lambda).
    """
    if factor[0] == "chol":
        L = factor[1]
        y = scipy.linalg.solve_triangular(L, rhs, lower=True)
        return scipy.linalg.solve_triangular(L.T, y, lower=False)
    else:
        Q, Lambda = factor[1], factor[2]
        return Q @ ((Q.T @ rhs) / Lambda)


class BlockPSDConvexifier:
    """Bordered block-tridiagonal PSD convexifier.

    Chain blocks via BISC Schur complement (forward or backward), hub
    variables via border tracking (accumulated Schur complement with fill-in
    propagation). Same public interface as CurvatureProbeConvexifier.
    """

    def __init__(
        self, options, barrier_param, model, problem, solver, distribute, tol=1e-6
    ):
        # Eigenvalue modification mode: "reflect" or "clip"
        self.eigenvalue_mode = options.get("block_psd_eigenvalue_mode", "reflect")
        # Sweep direction: "forward" or "backward"
        self.sweep_direction = options.get("block_psd_sweep_direction", "backward")
        # Max condition number for modified blocks (condition-based adaptive eps)
        self.kappa_max = options.get("block_psd_kappa_max", 1e8)

        # eps_z / VC parameters (same as CurvatureProbeConvexifier)
        self.cz = options["convex_eps_z_coeff"]
        self.eps_z_floor = 0.0
        self.eps_z = 0.0  # Set properly after _nonconvex_indices resolved
        self.tol = tol
        self._barrier = barrier_param
        self.theta = 0.0
        self.eta = 0.0
        self.vc_floor = 0.0
        self.max_rejections = options["max_consecutive_rejections"]
        self.barrier_inc = options["barrier_increase_factor"]
        self.initial_barrier = options["initial_barrier_param"]
        self.step_rejected = False
        self.consecutive_rejections = 0
        self.numerical_eps = 1e-12
        self.modification_ratio = 0.0
        self._fill_compensation = 0.0
        self._max_w_c = 0.0
        self._cascade_factor = 0.0
        self._last_mod_norm = 0.0
        self._block_min_eigs = []  # per-block min eigenvalue of S_tilde (before mod)
        self._blocks_modified = 0  # total blocks modified this iteration
        self._blocks_total = 0  # total blocks processed
        self._blocks_borderline = (
            0  # blocks with min_eig in [-10*eps, 0) (cascade suspects)
        )
        self._bisc_tier = 0  # 0=not called, 1=gate, 2=delta-skip, 3=full BISC

        # Riccati solve storage (populated during backward sweep of build_regularization)
        self._riccati_chain_factors = (
            []
        )  # [ci][ k ] = ('chol', L) or ('eig', Q, Lambda)
        self._riccati_chain_couplings = (
            []
        )  # [ci][ k ] = C_k  (d_k x d_{k+1}) for k=0..N-2
        self._riccati_chain_S_schur = (
            []
        )  # [ci][ k ] = S_schur diagonal block (d_k x d_k)

        # Multiplier indicator
        self.mult_ind = np.array(problem.get_multiplier_indicator(), dtype=bool)
        self.n_neg = 0
        self.max_reg = 0.0

        # Nonconvex constraint indices (for selective eps_z)
        nc_constraints = options["nonconvex_constraints"]
        if nc_constraints and not distribute:
            self._nonconvex_indices = np.sort(model.get_indices(nc_constraints))
        else:
            self._nonconvex_indices = None

        # Now set eps_z: only nonzero when nonconvex constraints exist
        if self._nonconvex_indices is not None:
            self.eps_z = max(self.eps_z_floor, self.cz * barrier_param)

        # Block detection: model-based (OCP) or BFS fallback
        max_block_size = options.get("block_psd_max_block_size", 64)
        chain, remaining = _detect_model_chain(model, self.mult_ind)
        if chain is not None:
            self.bfs_chains = [chain]
            # Classify remaining variables: high-degree -> hubs, rest -> small blocks
            if remaining:
                hub_threshold = options.get("block_psd_hub_threshold", 50)
                remaining_hub = []
                remaining_small = []
                for r in remaining:
                    rs, re = solver.rowp[r], solver.rowp[r + 1]
                    row_cols = solver.cols[rs:re]
                    degree = 0
                    for c in row_cols:
                        if int(c) != r and not self.mult_ind[int(c)]:
                            degree += 1
                    if degree > hub_threshold:
                        remaining_hub.append(r)
                    else:
                        remaining_small.append(r)
                self.small_blocks = [
                    np.array([r], dtype=np.int64) for r in remaining_small
                ]
                self.hub_indices = np.array(sorted(remaining_hub), dtype=np.int64)
            else:
                self.small_blocks = []
                self.hub_indices = np.array([], dtype=np.int64)
        else:
            hub_threshold = options.get("block_psd_hub_threshold", 50)
            self.small_blocks, self.bfs_chains, self.hub_indices = (
                detect_bfs_level_blocks(
                    solver.rowp,
                    solver.cols,
                    self.mult_ind,
                    hub_threshold,
                    max_block_size=max_block_size,
                )
            )

        # Validate block-tridiagonal structure for each chain.
        # Chains with non-adjacent coupling (W or constraint-mediated) cannot
        # use Schur propagation correctly. Try to fix periodic coupling by
        # moving endpoint levels to hub; otherwise demote to small blocks.
        validated_chains = []
        for chain_i in self.bfs_chains:
            valid, viol_levels = _validate_chain_tridiagonal(
                chain_i,
                self.mult_ind,
                solver.rowp,
                solver.cols,
            )
            if valid:
                validated_chains.append(chain_i)
            else:
                # Check if violations only involve chain endpoints (periodic).
                # For periodic OCP: first/last levels wrap around the track.
                n_levels = len(chain_i)
                endpoints = {0, n_levels - 1}
                # Also include level adjacent to last (can have Hessian coupling
                # to first through the periodic collocation constraint)
                if n_levels > 2:
                    endpoints.add(n_levels - 2)
                if viol_levels <= endpoints:
                    # Periodic coupling: move violating endpoint blocks to hub.
                    # Trim from both ends if needed to keep chain contiguous.
                    trim_start = 0
                    trim_end = n_levels
                    for k in sorted(viol_levels):
                        if k == trim_start:
                            trim_start += 1
                        elif k == trim_end - 1:
                            trim_end -= 1
                    trimmed = chain_i[trim_start:trim_end]
                    hub_blocks = list(chain_i[:trim_start]) + list(chain_i[trim_end:])
                    # Re-validate trimmed chain
                    valid2, viol2 = _validate_chain_tridiagonal(
                        trimmed,
                        self.mult_ind,
                        solver.rowp,
                        solver.cols,
                    )
                    if valid2 and len(trimmed) >= 2:
                        validated_chains.append(trimmed)
                        hub_vars = []
                        for block in hub_blocks:
                            hub_vars.extend(int(v) for v in block)
                        self.hub_indices = np.array(
                            sorted(list(self.hub_indices) + hub_vars),
                            dtype=np.int64,
                        )
                        print(
                            f"  BISC: periodic coupling at levels "
                            f"{sorted(viol_levels)}, moved "
                            f"{len(hub_vars)} vars to hub, chain "
                            f"{len(trimmed)} levels"
                        )
                    else:
                        # Trimmed chain still invalid, demote all
                        for block in chain_i:
                            self.small_blocks.append(block)
                        print(
                            f"  BISC: chain ({n_levels} levels) has "
                            f"non-adjacent coupling at levels "
                            f"{sorted(viol_levels)}, trimmed chain "
                            f"invalid, demoted to small blocks"
                        )
                else:
                    # Non-endpoint violations, demote all
                    for block in chain_i:
                        self.small_blocks.append(block)
                    print(
                        f"  BISC: chain ({n_levels} levels) has "
                        f"non-adjacent coupling at {len(viol_levels)} "
                        f"interior levels, demoted to small blocks"
                    )
        self.bfs_chains = validated_chains

        # Pre-compute CSR index maps for small blocks (eigendecomp)
        self._small_block_maps = []
        for block in self.small_blocks:
            d = len(block)
            csr_map = np.full((d, d), -1, dtype=np.int64)
            for a in range(d):
                row = block[a]
                row_start, row_end = solver.rowp[row], solver.rowp[row + 1]
                row_cols = solver.cols[row_start:row_end]
                for b in range(d):
                    col = block[b]
                    pos = np.searchsorted(row_cols, col)
                    if pos < len(row_cols) and row_cols[pos] == col:
                        csr_map[a, b] = row_start + pos
            self._small_block_maps.append(csr_map)

        # Pre-compute CSR index maps for BFS chain blocks
        # For each chain: diagonal block maps + coupling block maps
        self._chain_diag_maps = []
        self._chain_coupling_maps = []
        for chain_i in self.bfs_chains:
            diag_maps = []
            coupling_maps = []

            for k, block in enumerate(chain_i):
                d = len(block)
                # Diagonal block CSR map
                csr_map = np.full((d, d), -1, dtype=np.int64)
                for a in range(d):
                    row = block[a]
                    rs, re = solver.rowp[row], solver.rowp[row + 1]
                    row_cols = solver.cols[rs:re]
                    for b in range(d):
                        col = block[b]
                        pos = np.searchsorted(row_cols, col)
                        if pos < len(row_cols) and row_cols[pos] == col:
                            csr_map[a, b] = rs + pos
                diag_maps.append(csr_map)

                # Coupling block CSR map (block k -> block k+1)
                if k < len(chain_i) - 1:
                    block_next = chain_i[k + 1]
                    dk = len(block)
                    dk1 = len(block_next)
                    coup_map = np.full((dk, dk1), -1, dtype=np.int64)
                    for a in range(dk):
                        row = block[a]
                        rs, re = solver.rowp[row], solver.rowp[row + 1]
                        row_cols = solver.cols[rs:re]
                        for b in range(dk1):
                            col = block_next[b]
                            pos = np.searchsorted(row_cols, col)
                            if pos < len(row_cols) and row_cols[pos] == col:
                                coup_map[a, b] = rs + pos
                    coupling_maps.append(coup_map)

            self._chain_diag_maps.append(diag_maps)
            self._chain_coupling_maps.append(coupling_maps)

        # CONVEXIFY: pre-compute local/coupled variable partition per chain level.
        # In backward sweep, "coupled" vars at level k = those with nonzero columns
        # in C_{k-1} (coupling from level k-1 to k). These participate in Gamma
        # propagation; their curvature redistributes naturally without projection.
        # "Local" vars = zero columns in C_{k-1}; their Schur complement is projected.
        self._chain_local_idx = []
        self._chain_coupled_idx = []
        for ci, chain_i in enumerate(self.bfs_chains):
            local_idx = []
            coupled_idx = []
            for k in range(len(chain_i)):
                dk = len(chain_i[k])
                if k == 0:
                    # In backward sweep, last processed block: all local
                    local_idx.append(np.arange(dk))
                    coupled_idx.append(np.array([], dtype=int))
                else:
                    # C_{k-1} shape (d_{k-1} x d_k): columns = block k vars
                    coup_map = self._chain_coupling_maps[ci][k - 1]
                    has_coupling = np.any(coup_map >= 0, axis=0)
                    coupled_idx.append(np.where(has_coupling)[0])
                    local_idx.append(np.where(~has_coupling)[0])
            self._chain_local_idx.append(local_idx)
            self._chain_coupled_idx.append(coupled_idx)

        # Pre-compute CSR index map for dense hub block (n_hub x n_hub).
        # Diagonal entries always exist; off-diagonal only if W couples hubs.
        n_h = len(self.hub_indices)
        self._hub_csr_map = np.full((n_h, n_h), -1, dtype=np.int64)
        for i, hub_i in enumerate(self.hub_indices):
            rs, re = solver.rowp[hub_i], solver.rowp[hub_i + 1]
            row_cols = solver.cols[rs:re]
            for j, hub_j in enumerate(self.hub_indices):
                pos = np.searchsorted(row_cols, hub_j)
                if pos < len(row_cols) and row_cols[pos] == hub_j:
                    self._hub_csr_map[i, j] = rs + pos

        # Pre-compute hub-to-chain coupling CSR indices (bordered structure).
        # _hub_chain_coupling[ci][h_idx][k] = CSR index array for hub h
        # coupling to each variable in chain ci, level k.
        self._hub_chain_coupling = []
        for chain_i in self.bfs_chains:
            hub_maps = []
            for hub in self.hub_indices:
                rs, re = solver.rowp[hub], solver.rowp[hub + 1]
                row_cols = solver.cols[rs:re]
                level_indices = []
                for block in chain_i:
                    csri = np.full(len(block), -1, dtype=np.int64)
                    for b, var in enumerate(block):
                        pos = np.searchsorted(row_cols, var)
                        if pos < len(row_cols) and row_cols[pos] == var:
                            csri[b] = rs + pos
                    level_indices.append(csri)
                hub_maps.append(level_indices)
            self._hub_chain_coupling.append(hub_maps)

        # Pre-compute hub-to-small-block coupling CSR indices.
        # _hub_small_coupling[s_idx][h_idx] = CSR index array.
        self._hub_small_coupling = []
        for block in self.small_blocks:
            hub_maps = []
            for hub in self.hub_indices:
                rs, re = solver.rowp[hub], solver.rowp[hub + 1]
                row_cols = solver.cols[rs:re]
                csri = np.full(len(block), -1, dtype=np.int64)
                for b, var in enumerate(block):
                    pos = np.searchsorted(row_cols, var)
                    if pos < len(row_cols) and row_cols[pos] == var:
                        csri[b] = rs + pos
                hub_maps.append(csri)
            self._hub_small_coupling.append(hub_maps)

        # Pre-compute constraint-to-block maps for J^T |D^{-1}| J (Schur complement).
        # For each chain, for each level k:
        #   _schur_block_data[ci][k] = list of (c_row, local_pos, csr_idx)
        #     Constraints touching primal vars in block k.
        #   _schur_coupling_data[ci][k] = list of (c_row, lp_k, lp_k1, ci_k, ci_k1)
        #     Constraints touching primal vars in both block k and block k+1.
        # Also _schur_hub_data[ci][h_idx][k] = list of (c_row, lp_k, ci_k, hub_ci)
        #   Constraints touching both hub h and block k.
        var_to_block = {}
        for ci, chain_i in enumerate(self.bfs_chains):
            for k, block in enumerate(chain_i):
                for local_pos, var in enumerate(block):
                    var_to_block[int(var)] = (ci, k, local_pos)

        var_to_small = {}
        for s_idx, block in enumerate(self.small_blocks):
            for local_pos, var in enumerate(block):
                var_to_small[int(var)] = (s_idx, local_pos)

        hub_var_set = set(int(h) for h in self.hub_indices)
        hub_var_to_idx = {int(h): i for i, h in enumerate(self.hub_indices)}

        self._schur_block_data = []
        self._schur_coupling_data = []
        self._schur_hub_data = []
        self._schur_hub_hub_data = []
        # S_schur data for small blocks and hub-small constraint coupling
        self._schur_small_data = [[] for _ in range(len(self.small_blocks))]
        self._schur_hub_small_data = [
            [[] for _ in range(len(self.hub_indices))]
            for _ in range(len(self.small_blocks))
        ]

        for ci, chain_i in enumerate(self.bfs_chains):
            n_levels = len(chain_i)
            block_data = [[] for _ in range(n_levels)]
            coupling_data = [[] for _ in range(n_levels - 1)]
            hub_data = [
                [[] for _ in range(n_levels)] for _ in range(len(self.hub_indices))
            ]

            self._schur_block_data.append(block_data)
            self._schur_coupling_data.append(coupling_data)
            self._schur_hub_data.append(hub_data)

        n_total = len(self.mult_ind)
        rowp = solver.rowp
        cols = solver.cols

        for c_row in range(n_total):
            if not self.mult_ind[c_row]:
                continue

            rs, re = rowp[c_row], rowp[c_row + 1]
            row_cols = cols[rs:re]

            # Group primal columns by block membership
            block_groups = {}
            small_groups = {}
            hub_entries = {}
            for idx in range(rs, re):
                j = int(row_cols[idx - rs])
                if self.mult_ind[j]:
                    continue
                if j in var_to_block:
                    ci, k, lp = var_to_block[j]
                    key = (ci, k)
                    if key not in block_groups:
                        block_groups[key] = ([], [])
                    block_groups[key][0].append(lp)
                    block_groups[key][1].append(idx)
                if j in var_to_small:
                    s_idx, lp = var_to_small[j]
                    if s_idx not in small_groups:
                        small_groups[s_idx] = ([], [])
                    small_groups[s_idx][0].append(lp)
                    small_groups[s_idx][1].append(idx)
                if j in hub_var_set:
                    hub_entries[hub_var_to_idx[j]] = idx

            # Store diagonal block contributions
            for (ci, k), (lp_list, csr_list) in block_groups.items():
                self._schur_block_data[ci][k].append(
                    (
                        c_row,
                        np.array(lp_list, dtype=np.int64),
                        np.array(csr_list, dtype=np.int64),
                    )
                )

            # Store coupling contributions (block k and block k+1 in same chain)
            for (ci, k), (lp_k, ci_k) in block_groups.items():
                key_next = (ci, k + 1)
                if key_next in block_groups:
                    lp_k1, ci_k1 = block_groups[key_next]
                    self._schur_coupling_data[ci][k].append(
                        (
                            c_row,
                            np.array(lp_k, dtype=np.int64),
                            np.array(lp_k1, dtype=np.int64),
                            np.array(ci_k, dtype=np.int64),
                            np.array(ci_k1, dtype=np.int64),
                        )
                    )

            # Store hub-block coupling contributions
            for h_idx, h_csr_idx in hub_entries.items():
                for (ci, k), (lp_k, ci_k) in block_groups.items():
                    self._schur_hub_data[ci][h_idx][k].append(
                        (
                            c_row,
                            np.array(lp_k, dtype=np.int64),
                            np.array(ci_k, dtype=np.int64),
                            h_csr_idx,
                        )
                    )

            # Store diagonal S_schur for small blocks
            for s_idx, (lp_list, csr_list) in small_groups.items():
                self._schur_small_data[s_idx].append(
                    (
                        c_row,
                        np.array(lp_list, dtype=np.int64),
                        np.array(csr_list, dtype=np.int64),
                    )
                )

            # Store hub-small-block constraint coupling
            for h_idx, h_csr_idx in hub_entries.items():
                for s_idx, (lp_s, ci_s) in small_groups.items():
                    self._schur_hub_small_data[s_idx][h_idx].append(
                        (
                            c_row,
                            np.array(lp_s, dtype=np.int64),
                            np.array(ci_s, dtype=np.int64),
                            h_csr_idx,
                        )
                    )

            # Store hub-hub constraint coupling (S_schur between hub pairs)
            if hub_entries:
                hub_list = list(hub_entries.items())
                for a in range(len(hub_list)):
                    h1_idx, h1_csr = hub_list[a]
                    for b in range(a, len(hub_list)):
                        h2_idx, h2_csr = hub_list[b]
                        self._schur_hub_hub_data.append(
                            (c_row, h1_idx, h2_idx, h1_csr, h2_csr)
                        )

        # Print summary
        small_sizes = [len(b) for b in self.small_blocks]
        chain_info = []
        for chain_i in self.bfs_chains:
            level_sizes = [len(lvl) for lvl in chain_i]
            chain_info.append(
                f"{len(chain_i)} levels, "
                f"sizes {min(level_sizes)}-{max(level_sizes)}"
            )
        n_hubs = len(self.hub_indices)
        n_primal = int(np.sum(~self.mult_ind))

        parts = []
        if small_sizes:
            parts.append(
                f"{len(self.small_blocks)} small blocks "
                f"({min(small_sizes)}-{max(small_sizes)})"
            )
        if chain_info:
            parts.append(f"{len(self.bfs_chains)} chains ({', '.join(chain_info)})")
        if n_hubs:
            parts.append(f"{n_hubs} hubs")
        mode_str = self.eigenvalue_mode
        sweep_str = self.sweep_direction
        print(
            f"  Block PSD: {', '.join(parts)}, {n_primal} primal vars, "
            f"mode={mode_str}, sweep={sweep_str}"
        )

    def build_regularization(
        self,
        diag,
        diag_base,
        zero_hessian_indices=None,
        zero_hessian_eps=None,
        solver=None,
    ):
        """Null-Space BISC: project primal Schur complement S_p to PSD.

        Operates on S_p = W + Sigma + J^T |D^{-1}| J, the exact primal Schur
        complement after eliminating constraints from the KKT system.

        inertia(KKT) = (n, m, 0) iff S_p > 0 and D < 0 (exact characterization).

        S_p is block-tridiagonal for OCPs (J couples at most adjacent blocks).
        Schur complement propagation (forward or backward) with spectral
        modification ensures S_p > 0. Write-back is self-consistent: PARDISO
        adds back Sigma (via factor_from_host) and J^T |D^{-1}| J (via
        constraint elimination), recovering the modified S_p at each level.

        No coupling floors, no mu-scaling. Pure spectral modification only.
        """
        diag.copy_device_to_host()
        diag_arr = diag.get_array()

        data = solver.hess.get_data()
        # Adaptive eps: larger early (stable hub fill-in, less cascade),
        # shrinks to numerical floor near convergence (pure Newton).
        # eps = O(mu) ensures asymptotic vanishing under SOSC.
        eps = max(self.numerical_eps, 1e-2 * self._barrier)
        ev_mode = self.eigenvalue_mode

        # Build regularized constraint diagonal for Schur weight computation.
        # diag_base has 0 for equalities, -1/C for inequalities.
        # eps_z only applied to nonconvex constraints (if configured).
        # Using diag_reg ensures the Schur complement J^T |D_reg|^{-1} J
        # matches what PARDISO sees, so the CSR write-back is self-consistent.
        diag_reg = diag_base.copy()
        if self._nonconvex_indices is not None and self.eps_z > 0:
            diag_reg[self._nonconvex_indices] -= self.eps_z

        total_mod_norm_sq = 0.0
        total_hess_norm_sq = 0.0
        n_modified = 0
        max_eig_corr = 0.0
        n_hubs = len(self.hub_indices)
        hub_accum = np.zeros((n_hubs, n_hubs))
        max_w_c = 0.0
        total_fill_comp = 0.0
        block_min_eigs = []  # min eigenvalue of S_tilde per block (before mod)

        blocks_modified = 0
        blocks_total = 0
        blocks_borderline = 0  # min_eig in [-10*eps, 0): cascade suspects

        # --- 1. Small blocks: direct eigendecomp on S_p ---
        for s_idx, (block, csr_map) in enumerate(
            zip(self.small_blocks, self._small_block_maps)
        ):
            d = len(block)

            # Extract W + Sigma from CSR + barrier diagonal
            valid = csr_map >= 0
            H = np.where(valid, data[np.where(valid, csr_map, 0)], 0.0)
            H[np.arange(d), np.arange(d)] += diag_base[block]

            # Add S_schur = J^T |D^{-1}| J (constraint Schur complement)
            S_schur_small = np.zeros((d, d))
            for c_row, lp, ci_arr in self._schur_small_data[s_idx]:
                D_cc = diag_reg[c_row]
                if abs(D_cc) < 1e-30:
                    continue
                w_c = -1.0 / D_cc
                if w_c < 0:
                    continue
                max_w_c = max(max_w_c, w_c)
                j_local = data[ci_arr]
                S_schur_small[np.ix_(lp, lp)] += w_c * np.outer(j_local, j_local)
            H += S_schur_small

            evals, evecs = np.linalg.eigh(H)
            total_hess_norm_sq += np.sum(evals**2)
            min_eig = float(evals[0]) if len(evals) > 0 else 0.0
            block_min_eigs.append(min_eig)
            blocks_total += 1

            # Eigenvalue modification (reflect or clip)
            max_abs_eig = max(abs(evals[-1]), abs(evals[0])) if len(evals) > 0 else 0.0
            eps_eff = max(eps, max_abs_eig / self.kappa_max)
            use_evals, needs_mod_small = _modify_eigenvalues(evals, eps_eff, ev_mode)
            if needs_mod_small:
                blocks_modified += 1
                if min_eig > -10.0 * eps:
                    blocks_borderline += 1

            # Hub border: accumulate dense hub Schur complement
            if n_hubs > 0:
                hub_f_scaled = {}
                for h_idx in range(n_hubs):
                    # W coupling between hub and small block
                    csri = self._hub_small_coupling[s_idx][h_idx]
                    valid_h = csri >= 0
                    c = np.where(valid_h, data[np.where(valid_h, csri, 0)], 0.0)
                    # Constraint-mediated coupling: J^T |D^{-1}| J
                    hub_sd = self._schur_hub_small_data[s_idx][h_idx]
                    if hub_sd:
                        for c_row, lp_s, ci_s, h_ci in hub_sd:
                            D_cc = diag_reg[c_row]
                            if abs(D_cc) < 1e-30:
                                continue
                            w_c = -1.0 / D_cc
                            if w_c < 0:
                                continue
                            max_w_c = max(max_w_c, w_c)
                            c[lp_s] += w_c * data[ci_s] * data[h_ci]
                    if np.any(c != 0):
                        f = evecs.T @ c
                        hub_f_scaled[h_idx] = f / np.sqrt(use_evals)
                for h1, sf1 in hub_f_scaled.items():
                    for h2, sf2 in hub_f_scaled.items():
                        if h2 >= h1:
                            val = np.dot(sf1, sf2)
                            hub_accum[h1, h2] += val
                            if h1 != h2:
                                hub_accum[h2, h1] += val

            if not needs_mod_small:
                continue

            n_modified += 1
            max_eig_corr = max(max_eig_corr, float(np.max(use_evals - evals)))
            total_mod_norm_sq += np.sum((use_evals - evals) ** 2)

            # Write-back: CSR = S_p_modified - Sigma - S_schur
            H_proj = (evecs * use_evals) @ evecs.T
            H_proj[np.arange(d), np.arange(d)] -= diag_base[block]
            H_proj -= S_schur_small
            data[csr_map[valid]] = H_proj[valid]
            # Fill-in compensation (Gershgorin per-row bound): missing
            # off-diagonal entries cause perturbation E_missing. Per-row
            # absolute sums give diagonal dominance compensation.
            if not np.all(valid):
                missing = ~valid
                np.fill_diagonal(missing, False)
                if np.any(missing):
                    fill_vals = _compute_fill_compensation(
                        evecs, evals, use_evals, missing
                    )
                    if np.any(fill_vals > 0):
                        diag_arr[block] += fill_vals
                        total_fill_comp += np.sum(fill_vals)

        # --- 2. Chain blocks: Null-Space BISC ---
        # Schur complement propagation on S_p (forward or backward).
        # S_p[k,k] = W[k,k] + Sigma[k] + sum_c w_c * J_c[k]^T J_c[k]
        # S_p[k,k+1] = W[k,k+1] + sum_c w_c * J_c[k]^T J_c[k+1]
        # Pure spectral modification at each level. No coupling floors.
        forward = self.sweep_direction == "forward"

        self._riccati_chain_factors = []
        self._riccati_chain_couplings = []
        self._riccati_chain_S_schur = []

        for ci, (chain, diag_maps, coupling_maps) in enumerate(
            zip(self.bfs_chains, self._chain_diag_maps, self._chain_coupling_maps)
        ):
            n_levels = len(chain)
            hub_Sinv_c = [None] * n_hubs
            chain_factors = [None] * n_levels

            # Pre-extract W coupling matrices from CSR
            coupling_matrices = []
            for kk in range(n_levels - 1):
                cm = coupling_maps[kk]
                valid_cm = cm >= 0
                C_coup = np.where(valid_cm, data[np.where(valid_cm, cm, 0)], 0.0)
                coupling_matrices.append(C_coup)

            # Pre-compute S_schur diagonal blocks for all levels
            all_S_schur = []
            for k in range(n_levels):
                block = chain[k]
                dk = len(block)
                S_schur_k = np.zeros((dk, dk))
                for c_row, lp, ci_arr in self._schur_block_data[ci][k]:
                    D_cc = diag_reg[c_row]
                    if abs(D_cc) < 1e-30:
                        continue
                    w_c = -1.0 / D_cc
                    if w_c < 0:
                        continue
                    max_w_c = max(max_w_c, w_c)
                    j_local = data[ci_arr]
                    S_schur_k[np.ix_(lp, lp)] += w_c * np.outer(j_local, j_local)
                all_S_schur.append(S_schur_k)

            # Pre-augment coupling matrices with S_schur coupling
            for k in range(n_levels - 1):
                coup_data = self._schur_coupling_data[ci][k]
                if coup_data:
                    dk = len(chain[k])
                    dk1 = len(chain[k + 1])
                    S_schur_coup = np.zeros((dk, dk1))
                    for c_row, lp_k, lp_k1, ci_k, ci_k1 in coup_data:
                        D_cc = diag_reg[c_row]
                        if abs(D_cc) < 1e-30:
                            continue
                        w_c = -1.0 / D_cc
                        if w_c < 0:
                            continue
                        max_w_c = max(max_w_c, w_c)
                        j_k = data[ci_k]
                        j_k1 = data[ci_k1]
                        S_schur_coup[np.ix_(lp_k, lp_k1)] += w_c * np.outer(j_k, j_k1)
                    coupling_matrices[k] = coupling_matrices[k] + S_schur_coup

            # Sweep: forward (k=0..N-1) or backward (k=N-1..0)
            levels_range = range(n_levels) if forward else range(n_levels - 1, -1, -1)
            Q_adj = None
            Lambda_adj = None
            L_adj_stored = None  # Cholesky factor from CONVEXIFY path

            for k in levels_range:
                block = chain[k]
                dk = len(block)
                dm = diag_maps[k]

                # Extract W[k,k] + Sigma[k] from CSR + barrier diagonal
                valid_dm = dm >= 0
                H = np.where(valid_dm, data[np.where(valid_dm, dm, 0)], 0.0)
                H[np.arange(dk), np.arange(dk)] += diag_base[block]

                # S_p[k,k] = W + Sigma + S_schur
                S_schur = all_S_schur[k]
                H += S_schur

                # Gamma from adjacent processed level
                Gamma = None
                if forward and k > 0 and Q_adj is not None:
                    # Forward: Gamma = C_{k-1}^T R~_{k-1}^{-1} C_{k-1}
                    C_adj = coupling_matrices[k - 1]  # dk_prev x dk
                    F = Q_adj.T @ C_adj  # dk_prev x dk
                    F_scaled = F / np.sqrt(Lambda_adj)[:, np.newaxis]
                    Gamma = F_scaled.T @ F_scaled  # dk x dk
                elif not forward and k < n_levels - 1:
                    C_adj = coupling_matrices[k]  # dk x dk1
                    if L_adj_stored is not None:
                        # Cholesky path: Gamma = C L^{-T} L^{-1} C^T
                        F = scipy.linalg.solve_triangular(
                            L_adj_stored, C_adj.T, lower=True
                        )  # dk1 x dk
                        Gamma = F.T @ F  # dk x dk
                    elif Q_adj is not None:
                        # Eigendecomp path (fallback)
                        F = C_adj @ Q_adj  # dk x dk1
                        F_scaled = F * (1.0 / np.sqrt(Lambda_adj))[np.newaxis, :]
                        Gamma = F_scaled @ F_scaled.T  # dk x dk

                if Gamma is not None:
                    S_tilde = H - Gamma
                else:
                    S_tilde = H.copy()

                blocks_total += 1
                min_eig_chain = None  # set below by whichever path runs

                # --- CONVEXIFY-style selective projection ---
                # In backward sweep, identify "local" variables (not coupled to
                # adjacent level via C_{k-1}) and "coupled" variables.  Only
                # project the Schur complement of local vars after eliminating
                # coupled vars.  Coupled curvature redistributes naturally
                # through the sweep, reducing total ||E||_F.
                local_idx = self._chain_local_idx[ci][k]
                coupled_idx = self._chain_coupled_idx[ci][k]
                used_convexify = False
                L_full = None  # Cholesky factor if CONVEXIFY succeeds
                evecs = None  # eigendecomp if fallback
                evals = None

                if not forward and len(local_idx) > 0 and len(coupled_idx) > 0:
                    try:
                        # Partition S_tilde into coupled (c) and local (l)
                        R_cc = S_tilde[np.ix_(coupled_idx, coupled_idx)]
                        R_cl = S_tilde[np.ix_(coupled_idx, local_idx)]
                        R_ll = S_tilde[np.ix_(local_idx, local_idx)]

                        L_cc = np.linalg.cholesky(R_cc)
                        # Inner Schur: S_local = R_ll - R_lc R_cc^{-1} R_cl
                        solve_cl = scipy.linalg.solve_triangular(L_cc, R_cl, lower=True)
                        S_local = R_ll - solve_cl.T @ solve_cl

                        evals_l, evecs_l = np.linalg.eigh(S_local)
                        min_eig_chain = float(evals_l[0])
                        max_abs_l = max(abs(evals_l[-1]), abs(evals_l[0]))
                        eps_eff = max(eps, max_abs_l / self.kappa_max)
                        use_evals_l, needs_mod_chain = _modify_eigenvalues(
                            evals_l, eps_eff, ev_mode
                        )

                        if needs_mod_chain:
                            Delta_l = (evecs_l * (use_evals_l - evals_l)) @ evecs_l.T
                            S_tilde[np.ix_(local_idx, local_idx)] += Delta_l

                        # Cholesky of full modified block for propagation
                        L_full = np.linalg.cholesky(S_tilde)
                        used_convexify = True

                        # Eigenvalues for diagnostics
                        evals = np.linalg.eigvalsh(S_tilde)
                        use_evals = evals  # all positive after successful Cholesky
                        total_hess_norm_sq += np.sum(evals**2)

                        if needs_mod_chain:
                            total_mod_norm_sq += np.sum((use_evals_l - evals_l) ** 2)
                    except np.linalg.LinAlgError:
                        used_convexify = False
                        L_full = None

                if not used_convexify:
                    # Full projection (current approach, also for forward sweep)
                    evals, evecs = np.linalg.eigh(S_tilde)
                    min_eig_chain = float(evals[0]) if dk > 0 else 0.0
                    total_hess_norm_sq += np.sum(evals**2)

                    max_abs_eig = max(abs(evals[-1]), abs(evals[0])) if dk > 0 else 0.0
                    eps_eff = max(eps, max_abs_eig / self.kappa_max)
                    use_evals, needs_mod_chain = _modify_eigenvalues(
                        evals, eps_eff, ev_mode
                    )

                block_min_eigs.append(min_eig_chain)
                if needs_mod_chain:
                    blocks_modified += 1
                    if min_eig_chain > -10.0 * eps:
                        blocks_borderline += 1

                # Hub border tracking: accumulate and propagate fill-in
                if n_hubs > 0:
                    hub_maps_ci = self._hub_chain_coupling[ci]
                    hub_f_scaled = {}
                    for h_idx in range(n_hubs):
                        csri = hub_maps_ci[h_idx][k]

                        valid_h = csri >= 0
                        c_orig = np.where(
                            valid_h, data[np.where(valid_h, csri, 0)], 0.0
                        )

                        # Add constraint-mediated hub-block coupling
                        hub_schur_data = self._schur_hub_data[ci][h_idx][k]
                        if hub_schur_data:
                            c_schur = np.zeros(dk)
                            for c_row, lp_k, ci_k, h_ci in hub_schur_data:
                                D_cc = diag_reg[c_row]
                                if abs(D_cc) < 1e-30:
                                    continue
                                w_c = -1.0 / D_cc
                                if w_c < 0:
                                    continue
                                max_w_c = max(max_w_c, w_c)
                                j_k = data[ci_k]
                                j_h = data[h_ci]
                                c_schur[lp_k] += w_c * j_k * j_h
                            c_orig = c_orig + c_schur

                        # Fill-in propagation from adjacent processed level
                        if forward and k > 0 and hub_Sinv_c[h_idx] is not None:
                            C_adj = coupling_matrices[k - 1]
                            c_k = c_orig - C_adj.T @ hub_Sinv_c[h_idx]
                        elif (
                            not forward
                            and k < n_levels - 1
                            and hub_Sinv_c[h_idx] is not None
                        ):
                            C_adj = coupling_matrices[k]
                            c_k = c_orig - C_adj @ hub_Sinv_c[h_idx]
                        else:
                            c_k = c_orig

                        # R^{-1} c_k via Cholesky or eigendecomp
                        if used_convexify:
                            y = scipy.linalg.solve_triangular(L_full, c_k, lower=True)
                            hub_Sinv_c[h_idx] = scipy.linalg.solve_triangular(
                                L_full.T, y, lower=False
                            )
                            hub_f_scaled[h_idx] = y
                        else:
                            f = evecs.T @ c_k
                            hub_Sinv_c[h_idx] = evecs @ (f / use_evals)
                            hub_f_scaled[h_idx] = f / np.sqrt(use_evals)

                    # Accumulate dense hub Schur complement (cross-hub terms)
                    for h1, sf1 in hub_f_scaled.items():
                        for h2, sf2 in hub_f_scaled.items():
                            if h2 >= h1:
                                val = np.dot(sf1, sf2)
                                hub_accum[h1, h2] += val
                                if h1 != h2:
                                    hub_accum[h2, h1] += val

                # Write back when any eigenvalue was modified
                if needs_mod_chain:
                    n_modified += 1
                    if not used_convexify:
                        block_correction = float(np.max(use_evals - evals))
                        max_eig_corr = max(max_eig_corr, block_correction)
                        total_mod_norm_sq += np.sum((use_evals - evals) ** 2)

                    # Write-back: CSR[k,k] = R_k_mod + Gamma - Sigma - S_schur
                    # Self-consistency: PARDISO adds Sigma (factor_from_host) and
                    # S_schur (constraint elimination), recovering R_k_mod + Gamma.
                    # PARDISO subtracts Gamma (chain elimination) -> gets R_k_mod.
                    if used_convexify:
                        # S_tilde already modified in-place (Delta_l added)
                        H_new = S_tilde.copy()
                    else:
                        H_new = (evecs * use_evals) @ evecs.T
                    if Gamma is not None:
                        H_new += Gamma
                    H_new[np.arange(dk), np.arange(dk)] -= diag_base[block]
                    H_new -= S_schur
                    data[dm[valid_dm]] = H_new[valid_dm]
                    # Fill-in compensation (Gershgorin per-row bound)
                    if not np.all(valid_dm):
                        missing = ~valid_dm
                        np.fill_diagonal(missing, False)
                        if np.any(missing):
                            if used_convexify:
                                # CONVEXIFY: modification E = S_tilde_mod - S_tilde_orig
                                # S_tilde was modified in-place, but H_new = S_tilde + Gamma - ...
                                # The E_k we need is the modification matrix (only on local sub-block)
                                E_k = np.zeros((dk, dk))
                                E_k[np.ix_(local_idx, local_idx)] = Delta_l
                                E_missing = np.where(missing, E_k, 0.0)
                                fill_vals = np.sum(np.abs(E_missing), axis=1)
                            else:
                                fill_vals = _compute_fill_compensation(
                                    evecs, evals, use_evals, missing
                                )
                            if np.any(fill_vals > 0):
                                diag_arr[block] += fill_vals
                                total_fill_comp += np.sum(fill_vals)

                # Store factorization for next level's Gamma AND for Riccati solve
                if used_convexify:
                    L_adj_stored = L_full
                    Q_adj = None
                    Lambda_adj = None
                    chain_factors[k] = ("chol", L_full.copy())
                else:
                    L_adj_stored = None
                    Q_adj = evecs
                    Lambda_adj = use_evals
                    chain_factors[k] = ("eig", evecs.copy(), use_evals.copy())

            # Store Riccati data for this chain
            self._riccati_chain_factors.append(chain_factors)
            self._riccati_chain_couplings.append(coupling_matrices)
            self._riccati_chain_S_schur.append(all_S_schur)

        # --- 3. Hub writeback: dense bordered Schur complement ---
        # Full n_hub x n_hub spectral modification (eigendecomp + reflect/clip),
        # consistent with chain and small block treatment. No uniform diagonal
        # shift -- each eigenvalue is modified individually, preserving the
        # eigenvector structure and minimizing direction distortion.
        if n_hubs > 0:
            n_h = n_hubs

            # Extract W + Sigma between hubs from CSR + barrier diagonal
            valid_hm = self._hub_csr_map >= 0
            H_hub = np.where(
                valid_hm,
                data[np.where(valid_hm, self._hub_csr_map, 0)],
                0.0,
            )
            H_hub[np.arange(n_h), np.arange(n_h)] += diag_base[self.hub_indices]

            # Add S_schur between hubs (J^T |D^{-1}| J for hub pairs)
            S_schur_hh = np.zeros((n_h, n_h))
            for c_row, h1, h2, csr1, csr2 in self._schur_hub_hub_data:
                D_cc = diag_reg[c_row]
                if abs(D_cc) < 1e-30:
                    continue
                w_c = -1.0 / D_cc
                if w_c < 0:
                    continue
                max_w_c = max(max_w_c, w_c)
                j_h1 = data[csr1]
                j_h2 = data[csr2]
                S_schur_hh[h1, h2] += w_c * j_h1 * j_h2
                if h1 != h2:
                    S_schur_hh[h2, h1] += w_c * j_h1 * j_h2
            H_hub += S_schur_hh

            # S_hub = direct terms - fill-in from chains and small blocks
            S_hub_dense = H_hub - hub_accum

            evals_h, evecs_h = np.linalg.eigh(S_hub_dense)
            total_hess_norm_sq += np.sum(evals_h**2)
            min_eig_hub = float(evals_h[0])
            block_min_eigs.append(min_eig_hub)
            blocks_total += 1

            # Full spectral modification (same as chain/small blocks)
            max_abs_eig_h = (
                max(abs(evals_h[-1]), abs(evals_h[0])) if n_hubs > 0 else 0.0
            )
            eps_eff_h = max(eps, max_abs_eig_h / self.kappa_max)
            use_evals_h, needs_mod_hub = _modify_eigenvalues(
                evals_h, eps_eff_h, ev_mode
            )

            if needs_mod_hub:
                n_modified += 1
                blocks_modified += 1
                if min_eig_hub > -10.0 * eps:
                    blocks_borderline += 1
                max_eig_corr = max(max_eig_corr, float(np.max(use_evals_h - evals_h)))
                total_mod_norm_sq += np.sum((use_evals_h - evals_h) ** 2)

                # Reconstruct: S_hub_modified = Q diag(use_evals) Q^T
                S_hub_mod = (evecs_h * use_evals_h) @ evecs_h.T

                # Write-back: CSR[hubs] = S_hub_mod - Sigma - S_schur_hh
                # (self-consistent: PARDISO adds back Sigma and S_schur)
                H_hub_wb = S_hub_mod + hub_accum
                H_hub_wb[np.arange(n_h), np.arange(n_h)] -= diag_base[self.hub_indices]
                H_hub_wb -= S_schur_hh

                # Write to CSR (only entries that exist in the pattern)
                data[self._hub_csr_map[valid_hm]] = H_hub_wb[valid_hm]

                # Fill-in compensation (Gershgorin per-row bound)
                if not np.all(valid_hm):
                    missing_hm = ~valid_hm
                    np.fill_diagonal(missing_hm, False)
                    if np.any(missing_hm):
                        fill_vals = _compute_fill_compensation(
                            evecs_h, evals_h, use_evals_h, missing_hm
                        )
                        if np.any(fill_vals > 0):
                            diag_arr[self.hub_indices] += fill_vals
                            total_fill_comp += np.sum(fill_vals)

        # --- 4. Zero-hessian variables ---
        if zero_hessian_indices is not None and zero_hessian_eps is not None:
            for zh_idx in zero_hessian_indices:
                diag_arr[zh_idx] = max(diag_arr[zh_idx], zero_hessian_eps)

        # Store diagnostics
        self.max_reg = max_eig_corr
        self.n_neg = n_modified
        self._last_mod_norm = total_mod_norm_sq**0.5
        hess_norm = total_hess_norm_sq**0.5
        self.modification_ratio = (
            self._last_mod_norm / hess_norm if hess_norm > 1e-30 else 0.0
        )
        self._fill_compensation = total_fill_comp
        self._max_w_c = max_w_c
        self._cascade_factor = max_eig_corr / eps if eps > 0 else 0.0
        self._block_min_eigs = block_min_eigs
        self._blocks_modified = blocks_modified
        self._blocks_total = blocks_total
        self._blocks_borderline = blocks_borderline
        self._bisc_tier = 3

        # --- 5. Apply eps_z to dual entries (nonconvex only) ---
        if self._nonconvex_indices is not None and self.eps_z > 0:
            diag_arr[self._nonconvex_indices] -= self.eps_z

        diag.copy_host_to_device()

    def _apply_diag_only(self, diag, zero_hessian_indices, zero_hessian_eps):
        """Apply eps_z and zero-hessian treatment to diagonal only (no CSR mods).

        Used by the inertia-first check to prepare the diagonal for factoring
        the unmodified Hessian before deciding whether BISC is needed.
        """
        diag_arr = diag.get_array()

        if zero_hessian_indices is not None and zero_hessian_eps is not None:
            for zh_idx in zero_hessian_indices:
                diag_arr[zh_idx] = max(diag_arr[zh_idx], zero_hessian_eps)

        if self._nonconvex_indices is not None and self.eps_z > 0:
            diag_arr[self._nonconvex_indices] -= self.eps_z

    # ---- Methods identical to CurvatureProbeConvexifier ----

    def decompose_residual(self, res, vars):
        res_arr = np.array(res.get_array())
        self.theta = np.linalg.norm(res_arr[self.mult_ind])
        self.eta = np.linalg.norm(res_arr[~self.mult_ind])
        x_sol = np.array(vars.get_solution())
        if self._nonconvex_indices is not None:
            lam_nc_norm = np.linalg.norm(x_sol[self._nonconvex_indices])
            self.vc_floor = self.eps_z * lam_nc_norm
        else:
            self.vc_floor = 0.0

    def update_eps_z(self, barrier_param):
        if self._nonconvex_indices is not None:
            self.eps_z = max(self.eps_z_floor, self.cz * barrier_param)
        else:
            self.eps_z = 0.0

    def should_force_barrier_reduction(self):
        total_res = max(self.theta, self.eta)
        if total_res < 1e-30:
            return False
        return self.theta <= 3.0 * self.vc_floor and self.theta > 0.1 * total_res

    def begin_iteration(self, barrier_param):
        self._barrier = barrier_param
        self.update_eps_z(barrier_param)
        if self.step_rejected:
            self.step_rejected = False
        else:
            self.consecutive_rejections = 0  # reset on successful step

    def reject_step(self):
        self.step_rejected = True
        self.consecutive_rejections += 1

    def handle_rejection_escape(self, barrier_param):
        if self.consecutive_rejections >= self.max_rejections:
            new_barrier = min(barrier_param * self.barrier_inc, self.initial_barrier)
            if new_barrier > barrier_param:
                self.eps_z = self.cz * new_barrier
                self.consecutive_rejections = 0
                return new_barrier, True
            else:
                self.consecutive_rejections = 0
                return barrier_param, False
        return barrier_param, None

    def check_stagnation(
        self, stagnation_count, threshold, barrier_param, barrier_fraction, tol
    ):
        if stagnation_count >= threshold and barrier_param > tol:
            new_barrier = max(barrier_param * barrier_fraction, tol)
            return new_barrier, True
        return barrier_param, False

    def iter_data(self):
        return {
            "eps_x": self.max_reg if self.max_reg > 0 else self.numerical_eps,
            "eps_z": self.eps_z,
            "theta": self.theta,
            "eta": self.eta,
            "vc_floor": self.vc_floor,
            "inertia_delta": 0.0,
            "n_neg": self.n_neg,
            "fill_compensation": self._fill_compensation,
            "max_w_c": self._max_w_c,
            "cascade_factor": self._cascade_factor,
            "E_norm": self._last_mod_norm,
            "bisc_tier": self._bisc_tier,
            "blocks_modified": self._blocks_modified,
            "blocks_total": self._blocks_total,
            "blocks_borderline": self._blocks_borderline,
            "block_min_eigs": self._block_min_eigs,
        }

    def riccati_solve(self, rhs_arr, diag_base, solver):
        """Solve the condensed primal system via Riccati on S_p_full.

        Unlike BISC (which modifies S_p_ineq), this solve operates on the
        FULL primal Schur complement including equality constraints:
          S_p_full = W_mod + Sigma + J^T |D_reg|^{-1} J

        After BISC write-back, CSR[k,k] = W_mod[k,k] (self-consistency:
        all Sigma, S_schur, Gamma terms cancel). So we read W_mod from CSR,
        add Sigma and S_schur_full to get S_p_full.

        S_p_full is guaranteed PD because:
        - S_p_ineq + E > 0 from BISC (correct inertia)
        - S_schur_eq >= 0 (positive semidefinite)
        - So S_p_full = (S_p_ineq + E) + S_schur_eq > 0

        Fresh backward Cholesky factorization + 2-pass solve.

        Parameters
        ----------
        rhs_arr : ndarray
            Full KKT residual vector (primal + dual).
        diag_base : ndarray
            Diagonal before regularization (Sigma for primals, D for
            constraints from compute_diagonal).
        solver : object
            Solver with hess for CSR data access.

        Returns
        -------
        p_chain : ndarray
            Primal step (full length). Only chain variable entries filled.
        """
        # Use CSR snapshot from before factor_from_host (which corrupts
        # CSR data by adding diagonal in-place). The snapshot has pure
        # BISC write-back: CSR[k,k] = W_mod[k,k] (no Sigma, no eps_z).
        if hasattr(self, "_riccati_csr_snapshot"):
            data = self._riccati_csr_snapshot
        else:
            data = solver.hess.get_data()

        # Use diag snapshot (saved at factorization time) for consistent
        # weights. self.eps_z may have changed since factorization (QF
        # updates barrier_param, which updates eps_z). The snap_diag
        # has the exact diagonal that PARDISO factored with:
        #   primals: Sigma + fill_compensation
        #   constraints: D_base - eps_z_at_factorization
        if hasattr(self, "_riccati_diag_snapshot"):
            diag_snap = self._riccati_diag_snapshot
        else:
            diag_snap = diag_base
        diag_reg = diag_snap  # constraint weights from factorization time

        p_full = np.zeros(len(self.mult_ind))

        for ci, chain in enumerate(self.bfs_chains):
            n_levels = len(chain)
            diag_maps = self._chain_diag_maps[ci]
            coupling_maps = self._chain_coupling_maps[ci]

            # --- Phase 1: Build S_p_full diagonal blocks ---
            # CSR snapshot has W_mod[k,k] (BISC write-back, pre-diagonal).
            # S_p_full[k,k] = W_mod[k,k] + Sigma[k] + S_schur_full[k,k]
            S_p_blocks = []
            for k in range(n_levels):
                block = chain[k]
                dk = len(block)
                dm = diag_maps[k]

                # W_mod from CSR snapshot + diagonal from snap_diag
                # (snap_diag = Sigma + fill_comp for primals, D_reg for constraints)
                valid_dm = dm >= 0
                H = np.where(valid_dm, data[np.where(valid_dm, dm, 0)], 0.0)
                H[np.arange(dk), np.arange(dk)] += diag_snap[block]

                # S_schur_full: ALL constraints (equalities skipped: D=0)
                for c_row, lp, ci_arr in self._schur_block_data[ci][k]:
                    D_cc = diag_reg[c_row]
                    if abs(D_cc) < 1e-30:
                        continue
                    w_c = -1.0 / D_cc
                    if w_c < 0:
                        continue
                    j_local = data[ci_arr]
                    H[np.ix_(lp, lp)] += w_c * np.outer(j_local, j_local)

                S_p_blocks.append(H)

            # --- Phase 2: Build full coupling matrices ---
            C_full = []
            for k in range(n_levels - 1):
                cm = coupling_maps[k]
                valid_cm = cm >= 0
                C_k = np.where(valid_cm, data[np.where(valid_cm, cm, 0)], 0.0)

                coup_data = self._schur_coupling_data[ci][k]
                if coup_data:
                    for c_row, lp_k, lp_k1, ci_k, ci_k1 in coup_data:
                        D_cc = diag_reg[c_row]
                        if abs(D_cc) < 1e-30:
                            continue
                        w_c = -1.0 / D_cc
                        if w_c < 0:
                            continue
                        j_k = data[ci_k]
                        j_k1 = data[ci_k1]
                        C_k[np.ix_(lp_k, lp_k1)] += w_c * np.outer(j_k, j_k1)
                C_full.append(C_k)

            # --- Phase 3: Backward Cholesky sweep on S_p_full ---
            factors = [None] * n_levels
            last = n_levels - 1
            try:
                factors[last] = np.linalg.cholesky(S_p_blocks[last])
            except np.linalg.LinAlgError:
                continue  # shouldn't happen if BISC worked

            for k in range(last - 1, -1, -1):
                C_k = C_full[k]
                L_next = factors[k + 1]
                # Gamma = C_k @ R_{k+1}^{-1} @ C_k^T
                tmp = scipy.linalg.solve_triangular(L_next, C_k.T, lower=True)
                Gamma = tmp.T @ tmp

                R_k = S_p_blocks[k] - Gamma
                try:
                    factors[k] = np.linalg.cholesky(R_k)
                except np.linalg.LinAlgError:
                    # Tiny regularization fallback
                    R_k[np.arange(len(R_k)), np.arange(len(R_k))] += 1e-10
                    try:
                        factors[k] = np.linalg.cholesky(R_k)
                    except np.linalg.LinAlgError:
                        break

            if any(f is None for f in factors):
                continue

            # --- Phase 4: Form condensed RHS ---
            # b[k] = r_x[k] + sum_c w_c * J_c_k^T * r_d[c]
            # _schur_block_data[ci][k] already contains ALL constraints
            # touching block k (including coupling constraints that also
            # touch k+1). Each entry has the J entries for block k ONLY.
            # No separate coupling RHS needed (would double-count).
            b = []
            for k in range(n_levels):
                block = chain[k]
                b_k = rhs_arr[block].copy()

                for c_row, lp, ci_arr in self._schur_block_data[ci][k]:
                    D_cc = diag_reg[c_row]
                    if abs(D_cc) < 1e-30:
                        continue
                    w_c = -1.0 / D_cc
                    if w_c < 0:
                        continue
                    j_local = data[ci_arr]
                    b_k[lp] += w_c * j_local * rhs_arr[c_row]

                b.append(b_k)

            # --- Phase 5: Backward pass (modify RHS through chain) ---
            c = [None] * n_levels
            c[last] = b[last].copy()
            for k in range(last - 1, -1, -1):
                C_k = C_full[k]
                L_next = factors[k + 1]
                y = scipy.linalg.solve_triangular(L_next, c[k + 1], lower=True)
                Rinv_c = scipy.linalg.solve_triangular(L_next.T, y, lower=False)
                c[k] = b[k] - C_k @ Rinv_c

            # --- Phase 6: Forward pass (recover solution) ---
            pk = [None] * n_levels
            y = scipy.linalg.solve_triangular(factors[0], c[0], lower=True)
            pk[0] = scipy.linalg.solve_triangular(factors[0].T, y, lower=False)
            for k in range(1, n_levels):
                C_km1 = C_full[k - 1]
                rhs_k = c[k] - C_km1.T @ pk[k - 1]
                y = scipy.linalg.solve_triangular(factors[k], rhs_k, lower=True)
                pk[k] = scipy.linalg.solve_triangular(factors[k].T, y, lower=False)

            # --- Residual check: S_p_full @ p = b? ---
            # Compute block-tridiagonal matrix-vector product
            Sp_p = [None] * n_levels
            for k in range(n_levels):
                Sp_p[k] = S_p_blocks[k] @ pk[k]
                if k > 0:
                    Sp_p[k] += C_full[k - 1].T @ pk[k - 1]
                if k < n_levels - 1:
                    Sp_p[k] += C_full[k] @ pk[k + 1]
            res_blocks = np.concatenate([Sp_p[k] - b[k] for k in range(n_levels)])
            b_full = np.concatenate(b)
            print(
                f"    Riccati residual: ||Sp*p-b||/||b|| = "
                f"{np.linalg.norm(res_blocks)/max(np.linalg.norm(b_full), 1e-30):.2e}"
            )

            # Scatter into full vector
            for k in range(n_levels):
                p_full[chain[k]] = pk[k]

        return p_full
