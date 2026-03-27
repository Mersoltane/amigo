"""Block detection via BFS on KKT sparsity pattern.

Identifies connected components among primal variables in the Hessian,
with hub detection to isolate high-degree variables (e.g. shared final
time tf) that would otherwise merge all time-step blocks into one.

For large connected components (e.g. chain-coupled OCP with fixed tf),
BFS from a leaf vertex recovers the natural time-step block structure.
Adjacent BFS levels form the block-tridiagonal structure needed for
Schur complement convexification.
"""

import numpy as np


def _build_primal_adjacency(rowp, cols, mult_ind, hub_set):
    """Build adjacency list for non-hub primal variables."""
    n = len(mult_ind)
    primal_indices = []
    for i in range(n):
        if not mult_ind[i] and i not in hub_set:
            primal_indices.append(i)

    primal_set = set(primal_indices)
    adj = {p: [] for p in primal_indices}
    for i in primal_indices:
        for idx in range(rowp[i], rowp[i + 1]):
            j = int(cols[idx])
            if j != i and j in primal_set:
                adj[i].append(j)

    return primal_indices, primal_set, adj


def _compute_degrees(rowp, cols, mult_ind):
    """Compute primal-primal degree for each variable."""
    n = len(mult_ind)
    degree = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if mult_ind[i]:
            continue
        for idx in range(rowp[i], rowp[i + 1]):
            j = cols[idx]
            if j != i and not mult_ind[j]:
                degree[i] += 1
    return degree


def _find_connected_components(primal_indices, primal_set, adj):
    """BFS-based connected component detection."""
    visited = set()
    components = []
    for start in primal_indices:
        if start in visited:
            continue
        comp = []
        queue = [start]
        visited.add(start)
        while queue:
            v = queue.pop(0)
            comp.append(v)
            for nb in adj[v]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def _bfs_levels(comp, adj):
    """Run BFS from the minimum-degree vertex, return level groupings.

    Each level corresponds to a natural block in the block-tridiagonal
    structure (e.g., one time step in OCP).
    """
    comp_set = set(comp)

    # Start from minimum-degree vertex (leaf of the chain)
    min_deg_v = min(comp, key=lambda v: len([nb for nb in adj[v] if nb in comp_set]))

    levels = []
    bfs_visited = {min_deg_v}
    queue = [min_deg_v]
    while queue:
        levels.append(np.array(sorted(queue), dtype=np.int64))
        next_queue = []
        for v in queue:
            for nb in adj[v]:
                if nb in comp_set and nb not in bfs_visited:
                    bfs_visited.add(nb)
                    next_queue.append(nb)
        queue = next_queue

    return levels


def detect_bfs_level_blocks(
    rowp,
    cols,
    mult_ind,
    hub_degree_threshold=50,
    max_eigendecomp_size=20,
    max_block_size=64,
):
    """Detect blocks via BFS levels on primal-primal adjacency.

    For small connected components (size <= max_eigendecomp_size), returns
    them as direct eigendecomp blocks. For large components (e.g. chain-
    coupled OCP), runs BFS from a leaf vertex to recover the natural
    time-step block structure. Adjacent BFS levels form the block-
    tridiagonal structure needed for Schur complement processing.

    If any BFS level in a chain exceeds max_block_size, the entire
    component is returned as individual small blocks instead of a chain
    (eigendecomp cost O(d^3) per level becomes prohibitive for large d).

    Parameters
    ----------
    rowp, cols : np.ndarray
        CSR structure arrays.
    mult_ind : np.ndarray of bool
        True for constraint/dual rows, False for primal.
    hub_degree_threshold : int
        Variables with degree above this are isolated as hubs.
    max_eigendecomp_size : int
        Components this size or smaller use direct eigendecomp.
    max_block_size : int
        Maximum allowed block size in a chain. Chains with any level
        exceeding this are converted to independent small blocks.

    Returns
    -------
    small_blocks : list of np.ndarray
        Small connected components for direct eigendecomp.
    bfs_chains : list of list of np.ndarray
        Each chain is a sequence of BFS-level blocks. Adjacent entries
        in each chain are coupled in the block-tridiagonal structure.
    hub_indices : np.ndarray
        High-degree variables isolated as scalar blocks.
    """
    n = len(mult_ind)
    degree = _compute_degrees(rowp, cols, mult_ind)

    hub_set = set()
    for i in range(n):
        if not mult_ind[i] and degree[i] > hub_degree_threshold:
            hub_set.add(i)

    primal_indices, primal_set, adj = _build_primal_adjacency(
        rowp, cols, mult_ind, hub_set
    )

    if len(primal_indices) == 0:
        return [], [], np.array(sorted(hub_set), dtype=np.int64)

    components = _find_connected_components(primal_indices, primal_set, adj)

    small_blocks = []
    bfs_chains = []

    for comp in components:
        if len(comp) <= max_eigendecomp_size:
            small_blocks.append(np.array(sorted(comp), dtype=np.int64))
        else:
            levels = _bfs_levels(comp, adj)
            if any(len(lvl) > max_block_size for lvl in levels):
                # Levels too large for chain Schur propagation.
                # Convert to independent small blocks (direct eigendecomp).
                for lvl in levels:
                    small_blocks.append(lvl)
            else:
                bfs_chains.append(levels)

    hub_indices = np.array(sorted(hub_set), dtype=np.int64)
    return small_blocks, bfs_chains, hub_indices
