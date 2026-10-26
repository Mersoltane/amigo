"""Structural Hessian sparsity of a component Lagrangian and its compressed evaluation.

The pattern is read from the expression graph, so it is exact up to
cancellation and never misses an entry that can be nonzero. Columns that
share no row are then seeded together, which recovers the Hessian from far
fewer Hessian-vector products than one per input component.
"""

import numpy as np

from .expressions import (
    BinaryNode,
    ConstNode,
    FourierNode,
    IndexNode,
    InterpNode,
    PassiveNode,
    UnaryNode,
    VarNode,
)

_EMPTY = (frozenset(), frozenset())

# Largest number of dense-column thresholds tried when compressing
_MAX_THRESHOLDS = 24


class UnsupportedExpression(Exception):
    pass


def _pairs(a, b):
    """Symmetric index pairs coupling every index of a with every index of b."""
    return frozenset((i, j) if i <= j else (j, i) for i in a for j in b)


class HessianSparsity:
    """Hessian pattern of a scalar expression over the flattened input components."""

    def __init__(self, inputs):
        # Flat offset and shape of each active input, in the order of the Input tuple
        self.offsets = {}
        offset = 0
        for expr in inputs:
            node = expr.node
            self.offsets[node.name] = (offset, node.shape)
            offset += 1 if node.shape is None else int(np.prod(node.shape))
        self.ncomp = offset

    def pattern(self, expr):
        """Upper triangular index pairs of the structurally nonzero Hessian entries."""
        memo = {}
        stack = [(expr, False)]
        while stack:
            item, ready = stack.pop()
            key = id(item.node)
            if key in memo:
                continue
            children = self._children(item.node)
            if ready:
                memo[key] = self._combine(item.node, [memo[id(c.node)] for c in children])
            else:
                stack.append((item, True))
                stack.extend((c, False) for c in children if id(c.node) not in memo)
        return memo[id(expr.node)][1]

    def dense_pattern(self):
        """Every pair, for expressions the analysis does not cover."""
        return _pairs(range(self.ncomp), range(self.ncomp))

    def _children(self, node):
        # The table of an interpolation or a series is data and adds no dependency
        if isinstance(node, (UnaryNode, InterpNode, FourierNode)):
            return [node.expr]
        if isinstance(node, BinaryNode):
            return [node.left, node.right]
        return []

    def _leaf(self, node):
        if isinstance(node, (ConstNode, PassiveNode)):
            return _EMPTY
        if isinstance(node, VarNode):
            if not node.active:
                return _EMPTY
            if node.name not in self.offsets:
                raise UnsupportedExpression(f"Unknown active variable {node.name}")
            offset, shape = self.offsets[node.name]
            size = 1 if shape is None else int(np.prod(shape))
            return (frozenset(range(offset, offset + size)), frozenset())
        if isinstance(node, IndexNode):
            base = node.expr.node
            if isinstance(base, PassiveNode) or not base.is_active():
                return _EMPTY
            if not isinstance(base, VarNode) or base.name not in self.offsets:
                raise UnsupportedExpression("Indexing of an unknown expression")
            offset, shape = self.offsets[base.name]
            if isinstance(node.index, tuple):
                flat = node.index[0] * shape[1] + node.index[1]
            else:
                flat = node.index
            return (frozenset((offset + flat,)), frozenset())
        raise UnsupportedExpression(f"Unknown node {type(node).__name__}")

    def _combine(self, node, results):
        if not results:
            return self._leaf(node)

        # Linear interpolation has a zero second derivative
        if isinstance(node, InterpNode):
            return results[0]

        # A Fourier series curves in its argument like a nonlinear unary function
        if isinstance(node, FourierNode):
            deps, hess = results[0]
            return deps, hess | _pairs(deps, deps)

        if isinstance(node, UnaryNode):
            deps, hess = results[0]
            if node.op == "-":
                return deps, hess
            return deps, hess | _pairs(deps, deps)

        (du, hu), (dv, hv) = results
        deps = du | dv
        hess = hu | hv
        if node.op in ("+", "-"):
            return deps, hess
        if node.op == "*":
            return deps, hess | _pairs(du, dv)
        if node.op == "/":
            return deps, hess | _pairs(du, dv) | _pairs(dv, dv)
        if node.op in ("**", "pow") and not dv:
            exponent = node.right.node
            if isinstance(exponent, ConstNode) and exponent.value in (0, 1):
                return deps, hess
            return deps, hess | _pairs(du, du)
        # Other binary operations couple all their dependencies
        return deps, hess | _pairs(deps, deps)


def _color(masks, order):
    """Greedy column coloring where columns in one color share no row."""
    color = {}
    used = []
    for c in order:
        for k, rows in enumerate(used):
            if not rows & masks[c]:
                used[k] = rows | masks[c]
                color[c] = k
                break
        else:
            color[c] = len(used)
            used.append(masks[c])
    return color, len(used)


def compress_hessian(pattern, ncomp):
    """Directions and recovery map that evaluate a symmetric Hessian pattern.

    Dense columns are seeded one at a time. The remaining columns are colored
    so that no two in a color share a row outside the dense set. An entry in
    a dense row is read from that row's own direction by symmetry, and every
    other entry is the unique contribution to its row in its column's color.
    The dense set is the one of the tried degree thresholds that needs the
    fewest directions.
    """
    rows = [set() for _ in range(ncomp)]
    for i, j in pattern:
        rows[i].add(j)
        rows[j].add(i)
    degree = [len(r) for r in rows]
    active = [c for c in range(ncomp) if degree[c] > 0]
    full_mask = [sum(1 << r for r in rows[c]) for c in range(ncomp)]

    thresholds = sorted({degree[c] for c in active} | {0}, reverse=True)
    if len(thresholds) > _MAX_THRESHOLDS:
        picks = np.linspace(0, len(thresholds) - 1, _MAX_THRESHOLDS).astype(int)
        thresholds = [thresholds[k] for k in sorted(set(picks))]

    best = None
    for t in thresholds:
        dense = sorted(c for c in active if degree[c] > t)
        dense_bits = sum(1 << c for c in dense)
        masks = {c: full_mask[c] & ~dense_bits for c in active if degree[c] <= t}

        # A column whose rows are all dense is read from the dense directions
        order = sorted(
            (c for c in masks if masks[c]), key=lambda c: -bin(masks[c]).count("1")
        )
        color, ncolors = _color(masks, order)
        ndirs = len(dense) + ncolors
        if best is None or ndirs < best[0]:
            best = (ndirs, dense, color, ncolors)

    ndirs, dense, color, ncolors = best
    dense_dir = {c: d for d, c in enumerate(dense)}

    seeds = [[c] for c in dense] + [[] for _ in range(ncolors)]
    for c, k in color.items():
        seeds[len(dense) + k].append(c)

    # Each entry of the full symmetric pattern is recovered by exactly one direction
    entries = [[] for _ in range(ndirs)]
    for r in range(ncomp):
        for c in sorted(rows[r]):
            if c in dense_dir:
                entries[dense_dir[c]].append((r, c, r))
            elif r in dense_dir:
                entries[dense_dir[r]].append((r, c, c))
            else:
                entries[len(dense) + color[c]].append((r, c, r))

    return seeds, entries
