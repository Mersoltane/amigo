"""
Model orderings
---------------

- Every ordering returns a permutation, and the block ordering puts the multipliers last
"""

import numpy as np
import pytest

import amigo as am
from amigo.amigo import reorder_model

ORDERINGS = [
    am.OrderingType.AMD,
    am.OrderingType.NESTED_DISSECTION,
    am.OrderingType.MULTI_COLOR,
    am.OrderingType.NATURAL,
]

NAMES = ["AMD", "NESTED_DISSECTION", "MULTI_COLOR", "NATURAL"]


def make_grid(m, seed=0):
    """Quad elements on an m by m grid with one constraint node each, in shuffled numbering"""
    nodes = np.arange((m + 1) * (m + 1)).reshape(m + 1, m + 1)

    # The four corners of every quad
    corners = [nodes[:-1, :-1], nodes[1:, :-1], nodes[1:, 1:], nodes[:-1, 1:]]
    quads = np.stack(corners, axis=-1).reshape(-1, 4)

    # One multiplier node per quad after the grid nodes
    mult = (m + 1) * (m + 1) + np.arange(quads.shape[0])
    elems = np.hstack([quads, mult[:, None]])

    # Shuffle the numbering
    shuffle = np.random.default_rng(seed).permutation(mult[-1] + 1)

    return shuffle[elems].astype(np.int32), shuffle[mult].astype(np.int32)


@pytest.mark.parametrize("order", ORDERINGS, ids=NAMES)
def test_ordering_is_a_permutation(order):
    elems = make_grid(12)[0]
    n = elems.max() + 1

    iperm = np.asarray(reorder_model(order, [elems], np.array([1])))

    assert np.array_equal(np.sort(iperm), np.arange(n))


@pytest.mark.parametrize("order", ORDERINGS, ids=NAMES)
def test_block_ordering_puts_the_multipliers_last(order):
    # At this size every degree list is headed by a multiplier at some pivot
    grid = make_grid(48)
    elems = grid[0]
    mult = grid[1]

    n = elems.max() + 1

    iperm = np.asarray(reorder_model(order, [elems], np.array([1]), mult))

    assert np.array_equal(np.sort(iperm), np.arange(n))
    assert np.all(iperm[mult] >= n - mult.size)
