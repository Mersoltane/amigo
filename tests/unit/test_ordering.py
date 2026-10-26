"""Tests that every model ordering returns a permutation with the multipliers last."""

import numpy as np
import pytest

import amigo as am
from amigo.amigo import reorder_model

ORDERINGS = ["AMD", "NESTED_DISSECTION", "MULTI_COLOR", "NATURAL"]


def make_grid(m, seed=0):
    """Quad elements on an m by m grid with one constraint node each, in shuffled numbering."""
    nodes = np.arange((m + 1) * (m + 1)).reshape(m + 1, m + 1)
    quads = np.stack(
        [nodes[:-1, :-1], nodes[1:, :-1], nodes[1:, 1:], nodes[:-1, 1:]], axis=-1
    ).reshape(-1, 4)
    mult = (m + 1) * (m + 1) + np.arange(quads.shape[0])
    elems = np.hstack([quads, mult[:, None]])
    shuffle = np.random.default_rng(seed).permutation(mult[-1] + 1)
    return shuffle[elems].astype(np.int32), shuffle[mult].astype(np.int32)


@pytest.mark.parametrize("name", ORDERINGS)
def test_ordering_is_a_permutation(name):
    elems, _ = make_grid(12)
    n = elems.max() + 1
    order = getattr(am.OrderingType, name)
    iperm = np.asarray(reorder_model(order, [elems], np.array([1])))
    assert np.array_equal(np.sort(iperm), np.arange(n))


@pytest.mark.parametrize("name", ORDERINGS)
def test_block_ordering_puts_the_multipliers_last(name):
    # At this size every degree list is headed by a multiplier at some pivot
    elems, mult = make_grid(48)
    n = elems.max() + 1
    order = getattr(am.OrderingType, name)
    iperm = np.asarray(reorder_model(order, [elems], np.array([1]), mult))
    assert np.array_equal(np.sort(iperm), np.arange(n))
    assert np.all(iperm[mult] >= n - mult.size)
