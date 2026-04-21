"""Barrier parameter update strategies.

Two implementations:
    HeuristicBarrier        - LOQO-style + monotone fallback (classical)
    QualityFunctionBarrier  - Mehrotra PC / golden-section QF oracle
                              with adaptive-mu globalization
"""

from .base import BarrierStrategy
from .heuristic import HeuristicBarrier
from .quality_function import QualityFunctionBarrier


def make_barrier_strategy(opt, options):
    """Factory: pick the barrier strategy based on options."""
    if options["barrier_strategy"] == "quality_function":
        return QualityFunctionBarrier(opt, options)
    return HeuristicBarrier(opt, options)


__all__ = [
    "BarrierStrategy",
    "HeuristicBarrier",
    "QualityFunctionBarrier",
    "make_barrier_strategy",
]
