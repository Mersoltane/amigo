"""Barrier parameter update strategies.

MonotoneBarrierStrategy           - monotone decrease with a progress gate
HeuristicBarrierStrategy          - complementarity heuristic with fallback
QualityFunctionBarrierStrategy    - predictor-corrector or golden section
"""

from .base import BarrierStrategy, barrier_floor, barrier_minimum, fraction_to_boundary
from .monotone import MonotoneBarrierStrategy
from .heuristic import HeuristicBarrierStrategy
from .quality_function import QualityFunctionBarrierStrategy


def make_barrier_strategy(options, problem, optimizer):
    """Factory: pick the barrier strategy based on options."""
    if isinstance(options["barrier_strategy"], BarrierStrategy):
        return options["barrier_strategy"]
    elif options["barrier_strategy"] == "monotone":
        return MonotoneBarrierStrategy(options, problem, optimizer)
    elif options["barrier_strategy"] == "heuristic":
        return HeuristicBarrierStrategy(options, problem, optimizer)
    elif options["barrier_strategy"] == "quality_function":
        return QualityFunctionBarrierStrategy(options, problem, optimizer)
    else:
        strategy = options["barrier_strategy"]
        raise ValueError(f"Unrecognized barrier_strategy {strategy}")
