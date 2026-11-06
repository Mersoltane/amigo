"""
Mesh refinement
---------------

- The refinement loop and the records it keeps
- The hp refinement of Liu, Hager, and Rao
- The phs refinement of Haman and Rao
"""

from .base import (
    Assessment,
    MeshIteration,
    MeshRefinement,
    MeshSolution,
    RefinementHistory,
    best_iteration,
)
from .hp import HPRefinement, RefinementOptions, relative_errors
from .phs import (
    FlightError,
    PHSOptions,
    PHSRefinement,
    path_violations,
    simulated_errors,
    touch_points,
)
