"""
Radau collocation
-----------------

- Radau points, weights, matrices, and the hp mesh of intervals
- Costates from the multipliers of the collocated dynamics
- Declarations of one phase of a trajectory problem
"""

from .mesh import (
    RadauMesh,
    carry_densities,
    differentiation_matrix,
    integration_matrix,
    lagrange_matrix,
    mesh_time,
    radau_points,
    radau_weights,
    support_points,
)
from .costate import defect_costate, defect_weights, quadrature_costate
