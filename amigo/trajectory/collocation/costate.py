"""
Costate estimates
-----------------

- The multipliers of the collocated dynamics give the costate
- A multiplier is the costate times the quadrature weight of its point with the opposite sign
- Follows Eq. 22 of Agamawi, Hager, and Rao
"""

import numpy as np

from .mesh import differentiation_matrix, radau_weights


def defect_weights(mesh):
    """Weights that turn the defect multipliers into costates"""
    weights = []

    for n in mesh.degrees:
        D = differentiation_matrix(n, mesh.flipped)
        w = radau_weights(n, mesh.flipped)

        # Each defect row is scaled by one over its largest entry
        row_scale = 1.0 / np.abs(D).max(axis=1)

        weights.append(w / row_scale)

    return np.concatenate(weights)


def defect_costate(rows, mesh):
    """Costate at the collocation points from the multipliers of the defect rows"""
    weights = defect_weights(mesh)

    return -np.asarray(rows) / weights[:, None]


def quadrature_costate(rows, mesh, duration):
    """Costate at the collocation points from the multipliers of the dynamics residuals"""
    # Each multiplier carries the quadrature weight and half the phase duration
    weights = 0.5 * duration * mesh.quadrature_weights()

    return -np.asarray(rows) / weights[:, None]
