"""Costate estimates from the multipliers of collocated defects, Eq. 22 of Agamawi, Hager, and Rao.

The multipliers enter the Lagrangian of amigo with a plus sign.
Each stands for the costate times the quadrature weight of its point.
Dividing by the weight and flipping the sign gives the costate of the minimum principle.
"""

import numpy as np

from .mesh import differentiation_matrix, radau_weights


def defect_weights(mesh):
    """Radau weight over the row scaling of the derivative defect at every collocation point of a mesh.

    A defect row scales by its interval duration and by a row weight.
    The row weight is one over the largest magnitude of its differentiation row.
    The multiplier times the row weight over the Radau weight gives the costate.
    """
    weights = []
    for n in mesh.degrees:
        row_weight = 1.0 / np.abs(differentiation_matrix(n, mesh.flipped)).max(axis=1)
        weights.append(radau_weights(n, mesh.flipped) / row_weight)
    return np.concatenate(weights)


def defect_costate(rows, mesh):
    """Costate at the collocation points from the multipliers of derivative defect rows in the order of the points."""
    return -np.asarray(rows) / defect_weights(mesh)[:, None]


def quadrature_costate(rows, mesh, duration):
    """Costate at the collocation points from the multipliers of collocated dynamics residuals over a duration.

    This is the integral form, where the residual of each point is the rate
    minus the dynamics unscaled, so its multiplier is the costate times the
    quadrature weight of the point and half the duration of the phase.
    """
    return -np.asarray(rows) / (0.5 * duration * mesh.quadrature_weights())[:, None]
