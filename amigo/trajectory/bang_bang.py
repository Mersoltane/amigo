"""Switch estimates of the bang-bang mesh refinement of Agamawi, Hager, and Rao."""

import numpy as np

from .mesh import RadauMesh


def switch_signs(sigma, noise=0.0):
    """Sign of a switching function at every collocation point, where a value within noise times its largest magnitude takes the sign before it."""
    sigma = np.asarray(sigma, dtype=float)
    sign = np.where(np.abs(sigma) > noise * np.abs(sigma).max(), np.sign(sigma), 0.0)
    nonzero = np.flatnonzero(sign)
    if len(nonzero) == 0:
        raise ValueError("the switching function vanishes at every collocation point")
    sign[: nonzero[0]] = sign[nonzero[0]]
    for i in range(nonzero[0] + 1, len(sign)):
        if sign[i] == 0.0:
            sign[i] = sign[i - 1]
    return sign


def split_at_switches(mesh, sigma, parts=4, noise=0.0):
    """A mesh whose intervals that hold a sign change of the switching function split into equal parts of their degree.

    A sign change between the last point of one interval and the first point
    of the next lies inside the next, so both intervals split. The mesh comes
    back as it is when the switching function keeps its sign.
    """
    if len(sigma) != mesh.num_collocation:
        raise ValueError("sigma needs one value at every collocation point")
    sign = switch_signs(sigma, noise)
    owner = np.repeat(np.arange(mesh.num_intervals), mesh.degrees)
    marked = set()
    for i in np.flatnonzero(sign[:-1] != sign[1:]):
        marked.update((int(owner[i]), int(owner[i + 1])))
    if not marked:
        return mesh
    points, degrees = [mesh.points[0]], []
    for k in range(mesh.num_intervals):
        n = parts if k in marked else 1
        a, b = mesh.points[k], mesh.points[k + 1]
        points.extend(a + (b - a) * np.arange(1, n + 1) / n)
        degrees.extend([int(mesh.degrees[k])] * n)
    return RadauMesh(points, degrees, mesh.flipped)


def switch_estimates(mesh, sigma, control, noise=0.0):
    """Switches of one control-linear component on a mesh from its switching function, Eqs. 25 and 26.

    A sign change of sigma between two collocation points of one interval puts
    a switch at the average of the midpoint of those points and the midpoint of
    the two neighboring collocation points with the largest jump in the
    control in that interval, or at the first midpoint alone when the control
    does not jump there. A sign change between two intervals puts the switch
    at their common mesh point. Values of sigma within noise times its largest
    magnitude count as zero. The switches come back in mesh coordinates with
    the sign of sigma on each domain they bound, where a negative sign puts the
    control at its upper limit.
    """
    sigma = np.asarray(sigma, dtype=float)
    control = np.asarray(control, dtype=float)
    if len(sigma) != mesh.num_collocation or len(control) != mesh.num_collocation:
        raise ValueError("sigma and the control need one value at every collocation point")
    sign = switch_signs(sigma, noise)

    tau = mesh.collocation_points()
    owner = np.repeat(np.arange(mesh.num_intervals), mesh.degrees)
    switches, signs = [], [sign[0]]
    for i in range(len(sign) - 1):
        if sign[i] == sign[i + 1]:
            continue
        k = owner[i + 1]
        if owner[i] != k:
            switches.append(mesh.points[k])
        else:
            members = mesh.collocation_indices(k)
            jumps = np.abs(np.diff(control[members]))
            t_sigma = 0.5 * (tau[i] + tau[i + 1])
            if jumps.max() > 0.0:
                j = members[int(np.argmax(jumps))]
                t_u = 0.5 * (tau[j] + tau[j + 1])
                switches.append(0.5 * (t_sigma + t_u))
            else:
                # A control that does not jump in the interval says nothing about where it switches
                switches.append(t_sigma)
        signs.append(sign[i + 1])
    return np.array(switches), np.array(signs)
