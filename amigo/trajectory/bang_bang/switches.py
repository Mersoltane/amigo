"""
Bang-bang switches
------------------

- The switching function changes sign where a bang-bang control switches
- A negative sign puts the control at its upper limit
- Follows the bang-bang mesh refinement of Agamawi, Hager, and Rao
"""

import numpy as np

from ..collocation.mesh import RadauMesh


def switch_signs(sigma, noise=0.0):
    """Sign of the switching function at each collocation point"""
    sigma = np.asarray(sigma, dtype=float)

    # Values below the noise level count as zero
    threshold = noise * np.abs(sigma).max()

    sign = np.zeros(len(sigma))

    for i in range(len(sigma)):
        if abs(sigma[i]) > threshold:
            sign[i] = np.sign(sigma[i])

    # Find the first point with a sign
    nonzero = np.flatnonzero(sign)

    if len(nonzero) == 0:
        raise ValueError("the switching function vanishes at every collocation point")

    first = nonzero[0]

    # The points before it take its sign
    sign[:first] = sign[first]

    # Every later zero takes the sign of the point before it
    for i in range(first + 1, len(sign)):
        if sign[i] == 0.0:
            sign[i] = sign[i - 1]

    return sign


def split_at_switches(mesh, sigma, parts=4, noise=0.0):
    """Split the intervals where the switching function changes sign"""
    if len(sigma) != mesh.num_collocation:
        raise ValueError("sigma needs one value at every collocation point")

    sign = switch_signs(sigma, noise)

    # The interval of each collocation point
    owner = np.repeat(np.arange(mesh.num_intervals), mesh.degrees)

    # Mark the intervals on both sides of every sign change
    marked = set()

    for i in range(len(sign) - 1):
        if sign[i] != sign[i + 1]:
            marked.add(int(owner[i]))
            marked.add(int(owner[i + 1]))

    # Keep the mesh when the sign never changes
    if len(marked) == 0:
        return mesh

    # Split every marked interval into equal parts of the same degree
    points = [mesh.points[0]]
    degrees = []

    for k in range(mesh.num_intervals):
        a = mesh.points[k]
        b = mesh.points[k + 1]

        n = 1
        if k in marked:
            n = parts

        for j in range(1, n + 1):
            points.append(a + (b - a) * j / n)
            degrees.append(int(mesh.degrees[k]))

    return RadauMesh(points, degrees, mesh.flipped)


def switch_estimates(mesh, sigma, control, noise=0.0):
    """Switch times and the sign of the switching function between them"""
    sigma = np.asarray(sigma, dtype=float)
    control = np.asarray(control, dtype=float)

    if len(sigma) != mesh.num_collocation or len(control) != mesh.num_collocation:
        raise ValueError("sigma and the control need one value at every collocation point")

    sign = switch_signs(sigma, noise)

    # The collocation points and the interval of each point
    tau = mesh.collocation_points()
    owner = np.repeat(np.arange(mesh.num_intervals), mesh.degrees)

    # Collect the switch times and the sign on each side of them
    switches = []
    signs = [sign[0]]

    for i in range(len(sign) - 1):
        if sign[i] == sign[i + 1]:
            continue

        signs.append(sign[i + 1])

        # A sign change between two intervals switches at the mesh point
        k = owner[i + 1]

        if owner[i] != k:
            switches.append(mesh.points[k])
            continue

        # A sign change inside an interval starts from its midpoint
        t_sigma = 0.5 * (tau[i] + tau[i + 1])

        # Find the largest jump of the control in the interval
        members = mesh.collocation_indices(k)
        jumps = np.abs(np.diff(control[members]))

        # Average the two midpoints when the control jumps
        if jumps.max() > 0.0:
            largest = int(np.argmax(jumps))
            j = members[largest]
            t_u = 0.5 * (tau[j] + tau[j + 1])

            switches.append(0.5 * (t_sigma + t_u))

        # A control without a jump does not show the switch
        else:
            switches.append(t_sigma)

    return np.array(switches), np.array(signs)
