"""
Bang-bang arcs
--------------

- A solution held as consecutive arcs that meet at the switches
- Follows the bang-bang method of Agamawi, Hager, and Rao
"""

import numpy as np

from ..collocation.mesh import RadauMesh, mesh_time
from ..refinement.base import MeshSolution
from .switches import switch_estimates


def arc_edges(arcs):
    """Start of the first arc and the end of every arc"""
    edges = [arcs[0][1]]

    for arc in arcs:
        edges.append(arc[2])

    return edges


def solution_edges(solutions):
    """Start of the first solved arc and the end of every one"""
    edges = [solutions[0].t0]

    for solution in solutions:
        edges.append(solution.tf)

    return edges


def arc_times(kinds, solutions):
    """Kind, start, and end of every solved arc"""
    arcs = []

    for i in range(len(kinds)):
        arcs.append((kinds[i], solutions[i].t0, solutions[i].tf))

    return arcs


def combine_arcs(solutions):
    """One solution on a single mesh from the solutions of consecutive arcs"""
    t0 = solutions[0].t0
    tf = solutions[-1].tf

    # The mesh points of every arc on the combined mesh, so every switch is a mesh point
    points = [-1.0]
    degrees = []

    # The first state, then the states and controls of every arc after its first point
    states = [solutions[0].states[:1]]
    controls = []

    for arc in solutions:
        times = mesh_time(arc.mesh.points[1:], arc.t0, arc.tf)

        points.extend(2.0 * (times - t0) / (tf - t0) - 1.0)
        degrees.extend(arc.mesh.degrees)

        states.append(arc.states[1:])
        controls.append(arc.controls)

    # The last point is the end of the last arc exactly
    points[-1] = 1.0

    mesh = RadauMesh(points, degrees, solutions[0].mesh.flipped)

    # The combined solution converged when every arc did
    converged = True

    for arc in solutions:
        if not arc.converged:
            converged = False
            break

    states = np.vstack(states)
    controls = np.vstack(controls)
    parameters = dict(solutions[-1].parameters)

    return MeshSolution(mesh, states, controls, t0, tf, parameters, converged=converged)


def split_arcs(solution, edges):
    """Solutions of the arcs between consecutive edge times on their own meshes"""
    mesh = solution.mesh
    times = mesh_time(mesh.points, solution.t0, solution.tf)

    # The mesh point at every edge time
    index = []

    for t in edges:
        k = int(np.argmin(np.abs(times - t)))

        if abs(times[k] - t) > 1e-9 * (solution.tf - solution.t0):
            raise ValueError(f"the edge time {t} is not a mesh point of the solution")

        index.append(k)

    if np.any(np.diff(index) <= 0):
        raise ValueError("the edge times must increase")

    # Cut the solution at the edges
    pieces = []

    for j in range(len(index) - 1):
        a = index[j]
        b = index[j + 1]

        # The mesh points of the arc mapped onto [-1, 1]
        local = 2.0 * (times[a : b + 1] - times[a]) / (times[b] - times[a]) - 1.0
        local[0] = -1.0
        local[-1] = 1.0

        arc_mesh = RadauMesh(local, mesh.degrees[a:b], mesh.flipped)

        # The states and the controls of the arc
        states = solution.states[mesh.offsets[a] : mesh.offsets[b] + 1]
        controls = solution.controls[mesh.offsets[a] : mesh.offsets[b]]

        piece = MeshSolution(
            arc_mesh,
            states,
            controls,
            float(times[a]),
            float(times[b]),
            dict(solution.parameters),
            converged=solution.converged,
        )

        pieces.append(piece)

    return pieces


def stretch_arcs(solutions, tf):
    """Solutions of consecutive arcs with every time scaled so the last arc ends at tf"""
    t0 = solutions[0].t0

    # Every time scales about the first start
    scale = (tf - t0) / (solutions[-1].tf - t0)

    stretched = []

    for arc in solutions:
        start = t0 + scale * (arc.t0 - t0)
        end = t0 + scale * (arc.tf - t0)

        piece = MeshSolution(
            arc.mesh,
            arc.states,
            arc.controls,
            start,
            end,
            dict(arc.parameters),
            converged=arc.converged,
        )

        stretched.append(piece)

    return stretched


def switching_arcs(solution, sigma, control, noise=0.0):
    """Edge times and arc signs of a switching function on a solution"""
    estimates = switch_estimates(solution.mesh, sigma, control, noise)
    switches = estimates[0]
    signs = estimates[1]

    # The edges are the two ends of the phase and every switch between them
    tau = np.concatenate([[-1.0], switches, [1.0]])
    edges = mesh_time(tau, solution.t0, solution.tf)

    return edges, signs


def switch_brackets(edges):
    """Bounds on every switch between two arcs"""
    edges = np.asarray(edges, dtype=float)
    last = len(edges) - 2

    # The bounds sit halfway to the next switches and stay open beyond the first and the last
    brackets = []

    for j in range(1, last + 1):
        lower = -np.inf
        if j > 1:
            lower = 0.5 * (edges[j - 1] + edges[j])

        upper = np.inf
        if j < last:
            upper = 0.5 * (edges[j] + edges[j + 1])

        brackets.append((lower, upper))

    return brackets


def sub_mesh(mesh, t0, tf, a, b, min_width=0.0):
    """The part of a mesh over t0 to tf between the times a and b as a mesh of its own"""
    times = mesh_time(mesh.points, t0, tf)

    # A mesh point too close to its neighbors drops out, so no sliver is left behind
    gap = max(min_width, 1e-9 * (tf - t0))

    edges = [a]

    for t in times:
        if edges[-1] + gap < t < b - gap:
            edges.append(t)

    edges.append(b)

    # Each piece keeps the degree of the interval that holds its middle
    last = mesh.num_intervals - 1
    degrees = []

    for j in range(len(edges) - 1):
        middle = 0.5 * (edges[j] + edges[j + 1])

        k = int(np.searchsorted(times, middle)) - 1
        k = min(max(k, 0), last)

        degrees.append(int(mesh.degrees[k]))

    # The edges mapped onto [-1, 1]
    points = 2.0 * (np.asarray(edges) - a) / (b - a) - 1.0
    points[0] = -1.0
    points[-1] = 1.0

    return RadauMesh(points, degrees, mesh.flipped)


def merge_short_arcs(kinds, edges, floor):
    """Kinds and edges of the arcs after every arc no longer than twice the floor goes to its neighbors"""
    kinds = list(kinds)
    edges = list(edges)

    # Hand every arc no longer than twice the floor to its neighbors
    p = 0

    while len(kinds) > 1 and p < len(kinds):
        if edges[p + 1] - edges[p] > 2.0 * floor:
            p += 1
            continue

        # A short first arc goes to the arc after it
        if p == 0:
            del kinds[0]
            del edges[1]

        # A short last arc, or one after an arc of its own kind, goes to the arc before it
        elif p == len(kinds) - 1 or kinds[p - 1] == kinds[p]:
            del kinds[p]
            del edges[p]

        # A short arc before an arc of its own kind goes to that arc
        elif kinds[p + 1] == kinds[p]:
            del kinds[p]
            del edges[p + 1]

        # A short arc between two arcs of the other kind joins them into one
        else:
            del kinds[p : p + 2]
            del edges[p : p + 2]

        # Check again from the arc before
        p = max(p - 1, 0)

    return kinds, edges


def sigma_zero(t1, s1, t2, s2):
    """Time where the line through two values of the switching function crosses zero, or their midpoint"""
    if s1 * s2 <= 0.0 and s1 != s2:
        return t1 + (t2 - t1) * s1 / (s1 - s2)

    return 0.5 * (t1 + t2)


def insert_arcs(kinds, solutions, sigma, signs, other, noise=0.0, floor=0.0):
    """Arc kinds and edges with the other kind inserted where the switching function has the wrong sign"""
    # The switching function at the collocation points in arc order
    sigma = np.asarray(sigma, dtype=float)

    # Values below the noise level carry no sign
    level = noise * np.abs(sigma).max()

    pieces = []
    k = 0

    for i in range(len(solutions)):
        kind = kinds[i]
        sign = signs[i]
        arc = solutions[i]

        # The switching function on this arc
        n = arc.mesh.num_collocation
        t = mesh_time(arc.mesh.collocation_points(), arc.t0, arc.tf)

        values = sigma[k : k + n]
        k += n

        # The points where the switching function has the wrong sign
        wrong = (np.sign(values) == -np.sign(sign)) & (np.abs(values) > level)

        # Cut out every run of wrong points as an arc of the other kind
        start = arc.t0
        j = 0

        while j < n:
            if not wrong[j]:
                j += 1
                continue

            # The run of wrong points from j to r
            r = j

            while r + 1 < n and wrong[r + 1]:
                r += 1

            # The run starts and ends where the switching function crosses zero
            low = arc.t0
            if j > 0:
                low = sigma_zero(t[j - 1], values[j - 1], t[j], values[j])

            high = arc.tf
            if r < n - 1:
                high = sigma_zero(t[r], values[r], t[r + 1], values[r + 1])

            # A run shorter than twice the floor is left alone
            if high - low >= 2.0 * floor:
                pieces.append((kind, start, low))
                pieces.append((other(kind), low, high))
                start = high

            j = r + 1

        pieces.append((kind, start, arc.tf))

    # Nothing was inserted
    if len(pieces) == len(kinds):
        return list(kinds), solution_edges(solutions)

    # The kinds and the edges of the pieces
    new_kinds = []
    new_edges = [pieces[0][1]]

    for piece in pieces:
        new_kinds.append(piece[0])
        new_edges.append(piece[2])

    # A piece of no length between two of one kind lets them join through merge_short_arcs
    return merge_short_arcs(new_kinds, new_edges, floor)


def uniform_arc_meshes(edges, intervals=40, degree=5, min_intervals=2, flipped=True):
    """Uniform meshes of the arcs between consecutive edge times"""
    edges = np.asarray(edges, dtype=float)
    total = edges[-1] - edges[0]

    # The arcs share the intervals in proportion to their length, with at least min_intervals each
    meshes = []

    for j in range(len(edges) - 1):
        a = edges[j]
        b = edges[j + 1]

        count = max(min_intervals, int(np.ceil(intervals * (b - a) / total)))
        meshes.append(RadauMesh.uniform(count, degree, flipped))

    return meshes


def sign_agreement(signs, solutions, sigma):
    """Share of the collocation points where sigma has the sign of its arc, and the worst disagreement"""
    sigma = np.asarray(sigma, dtype=float)

    # The sign every arc expects at its collocation points
    expected = []

    for i in range(len(solutions)):
        n = solutions[i].mesh.num_collocation
        expected.append(np.full(n, np.sign(signs[i])))

    expected = np.concatenate(expected)

    if len(expected) != len(sigma):
        raise ValueError("sigma needs one value at every collocation point of the arcs")

    wrong = np.sign(sigma) != expected
    share = float(1.0 - wrong.mean())

    # The worst disagreement over the largest magnitude
    worst = 0.0
    if wrong.any():
        worst = float(np.abs(sigma[wrong]).max() / np.abs(sigma).max())

    return share, worst
