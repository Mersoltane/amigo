"""Solutions held as consecutive arcs that meet at switches, as the bang-bang method of Agamawi, Hager, and Rao solves them."""

import numpy as np

from .bang_bang import switch_estimates
from .mesh import RadauMesh, mesh_time
from .refinement import MeshSolution


def combine_arcs(solutions):
    """One solution on a single mesh whose points include every switch, from the solutions of consecutive arcs."""
    t0, tf = solutions[0].t0, solutions[-1].tf
    points, degrees = [-1.0], []
    states, controls = [solutions[0].states[:1]], []
    for s in solutions:
        points.extend(2.0 * (mesh_time(s.mesh.points[1:], s.t0, s.tf) - t0) / (tf - t0) - 1.0)
        degrees.extend(s.mesh.degrees)
        states.append(s.states[1:])
        controls.append(s.controls)

    # The last point is the end of the last arc exactly
    points[-1] = 1.0
    mesh = RadauMesh(points, degrees, solutions[0].mesh.flipped)
    converged = all(s.converged for s in solutions)
    return MeshSolution(
        mesh, np.vstack(states), np.vstack(controls), t0, tf, dict(solutions[-1].parameters), converged=converged
    )


def split_arcs(solution, edges):
    """Solutions of the arcs between consecutive edge times on their own meshes, from one solution with a mesh point at every edge."""
    mesh = solution.mesh
    times = mesh_time(mesh.points, solution.t0, solution.tf)
    index = []
    for t in edges:
        k = int(np.argmin(np.abs(times - t)))
        if abs(times[k] - t) > 1e-9 * (solution.tf - solution.t0):
            raise ValueError(f"the edge time {t} is not a mesh point of the solution")
        index.append(k)
    if np.any(np.diff(index) <= 0):
        raise ValueError("the edge times must increase")

    pieces = []
    for a, b in zip(index[:-1], index[1:]):
        local = 2.0 * (times[a : b + 1] - times[a]) / (times[b] - times[a]) - 1.0
        local[0], local[-1] = -1.0, 1.0
        pieces.append(
            MeshSolution(
                RadauMesh(local, mesh.degrees[a:b], mesh.flipped),
                solution.states[mesh.offsets[a] : mesh.offsets[b] + 1],
                solution.controls[mesh.offsets[a] : mesh.offsets[b]],
                float(times[a]),
                float(times[b]),
                dict(solution.parameters),
                converged=solution.converged,
            )
        )
    return pieces


def stretch_arcs(solutions, tf):
    """Solutions of consecutive arcs with every time scaled about the first start so the last arc ends at tf."""
    t0 = solutions[0].t0
    scale = (tf - t0) / (solutions[-1].tf - t0)
    return [
        MeshSolution(
            s.mesh,
            s.states,
            s.controls,
            t0 + scale * (s.t0 - t0),
            t0 + scale * (s.tf - t0),
            dict(s.parameters),
            converged=s.converged,
        )
        for s in solutions
    ]


def switching_arcs(solution, sigma, control, noise=0.0):
    """Edge times of the arcs a switching function bounds on a solution and its sign on each arc, Section 5.2 of Agamawi, Hager, and Rao.

    Values of sigma within noise times its largest magnitude count as zero.
    """
    switches, signs = switch_estimates(solution.mesh, sigma, control, noise)
    edges = mesh_time(np.concatenate([[-1.0], switches, [1.0]]), solution.t0, solution.tf)
    return edges, signs


def switch_brackets(edges):
    """Bounds on every switch between two arcs, halfway to the switches on either side and open beyond the first and last."""
    edges = np.asarray(edges, dtype=float)
    last = len(edges) - 2
    brackets = []
    for j in range(1, last + 1):
        lower = -np.inf if j == 1 else 0.5 * (edges[j - 1] + edges[j])
        upper = np.inf if j == last else 0.5 * (edges[j] + edges[j + 1])
        brackets.append((lower, upper))
    return brackets


def sub_mesh(mesh, t0, tf, a, b, min_width=0.0):
    """The part of a mesh over t0 to tf between the times a and b as a mesh of its own.

    A piece of an interval keeps its degree, and a mesh point closer than
    min_width to the point kept before it or to b drops out, so an arc that
    its neighbors take over leaves no sliver behind.
    """
    times = mesh_time(mesh.points, t0, tf)
    gap = max(min_width, 1e-9 * (tf - t0))
    edges = [a]
    for t in times:
        if edges[-1] + gap < t < b - gap:
            edges.append(t)
    edges.append(b)
    last = mesh.num_intervals - 1
    degrees = [
        int(mesh.degrees[min(max(int(np.searchsorted(times, 0.5 * (left + right))) - 1, 0), last)])
        for left, right in zip(edges[:-1], edges[1:])
    ]
    points = 2.0 * (np.asarray(edges) - a) / (b - a) - 1.0
    points[0], points[-1] = -1.0, 1.0
    return RadauMesh(points, degrees, mesh.flipped)


def merge_short_arcs(kinds, edges, floor):
    """Kinds and edges of arcs with every arc no longer than twice the floor given to its neighbors.

    A short arc between two arcs of the other kind joins them into one, while
    a short arc at either end or beside an arc of its own kind goes to that
    neighbor.
    """
    kinds, edges = list(kinds), list(edges)
    p = 0
    while len(kinds) > 1 and p < len(kinds):
        if edges[p + 1] - edges[p] > 2.0 * floor:
            p += 1
            continue
        if p == 0:
            del kinds[0], edges[1]
        elif p == len(kinds) - 1 or kinds[p - 1] == kinds[p]:
            del kinds[p], edges[p]
        elif kinds[p + 1] == kinds[p]:
            del kinds[p], edges[p + 1]
        else:
            del kinds[p : p + 2], edges[p : p + 2]
        p = max(p - 1, 0)
    return kinds, edges


def sigma_zero(t1, s1, t2, s2):
    """Time between two points where the line through the switching function at them crosses zero, or their midpoint."""
    if s1 * s2 <= 0.0 and s1 != s2:
        return t1 + (t2 - t1) * s1 / (s1 - s2)
    return 0.5 * (t1 + t2)


def insert_arcs(kinds, solutions, sigma, signs, other, noise=0.0, floor=0.0):
    """Kinds and edges of solved arcs with an arc of the other kind over every region where the switching function has the wrong sign.

    sigma holds the switching function at the collocation points of the arcs
    in order and signs its sign on each arc. A value within noise times its
    largest magnitude has no sign, and a region shorter than twice the floor
    stays out. A region ends where sigma crosses zero or at the edge of its
    arc. Arcs of one kind that come to meet join, while two arcs of one kind
    that met at a contact event stay apart.
    """
    sigma = np.asarray(sigma, dtype=float)
    level = noise * np.abs(sigma).max()
    pieces, k = [], 0
    for kind, sign, s in zip(kinds, signs, solutions):
        n = s.mesh.num_collocation
        t = mesh_time(s.mesh.collocation_points(), s.t0, s.tf)
        values = sigma[k : k + n]
        k += n
        wrong = (np.sign(values) == -np.sign(sign)) & (np.abs(values) > level)
        start, j = s.t0, 0
        while j < n:
            if not wrong[j]:
                j += 1
                continue
            r = j
            while r + 1 < n and wrong[r + 1]:
                r += 1
            low = s.t0 if j == 0 else sigma_zero(t[j - 1], values[j - 1], t[j], values[j])
            high = s.tf if r == n - 1 else sigma_zero(t[r], values[r], t[r + 1], values[r + 1])
            if high - low >= 2.0 * floor:
                pieces += [(kind, start, low), (other(kind), low, high)]
                start = high
            j = r + 1
        pieces.append((kind, start, s.tf))
    if len(pieces) == len(kinds):
        return list(kinds), [solutions[0].t0] + [s.tf for s in solutions]

    # A piece of no length between two of one kind lets them join through merge_short_arcs
    return merge_short_arcs([kind for kind, _, _ in pieces], [pieces[0][1]] + [b for _, _, b in pieces], floor)


def uniform_arc_meshes(edges, intervals=40, degree=5, min_intervals=2, flipped=True):
    """Uniform meshes of the arcs between consecutive edge times with about a given number of intervals over all of them."""
    edges = np.asarray(edges, dtype=float)
    total = edges[-1] - edges[0]
    return [
        RadauMesh.uniform(max(min_intervals, int(np.ceil(intervals * (b - a) / total))), degree, flipped)
        for a, b in zip(edges[:-1], edges[1:])
    ]


def sign_agreement(signs, solutions, sigma):
    """Share of the collocation points of consecutive arcs where the switching function has the sign of its arc, and its largest disagreement over its largest magnitude."""
    sigma = np.asarray(sigma, dtype=float)
    expected = np.concatenate([np.full(s.mesh.num_collocation, np.sign(sign)) for sign, s in zip(signs, solutions)])
    if len(expected) != len(sigma):
        raise ValueError("sigma needs one value at every collocation point of the arcs")
    wrong = np.sign(sigma) != expected
    worst = float(np.abs(sigma[wrong]).max() / np.abs(sigma).max()) if wrong.any() else 0.0
    return float(1.0 - wrong.mean()), worst
