from dataclasses import dataclass
from enum import Enum, auto
from .fem_space import Space, FunctionSpace


class CellType(Enum):
    POINT = auto()
    SEGMENT = auto()
    TRIANGLE = auto()
    QUADRILATERAL = auto()
    TETRAHEDRON = auto()
    HEXAHEDRON = auto()
    PRISM = auto()
    PYRAMID = auto()


@dataclass(frozen=True)
class ReferenceCell:
    cell_type: CellType
    dimension: int
    vertices: tuple[tuple[float, ...], ...]
    edges: tuple[tuple[int, int], ...] = ()
    faces: tuple[tuple[int, ...], ...] = ()
    face_types: tuple[CellType, ...] = ()


REFERENCE_CELLS: dict[CellType, ReferenceCell] = {
    CellType.POINT: ReferenceCell(
        cell_type=CellType.POINT,
        dimension=0,
        vertices=((),),
    ),
    CellType.SEGMENT: ReferenceCell(
        cell_type=CellType.SEGMENT,
        dimension=1,
        vertices=(
            (-1.0,),  # 0
            (1.0,),  # 1
        ),
        edges=((0, 1),),
    ),
    CellType.TRIANGLE: ReferenceCell(
        cell_type=CellType.TRIANGLE,
        dimension=2,
        vertices=(
            (0.0, 0.0),  # 0
            (1.0, 0.0),  # 1
            (0.0, 1.0),  # 2
        ),
        edges=(
            (0, 1),
            (1, 2),
            (2, 0),
        ),
        faces=((0, 1, 2),),
        face_types=(CellType.TRIANGLE,),
    ),
    CellType.QUADRILATERAL: ReferenceCell(
        cell_type=CellType.QUADRILATERAL,
        dimension=2,
        vertices=(
            (-1.0, -1.0),  # 0
            (1.0, -1.0),  # 1
            (1.0, 1.0),  # 2
            (-1.0, 1.0),  # 3
        ),
        edges=(
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ),
        faces=((0, 1, 2, 3),),
        face_types=(CellType.QUADRILATERAL,),
    ),
    CellType.TETRAHEDRON: ReferenceCell(
        cell_type=CellType.TETRAHEDRON,
        dimension=3,
        vertices=(
            (0.0, 0.0, 0.0),  # 0
            (1.0, 0.0, 0.0),  # 1
            (0.0, 1.0, 0.0),  # 2
            (0.0, 0.0, 1.0),  # 3
        ),
        edges=(
            (0, 1),
            (1, 2),
            (2, 0),
            (0, 3),
            (1, 3),
            (2, 3),
        ),
        # Counterclockwise when viewed from outside.
        faces=(
            (0, 2, 1),
            (0, 1, 3),
            (1, 2, 3),
            (2, 0, 3),
        ),
        face_types=(
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
        ),
    ),
    CellType.HEXAHEDRON: ReferenceCell(
        cell_type=CellType.HEXAHEDRON,
        dimension=3,
        vertices=(
            (-1.0, -1.0, -1.0),  # 0
            (1.0, -1.0, -1.0),  # 1
            (1.0, 1.0, -1.0),  # 2
            (-1.0, 1.0, -1.0),  # 3
            (-1.0, -1.0, 1.0),  # 4
            (1.0, -1.0, 1.0),  # 5
            (1.0, 1.0, 1.0),  # 6
            (-1.0, 1.0, 1.0),  # 7
        ),
        edges=(
            # Bottom
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Top
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            # Vertical
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ),
        # Counterclockwise when viewed from outside.
        faces=(
            (0, 3, 2, 1),  # bottom
            (4, 5, 6, 7),  # top
            (0, 1, 5, 4),  # front
            (1, 2, 6, 5),  # right
            (2, 3, 7, 6),  # back
            (3, 0, 4, 7),  # left
        ),
        face_types=(
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
        ),
    ),
    CellType.PRISM: ReferenceCell(
        cell_type=CellType.PRISM,
        dimension=3,
        vertices=(
            (0.0, 0.0, -1.0),  # 0
            (1.0, 0.0, -1.0),  # 1
            (0.0, 1.0, -1.0),  # 2
            (0.0, 0.0, 1.0),  # 3
            (1.0, 0.0, 1.0),  # 4
            (0.0, 1.0, 1.0),  # 5
        ),
        edges=(
            # Bottom triangle
            (0, 1),
            (1, 2),
            (2, 0),
            # Top triangle
            (3, 4),
            (4, 5),
            (5, 3),
            # Vertical
            (0, 3),
            (1, 4),
            (2, 5),
        ),
        # Counterclockwise when viewed from outside.
        faces=(
            (0, 2, 1),  # bottom triangle
            (3, 4, 5),  # top triangle
            (0, 1, 4, 3),  # side
            (1, 2, 5, 4),  # side
            (2, 0, 3, 5),  # side
        ),
        face_types=(
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
            CellType.QUADRILATERAL,
        ),
    ),
    CellType.PYRAMID: ReferenceCell(
        cell_type=CellType.PYRAMID,
        dimension=3,
        vertices=(
            (-1.0, -1.0, 0.0),  # 0
            (1.0, -1.0, 0.0),  # 1
            (1.0, 1.0, 0.0),  # 2
            (-1.0, 1.0, 0.0),  # 3
            (0.0, 0.0, 1.0),  # 4: apex
        ),
        edges=(
            # Base
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Apex edges
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
        ),
        # Counterclockwise when viewed from outside.
        faces=(
            (0, 3, 2, 1),  # base
            (0, 1, 4),  # front
            (1, 2, 4),  # right
            (2, 3, 4),  # back
            (3, 0, 4),  # left
        ),
        face_types=(
            CellType.QUADRILATERAL,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
            CellType.TRIANGLE,
        ),
    ),
}


@dataclass(frozen=True)
class DofLayout:
    # Reference cell that defines the cell type
    ref_cell: ReferenceCell

    # Function space type
    space: FunctionSpace

    # Points in parametric space where the degrees of freedom are located
    pts: tuple[tuple[float, ...], ...] = ()

    # Directions associated with vector elements
    dirs: tuple[tuple[float, ...] | None, ...] = ()

    # Entity dofs associated with the vertices, edges, faces and interior points.
    # Dof are ordered as follows: vertice, edges, faces then interior dof
    vertex_dofs: tuple[tuple[int, ...], ...] = ()
    edge_dofs: tuple[tuple[int, ...], ...] = ()
    face_dofs: tuple[tuple[int, ...], ...] = ()
    cell_dofs: tuple[int, ...] = ()

    @property
    def ndof(self):
        return len(self.pts)

    @classmethod
    def make_layout(cls, space: FunctionSpace, cell_type: CellType):
        if space.func_space == Space.CONST:
            return cls.make_const(space, cell_type)
        elif space.func_space == Space.H1:
            return cls.make_h1(space, cell_type)
        elif space.func_space == Space.HDIV:
            return cls.make_hdiv(space, cell_type)
        else:
            raise NotImplementedError

    @classmethod
    def make_const(cls, space: FunctionSpace, cell_type: CellType):
        ref_cell = REFERENCE_CELLS[cell_type]

        return cls(
            ref_cell=ref_cell,
            pts=((0.0, 0.0, 0.0),),
            space=space,
            cell_dofs=((0,)),
        )

    @classmethod
    def make_h1(cls, space: FunctionSpace, cell_type: CellType):
        """
        Construct nodal H1 Lagrange DOFs on a reference cell.

        The ordering is

            vertices
            edge interiors
            face interiors
            cell interiors

        Parameters
        ----------
        ref_cell : ReferenceCell
            Reference-cell topology and vertex coordinates.

        degree : int
            Polynomial degree p >= 1.

        Returns
        -------
        pts : list[tuple[float, ...]]
            Coordinates of all interpolation points.

        dirs : list[None]
            Direction information associated with each DOF.
            H1 scalar DOFs have no direction, so every entry is None.

        vertex_dofs : tuple[int, ...]
            DOF number associated with each vertex.

        edge_dofs : tuple[tuple[int, ...], ...]
            Interior DOFs associated with each edge, ordered according
            to ref_cell.edges.

        face_dofs : tuple[tuple[int, ...], ...]
            Interior DOFs associated with each face, ordered according
            to ref_cell.faces.

        cell_dofs : tuple[int, ...]
            DOFs in the interior of the cell.
        """

        ref_cell = REFERENCE_CELLS[cell_type]

        if space.degree < 1:
            raise ValueError("H1 Lagrange degree must be >= 1")

        p = space.degree
        verts = ref_cell.vertices

        pts = []
        dirs = []

        vertex_dofs = []
        edge_dofs = []
        face_dofs = []
        cell_dofs = []

        # ------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------

        def add_point(x):
            dof = len(pts)
            pts.append(tuple(float(v) for v in x))
            dirs.append(None)
            return dof

        def weighted_point(indices, weights):
            """
            Affine combination of reference-cell vertices.
            """
            dim = ref_cell.dimension

            return tuple(
                sum(weights[k] * verts[indices[k]][d] for k in range(len(indices)))
                for d in range(dim)
            )

        def edge_point(v0, v1, i):
            """
            Point i/p along the directed edge v0 -> v1.
            """
            t = i / p
            return weighted_point(
                (v0, v1),
                (1.0 - t, t),
            )

        def triangle_point(v0, v1, v2, i, j, denom=None):
            """
            Triangle point with barycentric coordinates

                lambda1 = i / denom
                lambda2 = j / denom
                lambda0 = 1 - lambda1 - lambda2
            """
            if denom is None:
                denom = p

            l1 = i / denom
            l2 = j / denom
            l0 = 1.0 - l1 - l2

            return weighted_point(
                (v0, v1, v2),
                (l0, l1, l2),
            )

        def quad_point(v0, v1, v2, v3, i, j, denom=None):
            """
            Tensor-product point on a quadrilateral.

            Vertex ordering is assumed to be cyclic:

                v3 ---- v2
                |       |
                |       |
                v0 ---- v1
            """
            if denom is None:
                denom = p

            s = i / denom
            t = j / denom

            return weighted_point(
                (v0, v1, v2, v3),
                (
                    (1.0 - s) * (1.0 - t),
                    s * (1.0 - t),
                    s * t,
                    (1.0 - s) * t,
                ),
            )

        # Vertices: all dimensions
        for v in range(len(verts)):
            vertex_dofs.append(add_point(verts[v]))

        # Edges: every topological edge, including a SEGMENT itself
        for v0, v1 in ref_cell.edges:
            dofs = []

            for i in range(1, p):
                dofs.append(add_point(edge_point(v0, v1, i)))

            edge_dofs.append(tuple(dofs))

        # Faces: every topological face, including a TRIANGLE/QUAD itself
        for face, face_type in zip(ref_cell.faces, ref_cell.face_types):
            dofs = []

            if face_type == CellType.TRIANGLE:
                v0, v1, v2 = face

                for j in range(1, p):
                    for i in range(1, p - j):
                        dofs.append(add_point(triangle_point(v0, v1, v2, i, j)))

            elif face_type == CellType.QUADRILATERAL:
                v0, v1, v2, v3 = face

                for j in range(1, p):
                    for i in range(1, p):
                        dofs.append(add_point(quad_point(v0, v1, v2, v3, i, j)))

            else:
                raise ValueError(f"Unsupported H1 face type: {face_type}")

            if len(dofs) > 0:
                face_dofs.append(tuple(dofs))

        # Cell interior
        cell_type = ref_cell.cell_type

        if cell_type == CellType.TETRAHEDRON:
            for k in range(1, p):
                for j in range(1, p - k):
                    for i in range(1, p - j - k):

                        l1 = i / p
                        l2 = j / p
                        l3 = k / p
                        l0 = 1.0 - l1 - l2 - l3

                        cell_dofs.append(
                            add_point(
                                weighted_point(
                                    (0, 1, 2, 3),
                                    (l0, l1, l2, l3),
                                )
                            )
                        )

        elif cell_type == CellType.HEXAHEDRON:

            # Standard hexahedron ordering:
            #
            #       7-------6
            #      /|      /|
            #     4-------5 |
            #     | |     | |
            #     | 3-----|-2
            #     |/      |/
            #     0-------1
            #

            for k in range(1, p):
                z = k / p

                for j in range(1, p):
                    y = j / p

                    for i in range(1, p):
                        x = i / p

                        weights = (
                            (1 - x) * (1 - y) * (1 - z),
                            x * (1 - y) * (1 - z),
                            x * y * (1 - z),
                            (1 - x) * y * (1 - z),
                            (1 - x) * (1 - y) * z,
                            x * (1 - y) * z,
                            x * y * z,
                            (1 - x) * y * z,
                        )

                        cell_dofs.append(
                            add_point(
                                weighted_point(
                                    tuple(range(8)),
                                    weights,
                                )
                            )
                        )

        elif cell_type == CellType.PRISM:

            # Standard triangular-prism ordering:
            #
            # bottom: 0, 1, 2
            # top:    3, 4, 5
            #
            # with
            #
            #     0--3
            #     1--4
            #     2--5
            #

            for k in range(1, p):
                z = k / p

                for j in range(1, p):
                    for i in range(1, p - j):

                        l1 = i / p
                        l2 = j / p
                        l0 = 1.0 - l1 - l2

                        weights = (
                            (1 - z) * l0,
                            (1 - z) * l1,
                            (1 - z) * l2,
                            z * l0,
                            z * l1,
                            z * l2,
                        )

                        cell_dofs.append(
                            add_point(
                                weighted_point(
                                    tuple(range(6)),
                                    weights,
                                )
                            )
                        )

        elif cell_type == CellType.PYRAMID:

            # Standard ordering:
            #
            # base:  0,1,2,3
            # apex:  4
            #
            # Each horizontal layer is a shrinking quadrilateral.
            #
            # This gives the natural nodal layout for a pyramid, although
            # the polynomial/rational function space associated with a
            # high-order pyramid deserves separate treatment.

            v0, v1, v2, v3, va = 0, 1, 2, 3, 4

            for k in range(1, p):
                t = k / p

                # Number of intervals across this layer.
                q = p - k

                # q == 1 has no layer-interior points.
                for j in range(1, q):
                    eta = j / q

                    for i in range(1, q):
                        xi = i / q

                        # Point on the base quadrilateral.
                        base_weights = (
                            (1 - xi) * (1 - eta),
                            xi * (1 - eta),
                            xi * eta,
                            (1 - xi) * eta,
                        )

                        weights = (
                            (1 - t) * base_weights[0],
                            (1 - t) * base_weights[1],
                            (1 - t) * base_weights[2],
                            (1 - t) * base_weights[3],
                            t,
                        )

                        cell_dofs.append(
                            add_point(
                                weighted_point(
                                    (v0, v1, v2, v3, va),
                                    weights,
                                )
                            )
                        )

        return cls(
            ref_cell=ref_cell,
            space=space,
            pts=pts,
            dirs=dirs,
            vertex_dofs=tuple(vertex_dofs),
            edge_dofs=tuple(edge_dofs),
            face_dofs=tuple(face_dofs),
            cell_dofs=tuple(cell_dofs),
        )

    @classmethod
    def make_hdiv(cls, space: FunctionSpace, cell_type: CellType):
        """
        Construct nodal H(div) DOFs on a reference cell.

        Topological ownership follows the mixed-dimensional embedding convention:

            POINT:
                no H(div) space

            SEGMENT:
                endpoint flux DOFs -> vertices
                remaining DOFs     -> edge

            TRIANGLE / QUADRILATERAL:
                normal-flux DOFs    -> edges
                remaining DOFs      -> face

            3D cells:
                normal-flux DOFs    -> faces
                remaining DOFs      -> cell

        Thus only a 3D reference cell has cell_dofs.

        Each DOF is a point evaluation

            u(pt) . dir

        with facet directions chosen as outward normals scaled by facet measure.
        """

        ref_cell = REFERENCE_CELLS[cell_type]

        if space.degree != 1:
            raise NotImplementedError
        if not (
            cell_type == CellType.SEGMENT
            or cell_type == CellType.TRIANGLE
            or cell_type == CellType.QUADRILATERAL
        ):
            raise NotImplementedError

        if cell_type == CellType.SEGMENT:
            pts = ((0.0,),)
            edge_dofs = ((0,),)
            dirs = ((1,),)

        elif cell_type == CellType.TRIANGLE:
            pts = ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
            edge_dofs = ((0,), (1,), (2,))
            dirs = ((0, -1), (1, 1), (-1, 0))

        elif cell_type == CellType.QUADRILATERAL:
            pts = ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
            edge_dofs = ((0,), (1,), (2,), (3,))
            dirs = ((0.0, -2.0), (2.0, 0.0), (0.0, 2.0), (-2.0, 0.0))

        return cls(
            ref_cell=ref_cell, space=space, pts=pts, dirs=dirs, edge_dofs=edge_dofs
        )

        # if space.degree < 1:
        #     raise ValueError("H(div) degree must be >= 1")

        # k = space.degree
        # dim = ref_cell.dimension
        # verts = ref_cell.vertices
        # cell_type = ref_cell.cell_type

        # if cell_type == CellType.POINT:
        #     raise ValueError("H(div) DOFs are not defined on a POINT cell")

        # if cell_type in (CellType.PRISM, CellType.PYRAMID) and k > 1:
        #     raise NotImplementedError(
        #         f"H(div) degree {k} is not implemented for {cell_type.name}; "
        #         "only the lowest-order (degree = 1) element is available"
        #     )

        # pts = []
        # dirs = []

        # # Allocate one entry for every topological entity represented
        # # by the reference cell.
        # vertex_dofs = [[] for _ in ref_cell.vertices]
        # edge_dofs = [[] for _ in ref_cell.edges]
        # face_dofs = [[] for _ in ref_cell.faces]
        # cell_dofs = []

        # def add_dof(x, d):
        #     dof = len(pts)

        #     pts.append(tuple(float(v) for v in x))
        #     dirs.append(tuple(float(v) for v in d))

        #     return dof

        # def weighted_point(indices, weights):
        #     return tuple(
        #         sum(weights[m] * verts[indices[m]][c] for m in range(len(indices)))
        #         for c in range(dim)
        #     )

        # def diff(a, b):
        #     return tuple(verts[a][c] - verts[b][c] for c in range(dim))

        # def cross(a, b):
        #     return (
        #         a[1] * b[2] - a[2] * b[1],
        #         a[2] * b[0] - a[0] * b[2],
        #         a[0] * b[1] - a[1] * b[0],
        #     )

        # def unit(c):
        #     return tuple(1.0 if d == c else 0.0 for d in range(dim))

        # def edge_point(v0, v1, t):
        #     return weighted_point(
        #         (v0, v1),
        #         (1.0 - t, t),
        #     )

        # def triangle_point(v0, v1, v2, l1, l2):
        #     return weighted_point(
        #         (v0, v1, v2),
        #         (1.0 - l1 - l2, l1, l2),
        #     )

        # def quad_point(v0, v1, v2, v3, s, t):
        #     return weighted_point(
        #         (v0, v1, v2, v3),
        #         (
        #             (1.0 - s) * (1.0 - t),
        #             s * (1.0 - t),
        #             s * t,
        #             (1.0 - s) * t,
        #         ),
        #     )

        # def hex_point(x, y, z):
        #     return weighted_point(
        #         tuple(range(8)),
        #         (
        #             (1.0 - x) * (1.0 - y) * (1.0 - z),
        #             x * (1.0 - y) * (1.0 - z),
        #             x * y * (1.0 - z),
        #             (1.0 - x) * y * (1.0 - z),
        #             (1.0 - x) * (1.0 - y) * z,
        #             x * (1.0 - y) * z,
        #             x * y * z,
        #             (1.0 - x) * y * z,
        #         ),
        #     )

        # # ------------------------------------------------------------
        # # Facet normals
        # # ------------------------------------------------------------

        # def edge_normal_2d(v0, v1):
        #     # Reference-cell edges are counterclockwise.
        #     # Rotating the tangent clockwise gives the outward normal.
        #     #
        #     # Its magnitude equals the edge length.
        #     tx, ty = diff(v1, v0)
        #     return (ty, -tx)

        # def triangle_face_normal(v0, v1, v2):
        #     # |cross| = 2 * face area
        #     n = cross(
        #         diff(v1, v0),
        #         diff(v2, v0),
        #     )
        #     return tuple(0.5 * c for c in n)

        # def quad_face_normal(v0, v1, v2, v3):
        #     # For a planar parallelogram:
        #     # |cross| = face area
        #     return cross(
        #         diff(v1, v0),
        #         diff(v3, v0),
        #     )

        # # ============================================================
        # # Vertices
        # # ============================================================
        # #
        # # Only the 1D H(div) element has flux DOFs associated with
        # # vertices. These are the two boundary facets of the segment.
        # #

        # if cell_type == CellType.SEGMENT:

        #     for v in range(len(verts)):

        #         other = 1 - v

        #         sign = 1.0 if verts[v][0] > verts[other][0] else -1.0

        #         vertex_dofs[v].append(
        #             add_dof(
        #                 verts[v],
        #                 (sign,),
        #             )
        #         )

        # # ============================================================
        # # Edges
        # # ============================================================

        # if cell_type == CellType.SEGMENT:

        #     # The SEGMENT itself is an edge in the mixed-dimensional
        #     # topology.
        #     #
        #     # The two endpoint DOFs above supply the boundary values of
        #     # P_k. The remaining k - 1 nodes belong to the edge itself.

        #     if len(edge_dofs) != 1:
        #         raise ValueError("SEGMENT reference cell must contain exactly one edge")

        #     for i in range(1, k):

        #         t = i / k

        #         edge_dofs[0].append(
        #             add_dof(
        #                 edge_point(0, 1, t),
        #                 unit(0),
        #             )
        #         )

        # elif dim == 2:

        #     # In 2D, the proper edges are the H(div) facets.
        #     #
        #     # Each edge receives k normal-flux DOFs, forming a
        #     # unisolvent set for P_{k-1} on the edge.

        #     for edge_index, (v0, v1) in enumerate(ref_cell.edges):

        #         normal = edge_normal_2d(v0, v1)

        #         for i in range(1, k + 1):

        #             t = i / (k + 1)

        #             edge_dofs[edge_index].append(
        #                 add_dof(
        #                     edge_point(v0, v1, t),
        #                     normal,
        #                 )
        #             )

        # # ============================================================
        # # Faces
        # # ============================================================

        # if dim == 2:

        #     # The TRIANGLE / QUADRILATERAL itself is a face in the
        #     # mixed-dimensional topology.
        #     #
        #     # Therefore the RT interior DOFs belong to face_dofs[0],
        #     # rather than cell_dofs.

        #     if len(face_dofs) != 1:
        #         raise ValueError(
        #             f"{cell_type.name} reference cell must contain " "exactly one face"
        #         )

        #     if cell_type == CellType.TRIANGLE:

        #         # 2 * dim(P_{k-2}) interior DOFs.
        #         #
        #         # Strictly interior points of the lattice of order k+1,
        #         # with both Cartesian components evaluated at each point.

        #         n = k + 1

        #         for c in range(2):

        #             for j in range(1, n - 1):
        #                 for i in range(1, n - j):

        #                     pt = triangle_point(0, 1, 2, i / n, j / n)

        #                     face_dofs[0].append(add_dof(pt, unit(c)))

        #     elif cell_type == CellType.QUADRILATERAL:

        #         # u_x in Q_{k,k-1}
        #         #
        #         #     x: i/k
        #         #     y: j/(k+1)

        #         for j in range(1, k + 1):
        #             for i in range(1, k):

        #                 pt = quad_point(
        #                     0,
        #                     1,
        #                     2,
        #                     3,
        #                     i / k,
        #                     j / (k + 1),
        #                 )

        #                 face_dofs[0].append(
        #                     add_dof(
        #                         pt,
        #                         unit(0),
        #                     )
        #                 )

        #         # u_y in Q_{k-1,k}
        #         #
        #         #     x: i/(k+1)
        #         #     y: j/k

        #         for j in range(1, k):
        #             for i in range(1, k + 1):

        #                 pt = quad_point(
        #                     0,
        #                     1,
        #                     2,
        #                     3,
        #                     i / (k + 1),
        #                     j / k,
        #                 )

        #                 face_dofs[0].append(
        #                     add_dof(
        #                         pt,
        #                         unit(1),
        #                     )
        #                 )

        # elif dim == 3:

        #     # In 3D the proper faces are the H(div) facets.
        #     #
        #     # These retain their normal-flux interpretation.

        #     if len(ref_cell.faces) != len(ref_cell.face_types):
        #         raise ValueError(
        #             "ref_cell.faces and ref_cell.face_types must "
        #             "have the same length"
        #         )

        #     for face_index, (face, face_type) in enumerate(
        #         zip(ref_cell.faces, ref_cell.face_types)
        #     ):

        #         if face_type == CellType.TRIANGLE:

        #             v0, v1, v2 = face

        #             normal = triangle_face_normal(
        #                 v0,
        #                 v1,
        #                 v2,
        #             )

        #             # P_{k-1} on the triangular face
        #             n = k + 2

        #             for j in range(1, n - 1):
        #                 for i in range(1, n - j):

        #                     pt = triangle_point(
        #                         v0,
        #                         v1,
        #                         v2,
        #                         i / n,
        #                         j / n,
        #                     )

        #                     face_dofs[face_index].append(
        #                         add_dof(
        #                             pt,
        #                             normal,
        #                         )
        #                     )

        #         elif face_type == CellType.QUADRILATERAL:

        #             v0, v1, v2, v3 = face

        #             normal = quad_face_normal(
        #                 v0,
        #                 v1,
        #                 v2,
        #                 v3,
        #             )

        #             # Q_{k-1} on the quadrilateral face

        #             for j in range(1, k + 1):
        #                 for i in range(1, k + 1):

        #                     pt = quad_point(
        #                         v0,
        #                         v1,
        #                         v2,
        #                         v3,
        #                         i / (k + 1),
        #                         j / (k + 1),
        #                     )

        #                     face_dofs[face_index].append(
        #                         add_dof(
        #                             pt,
        #                             normal,
        #                         )
        #                     )

        #         else:
        #             raise ValueError(f"Unsupported H(div) face type: {face_type}")

        # # ============================================================
        # # 3D cell interior
        # # ============================================================
        # #
        # # Only 3D cells have actual cell_dofs under this convention.
        # #

        # if cell_type == CellType.TETRAHEDRON:

        #     # 3 * dim(P_{k-2})

        #     n = k + 2

        #     for c in range(3):

        #         for l in range(1, n - 2):
        #             for j in range(1, n - 1 - l):
        #                 for i in range(1, n - j - l):

        #                     l1 = i / n
        #                     l2 = j / n
        #                     l3 = l / n
        #                     l0 = 1.0 - l1 - l2 - l3

        #                     pt = weighted_point(
        #                         (0, 1, 2, 3),
        #                         (l0, l1, l2, l3),
        #                     )

        #                     cell_dofs.append(
        #                         add_dof(
        #                             pt,
        #                             unit(c),
        #                         )
        #                     )

        # elif cell_type == CellType.HEXAHEDRON:

        #     # Component c:
        #     #
        #     #   coordinate c:
        #     #       i/k
        #     #
        #     #   transverse coordinates:
        #     #       i/(k+1)

        #     for c in range(3):

        #         counts = [k + 1 if d != c else k for d in range(3)]

        #         for l in range(1, counts[2]):
        #             for j in range(1, counts[1]):
        #                 for i in range(1, counts[0]):

        #                     pt = hex_point(
        #                         i / counts[0],
        #                         j / counts[1],
        #                         l / counts[2],
        #                     )

        #                     cell_dofs.append(
        #                         add_dof(
        #                             pt,
        #                             unit(c),
        #                         )
        #                     )

        # elif cell_type in (
        #     CellType.PRISM,
        #     CellType.PYRAMID,
        # ):

        #     # Lowest-order implementations contain only facet DOFs.
        #     pass

        # # ============================================================
        # # Freeze layout
        # # ============================================================

        # return cls(
        #     ref_cell=ref_cell,
        #     space=space,
        #     pts=pts,
        #     dirs=dirs,
        #     vertex_dofs=tuple(tuple(dofs) for dofs in vertex_dofs),
        #     edge_dofs=tuple(tuple(dofs) for dofs in edge_dofs),
        #     face_dofs=tuple(tuple(dofs) for dofs in face_dofs),
        #     cell_dofs=tuple(cell_dofs),
        # )
