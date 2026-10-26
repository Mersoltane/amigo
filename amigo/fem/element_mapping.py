from dataclasses import dataclass
from enum import Enum, auto
from .cell_types import CellType


class ElementFamily(Enum):
    LAGRANGE = auto()
    SERENDIPITY = auto()


@dataclass(frozen=True)
class MeshElementType:
    cell_type: CellType
    degree: int
    family: ElementFamily

    vertices: tuple[int, ...]
    edge_nodes: tuple[tuple[int, ...], ...] = ()
    face_nodes: tuple[tuple[int, ...], ...] = ()
    cell_nodes: tuple[int, ...] = ()

    def get_entity_dofs(self):
        """Get the indices of the local dof in Amigo ordering"""

        dofs = []
        for v in self.vertices:
            dofs.append(v)

        for e in self.edge_nodes:
            for v in e:
                dofs.append(v)

        for f in self.face_nodes:
            for v in f:
                dofs.append(v)

        for v in self.cell_nodes:
            dofs.append(v)

        return dofs


ABAQUS_ELEMENT_TYPES = {
    # ------------------------------------------------------------
    # 0D
    # ------------------------------------------------------------
    "MASS": MeshElementType(
        cell_type=CellType.POINT,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0,),
        edge_nodes=(),
    ),
    # ------------------------------------------------------------
    # 1D
    # ------------------------------------------------------------
    "T3D2": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1),
        edge_nodes=((),),
    ),
    "T3D3": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=2,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 2),
        edge_nodes=((1,),),
    ),
    "B21": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1),
        edge_nodes=((),),
    ),
    "B22": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=2,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1),
        edge_nodes=((2,),),
    ),
    "B31": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1),
        edge_nodes=((),),
    ),
    "B32": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=2,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1),
        edge_nodes=((2,),),
    ),
    # ------------------------------------------------------------
    # 2D triangles
    # Abaqus node ordering:
    #
    #       2
    #       o
    #      / \
    #   5 o   o 4
    #    /     \
    #   o---o---o
    #   0   3   1
    #
    # edges assumed:
    #   (0,1), (1,2), (2,0)
    # ------------------------------------------------------------
    "CPS3": MeshElementType(
        cell_type=CellType.TRIANGLE,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2),
        edge_nodes=((), (), ()),
    ),
    "CPS6": MeshElementType(
        cell_type=CellType.TRIANGLE,
        degree=2,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2),
        edge_nodes=((3,), (4,), (5,)),
    ),
    "CPE3": MeshElementType(
        cell_type=CellType.TRIANGLE,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2),
        edge_nodes=((), (), ()),
    ),
    "CPE6": MeshElementType(
        cell_type=CellType.TRIANGLE,
        degree=2,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2),
        edge_nodes=((3,), (4,), (5,)),
    ),
    # ------------------------------------------------------------
    # 2D quadrilaterals
    #
    # Abaqus Q8/Q9 ordering:
    #
    #   3----6----2
    #   |         |
    #   7    8    5
    #   |         |
    #   0----4----1
    #
    # edges assumed:
    #   (0,1), (1,2), (2,3), (3,0)
    # ------------------------------------------------------------
    "CPS4": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=((), (), (), ()),
    ),
    "CPS8": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=2,
        family=ElementFamily.SERENDIPITY,
        vertices=(0, 1, 2, 3),
        edge_nodes=((4,), (5,), (6,), (7,)),
    ),
    "CPS9": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=2,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=((4,), (5,), (6,), (7,)),
        cell_nodes=(8,),
    ),
    "CPE4": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=((), (), (), ()),
    ),
    "CPE8": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=2,
        family=ElementFamily.SERENDIPITY,
        vertices=(0, 1, 2, 3),
        edge_nodes=((4,), (5,), (6,), (7,)),
    ),
    # ------------------------------------------------------------
    # Shells / membranes
    # ------------------------------------------------------------
    "S3": MeshElementType(
        cell_type=CellType.TRIANGLE,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2),
        edge_nodes=((), (), ()),
    ),
    "S4": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=((), (), (), ()),
    ),
    "S8": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=2,
        family=ElementFamily.SERENDIPITY,
        vertices=(0, 1, 2, 3),
        edge_nodes=((4,), (5,), (6,), (7,)),
    ),
    "M3D3": MeshElementType(
        cell_type=CellType.TRIANGLE,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2),
        edge_nodes=((), (), ()),
    ),
    "M3D4": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=((), (), (), ()),
    ),
    # ------------------------------------------------------------
    # Tetrahedra
    #
    # C3D10:
    # vertices 0,1,2,3
    #
    # midside nodes:
    #   4 : 0-1
    #   5 : 1-2
    #   6 : 2-0
    #   7 : 0-3
    #   8 : 1-3
    #   9 : 2-3
    #
    # REFERENCE_CELLS[TETRAHEDRON].edges should therefore be:
    #   (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)
    # ------------------------------------------------------------
    "C3D4": MeshElementType(
        cell_type=CellType.TETRAHEDRON,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=((), (), (), (), (), ()),
    ),
    "C3D10": MeshElementType(
        cell_type=CellType.TETRAHEDRON,
        degree=2,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=(
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
        ),
    ),
    # ------------------------------------------------------------
    # Hexahedra
    #
    # Vertices:
    #
    #       7---------6
    #      /|        /|
    #     4---------5 |
    #     | |       | |
    #     | 3-------|-2
    #     |/        |/
    #     0---------1
    #
    # C3D20 midside nodes:
    #
    #  8 : 0-1
    #  9 : 1-2
    # 10 : 2-3
    # 11 : 3-0
    #
    # 12 : 4-5
    # 13 : 5-6
    # 14 : 6-7
    # 15 : 7-4
    #
    # 16 : 0-4
    # 17 : 1-5
    # 18 : 2-6
    # 19 : 3-7
    # ------------------------------------------------------------
    "C3D8": MeshElementType(
        cell_type=CellType.HEXAHEDRON,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3, 4, 5, 6, 7),
        edge_nodes=((), (), (), (), (), (), (), (), (), (), (), ()),
    ),
    "C3D20": MeshElementType(
        cell_type=CellType.HEXAHEDRON,
        degree=2,
        family=ElementFamily.SERENDIPITY,
        vertices=(0, 1, 2, 3, 4, 5, 6, 7),
        edge_nodes=(
            (8,),  # 0-1
            (9,),  # 1-2
            (10,),  # 2-3
            (11,),  # 3-0
            (12,),  # 4-5
            (13,),  # 5-6
            (14,),  # 6-7
            (15,),  # 7-4
            (16,),  # 0-4
            (17,),  # 1-5
            (18,),  # 2-6
            (19,),  # 3-7
        ),
    ),
    # ------------------------------------------------------------
    # Triangular prisms / wedges
    #
    # vertices:
    # bottom: 0,1,2
    # top:    3,4,5
    #
    # edges:
    # (0,1), (1,2), (2,0),
    # (3,4), (4,5), (5,3),
    # (0,3), (1,4), (2,5)
    # ------------------------------------------------------------
    "C3D6": MeshElementType(
        cell_type=CellType.PRISM,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3, 4, 5),
        edge_nodes=((), (), (), (), (), (), (), (), ()),
    ),
    "C3D15": MeshElementType(
        cell_type=CellType.PRISM,
        degree=2,
        family=ElementFamily.SERENDIPITY,
        vertices=(0, 1, 2, 3, 4, 5),
        edge_nodes=(
            (6,),
            (7,),
            (8,),
            (9,),
            (10,),
            (11,),
            (12,),
            (13,),
            (14,),
        ),
    ),
    # ------------------------------------------------------------
    # Pyramid
    # ------------------------------------------------------------
    "C3D5": MeshElementType(
        cell_type=CellType.PYRAMID,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3, 4),
        edge_nodes=((), (), (), (), (), (), (), ()),
    ),
}

ABAQUS_ELEMENT_ALIASES = {
    # Plane stress
    "CPS4R": "CPS4",
    "CPS4I": "CPS4",
    "CPS6M": "CPS6",
    "CPS8R": "CPS8",
    # Plane strain
    "CPE4R": "CPE4",
    "CPE4I": "CPE4",
    "CPE6M": "CPE6",
    "CPE8R": "CPE8",
    # Shells
    "S3R": "S3",
    "S4R": "S4",
    "S4R5": "S4",
    "S8R": "S8",
    # Membranes
    "M3D4R": "M3D4",
    # Solids
    "C3D4H": "C3D4",
    "C3D4T": "C3D4",
    "C3D10H": "C3D10",
    "C3D10M": "C3D10",
    "C3D10MH": "C3D10",
    "C3D10T": "C3D10",
    "C3D6H": "C3D6",
    "C3D6T": "C3D6",
    "C3D8R": "C3D8",
    "C3D8I": "C3D8",
    "C3D8H": "C3D8",
    "C3D8RH": "C3D8",
    "C3D8T": "C3D8",
    "C3D8RT": "C3D8",
    "C3D15H": "C3D15",
    "C3D20R": "C3D20",
    "C3D20H": "C3D20",
    "C3D20RH": "C3D20",
    "C3D20T": "C3D20",
    "C3D20RT": "C3D20",
}


def get_abaqus_element_type(name: str) -> MeshElementType:
    name = name.upper()

    if name in ABAQUS_ELEMENT_ALIASES:
        name = ABAQUS_ELEMENT_ALIASES[name]

    try:
        return ABAQUS_ELEMENT_TYPES[name]
    except KeyError:
        raise ValueError(f"Unsupported ABAQUS element type {name!r}")


NASTRAN_ELEMENT_TYPES = {
    # ------------------------------------------------------------
    # 1D
    # ------------------------------------------------------------
    "CROD": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1),
        edge_nodes=((),),
    ),
    "CTUBE": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1),
        edge_nodes=((),),
    ),
    "CBAR": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1),
        edge_nodes=((),),
    ),
    "CBEAM": MeshElementType(
        cell_type=CellType.SEGMENT,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1),
        edge_nodes=((),),
    ),
    # ------------------------------------------------------------
    # Triangles
    #
    # CTRIA6:
    #
    #       2
    #       o
    #      / \
    #   5 o   o 4
    #    /     \
    #   o---o---o
    #   0   3   1
    #
    # REFERENCE_CELLS[TRIANGLE].edges:
    #   (0,1), (1,2), (2,0)
    # ------------------------------------------------------------
    "CTRIA3": MeshElementType(
        cell_type=CellType.TRIANGLE,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2),
        edge_nodes=((), (), ()),
    ),
    "CTRIA6": MeshElementType(
        cell_type=CellType.TRIANGLE,
        degree=2,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2),
        edge_nodes=(
            (3,),  # 0-1
            (4,),  # 1-2
            (5,),  # 2-0
        ),
    ),
    # ------------------------------------------------------------
    # Quadrilaterals
    #
    # CQUAD8:
    #
    #   3----6----2
    #   |         |
    #   7         5
    #   |         |
    #   0----4----1
    #
    # edges:
    #   (0,1), (1,2), (2,3), (3,0)
    # ------------------------------------------------------------
    "CQUAD4": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=((), (), (), ()),
    ),
    "CQUADR": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=((), (), (), ()),
    ),
    "CQUAD8": MeshElementType(
        cell_type=CellType.QUADRILATERAL,
        degree=2,
        family=ElementFamily.SERENDIPITY,
        vertices=(0, 1, 2, 3),
        edge_nodes=(
            (4,),
            (5,),
            (6,),
            (7,),
        ),
    ),
    # ------------------------------------------------------------
    # Tetrahedra
    #
    # CTETRA10
    #
    # vertices: 0,1,2,3
    #
    # midside nodes:
    #   4 : 0-1
    #   5 : 1-2
    #   6 : 2-0
    #   7 : 0-3
    #   8 : 1-3
    #   9 : 2-3
    #
    # ------------------------------------------------------------
    "CTETRA4": MeshElementType(
        cell_type=CellType.TETRAHEDRON,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=((), (), (), (), (), ()),
    ),
    "CTETRA10": MeshElementType(
        cell_type=CellType.TETRAHEDRON,
        degree=2,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3),
        edge_nodes=(
            (4,),  # 0-1
            (5,),  # 1-2
            (6,),  # 2-0
            (7,),  # 0-3
            (8,),  # 1-3
            (9,),  # 2-3
        ),
    ),
    # ------------------------------------------------------------
    # Hexahedra
    #
    # vertices:
    #
    #       7---------6
    #      /|        /|
    #     4---------5 |
    #     | |       | |
    #     | 3-------|-2
    #     |/        |/
    #     0---------1
    #
    # CHEXA20 midside nodes:
    #
    #   8 : 0-1
    #   9 : 1-2
    #  10 : 2-3
    #  11 : 3-0
    #
    #  12 : 4-5
    #  13 : 5-6
    #  14 : 6-7
    #  15 : 7-4
    #
    #  16 : 0-4
    #  17 : 1-5
    #  18 : 2-6
    #  19 : 3-7
    # ------------------------------------------------------------
    "CHEXA8": MeshElementType(
        cell_type=CellType.HEXAHEDRON,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3, 4, 5, 6, 7),
        edge_nodes=((), (), (), (), (), (), (), (), (), (), (), ()),
    ),
    "CHEXA20": MeshElementType(
        cell_type=CellType.HEXAHEDRON,
        degree=2,
        family=ElementFamily.SERENDIPITY,
        vertices=(0, 1, 2, 3, 4, 5, 6, 7),
        edge_nodes=(
            (8,),
            (9,),
            (10,),
            (11,),
            (12,),
            (13,),
            (14,),
            (15,),
            (16,),
            (17,),
            (18,),
            (19,),
        ),
    ),
    # ------------------------------------------------------------
    # Prisms / wedges
    #
    # vertices:
    # bottom: 0,1,2
    # top:    3,4,5
    #
    # edges:
    #   (0,1), (1,2), (2,0)
    #   (3,4), (4,5), (5,3)
    #   (0,3), (1,4), (2,5)
    # ------------------------------------------------------------
    "CPENTA6": MeshElementType(
        cell_type=CellType.PRISM,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3, 4, 5),
        edge_nodes=((), (), (), (), (), (), (), (), ()),
    ),
    "CPENTA15": MeshElementType(
        cell_type=CellType.PRISM,
        degree=2,
        family=ElementFamily.SERENDIPITY,
        vertices=(0, 1, 2, 3, 4, 5),
        edge_nodes=(
            (6,),  # 0-1
            (7,),  # 1-2
            (8,),  # 2-0
            (9,),  # 3-4
            (10,),  # 4-5
            (11,),  # 5-3
            (12,),  # 0-3
            (13,),  # 1-4
            (14,),  # 2-5
        ),
    ),
    # ------------------------------------------------------------
    # Pyramids
    #
    # vertices:
    # base: 0,1,2,3
    # apex: 4
    #
    # edges:
    #   (0,1), (1,2), (2,3), (3,0)
    #   (0,4), (1,4), (2,4), (3,4)
    # ------------------------------------------------------------
    "CPYRAM5": MeshElementType(
        cell_type=CellType.PYRAMID,
        degree=1,
        family=ElementFamily.LAGRANGE,
        vertices=(0, 1, 2, 3, 4),
        edge_nodes=((), (), (), (), (), (), (), ()),
    ),
    "CPYRAM13": MeshElementType(
        cell_type=CellType.PYRAMID,
        degree=2,
        family=ElementFamily.SERENDIPITY,
        vertices=(0, 1, 2, 3, 4),
        edge_nodes=(
            (5,),  # 0-1
            (6,),  # 1-2
            (7,),  # 2-3
            (8,),  # 3-0
            (9,),  # 0-4
            (10,),  # 1-4
            (11,),  # 2-4
            (12,),  # 3-4
        ),
    ),
}

NASTRAN_SOLID_TYPES = {
    ("CTETRA", 4): "CTETRA4",
    ("CTETRA", 10): "CTETRA10",
    ("CPENTA", 6): "CPENTA6",
    ("CPENTA", 15): "CPENTA15",
    ("CHEXA", 8): "CHEXA8",
    ("CHEXA", 20): "CHEXA20",
    ("CPYRAM", 5): "CPYRAM5",
    ("CPYRAM", 13): "CPYRAM13",
}


def get_nastran_element_type(
    card_type: str,
    num_nodes: int,
) -> MeshElementType:

    card_type = card_type.upper()

    key = card_type

    if card_type in {"CTETRA", "CPENTA", "CHEXA", "CPYRAM"}:
        try:
            key = NASTRAN_SOLID_TYPES[(card_type, num_nodes)]
        except KeyError:
            raise ValueError(
                f"Unsupported NASTRAN {card_type} with " f"{num_nodes} nodes"
            )

    try:
        return NASTRAN_ELEMENT_TYPES[key]
    except KeyError:
        raise ValueError(f"Unsupported NASTRAN element type {card_type!r}")
