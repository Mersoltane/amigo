from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import re
from .cell_types import CellType
from .element_mapping import (
    ElementFamily,
    MeshElementType,
    get_abaqus_element_type,
    get_nastran_element_type,
)


@dataclass
class ElementBlock:
    domain: str
    cell_type: CellType
    degree: int
    family: ElementFamily
    connectivity: np.ndarray
    element_type: MeshElementType
    element_ids: np.ndarray | None = None
    source_type: str | None = None
    property_id: int | None = None


@dataclass
class DirichletBCs:
    index: int
    nodes: np.ndarray
    values: np.ndarray


class InpParser:
    def __init__(self):
        # Raw ABAQUS data during parsing
        self._nodes = {}
        self._elements = {}

        # File-format-independent output
        self.X = None
        self.element_blocks = []

        # Mappings from ABAQUS IDs -> internal contiguous IDs
        self.node_id_map = {}
        self.element_id_map = {}

        # Node sets from the INP
        self.node_sets = {}

        # Dirichlet bcs - set but not used for INP
        self.dirichlet_bcs = {}

        return

    def _read_file(self, filename):
        with open(filename, "r", errors="ignore") as fp:
            return [line.rstrip("\n") for line in fp]

    @staticmethod
    def _split_csv_line(s):
        return [p.strip() for p in s.split(",") if p.strip()]

    @staticmethod
    def _find_kw(header, key):
        m = re.search(
            rf"\b{re.escape(key)}\s*=\s*([^,\s]+)",
            header,
            flags=re.I,
        )
        return m.group(1) if m else None

    def parse(self, filename):
        section = None

        elset = None
        source_type = None
        nset_name = None

        for line in self._read_file(filename):
            raw = line.strip()

            if not raw or raw.startswith("**"):
                continue

            # Keywords
            if raw.startswith("*"):
                header = raw.upper()

                if re.match(r"\*NODE\s*(,|$)", header):
                    section = "NODE"

                elif re.match(r"\*ELEMENT\s*(,|$)", header):
                    section = "ELEMENT"
                    source_type = self._find_kw(header, "TYPE")
                    elset = self._find_kw(header, "ELSET")

                    if source_type is None:
                        raise ValueError(f"No element TYPE specified in {raw!r}")

                    source_type = source_type.upper()

                    # Validate now
                    get_abaqus_element_type(source_type)

                    if elset is None:
                        elset = "__DEFAULT__"
                    else:
                        elset = elset.upper()

                    key = (elset, source_type)
                    self._elements.setdefault(key, {})

                elif header.startswith("*NSET"):
                    section = "NSET"

                    nset_name = self._find_kw(header, "NSET")

                    if nset_name is None:
                        raise ValueError(f"No NSET name specified in {raw!r}")

                    nset_name = nset_name.upper()
                    self.node_sets.setdefault(nset_name, [])

                else:
                    section = None

                continue

            # Section data
            parts = self._split_csv_line(raw)

            if section == "NODE":
                abaqus_nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3]) if len(parts) > 3 else 0.0

                self._nodes[abaqus_nid] = (x, y, z)

            elif section == "ELEMENT":
                abaqus_eid = int(parts[0])

                conn = tuple(int(p) for p in parts[1:])

                self._elements[(elset, source_type)][abaqus_eid] = conn

            elif section == "NSET":
                self.node_sets[nset_name].extend(int(p) for p in parts)

        # Convert ABAQUS IDs into internal contiguous IDs
        self._finalize()

    def _finalize(self):
        self._build_nodes()
        self._build_element_blocks()
        self._convert_node_sets()

    def _build_nodes(self):
        """
        Convert arbitrary ABAQUS node IDs into contiguous internal
        numbering 0 ... num_nodes-1.
        """

        node_ids = sorted(self._nodes)
        self.node_id_map = {nid: i for i, nid in enumerate(node_ids)}
        self.X = np.array([self._nodes[nid] for nid in node_ids], dtype=float)

    def _build_element_blocks(self):
        """
        Convert parsed ABAQUS element sections into generic ElementBlock objects.
        """

        next_element = 0

        for (domain, source_type), elements in self._elements.items():
            element_type = get_abaqus_element_type(source_type)
            abaqus_eids = sorted(elements)
            nelem = len(abaqus_eids)

            if nelem == 0:
                continue

            nnode = len(elements[abaqus_eids[0]])
            conn = np.empty((nelem, nnode), dtype=np.int64)
            element_ids = np.empty(nelem, dtype=np.int64)

            for i, abaqus_eid in enumerate(abaqus_eids):
                abaqus_conn = elements[abaqus_eid]

                if len(abaqus_conn) != nnode:
                    raise ValueError(
                        f"Inconsistent number of nodes in " f"{source_type} block"
                    )

                conn[i, :] = [self.node_id_map[nid] for nid in abaqus_conn]
                element_ids[i] = next_element
                self.element_id_map[abaqus_eid] = next_element

                next_element += 1

            block = ElementBlock(
                domain=domain,
                cell_type=element_type.cell_type,
                degree=element_type.degree,
                family=element_type.family,
                element_type=element_type,
                connectivity=conn,
                element_ids=element_ids,
                source_type=source_type,
            )

            self.element_blocks.append(block)

    def _convert_node_sets(self):
        """
        Convert ABAQUS node IDs in NSETs into internal node IDs.
        """

        for name, node_ids in self.node_sets.items():
            self.node_sets[name] = np.array(
                [self.node_id_map[nid] for nid in node_ids], dtype=np.int64
            )

    def get_nodes(self):
        return self.X

    def get_element_blocks(self):
        return self.element_blocks

    def get_node_sets(self):
        return self.node_sets

    def get_dirichlet_bcs(self):
        return self.dirichlet_bcs


class BdfParser:
    def __init__(self):
        try:
            import pyNastran.bdf.bdf as pn

            self.pn = pn
        except:
            raise RuntimeError("Must install pyNastran to use .bdf files")

        self.X = None
        self.element_blocks = []
        self.node_sets = {}
        self.dirichlet_bcs = {}

        return

    def parse(self, filename: str, debug: bool = False):
        model = self.pn.read_bdf(filename, validate=False, xref=False, debug=debug)

        model.is_xrefed = False

        # Add dummy property cards
        model.missing_properties = False
        for elem_id in model.elements:
            element = model.elements[elem_id]
            if element.pid not in model.property_ids:
                matID = 1
                E = 70.0
                G = 35.0
                nu = 0.3
                model.add_mat1(matID, E, G, nu)
                model.add_pbar(element.pid, matID)

                # Set a warning flag
                model.missing_properties = True

                if debug:
                    print(
                        f"Element ID {elem_id} references undefined property ID "
                        "{element.pid} in bdf file."
                    )

        # Set the node locations
        node_ids = sorted(model.nodes)
        node_map = {nid: i for i, nid in enumerate(node_ids)}
        self.X = np.array(
            [model.nodes[nid].get_position() for nid in node_ids], dtype=float
        )

        # Populate list entries with default values
        domain_names = {}
        for p_id in model.property_ids:
            # Check if there is a Femap/HyperMesh/Patran label for this component
            comment = model.properties[p_id].comment

            # Femap format
            if "$ Femap Property" in comment:
                # Pick off last word from comment, this is the name
                domain_names[p_id] = comment.split()[-1]
            # HyperMesh format
            elif "$HMNAME PROP" in comment:
                # Locate property name line
                loc = comment.find("HMNAME PROP")
                comp_line = comment[loc:]
                # The component name is between double quotes
                domain_names[p_id] = comp_line.split('"')[1]
            # Patran format
            elif "$ Elements and Element Properties for region" in comment:
                # The component name is after the colon
                domain_names[p_id] = comment.split(":")[1]
            else:  # No format, default component name
                domain_names[p_id] = None

        for e_id, elem in model.elements.items():
            if "Shell element data for family" in elem.comment:
                p_id = elem.Pid()
                if domain_names[p_id] is None:
                    domain_names[p_id] = elem.comment.split()[-1]

        for p_id in model.property_ids:
            if domain_names[p_id] is None:
                domain_names[p_id] = f"PID_{p_id}"

        # Elements
        element_conn = {}
        element_ids = {}
        element_card_type = {}

        # Parse each element and add to pid/element type
        for e_id, elem in model.elements.items():
            p_id = elem.Pid()
            card_type = elem.type
            node_ids = [nid for nid in elem.node_ids if nid is not None]

            element_type = get_nastran_element_type(card_type, len(node_ids))
            conn = [node_map[nid] for nid in node_ids]

            key = (p_id, element_type)
            if key in element_conn:
                element_conn[key].append(conn)
                element_ids[key].append(e_id)
            else:
                element_conn[key] = [conn]
                element_ids[key] = [e_id]
            element_card_type[key] = card_type

        # Create the element blocks
        for p_id, element_type in element_conn:
            key = (p_id, element_type)
            conn = np.array(element_conn[key], dtype=np.int64)
            elem_ids = np.array(element_ids[key], dtype=np.int64)
            card_type = element_card_type[key]

            block = ElementBlock(
                domain=domain_names[p_id],
                cell_type=element_type.cell_type,
                degree=element_type.degree,
                family=element_type.family,
                element_type=element_type,
                connectivity=conn,
                element_ids=elem_ids,
                source_type=card_type,
            )
            self.element_blocks.append(block)

        # Parse the boundary conditions
        bcs = {}
        for i in range(6):
            bcs[i] = {"nodes": [], "values": []}

        for spc_id in model.spcs:
            for spc in model.spcs[spc_id]:
                # Loop through every node specifed in this spc and record bc info
                for j, node in enumerate(spc.nodes):
                    if node not in model.node_ids:
                        print(
                            f"Node ID {node} (Nastran ordering) is referenced by an SPC,  "
                            "but the node was not defined in the BDF file. Skipping SPC."
                        )
                        continue

                    # Convert the node number
                    inode = node_map[node]

                    for dof in range(6):
                        if spc.type == "SPC":
                            comp = spc.components[j]
                            val = spc.enforced[j]
                        else:  # SPC1
                            comp = spc.components
                            val = 0.0

                        if f"{dof + 1}" in comp:
                            bcs[dof]["nodes"].append(inode)
                            bcs[dof]["values"].append(val)

        for index in bcs:
            nodes = np.array(bcs[index]["nodes"], dtype=np.int64)
            values = np.array(bcs[index]["values"], dtype=float)
            self.dirichlet_bcs[index] = DirichletBCs(
                index=index, nodes=nodes, values=values
            )

        return

    def get_nodes(self):
        return self.X

    def get_element_blocks(self):
        return self.element_blocks

    def get_node_sets(self):
        return self.node_sets

    def get_dirichlet_bcs(self):
        return self.dirichlet_bcs
