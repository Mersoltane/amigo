import amigo as am
import numpy as np
from .dof_handler import DofHandler
from .fem_space import IntegrandFormulation


class ScaledBC(am.Component):
    def __init__(self, name, input_name=[], scale=[1.0, 1.0]):
        super().__init__(name)

        if len(scale) != 2:
            raise ValueError("scale must be of length 2")

        self.input_name = input_name

        self.add_constant(f"scale_left", value=scale[0])
        self.add_constant(f"scale_right", value=scale[1])

        for name in self.input_name:
            self.add_input(f"{name}_left", value=1.0)
            self.add_input(f"{name}_right", value=1.0)
            self.add_constraint(f"res_{name}")

        return

    def compute(self):
        scale_left = self.constants["scale_left"]
        scale_right = self.constants["scale_right"]

        for name in self.input_name:
            self.constraints[f"res_{name}"] = (
                scale_left * self.inputs[f"{name}_left"]
                + scale_right * self.inputs[f"{name}_right"]
            )
        return


class CustomOrderedBCs:
    def __init__(self, dir=1, start=True, end=True):
        self.dir = dir
        self.start = start
        self.end = end

    def __call__(self, name, targets, dof_handler):
        mesh = dof_handler.get_mesh()
        mapping = dof_handler.get_dof_to_node_map(name)
        inverse = np.empty_like(mapping)
        inverse[mapping] = np.arange(len(mapping))

        # Get the targets
        a, b = targets
        id_a = mesh.get_block_ids(a)
        id_b = mesh.get_block_ids(b)

        coord_left, nodes_left = self._get_ordered_dof(
            id_a, mesh, inverse, self.start, self.end
        )
        coord_right, nodes_right = self._get_ordered_dof(
            id_b, mesh, inverse, self.start, self.end
        )

        if len(nodes_left) != len(nodes_right):
            raise Exception(f"nnodes left != nnodes right")

        # Reorder the nodes to match
        return self._reorder_nodes(coord_left, nodes_left, coord_right, nodes_right)

    def _get_ordered_dof(
        self, ids, mesh, imapping, start: bool = True, end: bool = True
    ):
        # Get the node numbers from the mesh
        all_nodes = []
        for id in ids:
            all_nodes.extend(mesh.blocks[id].connectivity.flatten())
        unique_nodes = list(dict.fromkeys(all_nodes))

        if not start or not end:
            if not start:
                unique_nodes = unique_nodes[1:]
            if not end:
                unique_nodes = unique_nodes[:-1]

        y = mesh.X[unique_nodes, self.dir]

        return y, imapping[unique_nodes]

    def _reorder_nodes(self, coord_left, nodes_left, coord_right, nodes_right):
        nodes_left = np.array(nodes_left)
        nodes_right = np.array(nodes_right)

        idx_left = np.argsort(coord_left)
        idx_right = np.argsort(coord_right)

        return nodes_left[idx_left], nodes_right[idx_right]


class BoundaryConditions:
    def __init__(
        self,
        bc_descript: dict,
        dof_handler: DofHandler,
        integrand_formulation: IntegrandFormulation = IntegrandFormulation.POTENTIAL,
    ):
        # Check if the name of the boundary conditions is recognized
        for bc_name, bc in bc_descript.items():
            if not (
                bc["type"] == "dirichlet"
                or bc["type"] == "continuity"
                or bc["type"] == "scaled"
            ):
                typ = bc["type"]
                raise ValueError(
                    f"Unrecognized boundary condition type {typ} for {bc_name}"
                )
            if bc["type"] == "dirichlet":
                if not (key in bc for key in ["input", "target"]):
                    raise ValueError("Dirichlet BC missing input or target")
            elif bc["type"] == "continuity":
                if not (key in bc for key in ["input", "target", "callback"]):
                    raise ValueError("Continuity BC missing input, target or callback")
            elif bc["type"] == "scaled":
                if not (key in bc for key in ["input", "target", "callback", "scale"]):
                    raise ValueError(
                        "Continuity BC missing input, target, callback or scale"
                    )

        self.bc_descript = bc_descript
        self.dof_handler = dof_handler
        self.integrand_formulation = integrand_formulation
        return

    def _get_target_dof(self, name: str, targets: list[str]):
        all_dof = []
        for target in targets:
            dof = self.dof_handler.get_dof_in_domain(name, target)
            all_dof.extend(dof)

        return np.unique(all_dof)

    def add_bcs(self, model):
        """Add the boundary conditions to the model"""

        # Get the solution space
        solution_space = self.dof_handler.get_solution_space()

        # Extract any dirichlet boundary conditions direct from the DofHandler
        bcs = self.dof_handler.get_dirichlet_bcs()
        if len(bcs) > 0:
            for name in bcs:
                comp_name = solution_space.get_component_name(name)
                model.add_fixed(f"{comp_name}.{name}", bcs[name].dofs)

        for bc_name, bc in self.bc_descript.items():
            if bc["type"] == "dirichlet":
                # Add boundary conditions from the input specification
                input_names = bc["input"]
                for name in input_names:
                    dof = self._get_target_dof(name, bc["target"])

                    # Fix the DOF
                    comp_name = solution_space.get_component_name(name)
                    model.add_fixed(f"{comp_name}.{name}", dof)
                    if self.integrand_formulation == IntegrandFormulation.WEAK:
                        model.add_fixed(f"{comp_name}.res_{name}", dof)
            else:
                input_names = bc["input"]
                target = bc["target"]
                callback = bc["callback"]

                for name in input_names:
                    comp_name = solution_space.get_component_name(name)
                    left_dofs, right_dofs = callback(name, target, self.dof_handler)

                    if bc["type"] == "continuity":
                        model.link(
                            f"{comp_name}.{name}",
                            f"{comp_name}.{name}",
                            src_indices=left_dofs,
                            tgt_indices=right_dofs,
                        )

                    elif bc["type"] == "scaled":
                        scale = bc["scale"]
                        class_name = f"ScaledBC_{bc_name}"
                        bc_src = ScaledBC(class_name, input_names, scale=scale)

                        if len(left_dofs) > 0:
                            model.add_component(f"{bc_name}", len(left_dofs), bc_src)

                            model.link(
                                f"{comp_name}.{name}",
                                f"{bc_name}.{name}_left",
                                src_indices=left_dofs,
                            )
                            model.link(
                                f"{comp_name}.{name}",
                                f"{bc_name}.{name}_right",
                                src_indices=right_dofs,
                            )
        return
