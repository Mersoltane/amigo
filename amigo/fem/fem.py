import amigo as am
import numpy as np
from .element import FiniteElement, FiniteElementOutput, FiniteElementFieldOutput
from .fem_space import Space, SolutionSpace
from .basis import make_basis
from .dof_handler import DofSource, DegreesOfFreedom
from .boundary_conditions import BoundaryConditions
from .quadrature import make_quadrature, ReducedQuadQuadrature


class Problem:
    def __init__(
        self,
        mesh,
        soln_space: SolutionSpace,
        geo_space: SolutionSpace,
        data_space: SolutionSpace | None = None,
        integrand_map=None,
        integrand_formulation="potential",
        bc_map=None,
        element_objs=None,
    ):
        if integrand_map is None:
            integrand_map = {}
        if bc_map is None:
            bc_map = {}
        if element_objs is None:
            element_objs = {}
        self.mesh = mesh
        self.soln_space = soln_space
        self.data_space = data_space
        self.geo_space = geo_space

        self.integrand_map = integrand_map
        self.integrand_formulation = integrand_formulation
        self.bc_map = bc_map

        # Set the element objects, if any
        self.element_objs = element_objs

        # Set up constraints to be linked
        kind = ["input"]
        if self.integrand_formulation == "weak":
            kind.append("multiplier")

        # Initialize Dofs
        self.soln_dof = DegreesOfFreedom(self.mesh, self.soln_space, kind=kind)
        self.geo_dof = DegreesOfFreedom(self.mesh, self.geo_space, kind="data")
        if self.data_space is not None:
            self.data_dof = DegreesOfFreedom(self.mesh, self.data_space, kind="data")
        else:
            self.data_dof = None

        # Get the handler for the boundary conditions
        dof_handler = self.soln_dof.get_dof_handler()

        # Build the boundary conditions
        self.boundary_conditions = BoundaryConditions(
            bc_map, dof_handler, self.integrand_formulation
        )

        return

    def _create_element_objs(self):
        # Create the element objects
        for integrand_name in self.integrand_map:
            targets = self.integrand_map[integrand_name]["target"]
            integrand = self.integrand_map[integrand_name].get("integrand", None)
            integration_rule = self.integrand_map[integrand_name].get("rule", None)

            # Figure out the cell types that we need
            ctypes = self.mesh.get_cell_types(targets)

            # Loop over the element types for this integrand
            for ctype in ctypes:
                if (integrand_name, ctype) in self.element_objs:
                    continue

                # Set the element name
                elem_name = f"Element{integrand_name}{ctype.name}"

                # Get the basis objects for the element type
                test_basis = None
                if self.integrand_formulation == "weak":
                    test_basis = make_basis(self.soln_space, ctype, kind="multiplier")

                soln_basis = make_basis(self.soln_space, ctype, kind="input")
                geo_basis = make_basis(self.geo_space, ctype, kind="data")
                data_basis = None
                if self.data_space is not None:
                    data_basis = make_basis(self.data_space, ctype, kind="data")

                # reduced integration option if rule is given
                if integration_rule == ["reduced"]:
                    # override get quadrature with custom quadrature
                    quadrature = ReducedQuadQuadrature()
                elif integration_rule is not None:
                    raise ValueError("Non-standard integration rule not supported")
                else:
                    quadrature = make_quadrature(self.soln_space, ctype)

                # Create the element object
                obj = FiniteElement(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    integrand,
                    test_basis=test_basis,
                )

                # Set this into the element dictionary
                self.element_objs[(integrand_name, ctype)] = obj

        return

    def create_model(self, module_name: str):
        """Create and link the Amigo model"""
        model = am.Model(module_name)

        self.soln_dof.add_source(model)
        self.geo_dof.add_source(model)
        if self.data_dof is not None:
            self.data_dof.add_source(model)

        # Figure out which elements need to be created
        self._create_element_objs()

        # Get the dof handler
        dof_handler = self.soln_dof.get_dof_handler()

        # Add the element component objects
        for integrand_name in self.integrand_map:
            targets = self.integrand_map[integrand_name]["target"]
            block_ids = self.mesh.get_block_ids(targets)

            for block_id in block_ids:
                ctype = self.mesh.get_cell_type(block_id)

                elem = self.element_objs[(integrand_name, ctype)]
                comp_name = f"Element{integrand_name}{ctype.name}{block_id}"

                # Add the element/component
                nelems = self.mesh.get_num_elements(block_id)
                model.add_component(comp_name, nelems, elem)

                # Link all the element dof to the component
                self.soln_dof.link_dof(model, block_id, comp_name)
                self.geo_dof.link_dof(model, block_id, comp_name)
                if self.data_dof is not None:
                    self.data_dof.link_dof(model, block_id, comp_name)

                # Set the element signs if relevant
                for space in self.soln_space.get_spaces():
                    if space.func_space == Space.HDIV:
                        signs = dof_handler.get_dof_signs(space, block_id)
                        model.set_data(f"{comp_name}.hdiv_signs", signs)
                    elif space.func_space == Space.HCURL:
                        signs = dof_handler.get_dof_signs(space, block_id)
                        model.set_data(f"{comp_name}.hcurl_signs", signs)

        # Add BC components and links
        self.boundary_conditions.add_bcs(model)

        # Set the node locations directly
        spatial_dim = self.mesh.get_spatial_dim()
        spatial_names = ["x", "y", "z"][:spatial_dim]

        # Get the first (and what should be the only) function space
        geo_space = self.geo_space.get_spaces()[0]
        coords = self.geo_dof.get_coordinates()
        for k, name in enumerate(self.geo_space.get_names(geo_space)):
            if name in spatial_names:
                model.set_data(f"geo.{name}", coords[:, k])

        return model

    def _create_output_objs(self, output_map, output_objs):
        # Create the output objects
        for out_name in output_map:
            targets = output_map[out_name]["target"]
            output_names = output_map[out_name]["names"]
            output_func = output_map[out_name]["function"]

            # Figure out the cell types that we need
            ctypes = self.mesh.get_cell_types(targets)

            # Loop over the element types for generating the output function
            for ctype in ctypes:
                if (out_name, ctype) in output_objs:
                    continue

                elem_name = f"ElementOutput{out_name}{ctype.name}"

                # Get the basis objects for the element type
                soln_basis = make_basis(self.soln_space, ctype, kind="input")
                geo_basis = make_basis(self.geo_space, ctype, kind="data")
                data_basis = None
                if self.data_space is not None:
                    data_basis = make_basis(self.data_space, ctype, kind="data")

                # Create the quadrature instance
                quadrature = make_quadrature(self.soln_space, ctype)

                # Create the output object
                obj = FiniteElementOutput(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    output_names,
                    output_func,
                )

                # Set this into the output dictionary
                output_objs[(out_name, ctype)] = obj

        return

    def add_integrated_output(
        self,
        model: am.Model,
        output_group_name="outputs",
        output_map: dict = {},
        output_objs: dict = {},
    ):
        """
        Add integrated functional outputs to the model and link them.
        """
        if len(output_map) == 0:
            return

        # Make a list of all of the outputs
        all_outputs = []
        for out_name in output_map:
            for name in output_map[out_name]["names"]:
                if not (name in all_outputs):
                    all_outputs.append(name)

        # Add the outputs component
        model.add_component(output_group_name, 1, DofSource(output_names=all_outputs))

        self._create_output_objs(output_map, output_objs)

        for out_name in output_map:
            targets = output_map[out_name]["target"]
            output_names = output_map[out_name]["names"]
            block_ids = self.mesh.get_block_ids(targets)

            for block_id in block_ids:
                ctype = self.mesh.get_cell_type(block_id)
                obj = output_objs[(out_name, ctype)]
                comp_name = f"ElementOutput{out_name}{ctype.name}{block_id}"

                # Add the element/component
                nelems = self.mesh.get_num_elements(block_id)
                model.add_component(comp_name, nelems, obj)

                # Link all the element dof to the component
                self.soln_dof.link_dof(model, block_id, comp_name)
                self.geo_dof.link_dof(model, block_id, comp_name)
                if self.data_dof is not None:
                    self.data_dof.link_dof(model, block_id, comp_name)

                # Link the outputs
                for name in output_names:
                    model.link(f"{comp_name}.{name}", f"{output_group_name}.{name}[0]")

        return

    def _create_field_output_objs(
        self, output_space: SolutionSpace, output_map: dict = {}, output_objs: dict = {}
    ):
        # Create the output objects
        for out_name in output_map:
            targets = output_map[out_name]["target"]
            output_func = output_map[out_name]["function"]

            # Figure out the cell types that we need
            ctypes = self.mesh.get_cell_types(targets)

            # Loop over the element types for generating the output function
            for ctype in ctypes:
                if (out_name, ctype) in output_objs:
                    continue

                elem_name = f"ElementFieldOutput{out_name}{ctype.name}"

                # Get the basis objects for the element type
                soln_basis = make_basis(self.soln_space, ctype, kind="input")
                geo_basis = make_basis(self.geo_space, ctype, kind="data")
                data_basis = None
                if self.data_space is not None:
                    data_basis = make_basis(self.data_space, ctype, kind="data")

                output_basis = make_basis(output_space, ctype, kind="output")

                # Create the output object
                obj = FiniteElementFieldOutput(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    output_basis,
                    output_func,
                )

                # Set this into the output dictionary
                output_objs[(out_name, ctype)] = obj

        return

    def add_field_output(
        self,
        model,
        output_space: SolutionSpace,
        output_map: dict = {},
        output_objs: dict = {},
    ):
        # Create the output dof
        output_dof = DegreesOfFreedom(self.mesh, output_space, kind=["output"])
        output_dof.add_source(model)

        # Figure out which elements need to be created
        self._create_field_output_objs(output_space, output_map, output_objs)

        # Add the element component objects
        for output_name in output_map:
            targets = output_map[output_name]["target"]
            block_ids = self.mesh.get_block_ids(targets)

            for block_id in block_ids:
                ctype = self.mesh.get_cell_type(block_id)

                elem = output_objs[(output_name, ctype)]
                comp_name = f"ElementFieldOutput{output_name}{ctype.name}{block_id}"

                # Add the element/component
                nelems = self.mesh.get_num_elements(block_id)
                model.add_component(comp_name, nelems, elem)

                # Link all the element dof to the component
                self.soln_dof.link_dof(model, block_id, comp_name)
                self.geo_dof.link_dof(model, block_id, comp_name)
                if self.data_dof is not None:
                    self.data_dof.link_dof(model, block_id, comp_name)

                # Link the output dof
                output_dof.link_dof(model, block_id, comp_name)

        return
