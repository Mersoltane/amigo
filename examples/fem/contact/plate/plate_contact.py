# Author: Jack Turbush
# Description: Plate contact problem using reduced-integration shell elements.

import amigo as am
from amigo.fem.elements import MITCPlate
from amigo.fem import Mesh, Problem, CellType, build_grid
import numpy as np
import argparse


class GapConstraint(am.Component):
    """Floor contact: gap = w_coord <= z_contact"""

    def __init__(self):
        super().__init__()
        self.add_input("w")
        self.add_data("z_contact", value=0.2)
        self.add_constraint("gap", lower=-am.inf, upper=0.0)

    def compute(self):
        w = self.inputs["w"]
        z_contact = self.data["z_contact"]
        self.constraints["gap"] = w - z_contact


def traction(soln, data=None, geo=None):
    q0 = 1  # N/m^2, load intensity
    w = soln["w"].value
    return -q0 * w


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

surfaces = ["SURFACE1"]
integrand_map = {"plate": {"target": surfaces}, "traction": {"target": surfaces}}

clamped_domains = ["LINE1", "LINE2", "LINE3", "LINE4"]
bc_map = {
    "clamped": {
        "type": "dirichlet",
        "input": ["w", "rx", "ry"],
        "target": clamped_domains,
    },
}

# Load the plate
mesh = Mesh("plate.inp")

ctype = CellType.QUADRILATERAL

# Assign fixed material properties
E = 70e3  # Young's modulus
nu = 0.3  # Poisson's ratio
t = 0.01  # Plate thickness
mitc_plate = MITCPlate(E=E, nu=nu, t=t)
soln_space, geo_space, data_space, output_space = mitc_plate.get_default_spaces()

# Set the element objects
element_objs = {
    ("plate", ctype): mitc_plate.create_element(name="Plate", ctype=ctype),
    ("traction", ctype): mitc_plate.create_element(
        name="Traction", ctype=ctype, integrand=traction
    ),
}

# Create the finite element problem
problem = Problem(
    mesh,
    soln_space,
    geo_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
    element_objs=element_objs,
)

model = problem.create_model("plate")

# Find the nodes that are strictly in the interior of the mesh
handler = problem.soln_dof.get_dof_handler()
ndofs = handler.get_num_dof("w")
clamped_dofs = handler.get_dof_in_domain("w", clamped_domains)
dofs = np.setdiff1d(np.arange(ndofs), clamped_dofs)

# Add the contact constraint
model.add_component("contact", len(dofs), GapConstraint())
model.link("soln.w", "contact.w", src_indices=dofs)

if args.build:
    model.build_module()
model.initialize()

# Set the z-contact
data = model.get_data_vector()
data["contact.z_contact"] = 0.125

opt_options = {
    "max_iterations": 200,
    "convergence_tolerance": 1e-6,
    "max_line_search_iterations": 1,
    "initial_barrier_param": 0.1,
}

x = model.create_vector()

# Optimize for the plate
opt = am.Optimizer(model, x)
opt.optimize(opt_options)

alpha = 1.0
dof_to_node = handler.get_dof_to_node_map("w")
mesh.X[dof_to_node, 2] = alpha * x["soln.w"]

print("Plotting...")
grid = build_grid(problem, "w", x)
grid.plot(scalars="w", cmap="coolwarm", show_edges=True)
