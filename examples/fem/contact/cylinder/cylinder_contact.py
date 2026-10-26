# Author: Jack Turbush
# Description: MITC4 shell FEM contact problem on a cylindrical shell geometry.

"""
MITC4 shell FEM on a cylinder — end traction variant.
Load applied as axial traction integrated along top ring (LINE1, T3D2),
instead of a distributed surface pressure.
"""

import argparse
import numpy as np
import amigo as am
from amigo.fem import Mesh, Problem, CellType, build_grid
from amigo.fem.elements import MITCShell


class GapConstraint(am.Component):
    """Floor contact in x: (x + u) - x_floor <= 0"""

    def __init__(self):
        super().__init__()
        self.add_data("x")
        self.add_data("x_floor", value=1.25)
        self.add_input("u")
        self.add_constraint("gap", lower=-am.inf, upper=0.0)

    def compute(self):
        x = self.data["x"]
        u = self.inputs["u"]
        x_floor = self.data["x_floor"]
        self.constraints["gap"] = (x + u) - x_floor


def traction(soln, data=None, geo=None):
    """Add a surface traction in the z-direction"""
    tx = 1e3
    return -tx * soln["u"].value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

mesh = Mesh("cylinder.inp")

# Set the domain names
surfaces = ["SURFACE1"]
bottom_line = "LINE3"
lateral_line = "LINE2"
top_line = "LINE1"

# Create the shell element
ctype = CellType.QUADRILATERAL

# Set the material properties
E = 70e4
nu = 0.3
t = 0.01
mitc_shell = MITCShell(E=E, nu=nu, t=t)
soln_space, geo_space, data_space, output_space = mitc_shell.get_default_spaces()

integrand_map = {"shell": {"target": surfaces}, "traction": {"target": surfaces}}

bc_map = {
    "pinned_bottom": {
        "type": "dirichlet",
        "input": ["u", "v", "w", "rz"],
        "target": [bottom_line],
    },
    "pinned_top": {
        "type": "dirichlet",
        "input": ["u", "v", "w", "rz"],
        "target": [top_line],
    },
}

# Create the element object
element_objs = {
    ("shell", ctype): mitc_shell.create_element(name="Shell", ctype=ctype),
    ("traction", ctype): mitc_shell.create_element(
        name="Traction", ctype=ctype, integrand=traction
    ),
}

problem = Problem(
    mesh,
    soln_space,
    geo_space,
    data_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
    element_objs=element_objs,
)

model = problem.create_model("cylinder_shell")

# Add the shell normals to the output
MITCShell.add_shell_normals(problem, model, surfaces)

# Find the nodes that are strictly in the interior of the mesh
handler = problem.soln_dof.get_dof_handler()
ndofs = handler.get_num_dof("u")
clamped_dofs = handler.get_dof_in_domain("u", [bottom_line, top_line])
dofs = np.setdiff1d(np.arange(ndofs), clamped_dofs)

# Add the contact constraint
model.add_component("contact", len(dofs), GapConstraint())
model.link("geo.x", "contact.x", src_indices=dofs)
model.link("soln.u", "contact.u", src_indices=dofs)
model.link("contact.x_floor", "contact.x_floor[0]")

if args.build:
    model.build_module()

model.initialize()
MITCShell.set_shell_normals(model)

print("Number of variables... ", model.num_variables)

x = model.create_vector()

# Optimization for the nonlinear solve
opt_options = {
    "max_iterations": 200,
    "convergence_tolerance": 1e-8,
    "max_line_search_iterations": 1,
    "initial_barrier_param": 0.1,
}

opt = am.Optimizer(model, x)
opt.optimize(opt_options)

print("Plotting...")

# Hack - modify the mesh coordinates
alpha = 1.0

dof_to_node = handler.get_dof_to_node_map("u")
mesh.X[dof_to_node, 0] += alpha * x["soln.u"]
mesh.X[dof_to_node, 1] += alpha * x["soln.v"]
mesh.X[dof_to_node, 2] += alpha * x["soln.w"]

# Estimate the multipliers
mu = opt.state.mu
data = model.get_data_vector()
fx = np.sum(mu / (data["contact.x_floor[0]"] - (x["soln.u"] + data["geo.x"])))

print(f"Force on plane: {fx:10.6e}")

# Plot the mesh
grid = build_grid(problem, "u", x, surfaces)
grid.plot(scalars="u", cmap="coolwarm", show_edges=False)
