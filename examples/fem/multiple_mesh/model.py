import numpy as np
from amigo.fem import (
    dot_product,
    Problem,
    Mesh,
    basis,
    CustomOrderedBCs,
    build_grid,
    FunctionSpace,
    Space,
    Conformity,
)
import amigo as am
import argparse
import pyvista as pv


class VecWrapper:
    def __init__(self, v, name):
        self.v = v
        self.name = name

    def __getitem__(self, index: str | np.ndarray | int):
        if isinstance(index, str):
            print(np.linalg.norm(self.v[self.name + "." + index]))
            return self.v[self.name + "." + index]
        else:
            raise ValueError("only works for string inputs")


def ordered_line_dofs(dof_handler, mesh, line_name, field="u", direction=0):
    """
    Return the unique DOFs of ``field`` on the line domain ``line_name``,
    ordered by the nodal coordinate along ``direction``.
    """
    # Global DOFs
    line_dofs = np.unique(dof_handler.get_dof_in_domain(field, line_name))

    # Map each DOF back to its mesh node to recover the coordinate
    dof_to_node = dof_handler.get_dof_to_node_map(field)
    coords = mesh.X[dof_to_node[line_dofs]]

    # Sort by position along the requested direction
    order = np.argsort(coords[:, direction])
    return line_dofs[order]


def potential_1(soln, data=None, geo=None):
    ugrad = soln["u"].grad
    wf = 0.5 * dot_product(ugrad, ugrad, n=2)
    return wf


def potential_2(soln, data=None, geo=None):
    u = soln["u"].value
    ugrad = soln["u"].grad

    f = data["Jz"].value
    wf = 0.5 * dot_product(ugrad, ugrad, n=2) - f * u
    return wf


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Define mesh objects
meshes = {
    "Mesh0": Mesh("multidomain.inp"),
    "Mesh1": Mesh("multidomain.inp"),
}

# Declare custom boundary conditions
custom = CustomOrderedBCs(dir=1, start=False, end=False)

bc_map_mesh0 = {
    "DirichletLine3": {
        "type": "dirichlet",
        "target": ["LINE3"],
        "input": ["u"],
    },
    "SymmMesh0": {
        "type": "scaled",
        "input": ["u"],
        "callback": custom,
        "target": [["LINE2"], ["LINE4"]],
        "scale": [1.0, 1.0],
    },
}

bc_map_mesh1 = {
    "DirichletLine3": {
        "type": "dirichlet",
        "target": ["LINE1"],
        "input": ["u"],
    },
    "SymmMesh0": {
        "type": "scaled",
        "input": ["u"],
        "callback": custom,
        "target": [["LINE2"], ["LINE4"]],
        "scale": [1.0, 1.0],
    },
}

bc_map = {"Mesh0": bc_map_mesh0, "Mesh1": bc_map_mesh1}

# Weak form mapping for each mesh
integrand_map = {
    "Mesh0": {
        "air": {"target": ["SURFACE1"], "integrand": potential_1},
        "coil": {"target": ["SURFACE2", "SURFACE3"], "integrand": potential_2},
    },
    "Mesh1": {
        "air": {"target": ["SURFACE1"], "integrand": potential_1},
        "coil": {"target": ["SURFACE2", "SURFACE3"], "integrand": potential_2},
    },
}

# Initialize the spaces (same for all domains)
const = FunctionSpace(func_space=Space.CONST, conformity=Conformity.COMPONENT)
soln_space = basis.SolutionSpace({"soln": {"u": "H1"}})
data_space = basis.SolutionSpace({"data": {"Jz": const}})
geo_space = basis.SolutionSpace({"geo": {("x", "y"): "H1"}})

# Define the global amigo model
model = am.Model("multi_mesh_model")

# Create an amigo model for each mesh
sub_problems = []
for mesh_name, mesh in meshes.items():
    problem = Problem(
        mesh,
        soln_space,
        geo_space,
        data_space,
        integrand_map=integrand_map[mesh_name],
        bc_map=bc_map[mesh_name],
    )
    sub_problems.append(problem)
    sub_model = problem.create_model(mesh_name)
    model.add_model(mesh_name, sub_model)

mesh0 = meshes["Mesh0"]
mesh1 = meshes["Mesh1"]

# The DOF handlers for the solution field on each mesh
dof_handler_mesh0 = sub_problems[0].soln_dof.get_dof_handler()
dof_handler_mesh1 = sub_problems[1].soln_dof.get_dof_handler()

# The direction along which the shared edge varies (0 -> x, 1 -> y).
shared_dir = 0

# Ordered DOFs along the shared edge for each mesh
dofs_line_1 = ordered_line_dofs(
    dof_handler_mesh0, mesh0, "LINE1", field="u", direction=shared_dir
)
dofs_line_3 = ordered_line_dofs(
    dof_handler_mesh1, mesh1, "LINE3", field="u", direction=shared_dir
)

# Number of points along the shared edge
npts_shared = len(dofs_line_1)

# Domain1 slides to the right by an integer number of points.
slide_number = 0
edge_length = 5.0
x_offset = slide_number * (edge_length / npts_shared)

# Overlapping (shared) portion of the two edges after sliding Mesh1.
# Mesh1's LINE3 is shifted by ``slide_number`` points relative to Mesh0's
# LINE1, so the shared region is where the two overlap.
dofs_line_1_shared = dofs_line_1[slide_number:]
dofs_line_3_shared = (
    dofs_line_3[:] if slide_number == 0 else dofs_line_3[:-slide_number]
)

# Link the solution DOFs on the overlapping region so that they are the
# same variable in the global model (enforces field continuity).
model.link(
    "Mesh0.soln.u",
    "Mesh1.soln.u",
    src_indices=dofs_line_1_shared,
    tgt_indices=dofs_line_3_shared,
)

# Build the model
if args.build:
    model.build_module()

# Initialize the model
model.initialize()

# Set the problem data
data = model.get_data_vector()
data["Mesh0.data.Jz.SURFACE1"] = 0.0
data["Mesh0.data.Jz.SURFACE2"] = 10.0
data["Mesh0.data.Jz.SURFACE3"] = 10.0

data["Mesh1.data.Jz.SURFACE1"] = 0.0
data["Mesh1.data.Jz.SURFACE2"] = 10.0
data["Mesh1.data.Jz.SURFACE3"] = 10.0

x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

model.eval_gradient(x, g)
model.eval_hessian(x, mat)

ldl = am.SparseLDL(mat, am.SolverType.LDL)
ldl.factor()

x[:] = -g[:]
ldl.solve(x.get_vector())

# Visualize the solution field
grid0 = build_grid(sub_problems[0], "u", VecWrapper(x, "Mesh0"))
grid1 = build_grid(sub_problems[1], "u", VecWrapper(x, "Mesh1"))

# Offset Mesh1 so it stacks directly beneath Mesh0, and apply the sliding
# x-offset so the shared edges line up as configured.
grid1.points[:, 0] += x_offset
grid1.points[:, 1] -= 5.0

# Shared color scale for both meshes
clim = [
    min(grid0["u"].min(), grid1["u"].min()),
    max(grid0["u"].max(), grid1["u"].max()),
]

plotter = pv.Plotter()
plotter.add_mesh(grid0, scalars="u", show_edges=False, cmap="coolwarm", clim=clim)
plotter.add_mesh(grid1, scalars="u", show_edges=False, cmap="coolwarm", clim=clim)
plotter.view_xy()
plotter.show()
