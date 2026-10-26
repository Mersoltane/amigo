import numpy as np
from amigo.fem import Mesh, Problem, SolutionSpace, CellType
import amigo as am
import matplotlib.pyplot as plt
import argparse
import matplotlib.tri as tri


def line_dofs(name, handler):
    """Return the unique solution DOFs on a boundary line domain."""
    return np.unique(handler.get_dof_in_domain("dx", name))


def apply_morph_bcs(x, dofs):
    """Write the prescribed inner-boundary displacements into the state."""
    dx_state = x["submodel.soln.dx"]
    dy_state = x["submodel.soln.dy"]
    dx_state[dofs] = morph_dx[dofs]
    dy_state[dofs] = morph_dy[dofs]
    x["submodel.soln.dx"] = dx_state
    x["submodel.soln.dy"] = dy_state


def plot_mesh(mesh: Mesh, dx, dy):
    fig, ax = plt.subplots()

    # Plot the triangular surface blocks (original + deformed).
    for block_id in range(mesh.get_num_blocks()):
        if mesh.get_cell_type(block_id) != CellType.TRIANGLE:
            continue

        # Element connectivity into the mesh.X node ordering.
        conn = mesh.blocks[block_id].connectivity

        triang = tri.Triangulation(mesh.X[:, 0], mesh.X[:, 1], conn)
        triang_deformed = tri.Triangulation(mesh.X[:, 0] + dx, mesh.X[:, 1] + dy, conn)
        ax.triplot(triang, color="grey", linestyle="--", linewidth=1.0)
        ax.triplot(triang_deformed, color="blue", linestyle="-", linewidth=1.0)

    ax.set_aspect("equal")
    ax.set_axis_off()
    return


def strain_energy_integrand_linear(soln, data=None, geo=None):
    # Gradients of the solution field
    dx_grad = soln["dx"].grad
    dy_grad = soln["dy"].grad

    # Strains
    exx = dx_grad[0]
    eyy = dy_grad[1]
    exy = dx_grad[1] + dy_grad[0]

    # Constitutive model
    E = 1.0
    nu = 0.3

    # Assume uniform thickness
    t = 1.0

    # Plane strain
    alpha_plane_strain = E / ((1 + nu) * (1 - 2 * nu))
    C11 = 1.0 - nu
    C12 = nu
    C13 = 0.0
    C21 = nu
    C22 = 1.0 - nu
    C23 = 0.0
    C31 = 0.0
    C32 = 0.0
    C33 = 0.5 * (1 - 2 * nu)

    # Constant strain elements total internal strain energy
    alpha = alpha_plane_strain
    Wx = exx * (C11 * exx + C12 * eyy + C13 * exy)
    Wy = eyy * (C21 * exx + C22 * eyy + C23 * exy)
    Wxy = exy * (C31 * exx + C32 * eyy + C33 * exy)

    # Total strain potential energy integrand
    W_linear = 0.5 * alpha * (Wx + Wy + Wxy) * t

    # Total strain energy
    W_total = W_linear

    return W_total


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build",
    dest="build",
    action="store_true",
    default=False,
    help="Enable building",
)

args = parser.parse_args()

# Define the spaces for the solutions
soln_space = SolutionSpace({"soln": {("dx", "dy"): "H1"}})
geo_space = SolutionSpace({"geo": {("x", "y"): "H1"}})
data_space = None

integrand_map = {
    "plane_stress": {
        "target": ["SURFACE1", "SURFACE2"],
        "integrand": strain_energy_integrand_linear,
    },
}

outer_plate_boundary = ["LINE1", "LINE2", "LINE3", "LINE4"]
inner_plate_boundary = ["LINE5", "LINE6", "LINE7", "LINE8"]
bc_map = {
    "clamp_outer": {
        "type": "dirichlet",
        "target": outer_plate_boundary,
        "input": ["dx", "dy"],
    },
    "clamp_inner": {
        "type": "dirichlet",
        "target": inner_plate_boundary,
        "input": ["dx", "dy"],
    },
}

# Create a global model
model = am.Model("model")

# Initialize the mesh object
mesh = Mesh("mesh.inp")

# Create the submodel for the plane stress problem
problem = Problem(
    mesh,
    soln_space,
    geo_space,
    data_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
)
submodel = problem.create_model("mesh_morph")

# Add the submodel to the global model
model.add_model("submodel", submodel)

# Build the module
if args.build:
    model.build_module()

# Initilize the model
model.initialize()

# Declare the dof handler
dof_handler = problem.soln_dof.get_dof_handler()

# Number of solution DOFs for solution space dx
num_dof = dof_handler.get_num_dof(soln_space.get_space("dx"))

# Prescribed displacement values, indexed by solution DOF number.
morph_dx = np.zeros(num_dof)
morph_dy = np.zeros(num_dof)
morph_dx[line_dofs("LINE5", dof_handler)] = 1.0
morph_dy[line_dofs("LINE6", dof_handler)] = -0.1
morph_dx[line_dofs("LINE7", dof_handler)] = -0.5
morph_dy[line_dofs("LINE8", dof_handler)] = -0.1

# Collect the DOFs of the inner boundary where the morph BCs are applied.
inner_bc_dofs = np.unique(
    np.concatenate([line_dofs(name, dof_handler) for name in inner_plate_boundary])
)

# Create the vectors and matrices for the model
x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

# Make sure x is updated with the boundary conditions
apply_morph_bcs(x, inner_bc_dofs)

# Evaluate the gradient and the Hessian
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

# Only a single Newton iteration is required for the linear problem.
ldl = am.SparseLDL(mat, am.SolverType.LDL)
ldl.factor()
ldl.solve(g.get_vector())
x[:] -= g[:]

# Extract displacement fields (solution DOF order) and reorder them into mesh
# node order so they align with mesh.X for plotting.
dof_to_node = dof_handler.get_dof_to_node_map("dx")
dx = np.zeros(num_dof)
dy = np.zeros(num_dof)
dx[dof_to_node] = x["submodel.soln.dx"]
dy[dof_to_node] = x["submodel.soln.dy"]

# Plot the mesh
plot_mesh(mesh, dx, dy)
plt.show()
