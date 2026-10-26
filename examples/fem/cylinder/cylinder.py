import argparse
import numpy as np
import amigo as am
from utils import get_exact_solution
import time
from amigo.fem import Mesh, Problem, CellType
from amigo.fem.elements import MITCShell
from amigo.fem.visualization import build_grid
from scipy.sparse.linalg import spsolve


def traction(soln, geo=None, data=None):
    u = soln["u"].value
    v = soln["v"].value
    w = soln["w"].value

    x = geo["x"].value
    y = geo["y"].value
    z = geo["z"].value

    nx = geo["nx"].value
    ny = geo["ny"].value
    nz = geo["nz"].value

    R = 1.0
    L = 2.0

    M = 4.0
    N = 3.0

    alpha = M / R
    beta = N * np.pi / L

    theta = am.atan2(x, y)

    # Make the pressure dependent on the coordinate
    pressure = am.sin(alpha * R * theta) * am.sin(beta * z)

    # Set the work term
    W = pressure * (u * nx + v * ny + w * nz)

    return -W


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
parser.add_argument(
    "--solver",
    dest="solver",
    choices=["cholesky", "cholesky_left", "ldl", "scipy", "cuda"],
    default="cholesky",
)
args = parser.parse_args()

# Load the mesh
mesh = Mesh("cylinder.inp")

surfaces = ["SURFACE1"]
bottom_line = "LINE3"
lateral_line = "LINE2"
top_line = "LINE1"

# Create the shell element
ctype = CellType.QUADRILATERAL
mitc_shell = MITCShell()
soln_space, geo_space, data_space, output_space = mitc_shell.get_default_spaces()

integrand_map = {"shell": {"target": surfaces}, "traction": {"target": surfaces}}

bc_map = {
    "pinned_bottom": {
        "type": "dirichlet",
        "input": ["u", "v", "rz"],
        "target": [bottom_line],
    },
    "pinned_top": {
        "type": "dirichlet",
        "input": ["u", "v", "rz"],
        "target": [top_line],
    },
}

# Create the element objects
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

# Add a fixed boundary condition for the zeroth node.
model.add_fixed("soln.w[0]")

if args.build:
    model.build_module()

model.initialize()

# Set the shell normals
MITCShell.set_shell_normals(model)

print("Number of variables... ", model.num_variables)

# Create the vectors and matrices for the model
x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

# Copy the data over to the GPU
data = model.get_data_vector()
data.copy_host_to_device()

print("Evaluating the Hessian...")
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

num_factors = 1
if args.solver == "cuda":
    from amigo.amigo import CSRMatFactorCuda

    # Duplicate the matrix
    mat_copy = mat.duplicate()
    mat_copy.copy(mat)

    pivot_eps = 1e-12
    solver = CSRMatFactorCuda(mat_copy, pivot_eps)
    solver.factor()

    start_time = time.perf_counter()
    for i in range(num_factors):
        mat_copy.copy(mat)
        solver.factor()
    end_time = time.perf_counter()
    tfactor = (end_time - start_time) / num_factors

    solver.solve(g.get_vector(), x.get_vector())

    x.copy_device_to_host()
else:
    g.copy_device_to_host()
    mat.copy_data_device_to_host()

    if args.solver == "cholesky" or args.solver == "ldl":
        stype = am.SolverType.CHOLESKY
        if args.solver == "ldl":
            stype = am.SolverType.LDL

        ldl = am.SparseLDL(mat, stype, ustab=0.05)
        flag = ldl.factor()

        start_time = time.perf_counter()
        for i in range(num_factors):
            flag = ldl.factor()
        end_time = time.perf_counter()
        if flag != 0:
            print(f"LDL factor flag {flag}")

        x[:] = g[:]
        ldl.solve(x.get_vector())
        if stype == am.SolverType.LDL:
            print("Inertia: ", ldl.get_inertia())

        tfactor = (end_time - start_time) / num_factors
    elif args.solver == "cholesky_left":
        chol = am.SparseCholesky(mat)
        start_time = time.perf_counter()
        for i in range(num_factors):
            flag = chol.factor()
        end_time = time.perf_counter()
        if flag != 0:
            print(f"Cholesky factor flag {flag}")

        x[:] = g[:]
        chol.solve(x.get_vector())

        tfactor = (end_time - start_time) / num_factors
    elif args.solver == "scipy":
        csr = am.tocsr(mat)

        # This isn't a completely fair comparison
        start_time = time.perf_counter()
        x[:] = spsolve(csr, g[:])
        end_time = time.perf_counter()

        tfactor = end_time - start_time

print(f"Factor time... {tfactor:.6f} seconds")

u_ex, v_ex, w_ex = get_exact_solution(data["geo.x"], data["geo.y"], data["geo.z"])

w = x["soln.w"]
w_diff = w - w_ex
print("Error = ", np.max(np.absolute(w_diff)) / np.max(w))

# Plot the mesh
grid = build_grid(problem, "u", x, surfaces)
grid.plot(scalars="u", cmap="coolwarm", show_edges=False)
