import argparse
import amigo as am
from amigo.fem.elements import MITCPlate
from amigo.fem import Mesh, Problem, CellType, build_grid
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time


def traction(soln, data=None, geo=None):
    """Strain energy density (integrand of TPE equation)"""
    w = soln["w"].value
    fz = 1e3
    return -fz * w


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--solver",
    dest="solver",
    choices=["cholesky", "cholesky_left", "ldl", "scipy", "cuda"],
    default="cholesky",
)
args = parser.parse_args()

surfaces = ["SURFACE1"]
integrand_map = {"plate": {"target": surfaces}, "traction": {"target": surfaces}}

bc_map = {
    "clamped": {
        "type": "dirichlet",
        "input": ["w", "rx", "ry"],
        "target": ["LINE1", "LINE2", "LINE3", "LINE4"],
    },
}

# Load the plate
mesh = Mesh("plate.inp")

ctype = CellType.QUADRILATERAL
mitc_plate = MITCPlate()
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

if args.build:
    model.build_module()

model.initialize()

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

# Solve the equations
print("Solving...")

if args.solver == "cuda":
    from amigo.amigo import CSRMatFactorCuda

    # Duplicate the matrix
    mat_copy = mat.duplicate()
    mat_copy.copy(mat)

    pivot_eps = 1e-12
    solver = CSRMatFactorCuda(mat_copy, pivot_eps)
    solver.factor()

    start_time = time.perf_counter()
    for i in range(10):
        mat_copy.copy(mat)
        solver.factor()
    end_time = time.perf_counter()
    tfactor = (end_time - start_time) / 10

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
        for i in range(10):
            flag = ldl.factor()
        end_time = time.perf_counter()
        if flag != 0:
            print(f"LDL factor flag {flag}")

        x[:] = g[:]
        ldl.solve(x.get_vector())
        if stype == am.SolverType.LDL:
            print("Inertia: ", ldl.get_inertia())

        tfactor = (end_time - start_time) / 10
    elif args.solver == "cholesky_left":
        chol = am.SparseCholesky(mat)
        start_time = time.perf_counter()
        for i in range(10):
            flag = chol.factor()
        end_time = time.perf_counter()
        if flag != 0:
            print(f"Cholesky factor flag {flag}")

        x[:] = g[:]
        chol.solve(x.get_vector())

        tfactor = (end_time - start_time) / 10
    elif args.solver == "scipy":
        csr = am.tocsr(mat)

        # This isn't a completely fair comparison
        start_time = time.perf_counter()
        x[:] = spsolve(csr, g[:])
        end_time = time.perf_counter()

        tfactor = end_time - start_time

print(f"Factor time... {tfactor:.6f} seconds")

print("Plotting...")
grid = build_grid(problem, "w", x)
grid.plot(scalars="w", cmap="coolwarm", show_edges=True)
