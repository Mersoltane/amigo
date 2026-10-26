import numpy as np
from amigo.fem import dot_product, Problem, Mesh, basis, build_grid
import amigo as am
import argparse


def potential(soln, data=None, geo=None):
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
mesh = Mesh("plate.inp")

integrand_map = {
    "air": {"target": "SURFACE1", "integrand": potential},
}

lines = ["LINE1", "LINE2", "LINE5", "LINE3", "LINE4", "LINE6", "LINE7", "LINE8"]
bc_map = {
    "bcs": {"type": "dirichlet", "target": lines, "input": ["u"]},
}

# Initialize the spaces (same for all domains)
soln_space = basis.SolutionSpace({"soln": {"u": "H1"}})
geo_space = basis.SolutionSpace({"geo": {("x", "y"): "H1"}})
data_space = basis.SolutionSpace({"data": {"Jz": "const"}})

# Create an amigo model for each mesh
problem = Problem(
    mesh, soln_space, geo_space, data_space, integrand_map=integrand_map, bc_map=bc_map
)

# Create the finite-element module
model = problem.create_model("poisson")

# Build the model
if args.build:
    model.build_module()

model.initialize()

# Set the problem data
data = model.get_data_vector()
data["data.Jz"] = 10.0

mat = model.create_matrix()
x = model.create_vector()
g = model.create_vector()

model.eval_hessian(x, mat)
model.eval_gradient(x, g)

ldl = am.SparseLDL(mat, am.SolverType.CHOLESKY)
ldl.factor()

x[:] = -g[:]
ldl.solve(x.get_vector())

# Plot the solution field
grid = build_grid(problem, "u", x)
grid.plot(scalars="u", cmap="coolwarm", show_edges=True)
