import amigo as am
from amigo.fem import Mesh, Problem, SolutionSpace
import argparse


def potential(soln, data=None, geo=None):
    u = soln["u"].vec
    div = soln["u"].div
    alpha = 0.05

    x = geo["x"].value
    y = geo["y"].value

    f = 1.0 - x**2 - y**2
    return 0.5 * (alpha * (u[0] ** 2 + u[1] ** 2) + div**2 - 2 * f * div)


soln_space = SolutionSpace({"soln": {"u": "H(div)"}})
geo_space = SolutionSpace({"geo": {("x", "y"): "H1"}})
data_space = SolutionSpace({})

integrand_map = {
    "hdiv": {
        "target": ["SURFACE1"],
        "integrand": potential,
    },
}

bc_map = {
    "clamp_x": {
        "type": "dirichlet",
        "target": ["LINE1", "LINE2", "LINE3", "LINE4"],
        "input": ["u"],
    },
}


mesh = Mesh("mesh.inp")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)

args = parser.parse_args()

problem = Problem(
    mesh,
    soln_space,
    geo_space,
    data_space,
    bc_map=bc_map,
    integrand_map=integrand_map,
)

model = problem.create_model("hdiv")

if args.build:
    model.build_module()

model.initialize()

# Create the vectors and matrices for the model
x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

print("Evaluating the Hessian...")
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

# Solve the equations
print("Solving...")
chol = am.SparseLDL(mat, ustab=0.1, solver_type=am.SolverType.LDL)
flag = chol.factor()

# Solve the equations
x[:] = -g[:]
chol.solve(x.get_vector())
