import amigo as am
import numpy as np  # used for plotting/analysis
import argparse
from amigo.fem import (
    Mesh,
    Problem,
    SolutionSpace,
    build_grid,
)


def potential_plane_stress(soln, data=None, geo=None):
    """Strain energy density"""
    # Displacement gradients in physical space
    ux, uy = soln["u"].grad
    vx, vy = soln["v"].grad

    # Strain components
    exx = ux
    eyy = vy
    exy = vx + uy

    # Set up the material penalization
    p = data["p"].value
    rho = soln["rho"].value
    rho0 = data["rho0"].value

    # Material properties (hardcoded or pull from data)
    E0 = 1e-3
    E1 = 1.0
    nu = 0.3

    # Compute the penalty
    const = rho0**p
    factor = p * rho0 ** (p - 1)
    E = E0 + (E1 - E0) * (const + factor * (rho - rho0))
    c = E / (1.0 - nu**2)  # poisson's ratio

    # Constitutive matrix C acting on [e11, e22, e12]
    # W = 0.5 * eT C e
    W = 0.5 * c * (exx**2 + eyy**2 + 2.0 * nu * exx * eyy + 0.5 * (1.0 - nu) * exy**2)

    return W


def potential_helmholz_filter(soln, data=None, geo=None):
    """Potential filter"""
    rho = soln["rho"].value
    rhox, rhoy = soln["rho"].grad

    r = 0.1

    return 0.5 * (rho * rho + r**2 * (rhox**2 + rhoy**2)) - 1.0


def potential_traction(soln, data=None, geo=None):
    """External Work Line integral integrand"""
    u = soln["u"].value
    v = soln["v"].value

    # traction force for element
    # W = uT t
    tx = 0
    ty = -1

    W = u * tx + v * ty
    return -W


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Two displacement DOFs per node
soln_space = SolutionSpace({"soln": {("u", "v", "rho"): "H1"}})
geo_space = SolutionSpace({"geo": {("x", "y"): "H1"}})
data_space = SolutionSpace({"data": {"rho0": "H1"}, "penalty": {"p": "const"}})

integrand_map = {
    "plane_stress": {
        "target": ["SURFACE1"],
        "integrand": potential_plane_stress,
    },
    "plane_stress": {
        "target": ["SURFACE1"],
        "integrand": potential_helmholz_filter,
    },
    "traction": {
        "target": ["LINE2"],
        "integrand": potential_traction,
    },
}

bc_map = {
    "clamp_x": {
        "type": "dirichlet",
        "target": ["LINE4"],  # left edge — fix ux
        "input": ["u", "v"],
    },
}

mesh = Mesh("plate.inp")
problem = Problem(
    mesh,
    soln_space,
    geo_space,
    data_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
)

model = problem.create_model("compliance")

if args.build:
    model.build_module()

# Set the lower/upper values

model.initialize()

input, cons, data, output = model.get_names()

for name in data:
    print(name)

data = model.get_data_vector()
data["penalty.p"] = 1.0
data["data.rho0"] = 1.0

# Create the vectors and matrices for the model
x = model.create_vector()

x["soln.rho"] = 1.0

g = model.create_vector()
mat = model.create_matrix()

print("Evaluating the Hessian...")
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

# Solve the equations
print("Solving...")
chol = am.SparseLDL(mat, solver_type=am.SolverType.CHOLESKY)
flag = chol.factor()

# Solve the equations
x[:] = -g[:]
chol.solve(x.get_vector())

u = x["soln.u"]
v = x["soln.v"]

# Plot the solution
grid = build_grid(problem, "u", x)
grid.plot(scalars="u", cmap="coolwarm", show_edges=True)

#     # Update the data and the design variable bounds
#     pval = 3.0
#     data["topo.p"] = pval
#     data["topo.rho0"] = x["topo.rho"]
#     lower["src.x"] = (1.0 - 1.0 / pval) * x["src.x"]
#     lower["src.rho"] = (1.0 - 1.0 / pval) * x["src.rho"]
