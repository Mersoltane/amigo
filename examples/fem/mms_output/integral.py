import numpy as np
import amigo as am
from amigo.fem import (
    dot_product,
    Problem,
    Mesh,
    basis,
    FunctionSpace,
    Space,
    Conformity,
)
import argparse


# Method of Manufactured solutions
def output(soln, data=None, geo=None):
    u = soln["u"].value
    return {"integrand": u}


def potential(soln, data=None, geo=None):
    u = soln["u"].value
    ugrad = soln["u"].grad
    x = geo["x"].value
    y = geo["y"].value

    f = -2 * np.pi**2 * am.sin(np.pi * x) * am.sin(np.pi * y)

    wf = 0.5 * dot_product(ugrad, ugrad, n=2) - f * u
    return wf


def exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

# Define mesh objects
meshes = {"Mesh0": Mesh("plate.inp")}

integrand_map = {
    "Mesh0": {
        "air": {
            "target": ["SURFACE1"],
            "integrand": potential,
        },
    }
}

bc_map_mesh0 = {
    "DirichletBC": {
        "type": "dirichlet",
        "target": [
            "LINE1",
            "LINE2",
            "LINE3",
            "LINE4",
            "LINE5",
            "LINE6",
            "LINE7",
            "LINE8",
        ],
        "input": ["u"],
    }
}
bc_map = {"Mesh0": bc_map_mesh0}

output_map = {
    "Mesh0": {
        "integral": {
            "names": ["integrand"],
            "target": ["SURFACE1"],
            "function": output,
        },
    }
}

# Initialize the spaces (same for all domains)
const = FunctionSpace(func_space=Space.CONST, conformity=Conformity.COMPONENT)
soln_space = basis.SolutionSpace({"soln": {"u": "H1"}})
data_space = basis.SolutionSpace({"data": {"Jz": const}})
geo_space = basis.SolutionSpace({"geo": {("x", "y"): "H1"}})

# Define the global amigo model
model = am.Model("mms_module")

# Create an amigo model for each mesh
for mesh_name, mesh in meshes.items():
    problem = Problem(
        mesh,
        soln_space,
        geo_space,
        data_space,
        integrand_map=integrand_map[mesh_name],
        bc_map=bc_map[mesh_name],
    )
    sub_model = problem.create_model(mesh_name)
    problem.add_integrated_output(sub_model, output_map=output_map[mesh_name])
    model.add_model(mesh_name, sub_model)

# Build the model
if args.build:
    model.build_module()
model.initialize()

# Set the problem data
data = model.get_data_vector()
data["Mesh0.data.Jz.SURFACE1"] = 10.0

mat = model.create_matrix()
alpha = 1.0
x = model.create_vector()
ans = model.create_vector()
g = model.create_vector()
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

ldl = am.SparseLDL(mat, am.SolverType.CHOLESKY)
flag = ldl.factor()

ans[:] = g[:]
ldl.solve(ans.get_vector())

# Get the FEM solution
u = ans["Mesh0.soln.u"]

# Compute the exact solution field
data = model.get_data_vector()
u_exact = exact(data["Mesh0.geo.x"], data["Mesh0.geo.y"])

# Get the output
output = model.create_output_vector()
model.compute_output(ans, output)
exact_integral = 4 / np.pi**2
amigo_integral = output["Mesh0.outputs.integrand[0]"]
rel_err = np.abs(exact_integral - amigo_integral) / exact_integral
norm = np.linalg.norm(u_exact - u)
print(f"Amigo Integral: {exact_integral:.4e}")
print(f"Analytic Integral: {amigo_integral:.4e}")
print(f"Integral Rel. Err.: {rel_err:.4e}")
print(f"||u_exact - u||: {norm:.4e}")

# Plot the solution field
# fig, ax = plt.subplots(ncols=2)
# mesh.plot(u, ax=ax[0])
# mesh.plot(u_exact, ax=ax[1])
# plt.show()
