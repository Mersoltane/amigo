from amigo.fem import (
    Mesh,
    Problem,
    CellType,
    Space,
    FunctionSpace,
    SolutionSpace,
    build_grid,
)
import amigo as am
import matplotlib.pyplot as plt
import numpy as np
import argparse


def potential_plane_stress(soln, data=None, geo=None):
    """Strain energy density (integrand of TPE equation)"""

    t = data["t"].value

    # Displacement gradients in physical space
    u_grad = soln["u"].grad
    v_grad = soln["v"].grad

    # Strain components
    exx = u_grad[0]
    eyy = v_grad[1]
    exy = u_grad[1] + v_grad[0]

    # Material properties (hardcoded or pull from data)
    E = 1.0 * t
    nu = 0.3
    c = E / (1.0 - nu**2)  # poisson's ratio

    # Constitutive matrix C acting on [e11, e22, e12]
    # W = 0.5 * eT C e
    W = 0.5 * c * (exx**2 + eyy**2 + 2.0 * nu * exx * eyy + 0.5 * (1.0 - nu) * exy**2)

    return W


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


def output_function(soln, data=None, geo=None):
    outputs = {}
    outputs["volume"] = data["t"].value
    outputs["strain_energy"] = potential_plane_stress(soln, data=data, geo=geo)
    return outputs


# Two displacement DOFs per node
H1 = FunctionSpace(func_space=Space.H1, degree=1)
soln_space = SolutionSpace({"soln": {("u", "v"): H1}})
geo_space = SolutionSpace({"geo": {("x", "y"): H1}})
data_space = SolutionSpace({"data": {"t": H1}})

integrand_map = {
    "plane_stress": {
        "target": ["SURFACE1"],
        "integrand": potential_plane_stress,
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
args = parser.parse_args()

mesh = Mesh("plate.inp")
problem = Problem(
    mesh,
    soln_space,
    geo_space,
    data_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
)

model = problem.create_model("plane_stress")

output_names = ["strain_energy", "volume"]
output_map = {
    "shell": {
        "names": output_names,
        "target": ["SURFACE1"],
        "function": output_function,
    }
}
problem.add_integrated_output(model, output_map=output_map)

if args.build:
    model.build_module()

model.initialize()

# Set the thickness values
data = model.get_data_vector()
data["data.t"] = 1.0

solver = am.SimpleSolver(model)
solver.solve()

x = solver.get_solution()

# Compute the output
out1 = model.create_output_vector()
model.compute_output(x, out1)

# Compute the derivative
dfdx, of_map, wrt_map = solver.compute_derivatives(
    of=["outputs.volume", "outputs.strain_energy"], wrt=["data.t"]
)

# Extract the derivatives
dVdt = dfdx[of_map["outputs.volume"], wrt_map["data.t"]]
dUdt = dfdx[of_map["outputs.strain_energy"], wrt_map["data.t"]]

# Now prepare to perturb the solution
dh = 5e-7
shape = data["data.t"].shape
dt = np.ones(shape=shape)  #  np.random.uniform(size=np.prod(shape)).reshape(shape)

# Compute the result using
Vans = np.dot(dVdt, dt)
Uans = np.dot(dUdt, dt)

# Reset the data
data["data.t"] += dh * dt

# Re-solve
solver.solve()

# Now re-compute the output
out2 = model.create_output_vector()
model.compute_output(x, out2)

Vfd = ((out2["outputs.volume"] - out1["outputs.volume"]) / dh)[0]
Ufd = ((out2["outputs.strain_energy"] - out1["outputs.strain_energy"]) / dh)[0]

print(f"FD: {Vfd:15.8e} Ans: {Vans:15.8e}  Err: {(Vans - Vfd) / Vfd:15.8e}")
print(f"FD: {Ufd:15.8e} Ans: {Uans:15.8e}  Err: {(Uans - Ufd) / Ufd:15.8e}")

# Show the solution
u = x["soln.u"]
v = x["soln.v"]

# Plot the solution
grid = build_grid(problem, "u", x)
grid.plot(scalars="u", cmap="coolwarm", show_edges=True)
