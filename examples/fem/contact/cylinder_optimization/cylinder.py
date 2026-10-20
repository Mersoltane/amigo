import argparse
import numpy as np
import amigo as am
from amigo.fem import Mesh, Problem, CellType, SolutionSpace
from amigo.fem.elements import MITCShell
from amigo.fem.visualization import build_grid


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


class IsoShell:
    def __init__(self, E=70e9, nu=0.3, ks=5.0 / 6.0, penalty=1e-4):
        self.E = E
        self.nu = nu
        self.ks = ks
        self.penalty = penalty

        return

    def constitutive(self, soln, geo=None, data=None):
        t = data["thickness"].value
        G = 0.5 * self.E / (1.0 + self.nu)
        D = self.E / (1.0 - self.nu**2)

        A11 = D * t
        A12 = self.nu * A11
        A22 = A11
        A66 = 0.5 * A11 * (1 - self.nu)

        D11 = D * t**3 / 12.0
        D12 = self.nu * D11
        D22 = D11
        D66 = 0.5 * D11 * (1 - self.nu)

        As44 = self.ks * G * t
        As55 = As44

        # Drill penalty
        kdrill = t * self.E * self.penalty

        return (A11, A12, A22, A66), (D11, D12, D22, D66), (As44, As55), kdrill


def traction(soln, geo=None, data=None):
    tx = 1e3
    return -tx * soln["u"].value


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

# Load the mesh
mesh = Mesh("cylinder.inp")

surfaces = ["SURFACE1"]
bottom_line = "LINE3"
lateral_line = "LINE2"
top_line = "LINE1"

# Create the shell element
E = 70e4
nu = 0.3

iso = IsoShell(E=E, nu=nu)
ctype = CellType.QUADRILATERAL
mitc_shell = MITCShell(constitutive=iso.constitutive)
soln_space, geo_space, data_space, output_space = mitc_shell.get_default_spaces()

# Override the default data space
data_space = SolutionSpace({"data": {"thickness": "H1"}})

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

# Create the element objects
element_objs = {
    ("shell", ctype): mitc_shell.create_element(
        name="Shell", data_space=data_space, ctype=ctype
    ),
    ("traction", ctype): mitc_shell.create_element(
        name="Traction", data_space=data_space, ctype=ctype, integrand=traction
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

model.set_meta("lower", "data.thickness", 0.0, vtype="data")
model.set_meta("upper", "data.thickness", am.inf, vtype="data")
model.set_meta("value", "data.thickness", 0.01, vtype="data")

output_names = ["volume", "strain_energy"]


def output_function(soln, data=None, geo=None):
    outputs = {}
    outputs["volume"] = data["thickness"].value
    outputs["strain_energy"] = mitc_shell.potential(soln, data=data, geo=geo)
    return outputs


output_obj = mitc_shell.create_element_output(
    name="Outputs",
    data_space=data_space,
    ctype=ctype,
    output_names=output_names,
    output_function=output_function,
)
output_map = {"shell": {"names": output_names, "target": surfaces}}
output_objs = {("shell", ctype): output_obj}

problem.add_integrated_output(model, output_map=output_map, output_objs=output_objs)

if args.build:
    model.build_module()

model.initialize()

# Set the shell normals
MITCShell.set_shell_normals(model)

print("Number of variables... ", model.num_variables)

data = model.get_data_vector()
data["data.thickness"] = 0.01
x = model.create_vector()

# Optimization for the nonlinear solve
opt_options = {
    "nlp_scaling": False,
    "max_iterations": 25,
    "convergence_tolerance": 1e-15,
    "max_line_search_iterations": 1,
    "initial_barrier_param": 0.1,
}

opt = am.Optimizer(model, x)
opt.optimize(opt_options)

out1 = model.create_output_vector()
model.compute_output(x, out1)

dh = 1e-7
dt = np.ones(shape=data["data.thickness"].shape)

dfdx, of_map, wrt_map = opt.compute_post_opt_derivatives(
    of=["outputs.volume", "outputs.strain_energy"], wrt=["data.thickness"]
)

dVdt = dfdx[of_map["outputs.volume"], wrt_map["data.thickness"]]
dUdt = dfdx[of_map["outputs.strain_energy"], wrt_map["data.thickness"]]

Vans = np.dot(dVdt, dt)
Uans = np.dot(dUdt, dt)

# Reset the data
data["data.thickness"] += dh * dt

# Re-optimize
# opt = am.Optimizer(model, x)
opt.optimize(opt_options)

# Now re-compute the output
out2 = model.create_output_vector()
model.compute_output(x, out2)

Vfd = ((out2["outputs.volume"] - out1["outputs.volume"]) / dh)[0]
Ufd = ((out2["outputs.strain_energy"] - out1["outputs.strain_energy"]) / dh)[0]

print(f"FD: {Vfd:15.8e} Ans: {Vans:15.8e}  Err: {(Vans - Vfd) / Vfd:15.8e}")
print(f"FD: {Ufd:15.8e} Ans: {Uans:15.8e}  Err: {(Uans - Ufd) / Ufd:15.8e}")

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

x["soln.u"] = dfdx[of_map["outputs.strain_energy"], wrt_map["data.thickness"]]

# Plot the mesh
grid = build_grid(problem, "u", x, surfaces)
grid.plot(scalars="u", cmap="coolwarm", show_edges=False)
