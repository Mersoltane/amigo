import argparse
import amigo as am
from amigo.fem import Mesh, Problem, CellType
from amigo.fem.elements import MITCShell
from amigo.fem.visualization import build_grid

parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
parser.add_argument(
    "--solver",
    dest="solver",
    choices=["cholesky", "cholesky_left", "ldl", "scipy", "cuda"],
    default="cholesky",
)
args = parser.parse_args()


def traction(soln, geo=None, data=None):
    w = soln["w"].value
    fz = 1e3  # 1 kN / m^2

    return -w * fz


# Load the mesh
filename = "CRM_box_2nd.bdf"
mesh = Mesh(filename)

# Get the domain names
domains = mesh.get_domain_names()

integrand_map = {"shell": {"target": domains}, "traction": {"target": domains}}

# Create the shell element
ctype = CellType.QUADRILATERAL
mitc_shell = MITCShell()
soln_space, geo_space, data_space, output_space = mitc_shell.get_default_spaces()

# Create the element objects
element_objs = {
    ("shell", ctype): mitc_shell.create_element(name="Shell", ctype=ctype),
    ("traction", ctype): mitc_shell.create_element(
        name="Traction", ctype=ctype, integrand=traction
    ),
}

problem = Problem(
    mesh, soln_space, geo_space, integrand_map=integrand_map, element_objs=element_objs
)

model = problem.create_model("crm_model")

# Add the shell normals to the output
MITCShell.add_shell_normals(problem, model, domains)

if args.build:
    model.build_module()

model.initialize()

# Set the shell normals
MITCShell.set_shell_normals(model)

x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

model.eval_gradient(x, g)
model.eval_hessian(x, mat)

ldl = am.SparseLDL(mat, am.SolverType.CHOLESKY, ustab=0.05)
flag = ldl.factor()

x[:] = -g[:]
ldl.solve(x.get_vector())

grid = build_grid(problem, "w", x)
grid.plot(scalars="w", cmap="coolwarm", show_edges=True)
