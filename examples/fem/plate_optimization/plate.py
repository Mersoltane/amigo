import argparse
import amigo as am
from amigo.fem.elements import MITCPlate
from amigo.fem import Mesh, Problem, CellType, IntegrandFormulation, build_grid
from amigo.fem import Space, FunctionSpace, SolutionSpace, Conformity

# Minimize mass subject to a stress constraint
# Objective is proportional to t


def traction(test, soln, data=None, geo=None):
    """Strain energy density (integrand of TPE equation)"""
    w = test["w"].value
    fz = 1e3
    return -fz * w


def constitutive(soln, data=None, geo=None):
    t = soln["t"]

    E = 70e9
    nu = 0.3
    ks = 5.0 / 6.0

    G = 0.5 * E / (1.0 + nu)
    D = E / (1.0 - nu**2)

    D11 = D * t**3 / 12.0
    D12 = nu * D11
    D22 = D11
    D66 = 0.5 * D11 * (1 - nu)

    As44 = ks * G * t
    As55 = As44

    return (D11, D12, D22, D66), (As44, As55)


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

formulation = IntegrandFormulation.WEAK
ctype = CellType.QUADRILATERAL
mitc_plate = MITCPlate()
soln_space, geo_space, data_space, output_space = mitc_plate.get_default_spaces()

H1 = FunctionSpace(func_space=Space.H1, degree=1, conformity=Conformity.GLOBAL)
soln_space = SolutionSpace({"soln": {("t", "w", "rx", "ry"): H1}})

# Set the element objects
element_objs = {
    ("plate", ctype): mitc_plate.create_element(
        name="Plate", ctype=ctype, formulation=formulation
    ),
    ("traction", ctype): mitc_plate.create_element(
        name="Traction", ctype=ctype, integrand=traction, formulation=formulation
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
    integrand_formulation=formulation,
)
model = problem.create_model("plate")

if args.build:
    model.build_module()

model.initialize()

print("Number of variables... ", model.num_variables)

x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

model.eval_gradient(x, g)
model.eval_hessian(x, mat)

ldl = am.SparseLDL(mat, am.SolverType.LDL, ustab=0.001)
flag = ldl.factor()

x[:] = -g[:]
ldl.solve(x.get_vector())

print("Plotting...")
grid = build_grid(problem, "w", x)
grid.plot(scalars="w", cmap="coolwarm", show_edges=True)
