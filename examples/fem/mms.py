import numpy as np
from fem import (
    Mesh,
    Problem,
)
import basis
import amigo as am
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

"""
Method of Manufactured Solutions
    u = sin(2πx)sin(2πy)
    ∇·∇u - f = 0
    f=(8π**2)sin(2xπ)sin(2yπ)
"""


def output(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    pi = np.pi
    integrand = am.sin(2 * pi * x) * am.cos(2 * pi * y)
    # return {"integrand": integrand}
    return integrand


def weakform(soln, data=None, geo=None):
    u = soln["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    pi = np.pi
    f = 8 * pi**2 * am.sin(2 * x * pi) * am.sin(2 * y * pi)

    wf = 0.5 * basis.dot_product(ugrad, ugrad, n=2) + f * uvalue
    return wf


def exact(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


# Loop through each mesh refinemens and store the 2 norm
n = 20

norms = np.zeros(n)
number_elems = np.zeros(n)
# lc_vals = np.linspace(1e-1, 4e-3, n)
lc_vals = [8e-3]

recombine = False
order = 1
if order == 1:
    elem_type = "CPS3"
elif order == 2:
    elem_type = "CPS6"

for i, lc in enumerate(lc_vals):
    meshes = {"Mesh": Mesh("plate.inp")}
    dirichlet_bc_meshes = {
        "Mesh": {
            "DirichletLine1": {
                "type": "dirichlet",
                "target": "LINE1",
                "input": ["u"],
                "start": True,
                "end": True,
            },
            "DirichletLine2": {
                "type": "dirichlet",
                "target": "LINE2",
                "input": ["u"],
                "start": False,
                "end": False,
            },
            "DirichletLine3": {
                "type": "dirichlet",
                "target": "LINE3",
                "input": ["u"],
                "start": True,
                "end": True,
            },
            "DirichletLine4": {
                "type": "dirichlet",
                "target": "LINE4",
                "input": ["u"],
                "start": False,
                "end": False,
            },
        },
    }

    # Symmetric BCs mapping for each mesh
    symm_bc_meshes = {"Mesh": {}}

    # Weak form mapping for each mesh
    weakform_map = {
        "Mesh": {
            "name": "mms_wf",
            "SURFACE1": weakform,
        },
    }
    output_map = {
        "Mesh": {
            "integrand": {
                "target": ["SURFACE1"],
                "output_func": output,
            },
            "torque": {
                "target": ["SURFACE1"],
                "function": output,
            },
        }
    }

    # Initialize the spaces (same for all domains)
    soln_space = basis.SolutionSpace({"u": "H1"})
    data_space = basis.SolutionSpace({})
    geo_space = basis.SolutionSpace({"x": "H1", "y": "H1"})

    # Define the mesh object
    mesh = meshes["Mesh"]

    nelems = mesh.get_num_elements("SURFACE1", elem_type)
    number_elems[i] = nelems

    # Define the global amigo model
    global_model = am.Model("global_model")

    # Create an amigo model for each mesh
    for mesh_name, mesh in meshes.items():
        problem = Problem(
            mesh,
            soln_space,
            weakform_map[mesh_name],
            output_map[mesh_name],
            data_space=data_space,
            geo_space=geo_space,
            dirichlet_bc_map=dirichlet_bc_meshes[mesh_name],
            sym_bc_map=symm_bc_meshes[mesh_name],
            ndim=2,
        )
        model = problem.create_model(mesh_name)
        global_model.add_model(mesh_name, model)

    # Build the model
    global_model.build_module()
    global_model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
    p = global_model.get_problem()

    # Set the problem data
    data = global_model.get_data_vector()
    data["Mesh.src_geo.x"] = mesh.X[:, 0]
    data["Mesh.src_geo.y"] = mesh.X[:, 1]

    mat = p.create_matrix()
    alpha = 1.0
    x = p.create_vector()
    ans = p.create_vector()
    g = p.create_vector()
    rhs = p.create_vector()
    p.hessian(alpha, x, mat)
    p.gradient(alpha, x, g)
    csr_mat = am.tocsr(mat)

    ans.get_array()[:] = spsolve(csr_mat, g.get_array())
    ans_local = ans
    u = ans_local.get_array()[global_model.get_indices("Mesh.src_soln.u")]
    u_exact = exact(x=mesh.X[:, 0], y=mesh.X[:, 1])

    mesh.plot(u)

    delta_u = u - u_exact
    frobenius_norm = np.linalg.norm(delta_u)
    print(f"\n||err||2 = {frobenius_norm:4e}")
    print(f"nelems: {nelems}")
    norms[i] = frobenius_norm

    plt.show()

# np.save(f"norms_" + elem_type, norms)
# np.save(f"number_elems_" + elem_type, number_elems)
