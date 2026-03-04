import amigo as am
import numpy as np
import re
import basis
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from connectivity import InpParser
from matplotlib.collections import PolyCollection
from scipy.sparse.linalg import spsolve


class DofSource(am.Component):
    def __init__(self, input_names=[], geo_names=[], data_names=[], con_names=[]):
        super().__init__()

        # Geo and data added as data to the component
        for name in data_names:
            self.add_data(name)
        for name in geo_names:
            self.add_data(name)

        # Add inputs and constraints
        for name in input_names:
            self.add_input(name)
        for name in con_names:
            self.add_constraint(name)

        return


class DegreesOfFreedom:
    def __init__(self, mesh, space, kind="input"):
        """
        Allocate the degrees of freedom on the mesh
        """

        self.mesh = mesh
        self.space = space
        self.kind = kind

        return

    def _initialize(self):
        """
        Initialize the degrees of freedom associated with this mesh
        """

        # Allocate degrees of freedom for each of the nodes/

        return

    def add_source(self, model):
        # Get the names of things associated with H1
        names = self.space.get_names("H1")  # u, v

        # Create amigo source component with input names and geo names
        input_names = []
        geo_names = []
        data_names = []
        if self.kind == "input":
            input_names = names
        elif self.kind == "data":
            data_names = names
        elif self.kind == "geo":
            geo_names = names

        dof_src = DofSource(
            input_names=input_names, geo_names=geo_names, data_names=data_names
        )

        # Add global mesh source component
        nnodes = mesh.get_num_nodes()
        model.add_component(f"src_{self.kind}", nnodes, dof_src)

    def link_dof(self, model, domain, etype, elem_name):
        names = self.space.get_names("H1")
        conn = self.mesh.get_conn(domain, etype)
        for name in names:
            model.link(
                f"src_{self.kind}.{name}", f"{elem_name}.{name}", src_indices=conn
            )
        return

    def get_basis(self, etype):
        basis_list = []

        for sp in ["H1"]:
            # self._get_basis(etype, sp, names, kind) # names and kind come from space info
            names = self.space.get_names(sp)
            basis_list.append(self._get_basis(etype, sp, names, self.kind))
        return basis.BasisCollection(basis_list)

    def _get_basis(self, etype, space, names=[], kind="input"):
        if etype == "CPS3":
            if space == "H1":
                return basis.TriangleLagrangeBasis(1, names, kind=kind)
        elif etype == "CPS4":
            if space == "H1":
                return basis.QuadLagrangeBasis(1, names, kind=kind)
        elif etype == "CPS6":
            if space == "H1":
                return basis.TriangleLagrangeBasis(2, names, kind=kind)
        elif etype == "M3D9":
            if space == "H1":
                return basis.QuadLagrangeBasis(2, names, kind=kind)

        raise NotImplementedError(
            f"Basis for element {etype} with space {space} not implemented"
        )

    def get_quadrature(self, etype):
        if etype == "CPS3":
            return basis.TriangleQuadrature(2)
        elif etype == "CPS4":
            return basis.QuadQuadrature(2)
        elif etype == "CPS6":
            return basis.TriangleQuadrature(4)
        elif etype == "M3D9":
            return basis.QuadQuadrature(3)

        raise NotImplementedError(f"Quadrature for element {etype} not implemented")


class Mesh:
    def __init__(self, filename):
        self.parser = InpParser()
        self.parser.parse_inp(filename)

        self.X = self.parser.get_nodes()

    def get_num_nodes(self):
        return self.X.shape[0]

    def get_domains(self):
        domains = self.parser.get_domains()

        element_types = ["CPS3", "CPS4", "CPS6", "M3D9"]

        volumes = {}
        for name in domains:
            for etype in element_types:
                if etype in domains[name]:
                    volumes[name] = domains[name]
                    break

        return volumes

    def get_conn(self, name, etype):
        return self.parser.get_conn(name, etype)

    def get_num_elements(self, name, etype):
        return self.parser.get_conn(name, etype).shape[0]

    def plot(self, u, ax=None, nlevels=30, cmap="coolwarm", title=None):
        min_level = np.min(u)
        max_level = np.max(u)
        levels = np.linspace(min_level, max_level, nlevels)

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

        volumes = self.get_domains()
        x = self.X[:, 0]
        y = self.X[:, 1]

        for name in volumes:
            for etype in volumes[name]:
                # Get the connectivity
                conn = self.convert_conn(etype, self.get_conn(name, etype))
                tri = mtri.Triangulation(x, y, conn)

                # Set the contour plot
                ax.tricontourf(tri, u, levels=levels, cmap=cmap)
                ax.tricontour(
                    tri, u, levels=levels, colors="k", linewidths=0.3, alpha=0.5
                )

                # Overlay the mesh skeleton
                # gmsh_conn = self.get_conn(name, etype)
                # X2d = self.X[:, 0:2]
                # polygons = [X2d[row] for row in gmsh_conn]
                # mesh = PolyCollection(
                #     polygons,
                #     facecolor="none",
                #     edgecolor="black",
                #     linewidth=0.5,
                #     alpha=0.4,
                # )
                # ax.add_collection(mesh)

                if title is not None:
                    ax.set_title(title)

                ax.set_aspect("equal")
        return ax

    def convert_conn(self, etype, conn):
        if etype == "CPS3":
            return conn
        elif etype == "CPS4":
            c = [[0, 1, 2], [0, 2, 3]]
        elif etype == "CPS6":
            # 2
            # |  .
            # 5     4
            # |        .
            # 0 --- 3 --- 1
            c = [[0, 3, 5], [3, 4, 5], [3, 1, 4], [5, 4, 2]]
        elif etype == "M3D9":
            # 3 --- 6 --- 2
            # |           |
            # 7     8     5
            # |           |
            # 0 --- 4 --- 1
            c = [
                [0, 4, 7],
                [4, 8, 7],
                [4, 1, 8],
                [1, 5, 8],
                [7, 8, 3],
                [8, 6, 3],
                [8, 5, 6],
                [5, 2, 6],
            ]

        cs = []
        for c0 in c:
            cs.append(conn[:, c0])

        return np.vstack(cs)


class Problem:
    # soln_space = object
    def __init__(self, mesh, soln_space, weakform, data_space=[], geo_space=[], ndim=2):

        self.mesh = mesh
        self.ndim = ndim  # Dimension of the problem

        self.soln_space = soln_space
        self.data_space = data_space
        self.geo_space = geo_space

        self.weakform = weakform

        # Initialize Dof's
        # Take in the soln space -> removes "H1" input
        self.soln_dof = DegreesOfFreedom(mesh, self.soln_space, "soln")
        self.geo_dof = DegreesOfFreedom(mesh, self.geo_space, "geo")
        self.data_dof = DegreesOfFreedom(mesh, self.data_space, "data")

        return

    def create_model(self, module_name: str):
        """Create and link the Amigo model"""
        model = am.Model(module_name)

        self.soln_dof.add_source(model)
        self.data_dof.add_source(model)
        self.geo_dof.add_source(model)

        # Build the elements for all domains
        domains = self.mesh.get_domains()
        for domain in domains:
            for etype in domains[domain]:
                # Each element type has a dictionary of solution basis's
                soln_basis = {}

                # Build a finite-element for each weak form
                elem_name = f"Element{etype}_{domain}"

                soln_basis = self.soln_dof.get_basis(etype)
                data_basis = self.data_dof.get_basis(etype)
                geo_basis = self.geo_dof.get_basis(etype)

                # Create the quadrature instance
                quadrature = self.soln_dof.get_quadrature(etype)

                # Create the element object
                elem = FiniteElement(
                    elem_name,
                    soln_basis,
                    data_basis,
                    geo_basis,
                    quadrature,
                    self.weakform,
                )

                # Add the element/component
                nelems = self.mesh.get_num_elements(domain, etype)
                model.add_component(elem_name, nelems, elem)

                # Link all the element dof to the component
                self.soln_dof.link_dof(model, domain, etype, elem_name)
                self.data_dof.link_dof(model, domain, etype, elem_name)
                self.geo_dof.link_dof(model, domain, etype, elem_name)

        return model


class FiniteElement(am.Component):
    def __init__(
        self,
        name,
        soln_basis,
        data_basis,
        geo_basis,
        quadrature,
        weakform,
    ):
        super().__init__(name=name)

        # NOTE: soln_basis is a dict of objectes for each input
        self.soln_basis = soln_basis

        self.data_basis = data_basis
        self.geo_basis = geo_basis
        self.quadrature = quadrature
        self.weakform = weakform

        # From BasisCollection
        self.soln_basis.add_declarations(self)
        self.geo_basis.add_declarations(self)
        self.data_basis.add_declarations(self)

        # Set the arguments to the compute function for each quadrature point
        self.set_args(self.quadrature.get_args())

        # Add the objective to minimize
        self.add_objective("obj")

        return

    def compute(self, **args):

        quad_weight, quad_point = self.quadrature.get_point(**args)

        # Evaluate the solution fields/data fields (u)
        soln_xi = self.soln_basis.eval(self, quad_point)
        data_xi = self.data_basis.eval(self, quad_point)
        geo = self.geo_basis.eval(self, quad_point)

        # Perform the mapping from computational to physical coordinates (u)
        detJ, Jinv = self.geo_basis.compute_transform(geo)
        soln_phys = self.soln_basis.transform(detJ, Jinv, soln_xi)
        data_phys = self.data_basis.transform(detJ, Jinv, data_xi)

        # Add the contributions directly to the Lagrangian
        self.objective["obj"] = quad_weight * detJ * self.weakform(soln_phys, geo=geo)
        return


def weakform(soln, data=None, geo=None):
    soln_u = soln["u"]
    soln_v = soln["v"]

    u = soln_u["u"]
    uvalue = u["value"]
    ugrad = u["grad"]

    v = soln_v["v"]
    vvalue = v["value"]
    vgrad = v["grad"]

    x = geo["x"]["value"]
    y = geo["y"]["value"]

    # f = am.sin(x) ** 2 * am.cos(y) ** 2
    # comp1 = 0.5 * (uvalue**2 + basis.dot_product(ugrad, vgrad, n=2) - 2.0 * uvalue * f)

    alpha = 1.0
    pi = 3.14159265358979

    f1 = 2 * pi**2 * am.sin(pi * x) * am.sin(pi * y) + alpha * am.cos(
        pi * x
    ) * am.cos(pi * y)
    f2 = 2 * pi**2 * am.cos(pi * x) * am.cos(pi * y) + alpha * am.sin(
        pi * x
    ) * am.sin(pi * y)

    comp1 = (
        basis.dot_product(ugrad, vgrad, n=2)
        + alpha * vvalue * uvalue
        + f1 * uvalue
        + basis.dot_product(vgrad, ugrad, n=2)
        + alpha * uvalue * vvalue
        + f2 * vvalue
    )
    return comp1


# Initialize the spaces
soln_space = basis.SolutionSpace({"u": "H1", "v": "H1"})
data_space = basis.SolutionSpace({"x": "H1", "y": "H1"})
geo_space = basis.SolutionSpace({"x": "H1", "y": "H1"})

# mesh = Mesh("magnet_order_1.inp")
mesh = Mesh("plate.inp")
problem = Problem(
    mesh,
    soln_space,
    weakform,
    data_space=data_space,
    geo_space=geo_space,
    ndim=2,
)

model = problem.create_model("test")

model.build_module()
model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

# print("num_variables = ", model.num_variables)
problem = model.get_problem()

# Set the problem data
data = model.get_data_vector()
data["src.x"] = mesh.X[:, 0]
data["src.y"] = mesh.X[:, 1]

# x = problem.create_vector()
# mat = problem.create_matrix()
# rhs = model.create_vector()
# problem.gradient(1.0, x, rhs.get_vector())
# problem.hessian(1.0, x, mat)

# chol = am.SparseCholesky(mat)
# flag = chol.factor()
# print("flag = ", flag)
# chol.solve(rhs.get_vector())

mat = problem.create_matrix()
alpha = 1.0
x = problem.create_vector()
ans = problem.create_vector()
g = problem.create_vector()
rhs = problem.create_vector()
problem.hessian(alpha, x, mat)
problem.gradient(alpha, x, g)
csr_mat = am.tocsr(mat)

ans.get_array()[:] = spsolve(csr_mat, g.get_array())
ans_local = ans
u = ans_local.get_array()[model.get_indices("src.u")]
v = ans_local.get_array()[model.get_indices("src.v")]
print(u)
print(v)

fig, ax = plt.subplots(ncols=2)
mesh.plot(u, ax=ax[0], title="u")
mesh.plot(v, ax=ax[1], title="v")
plt.show()
