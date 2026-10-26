import numpy as np
from .model import Model
from .amigo import SparseLDL, SolverType
from .utils import tocsr


class SimpleSolver:
    def __init__(self, model: Model, solver_type=SolverType.LDL, ustab=1e-2):
        self.model = model

        self.x = self.model.create_vector()
        self.g = self.model.create_vector()
        self.mat = self.model.create_matrix()
        self.psi = self.model.create_vector()

        self.problem = self.model.get_problem()
        self.out_wrt_input = self.problem.create_output_jacobian_wrt_input()
        self.out_wrt_data = self.problem.create_output_jacobian_wrt_data()
        self.grad_wrt_data = self.problem.create_gradient_jacobian_wrt_data()

        self.ldl = SparseLDL(self.mat, solver_type=solver_type, ustab=ustab)

    def get_solution(self):
        return self.x

    def solve(self):
        """Solve the model as if it is unconstrained"""

        self.x[:] = 0.0
        self.model.eval_gradient(self.x, self.g)
        self.model.eval_hessian(self.x, self.mat)

        flag = self.ldl.factor()

        self.x[:] = -self.g[:]
        self.ldl.solve(self.x.get_vector())

        return

    def compute_derivatives(
        self,
        of: str | list[str] | None = None,
        wrt: str | list[str] | None = None,
        method="adjoint",
    ):
        """Compute the total derivatives"""

        if of is None:
            _, _, _, of = self.model.get_names()
        if wrt is None:
            _, _, wrt, _ = self.model.get_names()

        of_indices, of_map = self.model.get_indices_and_map(of)
        wrt_indices, wrt_map = self.model.get_indices_and_map(wrt)

        dfdx = np.zeros((len(of_indices), len(wrt_indices)))

        x = self.x.get_vector()
        self.problem.output_jacobian_wrt_input(x, self.out_wrt_input)
        out_wrt_input = tocsr(self.out_wrt_input)

        self.problem.output_jacobian_wrt_data(x, self.out_wrt_data)
        out_wrt_data = tocsr(self.out_wrt_data)

        self.problem.gradient_jacobian_wrt_data(x, self.grad_wrt_data)
        grad_wrt_data = tocsr(self.grad_wrt_data)

        self.model.eval_hessian(self.x, self.mat)
        flag = self.ldl.factor()

        if method == "adjoint":
            for i in range(len(of_indices)):
                idx = of_indices[i]
                self.psi[:] = -out_wrt_input[idx, :].toarray()
                self.model.zero_fixed_entries(self.psi)
                self.ldl.solve(self.psi.get_vector())
                adjx = grad_wrt_data.T @ self.psi[:]
                dfdx[i, :] = out_wrt_data[idx, wrt_indices] + adjx[wrt_indices]
        elif method == "direct":
            grad_wrt_data = grad_wrt_data.tocsc()
            for i in range(len(wrt_indices)):
                idx = wrt_indices[i]
                self.psi[:] = -grad_wrt_data[:, idx].toarray().flatten()
                self.model.zero_fixed_entries(self.psi)
                self.ldl.solve(self.psi.get_vector())
                dirx = out_wrt_input @ self.psi[:]
                dfdx[:, i] = out_wrt_data[of_indices, idx] + dirx[of_indices]

        return dfdx, of_map, wrt_map
