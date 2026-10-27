"""Full-KKT GPU direct solver using the cuDSS symmetric-indefinite LDL^T.

The augmented KKT system is factored on device with no condensation.
Static pivoting rests on quasi-definite regularization and a residual check.
The solver runs on a single GPU and in core.
"""

from . import LinearSolver


class DirectCudaSolver(LinearSolver):
    # Subclasses swap the compiled factorization class and the build option it needs
    _factor_class = "CSRMatFactorCuda"
    _requires = "CUDA"

    def __init__(self, options, state):
        try:
            from amigo import amigo as ext

            factor_class = getattr(ext, self._factor_class)
            Vector = ext.Vector
        except Exception:
            raise NotImplementedError(
                f"Amigo compiled without {self._requires} support"
            )

        get = options.get if hasattr(options, "get") else lambda k, d: d
        self.pivot_eps = float(get("cuda_pivot_eps", 1e-8))
        self.ir_steps = int(get("cuda_ir_steps", 2))
        self.check_residual = bool(get("cuda_check_residual", True))
        self.residual_rtol = float(get("cuda_residual_rtol", 1e-4))

        self.mat_copy = state.hessian.duplicate()
        self.solver = factor_class(self.mat_copy, self.pivot_eps)
        self.solver.set_ir_steps(self.ir_steps)

        self._Vector = Vector
        self._tmp = None
        self._eps_eff = self.pivot_eps
        self.last_rel_residual = 0.0
        self.num_residual_violations = 0
        self.last_perturbed_pivots = 0
        self.num_perturbed_factorizations = 0

        self._report_backend()

        if not get("perturb_always_cd", False):
            print(
                f"  {self._name}: set perturb_always_cd=True, static "
                "pivoting is only reliable on a quasi-definite KKT"
            )

    @property
    def _name(self):
        """Class name, so subclasses label their own output"""
        return type(self).__name__

    def _report_backend(self):
        """Hook for subclasses to report on the constructed factorization"""
        return

    def factor(self, hessian, diagonal):
        self.mat_copy.copy(hessian)
        self.mat_copy.add_diagonal(diagonal)
        self.solver.factor()

        # A perturbed pivot means the inertia and step belong to another matrix
        self.last_perturbed_pivots = self.solver.num_perturbed_pivots()
        if self.last_perturbed_pivots > 0:
            self.num_perturbed_factorizations += 1
            n = self.num_perturbed_factorizations
            if n <= 3 or n % 25 == 0:
                print(
                    f"  {self._name}: {self.last_perturbed_pivots} "
                    f"statically perturbed pivots (factorization {n})"
                )

    def solve(self, bx, px):
        self.solver.solve(bx, px)

        if self.check_residual:
            # r = K p - b on device, against the matrix as factorized
            if self._tmp is None:
                self._tmp = self._Vector(len(bx.get_array()))
            self.solver.residual(bx, px, self._tmp)
            bnorm = bx.dot(bx) ** 0.5
            rnorm = self._tmp.dot(self._tmp) ** 0.5
            self.last_rel_residual = rnorm / max(bnorm, 1e-300)
            if self.last_rel_residual > self.residual_rtol:
                self.num_residual_violations += 1
                n = self.num_residual_violations
                if n <= 3 or n % 25 == 0:
                    print(
                        f"  {self._name}: solve residual "
                        f"{self.last_rel_residual:.2e} > rtol "
                        f"{self.residual_rtol:.1e} (violation {n})"
                    )
                # Repeated violations mean the perturbations are too large
                if n % 3 == 0 and self.pivot_eps > 1e-14:
                    self.pivot_eps = max(self.pivot_eps / 10.0, 1e-14)
                    self._eps_eff = self.pivot_eps
                    self.solver.set_pivot_epsilon(self.pivot_eps)
                    print(
                        f"  {self._name}: tightening pivot epsilon to "
                        f"{self.pivot_eps:.1e}"
                    )

    def inertia_enabled(self):
        return True

    def get_inertia(self):
        return self.solver.get_inertia()

    def static_pivot_floor(self):
        # Keep the dual block above the perturbation threshold
        return 10.0 * self._eps_eff

    def add_log_info(self, info):
        info["cuda_rel_residual"] = self.last_rel_residual
        info["cuda_residual_violations"] = self.num_residual_violations
        info["cuda_perturbed_pivots"] = self.last_perturbed_pivots
