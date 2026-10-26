import numpy as np
import matplotlib.pylab as plt


class Vandermonde1D:
    def __init__(self, p, pts):
        self.n = p + 1
        self.p = p
        self.pts = pts
        self.C = self._build_vandermonde(self.p, self.pts)

    def _eval_polynomials(self, p, xi):
        pows = np.ones(p + 1, dtype=float)
        for i in range(1, p + 1):
            pows[i] = pows[i - 1] * xi
        return pows

    def _eval_polynomial_grad(self, p, xi):
        pows = np.ones(p + 1, dtype=float)
        for i in range(1, p + 1):
            pows[i] = pows[i - 1] * xi

        grad = np.zeros(p + 1)
        for i in range(1, p + 1):
            grad[i] = i * pows[i - 1]
        return grad

    def _build_vandermonde(self, p, pts):
        n = p + 1
        V = np.zeros((n, n), dtype=float)
        for a, xi in enumerate(pts):
            V[a, :] = self._eval_polynomials(p, xi[0])

        # Compute C = V^{-1}
        I = np.eye(n, dtype=float)
        return np.linalg.solve(V, I)

    def eval_basis(self, xi):
        B = self._eval_polynomials(self.p, xi)
        return B @ self.C

    def eval_basis_grad(self, xi):
        B = self._eval_polynomial_grad(self.p, xi)
        return B @ self.C


class Vandermonde2D:
    def __init__(self, n, exps, pts):
        self.n = n
        self.exps = exps
        self.pts = pts
        self.C = self._build_vandermonde(self.n, self.exps, self.pts)

    def eval_basis(self, xi, eta):
        B = self._eval_polynomials(xi, eta)
        N = B @ self.C
        return N

    def eval_basis_grad(self, xi, eta):
        Bxi, Beta = self._eval_polynomial_grad(xi, eta)
        Nxi = Bxi @ self.C
        Neta = Beta @ self.C
        return Nxi, Neta

    def _eval_polynomials(self, xi, eta):
        B = np.zeros(self.n)
        for i, u in enumerate(self.exps):
            B[i] = xi ** u[0] * eta ** u[1]
        return B

    def _eval_polynomial_grad(self, xi, eta):
        Bxi = np.zeros(self.n)
        Beta = np.zeros(self.n)
        for i, u in enumerate(self.exps):
            if u[0] >= 1:
                Bxi[i] = u[0] * xi ** (u[0] - 1) * eta ** u[1]
            if u[1] >= 1:
                Beta[i] = u[1] * xi ** u[0] * eta ** (u[1] - 1)
        return Bxi, Beta

    def _build_vandermonde(self, n, exps, pts):
        V = np.zeros((n, n), dtype=float)
        for a, (xi, eta) in enumerate(pts):
            V[a, :] = self._eval_polynomials(xi, eta)

        # Compute C = V^{-1}
        I = np.eye(n, dtype=float)
        return np.linalg.solve(V, I)


class VecVandermonde2D:
    def __init__(self, n, uexps, vexps, pts, dirs):
        self.n = n
        self.uexps = uexps
        self.vexps = vexps
        self.pts = pts
        self.dirs = dirs
        self.C = self._build_vandermonde(self.n, self.pts, self.dirs)
        return

    def eval_basis(self, xi, eta):
        B = self._eval_polynomials(xi, eta)
        N = B @ self.C
        return N

    def eval_basis_grad(self, xi, eta):
        Bxi, Beta = self._eval_polynomial_grad(xi, eta)
        Nxi = Bxi @ self.C
        Neta = Beta @ self.C
        return Nxi, Neta

    def _eval_polynomials(self, xi, eta):
        """
        Return the basis function evaluated at point (xi, eta)
        shape = (2 , num basis functions)
        """
        B = np.zeros((2, self.n))
        for i, (u, v) in enumerate(zip(self.uexps, self.vexps)):
            if u[0] >= 0 and u[1] >= 0:
                B[0, i] = xi ** u[0] * eta ** u[1]
            if v[0] >= 0 and v[1] >= 0:
                B[1, i] = xi ** v[0] * eta ** v[1]

        return B

    def _eval_polynomial_grad(self, xi, eta):
        Bxi = np.zeros((2, self.n))
        Beta = np.zeros((2, self.n))
        for i, (u, v) in enumerate(zip(self.uexps, self.vexps)):
            if u[0] >= 0 and u[1] >= 0:
                # B[0, i] = xi ** u[0] * eta ** u[1]
                if u[0] >= 1:
                    Bxi[0, i] = u[0] * xi ** (u[0] - 1) * eta ** u[1]
                if u[1] >= 1:
                    Beta[0, i] = xi ** u[0] * u[1] * eta ** (u[1] - 1)

            if v[0] >= 0 and v[1] >= 0:
                # B[1, i] = xi ** v[0] * eta ** v[1]
                if v[0] >= 1:
                    Bxi[1, i] = v[0] * xi ** (v[0] - 1) * eta ** v[1]
                if v[1] >= 1:
                    Beta[1, i] = xi ** v[0] * v[1] * eta ** (v[1] - 1)

        return Bxi, Beta

    def _build_vandermonde(self, n, pts, dirs):
        V = np.zeros((n, n), dtype=float)
        for a, (xi, eta) in enumerate(pts):
            basis = self._eval_polynomials(xi, eta)
            V[a, :] = dirs[a] @ basis

        # Compute C = V^{-1}
        I = np.eye(n, dtype=float)
        return np.linalg.solve(V, I)

    def visualize(self, analytic=None):
        # Create the figure
        fig, ax = plt.subplots(ncols=self.n)

        # Define xi and eta within a unit right triangle
        rng = np.random.default_rng()

        # Evaluate the basis functions
        for i in range(500):
            a1 = rng.random()
            a2 = rng.random()

            if a1 + a2 > 1.0:
                # Flip about the hypotnuse
                a1, a2 = 1.0 - a1, 1.0 - a2

            # Compute the sample point
            sample = a1 * np.array([1, 0]) + a2 * np.array([0, 1])

            # Extract components
            xi = sample[0]
            eta = sample[1]

            # Evaluate the basis
            N = self.eval_basis(xi, eta)

            # Plot each basis function seperately
            for i in range(self.n):
                ax[i].quiver(xi, eta, N[0, i], N[1, i])

            if analytic is not None:
                N_exact = analytic(xi, eta)
                for i in range(self.n):
                    ax[i].quiver(
                        xi,
                        eta,
                        N_exact[0, i],
                        N_exact[1, i],
                        edgecolor="red",
                        facecolor="none",
                        linewidth=0.8,
                    )
        return
