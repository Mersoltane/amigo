"""Bordered GPU direct solver for KKT systems with dense hub columns.

Dense columns from a globally coupled scalar are found in the sparsity pattern.
cuDSS factors the hub-free block and a Schur complement eliminates the border.
The elimination is exact and the residual check covers the whole bordered solve.
"""

from .cuda_solver import DirectCudaSolver


class BorderedCudaSolver(DirectCudaSolver):
    _factor_class = "CSRMatFactorCudaBordered"
    _requires = "CUDA/cuDSS"

    def _report_backend(self):
        if self.solver.num_bordered() == 0:
            print(
                f"  {self._name}: no hub columns detected, behaves as "
                "the plain cuDSS solver"
            )
