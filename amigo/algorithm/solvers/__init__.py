from .inertia_correction import InertiaCorrector
from .linear_solver import LinearSolver

from .cuda_solver import DirectCudaSolver
from .bordered_cuda_solver import BorderedCudaSolver
from .mumps_solver import MumpsSolver
from .amigo_solver import AmigoSolver

# from .lnks_solver import LNKSInexactSolver
# from .pardiso_solver import PardisoSolver
# from .petsc_solver import DirectPetscSolver
# from .scipy_solver import DirectScipySolver

import warnings


def make_solver(options, state, problem=None, optimizer=None):
    """Make the linear solver depending on the options"""
    if isinstance(options["solver"], LinearSolver):
        return options["solver"]
    elif options["solver"] == "amigo":
        return AmigoSolver(options, state)
    elif options["solver"] == "mumps":
        # Only a missing library falls back, any other error surfaces
        try:
            return MumpsSolver(options, state)
        except ImportError as e:
            warnings.warn(f"MUMPS unavailable, using AmigoSolver instead: {e}")
            return AmigoSolver(options, state)
    elif options["solver"] == "cuda":
        # Only a build without CUDA falls back, any other error surfaces
        try:
            return DirectCudaSolver(options, state)
        except NotImplementedError as e:
            warnings.warn(f"CUDA unavailable, using AmigoSolver instead: {e}")
            return AmigoSolver(options, state)
    elif options["solver"] == "cuda_bordered":
        # Same iterates as "cuda", with hub columns eliminated by Schur
        return BorderedCudaSolver(options, state)
    else:
        solver = options["solver"]
        raise ValueError(f"Unrecognized solver {solver}")
