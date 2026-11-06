"""
Unit test stubs
---------------

- A compiled extension or heavy dependency that cannot be imported is stubbed before any amigo import
- The pure Python helpers then import without a C++ build, MUMPS, or SMT
- A module that imports for real is never stubbed, so tests like test_ldl.py keep the real extension
"""

import importlib
import sys
from unittest.mock import MagicMock

# The modules that may be missing without a build
OPTIONAL = [
    "amigo.amigo",
    "mpi4py",
    "mpi4py.MPI",
    "pybind11",
    "scipy",
    "scipy.sparse",
    "scipy.sparse.linalg",
    "scipy.interpolate",
    "networkx",
]


def is_importable(name):
    """Whether a module imports for real"""
    try:
        importlib.import_module(name)
    except ImportError:
        return False

    return True


for name in OPTIONAL:
    if not is_importable(name):
        sys.modules.setdefault(name, MagicMock())
