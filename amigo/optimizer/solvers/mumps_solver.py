from . import LinearSolver
import os
import sys
import numpy as np
from amigo import MemoryLocation


class MumpsSolver(LinearSolver):
    """Sparse symmetric indefinite solver via MUMPS (LDL^T with inertia).

    Uses the MUMPS C interface (dmumps_c) via ctypes. Provides exact
    inertia counts from MUMPS info arrays after factorization.

    Requires coin-or/ThirdParty-Mumps (with METIS ordering and scaling).
    Windows: build via MSYS2 with mingw-w64-x86_64-metis.
    Linux: apt install libmumps-dev or conda install mumps-seq.
    Mac: brew install brewsci/num/mumps.
    """

    supports_inertia = True

    @staticmethod
    def _load_mumps_library():
        """Locate and load the MUMPS shared library.

        Search order: MUMPS_LIB_DIR env var, coin-or ThirdParty-Mumps
        install, conda environment, system PATH.
        """
        import ctypes

        lib_dir = os.environ.get("MUMPS_LIB_DIR", "")

        # Platform-specific library names and search paths
        if sys.platform == "win32":
            # Register dependency directories for Windows DLL resolution
            for d in [
                r"C:\msys64\mingw64\bin",
                os.path.expanduser("~/mumps-coinor/bin"),
            ]:
                if os.path.isdir(d):
                    os.add_dll_directory(d)

            names = ["libcoinmumps-3.dll", "libdmumps.dll", "dmumps.dll"]
            search_dirs = [
                lib_dir,
                os.path.expanduser("~/mumps-coinor/bin"),
            ]
            conda = os.environ.get("CONDA_PREFIX", "")
            if conda:
                search_dirs.append(os.path.join(conda, "Library", "bin"))
        elif sys.platform == "darwin":
            names = ["libdmumps.dylib"]
            brew_prefix = "/opt/homebrew/opt/brewsci-mumps/lib"
            brew_x86 = "/usr/local/opt/brewsci-mumps/lib"
            search_dirs = [d for d in [lib_dir, brew_prefix, brew_x86] if d]
        else:
            names = ["libdmumps.so"]
            search_dirs = [lib_dir] if lib_dir else []

        # Try each directory + name combination, then bare names for PATH
        for d in search_dirs:
            if not d:
                continue
            for name in names:
                path = os.path.join(d, name)
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    pass
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                pass

        raise ImportError(
            "MUMPS library not found. "
            "Windows: build coin-or/ThirdParty-Mumps via MSYS2. "
            "Linux: apt install libmumps-dev or conda install mumps-seq. "
            "Mac: brew tap brewsci/num && brew install brewsci-mumps. "
            "Or set MUMPS_LIB_DIR to the directory containing the library."
        )

    def __init__(self, problem):
        import ctypes

        self._ct = ctypes
        self._libmumps = self._load_mumps_library()
        self._dmumps_c = self._libmumps.dmumps_c
        self._dmumps_c.restype = None

        self.problem = problem
        loc = MemoryLocation.HOST_AND_DEVICE
        self.hess = self.problem.create_matrix(loc)
        self.nrows, self.ncols, self.nnz, self.rowp, self.cols = (
            self.hess.get_nonzero_structure()
        )
        self._diag_indices = self._find_diag_indices(self.rowp, self.cols, self.nrows)

        # Build COO triplet arrays from CSR (MUMPS uses 1-based COO)
        # Only store lower triangle for sym=2 (symmetric indefinite)
        row_idx = np.repeat(np.arange(self.nrows, dtype=np.int32), np.diff(self.rowp))
        col_idx = np.array(self.cols, dtype=np.int32)
        lower_mask = col_idx <= row_idx
        self._irn = row_idx[lower_mask] + 1
        self._jcn = col_idx[lower_mask] + 1
        self._data_map = np.nonzero(lower_mask)[0].astype(np.intc)
        self._nnz_lower = int(lower_mask.sum())
        self._a = np.empty(self._nnz_lower, dtype=np.float64)

        # Build the MUMPS struct via ctypes
        self._build_struct()

        # Initialize MUMPS (job=-1)
        self._mumps.job = -1
        self._mumps.par = 1
        self._mumps.sym = 2  # symmetric indefinite (LDL^T)
        self._mumps.comm_fortran = -987654  # MPISEQ sequential
        self._call_mumps()

        # MUMPS solver parameters
        self._mumps.icntl[0] = -1  # ICNTL(1):  suppress error output
        self._mumps.icntl[1] = -1  # ICNTL(2):  suppress diagnostic output
        self._mumps.icntl[2] = -1  # ICNTL(3):  suppress global info
        self._mumps.icntl[3] = 0  # ICNTL(4):  no output
        self._mumps.icntl[5] = 7  # ICNTL(6):  permuting and scaling
        self._mumps.icntl[6] = 7  # ICNTL(7):  pivot ordering (automatic)
        self._mumps.icntl[7] = 77  # ICNTL(8):  scaling (automatic)
        self._mumps.icntl[9] = 0  # ICNTL(10): no iterative refinement
        self._mumps.icntl[12] = 1  # ICNTL(13): proper inertia detection
        self._mumps.icntl[13] = 1000  # ICNTL(14): workspace increase %
        # ICNTL(24) = 0: null pivot detection off during normal factorization.
        # When enabled, near-zero negative pivots can be misclassified as
        # "null", corrupting the inertia count.
        self._mumps.icntl[23] = 0  # ICNTL(24): null pivot detection OFF
        self._mumps.cntl[0] = 1e-6  # CNTL(1):  pivot tolerance

        # Set matrix structure and values pointer
        self._mumps.n = self.nrows
        self._mumps.nz = int(self._nnz_lower) if self._nnz_lower < 2**31 else 0
        self._mumps.nnz = self._nnz_lower
        self._mumps.irn = self._irn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._mumps.jcn = self._jcn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._mumps.a = self._a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Symbolic analysis deferred to first factorization (with real values)
        self._have_symbolic = False

        self._rhs = np.empty(self.nrows, dtype=np.float64)
        self._mumps.rhs = self._rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._mumps.nrhs = 1
        self._mumps.lrhs = self.nrows

    def _build_struct(self):
        """Build the ctypes Structure matching DMUMPS_STRUC_C."""
        ct = self._ct
        import sys

        if sys.platform == "win32":
            # Layout matching the Windows MUMPS build (newer/custom version)
            _mumps_fields = [
                ("sym", ct.c_int),
                ("par", ct.c_int),
                ("job", ct.c_int),
                ("comm_fortran", ct.c_int),
                ("icntl", ct.c_int * 60),
                ("keep", ct.c_int * 500),
                ("cntl", ct.c_double * 15),
                ("dkeep", ct.c_double * 230),
                ("keep8", ct.c_int64 * 150),
                ("n", ct.c_int),
                ("nblk", ct.c_int),
                ("nz_alloc", ct.c_int),
                ("nz", ct.c_int),
                ("nnz", ct.c_int64),
                ("irn", ct.POINTER(ct.c_int)),
                ("jcn", ct.POINTER(ct.c_int)),
                ("a", ct.POINTER(ct.c_double)),
                ("nz_loc", ct.c_int),
                ("nnz_loc", ct.c_int64),
                ("irn_loc", ct.POINTER(ct.c_int)),
                ("jcn_loc", ct.POINTER(ct.c_int)),
                ("a_loc", ct.POINTER(ct.c_double)),
                ("nelt", ct.c_int),
                ("eltptr", ct.POINTER(ct.c_int)),
                ("eltvar", ct.POINTER(ct.c_int)),
                ("a_elt", ct.POINTER(ct.c_double)),
                ("blkptr", ct.POINTER(ct.c_int)),
                ("blkvar", ct.POINTER(ct.c_int)),
                ("perm_in", ct.POINTER(ct.c_int)),
                ("sym_perm", ct.POINTER(ct.c_int)),
                ("uns_perm", ct.POINTER(ct.c_int)),
                ("colsca", ct.POINTER(ct.c_double)),
                ("rowsca", ct.POINTER(ct.c_double)),
                ("colsca_from_mumps", ct.c_int),
                ("rowsca_from_mumps", ct.c_int),
                ("colsca_loc", ct.POINTER(ct.c_double)),
                ("rowsca_loc", ct.POINTER(ct.c_double)),
                ("rowind", ct.POINTER(ct.c_int)),
                ("colind", ct.POINTER(ct.c_int)),
                ("pivots", ct.POINTER(ct.c_double)),
                ("rhs", ct.POINTER(ct.c_double)),
                ("redrhs", ct.POINTER(ct.c_double)),
                ("rhs_sparse", ct.POINTER(ct.c_double)),
                ("sol_loc", ct.POINTER(ct.c_double)),
                ("rhs_loc", ct.POINTER(ct.c_double)),
                ("rhsintr", ct.POINTER(ct.c_double)),
                ("irhs_sparse", ct.POINTER(ct.c_int)),
                ("irhs_ptr", ct.POINTER(ct.c_int)),
                ("isol_loc", ct.POINTER(ct.c_int)),
                ("irhs_loc", ct.POINTER(ct.c_int)),
                ("glob2loc_rhs", ct.POINTER(ct.c_int)),
                ("glob2loc_sol", ct.POINTER(ct.c_int)),
                ("nrhs", ct.c_int),
                ("lrhs", ct.c_int),
                ("lredrhs", ct.c_int),
                ("nz_rhs", ct.c_int),
                ("lsol_loc", ct.c_int),
                ("nloc_rhs", ct.c_int),
                ("lrhs_loc", ct.c_int),
                ("nsol_loc", ct.c_int),
                ("schur_mloc", ct.c_int),
                ("schur_nloc", ct.c_int),
                ("schur_lld", ct.c_int),
                ("mblock", ct.c_int),
                ("nblock", ct.c_int),
                ("nprow", ct.c_int),
                ("npcol", ct.c_int),
                ("ld_rhsintr", ct.c_int),
                ("info", ct.c_int * 80),
                ("infog", ct.c_int * 80),
                ("rinfo", ct.c_double * 40),
                ("rinfog", ct.c_double * 40),
                ("deficiency", ct.c_int),
                ("pivnul_list", ct.POINTER(ct.c_int)),
                ("mapping", ct.POINTER(ct.c_int)),
                ("singular_values", ct.POINTER(ct.c_double)),
                ("size_schur", ct.c_int),
                ("listvar_schur", ct.POINTER(ct.c_int)),
                ("schur", ct.POINTER(ct.c_double)),
                ("wk_user", ct.POINTER(ct.c_double)),
                ("version_number", ct.c_char * 32),
                ("ooc_tmpdir", ct.c_char * 1024),
                ("ooc_prefix", ct.c_char * 256),
                ("write_problem", ct.c_char * 1024),
                ("lwk_user", ct.c_int),
                ("save_dir", ct.c_char * 1024),
                ("save_prefix", ct.c_char * 256),
                ("metis_options", ct.c_int * 40),
                ("instance_number", ct.c_int),
            ]
        else:
            # Layout matching MUMPS 5.x (Linux/Mac)
            _mumps_fields = [
                ("sym", ct.c_int),
                ("par", ct.c_int),
                ("job", ct.c_int),
                ("comm_fortran", ct.c_int),
                ("icntl", ct.c_int * 60),
                ("keep", ct.c_int * 500),
                ("cntl", ct.c_double * 15),
                ("dkeep", ct.c_double * 230),
                ("keep8", ct.c_int64 * 150),
                ("n", ct.c_int),
                ("nblk", ct.c_int),
                ("nz_alloc", ct.c_int),
                ("nz", ct.c_int),
                ("nnz", ct.c_int64),
                ("irn", ct.POINTER(ct.c_int)),
                ("jcn", ct.POINTER(ct.c_int)),
                ("a", ct.POINTER(ct.c_double)),
                ("nz_loc", ct.c_int),
                ("nnz_loc", ct.c_int64),
                ("irn_loc", ct.POINTER(ct.c_int)),
                ("jcn_loc", ct.POINTER(ct.c_int)),
                ("a_loc", ct.POINTER(ct.c_double)),
                ("nelt", ct.c_int),
                ("eltptr", ct.POINTER(ct.c_int)),
                ("eltvar", ct.POINTER(ct.c_int)),
                ("a_elt", ct.POINTER(ct.c_double)),
                ("blkptr", ct.POINTER(ct.c_int)),
                ("blkvar", ct.POINTER(ct.c_int)),
                ("perm_in", ct.POINTER(ct.c_int)),
                ("sym_perm", ct.POINTER(ct.c_int)),
                ("uns_perm", ct.POINTER(ct.c_int)),
                ("colsca", ct.POINTER(ct.c_double)),
                ("rowsca", ct.POINTER(ct.c_double)),
                ("colsca_from_mumps", ct.c_int),
                ("rowsca_from_mumps", ct.c_int),
                ("rhs", ct.POINTER(ct.c_double)),
                ("redrhs", ct.POINTER(ct.c_double)),
                ("rhs_sparse", ct.POINTER(ct.c_double)),
                ("sol_loc", ct.POINTER(ct.c_double)),
                ("rhs_loc", ct.POINTER(ct.c_double)),
                ("irhs_sparse", ct.POINTER(ct.c_int)),
                ("irhs_ptr", ct.POINTER(ct.c_int)),
                ("isol_loc", ct.POINTER(ct.c_int)),
                ("irhs_loc", ct.POINTER(ct.c_int)),
                ("nrhs", ct.c_int),
                ("lrhs", ct.c_int),
                ("lredrhs", ct.c_int),
                ("nz_rhs", ct.c_int),
                ("lsol_loc", ct.c_int),
                ("nloc_rhs", ct.c_int),
                ("lrhs_loc", ct.c_int),
                ("schur_mloc", ct.c_int),
                ("schur_nloc", ct.c_int),
                ("schur_lld", ct.c_int),
                ("mblock", ct.c_int),
                ("nblock", ct.c_int),
                ("nprow", ct.c_int),
                ("npcol", ct.c_int),
                ("info", ct.c_int * 80),
                ("infog", ct.c_int * 80),
                ("rinfo", ct.c_double * 40),
                ("rinfog", ct.c_double * 40),
                ("deficiency", ct.c_int),
                ("pivnul_list", ct.POINTER(ct.c_int)),
                ("mapping", ct.POINTER(ct.c_int)),
                ("size_schur", ct.c_int),
                ("listvar_schur", ct.POINTER(ct.c_int)),
                ("schur", ct.POINTER(ct.c_double)),
                ("instance_number", ct.c_int),
                ("wk_user", ct.POINTER(ct.c_double)),
                ("version_number", ct.c_char * 32),
                ("ooc_tmpdir", ct.c_char * 256),
                ("ooc_prefix", ct.c_char * 64),
                ("write_problem", ct.c_char * 256),
                ("lwk_user", ct.c_int),
                ("save_dir", ct.c_char * 256),
                ("save_prefix", ct.c_char * 256),
                ("metis_options", ct.c_int * 40),
            ]

        class DMUMPS_STRUC_C(ct.Structure):
            _fields_ = _mumps_fields

        self._DMUMPS_STRUC_C = DMUMPS_STRUC_C
        self._mumps = DMUMPS_STRUC_C()
        self._dmumps_c.argtypes = [ct.POINTER(DMUMPS_STRUC_C)]

    def _call_mumps(self):
        self._dmumps_c(self._ct.byref(self._mumps))

    def _factorize_current(self):
        if not self._have_symbolic:
            self._mumps.job = 1  # symbolic analysis with actual values
            self._call_mumps()
            if self._mumps.infog[0] < 0:
                raise RuntimeError(
                    f"MUMPS analysis failed: infog(1)={self._mumps.infog[0]}"
                )
            self._have_symbolic = True
        self._mumps.job = 2  # numerical factorization
        self._call_mumps()
        if self._mumps.infog[0] < 0:
            raise RuntimeError(
                f"MUMPS factorize failed: infog(1)={self._mumps.infog[0]}, "
                f"infog(2)={self._mumps.infog[1]}"
            )

    def _update_values(self):
        data = self.hess.get_data()
        self._a[:] = data[self._data_map]

    def add_diagonal_and_factor(self, diag):
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        self._update_values()
        self._factorize_current()

    def factor(self, alpha, x, diag, post_hessian=None):
        self.problem.hessian(alpha, x, self.hess)
        if post_hessian is not None:
            self.hess.copy_data_device_to_host()
            post_hessian(self.hess)
        self.problem.add_diagonal(diag, self.hess)
        self.hess.copy_data_device_to_host()
        self._update_values()
        self._factorize_current()

    def get_inertia(self):
        """Return (n_positive, n_negative) from MUMPS infog(12).

        infog(12) = number of negative pivots in LDL^T factorization.
        With ICNTL(24)=0 (null pivot detection off), all pivots are
        classified as positive or negative.
        """
        n_neg = int(self._mumps.infog[11])
        n_pos = self.nrows - n_neg
        return n_pos, n_neg

    def solve(self, bx, px):
        bx.copy_device_to_host()
        self._rhs[:] = bx.get_array()
        self._mumps.job = 3
        self._call_mumps()
        if self._mumps.infog[0] < 0:
            raise RuntimeError(f"MUMPS solve failed: infog(1)={self._mumps.infog[0]}")
        px.get_array()[:] = self._rhs
        px.copy_host_to_device()

    def __del__(self):
        try:
            self._mumps.job = -2
            self._call_mumps()
        except Exception:
            pass
