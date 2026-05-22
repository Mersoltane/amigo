"""Primal-dual interior-point optimizer.

The Optimizer class composes the algorithmic pieces from the sibling modules
and runs the main iteration loop.  Each iteration evaluates the KKT
residual, checks convergence, updates the barrier parameter, computes
a Newton direction, runs a line search, and handles step acceptance
or feasibility restoration.
"""

import time
import numpy as np

from ..model import ModelVector
from ..amigo import InteriorPointOptimizer, Vector

from .default_options import get_default_options
from .filter_acceptance import Filter
from .filter_line_search import FilterLineSearch, WatchdogState
from .merit_line_search import MeritLineSearch
from .ipm_state import IpmData, IpmState, StepContext

from .iterate_initialization import IterateInitialization
from .convergence_check import (
    ConvergenceCheck,
    CONVERGED,
    CONVERGED_ACCEPTABLE,
    DIVERGED,
    PRECISION_FLOOR,
)
from .barrier_strategy import make_barrier_strategy
from .newton_direction import NewtonDirection
from .optimality_scaling import OptimalityScaling
from .bound_safeguards import BoundSafeguards
from .multiplier_initialization import MultiplierInitialization
from .feasibility_restoration import FeasibilityRestoration
from .iteration_logger import IterationLogger
from .newton_diagnostics import NewtonDiagnostics
from .post_optimization import PostOptimization

from .solvers import (
    AmigoSolver,
    DirectPetscSolver,
    DirectScipySolver,
    MumpsSolver,
    PardisoSolver,
)

from .inertia_correction import InertiaCorrector


class Optimizer(
    IterateInitialization,
    ConvergenceCheck,
    NewtonDirection,
    OptimalityScaling,
    BoundSafeguards,
    MultiplierInitialization,
    MeritLineSearch,
    FilterLineSearch,
    FeasibilityRestoration,
    IterationLogger,
    NewtonDiagnostics,
    PostOptimization,
):
    """Primal-dual interior-point optimizer with filter line search.

    Composes a BarrierStrategy (self.barrier) for the mu update and
    inherits the remaining algorithmic pieces as mixins.
    """

    def __init__(
        self,
        model=None,
        x=None,
        problem=None,
        solver=None,
        comm=None,
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        model : Model
            The amigo model to optimize
        x : array-like, optional
            Initial point
        solver : Solver, optional
            Linear solver for the KKT system.
            May also be a string from ["scipy", "pardiso", "mumps"]
        comm : MPI communicator, optional
            For distributed optimization
        """
        self.barrier_param = 1.0

        # Set the model and problem
        if model is not None:
            self.model = model
            self.problem = self.model.get_problem()
        elif problem is not None:
            self.model = None
            self.problem = problem

        # Set the design vector
        if isinstance(x, ModelVector):
            self.x = x.get_vector()
        elif isinstance(x, Vector):
            self.x = x
        else:
            self.x = self.problem.create_vector()

        self.comm = comm
        self.distribute = False
        if self.comm is not None and self.comm.size > 1:
            self.distribute = True

        # Set up the vectors
        self._setup_initial_vectors()
        self._select_solver(solver)
        self._create_interior_point_backend()
        self._allocate_working_vectors()

    def _setup_initial_vectors(self):
        """Initialize the vectors"""
        x_init = self.problem.get_initial_point()
        self.x.copy(x_init)
        self.lower = self.problem.get_lower()
        self.upper = self.problem.get_upper()

    def _select_solver(self, solver):
        """Resolve solver spec (instance, string, or None) to a concrete solver."""
        if solver is None and self.distribute:
            self.solver = DirectPetscSolver(self.comm, self.problem)
        elif isinstance(solver, str):
            solver_pref = solver.lower()
            if solver_pref == "scipy":
                self.solver = DirectScipySolver(self.problem)
            elif solver_pref == "pardiso":
                self.solver = PardisoSolver(self.problem)
            elif solver_pref == "mumps":
                try:
                    self.solver = MumpsSolver(self.problem)
                except:
                    self.solver = AmigoSolver(self.problem)
            elif solver_pref == "amigo":
                self.solver = AmigoSolver(self.problem)
            else:
                raise ValueError(
                    f"Unknown solver string '{solver}'. "
                    "Expected one of: 'scipy', 'pardiso', 'mumps', 'amigo'."
                )
        elif solver is not None:
            self.solver = solver
        else:
            self.solver = AmigoSolver(self.problem)

    def _create_interior_point_backend(self):
        """Create the C++ InteriorPointOptimizer backend and slack mapping."""
        data_vec = self.problem.get_data_vector()
        self.x.copy_host_to_device()
        self.lower.copy_host_to_device()
        self.upper.copy_host_to_device()
        data_vec.copy_host_to_device()

        self.optimizer = InteriorPointOptimizer(self.problem)
        self.vars = self.optimizer.create_opt_vector(self.x)
        self.update = self.optimizer.create_opt_vector()
        self.temp = self.optimizer.create_opt_vector()

    def _allocate_working_vectors(self):
        """Allocate scratch vectors for gradient, residual, direction, etc."""
        self.grad = self.problem.create_vector()
        self.res = self.problem.create_vector()
        self.diag = self.problem.create_vector()
        self.px = self.problem.create_vector()
        self.ir_corr = self.problem.create_vector()

    def _build_inertia_corrector(self, tol, options, comm_rank):
        """Create an InertiaCorrector if the solver supports inertia queries."""
        inertia_corrector = None
        if getattr(self.solver, "supports_inertia", False):
            inertia_corrector = InertiaCorrector(
                self.problem, self.optimizer, self.barrier_param, options
            )
            if comm_rank == 0:
                n_primal = self.optimizer.get_num_primals()
                n_dual = self.optimizer.get_num_constraints()
                n_total = n_primal + n_dual
                solver_name = type(self.solver).__name__
                print(f"\n  Amigo IPM ({solver_name})")
                print(
                    f"  Variables: {n_total} ({n_primal} primal, {n_dual} constraints)"
                )
                print(f"  Tolerance: {tol:.0e}  mu_init: {self.barrier_param:.0e}\n")
        return inertia_corrector

    def _zero_hessian_indices(self, options, comm_rank):
        """Resolve zero-Hessian variable names to integer indices."""
        zero_hessian_indices = None
        zero_hessian_eps = options["regularization_eps_x_zero_hessian"]
        zh_vars = options["zero_hessian_variables"]
        if zh_vars and not self.distribute:
            zero_hessian_indices = np.sort(self.model.get_indices(zh_vars))
            if comm_rank == 0:
                print(
                    f"  Variable-specific regularization: {len(zero_hessian_indices)} "
                    f"zero-Hessian vars, eps_x_zero={zero_hessian_eps:.2e}"
                )
        return zero_hessian_indices, zero_hessian_eps

    def get_options(self, options={}):
        return get_default_options(options)

    def get_optimized_point(self):
        return ModelVector(self.model, x=self.x)

    def optimize(self, options={}):
        """Run the interior-point optimization algorithm.

        Returns a dict with keys "converged", "iterations", "options".
        """
        start_time = time.perf_counter()
        comm_rank = self.comm.rank if self.comm is not None else 0

        options = self.get_options(options=options)
        opt_data = {"options": options, "converged": False, "iterations": []}

        max_iters = options["max_iterations"]
        base_tau = options["fraction_to_boundary"]
        tau_min = options["tau_min"]
        use_adaptive_tau = options["adaptive_tau"]
        tol = options["convergence_tolerance"]
        compl_inf_tol = options["compl_inf_tol"]
        record_components = options["record_components"]
        continuation_control = options["continuation_control"]
        max_rejections = options["max_consecutive_rejections"]
        barrier_inc = options["barrier_increase_factor"]
        initial_barrier = options["initial_barrier_param"]
        filter_reset_trigger = options["filter_reset_trigger"]
        max_filter_resets = options["max_filter_resets"]
        self.barrier_param = options["initial_barrier_param"]

        # Place everything in a data class
        # self.data = IpmData(
        #     options=self.options,
        #     problem=self.problem,
        #     optimizer=self.optimizer,
        #     solver=self.solver,
        #     vars=self.vars,
        # )

        # Create the barrier strategy
        self.barrier = make_barrier_strategy(self, options)

        # Initialization
        self._initialize_iterate(options, comm_rank)

        x = self.vars.get_solution()
        xview = ModelVector(self.model, x=x) if not self.distribute else None

        # Loop state
        state = IpmState()
        state.res_norm_mu = self.barrier_param

        # Inertia corrector + zero-Hessian indices
        inertia_corrector = self._build_inertia_corrector(tol, options, comm_rank)
        zero_hessian_indices, zero_hessian_eps = self._zero_hessian_indices(
            options, comm_rank
        )

        # Barrier-strategy step context (shared across iterations; per-iteration
        # fields i, res_norm, diag_base, filter_monotone_* are updated in-loop)
        ctx = StepContext(
            comm_rank=comm_rank,
            tol=tol,
            compl_inf_tol=compl_inf_tol,
            x=x,
            inertia_corrector=inertia_corrector,
            zero_hessian_indices=zero_hessian_indices,
            zero_hessian_eps=zero_hessian_eps,
        )
        self.barrier.initialize(ctx)

        # Filter line search state
        filter_ls = options["filter_line_search"]
        outer_filter = Filter() if filter_ls else None
        inner_filter = (
            Filter(
                gamma_phi=options["filter_gamma_phi"],
                gamma_theta=options["filter_gamma_theta"],
            )
            if filter_ls
            else None
        )
        if filter_ls:
            self._filter_theta_0 = None

        # Watchdog
        watchdog = WatchdogState(self.optimizer)
        watchdog.trigger = options["watchdog_shortened_iter_trigger"]
        watchdog.max_trials = options["watchdog_trial_iter_max"]

        # Create vectors for the primal or dual variables only
        primal_vec = self.problem.create_primal_vector()
        con_vec = self.problem.create_constraint_vector()

        # Main loop
        for i in range(max_iters):
            primal_indices = self.problem.get_primal_indices()
            con_indices = self.problem.get_constraint_indices()

            # Step A: KKT residual
            res_norm = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )
            state.res_norm_mu = self.barrier_param
            if inertia_corrector:
                inertia_corrector.update_barrier(self.barrier_param)

            self.res.get_values_at(con_indices, con_vec)
            self.res.get_values_at(primal_indices, primal_vec)
            theta_res = self.problem.norm(con_vec)
            eta_res = self.problem.norm(primal_vec)

            if filter_ls and self._filter_theta_0 is None:
                self._filter_theta_0 = self._compute_filter_theta()

            if continuation_control is not None:
                continuation_control(i, res_norm)

            # Step B: Log
            elapsed_time = time.perf_counter() - start_time
            iter_data = self._build_iter_data(
                i,
                elapsed_time,
                res_norm,
                state.line_iters,
                state.alpha_x_prev,
                state.alpha_z_prev,
                state.x_index_prev,
                state.z_index_prev,
                inertia_corrector,
                theta_res,
                eta_res,
                filter_ls,
                outer_filter,
                options,
            )
            if comm_rank == 0:
                self.write_log(i, iter_data)
            iter_data["x"] = {}
            if xview is not None:
                for name in record_components:
                    iter_data["x"][name] = xview[name].tolist()
            opt_data["iterations"].append(iter_data)

            # Step C: Convergence
            status, _, state.acceptable_counter, state.precision_floor_count = (
                self._check_convergence(
                    i,
                    options,
                    res_norm,
                    state.prev_res_norm,
                    state.acceptable_counter,
                    state.precision_floor_count,
                    comm_rank,
                )
            )
            if status == CONVERGED:
                opt_data["converged"] = True
                break
            if status == CONVERGED_ACCEPTABLE:
                opt_data["converged"] = True
                opt_data["acceptable"] = True
                break
            if status == PRECISION_FLOOR:
                opt_data["converged"] = True
                opt_data["acceptable"] = True
                opt_data["precision_floor"] = True
                break
            if status == DIVERGED:
                break
            state.prev_res_norm = res_norm

            # Step D: Barrier update + direction
            step_rejected = False

            # Zero-step recovery (non-inertia path only)
            if not inertia_corrector:
                state.zero_step_count = self.barrier.handle_zero_step_recovery(
                    i,
                    state.alpha_x_prev,
                    state.alpha_z_prev,
                    state.zero_step_count,
                    comm_rank,
                )

            # Barrier diagonal Sigma = Z/S
            self.optimizer.compute_diagonal(self.vars, self.diag)
            self.diag.copy_device_to_host()
            diag_base = self.diag.get_array().copy()

            barrier_before = self.barrier_param

            ctx.i = i
            ctx.res_norm = res_norm
            ctx.diag_base = diag_base
            ctx.filter_monotone_mode = state.filter_monotone_mode
            ctx.filter_monotone_mu = state.filter_monotone_mu
            factorize_ok = self.barrier.step(ctx)
            state.filter_monotone_mu = ctx.filter_monotone_mu

            # Reset line search state when mu changed
            if self.barrier_param != barrier_before:
                if inertia_corrector:
                    inertia_corrector.update_barrier(self.barrier_param)
                if filter_ls and inner_filter is not None:
                    inner_filter.clear()
                    self._filter_theta_0 = self._compute_filter_theta()
                watchdog.reset()
                state.count_successive_filter_rejections = 0
                state.filter_reset_count = 0

            # Inertia correction failed: reject, increase barrier if needed
            if not factorize_ok:
                step_rejected = True
                state.consecutive_rejections += 1
                if comm_rank == 0:
                    print(
                        f"  Inertia correction FAILED "
                        f"({state.consecutive_rejections}x)"
                    )
                self.barrier_param = barrier_before
                state.line_iters = 0
                state.alpha_x_prev = state.alpha_z_prev = 0.0
                state.x_index_prev = state.z_index_prev = -1
                state.consecutive_rejections = (
                    self.barrier.increase_barrier_on_rejections(
                        state.consecutive_rejections,
                        max_rejections,
                        barrier_inc,
                        initial_barrier,
                        comm_rank,
                    )
                )
                self.barrier.on_barrier_increased()
                continue

            # Optional Newton diagnostics
            if options["check_update_step"]:
                self._run_check_update_diagnostics(comm_rank)

            rhs_norm = self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )
            if options["check_update_step"] and comm_rank == 0 and i > 0:
                self._print_newton_diagnostics(rhs_norm, state.res_norm_mu)

            # Compute maximum step sizes from fraction-to-boundary
            tau = (
                self._compute_adaptive_tau(self.barrier_param, tau_min)
                if use_adaptive_tau
                else base_tau
            )
            alpha_x, x_index, alpha_z, z_index = self.optimizer.compute_max_step(
                tau, self.vars, self.update
            )
            if options["equal_primal_dual_step"]:
                alpha_x = alpha_z = min(alpha_x, alpha_z)

            # Step E: Line search
            if filter_ls:
                alpha, state.line_iters, step_accepted, filter_rejected = (
                    self._filter_line_search_with_watchdog(
                        alpha_x,
                        alpha_z,
                        inner_filter,
                        options,
                        comm_rank,
                        tau,
                        watchdog,
                        factorize_ok,
                    )
                )

                # Filter reset heuristic
                if step_accepted:
                    if filter_rejected:
                        state.count_successive_filter_rejections += 1
                    else:
                        state.count_successive_filter_rejections = 0
                    if (
                        state.count_successive_filter_rejections >= filter_reset_trigger
                        and state.filter_reset_count < max_filter_resets
                    ):
                        inner_filter.clear()
                        state.filter_reset_count += 1
                        state.count_successive_filter_rejections = 0
                        if comm_rank == 0:
                            print(
                                f"  Filter reset "
                                f"({state.filter_reset_count}/{max_filter_resets})"
                            )

                # Step F: Restoration if LS failed
                if not step_accepted:
                    restored = self._restoration_phase(
                        inertia_corrector,
                        inner_filter,
                        options,
                        comm_rank,
                        x,
                        diag_base,
                        zero_hessian_indices,
                        zero_hessian_eps,
                    )
                    if restored:
                        step_accepted = True
                        state.line_iters = 0
                        watchdog.shortened_iter = 0
                    else:
                        step_rejected = True
                        state.consecutive_rejections += 1
                        if comm_rank == 0:
                            print(
                                f"  Filter+Restoration REJECTED "
                                f"({state.consecutive_rejections}x)"
                            )

                if step_accepted:
                    n_adj = self._ensure_positive_slacks(self.vars, self.barrier_param)
                    if n_adj > 0 and comm_rank == 0:
                        print(f"  Slack adjustment: {n_adj} variable(s)")
                    self.optimizer.reset_bound_multipliers(
                        self.barrier_param,
                        1e10,
                        self.vars,
                    )
                    self._update_gradient(self.vars.get_solution())
            else:

                def _reject_step():
                    nonlocal step_rejected
                    step_rejected = True
                    state.consecutive_rejections += 1

                reject_cb = _reject_step if inertia_corrector else None
                alpha, state.line_iters, step_accepted = self._line_search(
                    alpha_x,
                    alpha_z,
                    options,
                    comm_rank,
                    tau=tau,
                    reject_callback=reject_cb,
                )

            # Step G: Post-step update
            if step_rejected:
                state.alpha_x_prev = state.alpha_z_prev = 0.0
                state.x_index_prev = state.z_index_prev = -1
                self.barrier_param = barrier_before

                self.barrier.on_step_rejected(ctx)

                state.consecutive_rejections = (
                    self.barrier.increase_barrier_on_rejections(
                        state.consecutive_rejections,
                        max_rejections,
                        barrier_inc,
                        initial_barrier,
                        comm_rank,
                    )
                )
                self.barrier.on_barrier_increased()
            else:
                state.alpha_x_prev = alpha * alpha_x
                state.alpha_z_prev = alpha_z if filter_ls else alpha * alpha_z
                state.x_index_prev = x_index
                state.z_index_prev = z_index
                state.consecutive_rejections = 0

                if filter_ls and not watchdog.in_watchdog:
                    if alpha == 1.0 or state.line_iters == 1:
                        watchdog.shortened_iter = 0
                    else:
                        watchdog.shortened_iter += 1

        if comm_rank == 0 and not opt_data.get("converged", False):
            print(f"\n{'='*70}")
            print(f"  Amigo did NOT converge (max iterations: {max_iters})")
            print(f"{'='*70}")
            print(f"  Residual                {res_norm:>20.10e}")
            print(f"  Barrier parameter       {self.barrier_param:>20.10e}")
            print(f"{'='*70}")

        return opt_data
