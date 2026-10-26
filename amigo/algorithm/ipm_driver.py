"""Primal-dual interior-point optimizer.

The Optimizer class composes the algorithmic pieces from the sibling modules
and runs the main iteration loop.  Each iteration evaluates the KKT
residual, checks convergence, updates the barrier parameter, computes
a Newton direction, runs a line search, and handles step acceptance
or feasibility restoration.
"""

import gc
import time
import warnings
import numpy as np

# Raw pybind11 classes
from ..amigo import InteriorPointOptimizer, Vector

# Import from amigo classes
from ..model import ModelVector
from ..utils import tocsr

# Optimizer imports from algorithm classes
from .barrier_strategy import make_barrier_strategy
from .convergence_check import (
    ConvergenceCheck,
    CONTINUE,
    CONVERGED_ACCEPTABLE,
    DIVERGED,
    RESTORATION_NEEDED,
    LOCALLY_INFEASIBLE,
)
from .default_options import get_default_options
from .evaluator import Evaluator
from .globalization import FeasibilityRestoration, make_line_search
from .initialization import (
    FeasibilityPresolve,
    IterateCenterer,
    MultiplierInitializer,
    NLPScaling,
)
from .iteration_logger import OptimizationLogger
from .ipm_state import InteriorPointState
from .newton_direction import NewtonStep
from .solvers import InertiaCorrector, make_solver


class Optimizer:
    """Primal-dual interior-point optimizer."""

    def __init__(
        self,
        model=None,
        x=None,
        problem=None,
        comm=None,
        **kwargs,
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        model : Model
            The amigo model to optimize
        x : array-like, optional
            Initial point
        comm : MPI communicator, optional
            For distributed optimization
        """

        if "solver" in kwargs:
            warnings.warn("Set the solver through the options")

        # Set the model and problem
        if model is not None:
            self.model = model
            self.problem = self.model.get_problem()
        elif problem is not None:
            self.model = None
            self.problem = problem
        else:
            raise ValueError("Must provide a model or a problem instance")

        # Set the design vector
        if isinstance(x, ModelVector):
            self.x = x.get_vector()
        elif isinstance(x, Vector):
            self.x = x
        else:
            self.x = self.problem.create_vector()

        self.x_init = self.problem.get_initial_point()
        self.lower = self.problem.get_lower()
        self.upper = self.problem.get_upper()

        # The MPI communicator
        self.comm = comm

        # Set up the vectors
        self._create_interior_point_backend()

        # Objects created during optimization that are saved
        self.evaluator = None
        self.state = None
        self.solver = None

        return

    def _create_interior_point_backend(self):
        """Create the C++ InteriorPointOptimizer backend and slack mapping."""
        data_vec = self.problem.get_data_vector()
        self.x.copy_host_to_device()
        self.x_init.copy_host_to_device()
        self.lower.copy_host_to_device()
        self.upper.copy_host_to_device()
        data_vec.copy_host_to_device()

        self.optimizer = InteriorPointOptimizer(self.problem)
        return

    def get_options(self, options={}):
        return get_default_options(options)

    def get_optimized_point(self):
        return ModelVector(self.model, x=self.x)

    def optimize(self, options={}):
        """
        The set up of the new class structure:

        All data about the current state of the optimizer (scalars, vectors, Hessian etc.) are
        stored in the InteriorPointState object. This contains all info about the current design
        point.

        Evaluator is responsible for evaluating the quantities of interest (gradient, Hessian etc.)
        for the current state and trial points that may become the current state. Each algorithm
        is responsible for updating the state object so that its internal state remains consistent.

        FilterLineSearch performs a filter line search
        """

        # Check and normalize the options dictionary for internal use
        options = self.get_options(options=options)

        # The logger clock starts after the solver is built, so time setup here
        _t_optimize_start = time.perf_counter()
        _solver_build_time = 0.0

        # Relax bounds from the originals so repeated calls do not compound
        if options["bound_relax_factor"] > 0.0:
            self.optimizer.relax_bounds(
                options["bound_relax_factor"], options["constr_viol_tol"]
            )

        # Continuation control object, if any
        continuation_control = options["continuation_control"]

        # Class for evaluating problem-specific quantities
        self.evaluator = Evaluator(self.problem, self.optimizer)

        # The interior point state object contains information about the design point, the
        # gradient and the Hessian of the Lagrangian
        self.state = InteriorPointState(self.x, options, self.problem, self.optimizer)

        # Warm start: begin at a barrier consistent with the near-optimal point
        if options["warm_start"]:
            self.state.mu = options["warm_start_mu_init"]

        # Break the previous call's reference cycles to release the C++ payloads
        gc.collect()

        # No solver reuse across calls, and this runs the symbolic factorization
        _t_solver_build = time.perf_counter()
        self.solver = make_solver(options, self.state, self.problem, self.optimizer)
        _solver_build_time = time.perf_counter() - _t_solver_build

        # The inertia correction
        inertia_corrector = InertiaCorrector(options, self.problem, self.optimizer)

        # Initialize the line search algorithm
        line_search = make_line_search(options, self.problem, self.optimizer)

        # Allocate the Newton step
        newton_step = NewtonStep(options, self.problem, self.optimizer)

        last_restoration_iter = 0
        # Feasibility restoration phase algorithm
        feasible_resto = FeasibilityRestoration(options, self.problem, self.optimizer)
        restoration_count = 0

        # Initialize the barrier strategy correction algorithm
        barrier_strategy = make_barrier_strategy(options, self.problem, self.optimizer)

        # Initialize the convergence check
        check = ConvergenceCheck(options, self.problem, self.optimizer)

        # Initialize the logger. The logger takes in additional objects that may
        # provide logging info via "obj.get_log_info()"
        objs = [line_search, inertia_corrector]
        if hasattr(self.solver, "add_log_info"):
            objs.append(self.solver)
        logger = OptimizationLogger(objs, options, self.problem, self.optimizer)

        # pre_loop_time precedes the logger clock, solver_build_time is within it
        logger.opt_data["solver_build_time"] = _solver_build_time
        logger.opt_data["pre_loop_time"] = time.perf_counter() - _t_optimize_start

        # Set the initial point
        self.x.copy(self.x_init)

        # Scale at the start point so the rest of the algorithm runs scaled
        self.nlp_scaling = NLPScaling(options, self.problem, self.optimizer)
        self.nlp_scaling.compute(self.evaluator, self.state)

        # Center the iterate on the central path at the initial barrier
        centerer = IterateCenterer(options, self.model, self.problem, self.optimizer)
        centerer.center(
            self.evaluator, self.state, keep_multipliers=options["warm_start"]
        )

        # Initialize the multipliers
        multiplier_init = MultiplierInitializer(
            options, self.model, self.problem, self.optimizer
        )
        multiplier_init.initialize_multipliers(self.evaluator, self.solver, self.state)

        # Set the initial status
        status = CONTINUE

        def attempt_restoration(augment_filter):
            """Run the restoration phase and repair the resume state.

            Returns the RestorationInfo, or None when restoration is
            disabled or its budget is exhausted. The filter is augmented
            with the failure point only on line-search failures: the
            NaN/divergence verdicts may carry non-finite phi values.
            """
            nonlocal restoration_count, last_restoration_iter
            if not (
                options["feasibility_restoration"]
                and restoration_count < options["max_restorations"]
            ):
                return None
            if augment_filter:
                line_search.augment_filter(
                    self.state.barrier_objective,
                    self.state.con_infeasibility,
                )
            resto_info = feasible_resto.restore(
                self.solver, self.evaluator, self.state, line_search
            )
            if resto_info.success:
                restoration_count += 1
                last_restoration_iter = self.state.iter
                line_search.reset_after_restoration()
                check.reset_step_watchdog(self.state)
                # Least-squares multipliers at the restored point
                multiplier_init.compute_least_squares_multipliers(
                    self.evaluator, self.solver, self.state
                )
            return resto_info

        # Optional feasible and centered starting point
        presolve = FeasibilityPresolve(options)
        presolve.run(
            feasible_resto,
            multiplier_init,
            centerer,
            self.solver,
            self.evaluator,
            self.state,
            line_search,
        )

        # Initialize the barrier strategy prior to optimization
        barrier_strategy.initialize(self.evaluator, self.state)

        max_iters = options["max_iterations"]
        for counter in range(max_iters):
            # Update the iteration counter
            self.state.iter = counter

            # Steady progress refills the restoration budget
            refresh = options["restoration_budget_refresh_iters"]
            if (
                refresh > 0
                and restoration_count > 0
                and counter - last_restoration_iter >= refresh
            ):
                restoration_count -= 1
                last_restoration_iter = counter

            # Evaluate the objective and barrier function
            self.evaluator.evaluate_objective_and_infeasibility(self.state)

            # Evaluate the residuals for the convergence check
            self.evaluator.evaluate_residual(self.state)

            # Check for convergence based on the initial point
            status = check.test_convergence(self.evaluator, self.state)

            # Log the information about the iteration and the status of all the
            # internal objects within the optimizer
            logger.log_iteration(status, self.state)

            # Restoration-needed and diverging states attempt restoration first
            if status != CONTINUE:
                if status in (RESTORATION_NEEDED, DIVERGED):
                    resto_info = attempt_restoration(augment_filter=False)
                    if resto_info is not None and resto_info.success:
                        continue
                    if resto_info is not None and resto_info.infeasible:
                        status = LOCALLY_INFEASIBLE
                        logger.log_iteration(status, self.state)
                break

            # Callback for the continuation control
            if continuation_control is not None:
                continuation_control(self.state)

            # Perform an update of the barrier parameter prior to any factorization or step
            barrier_info = barrier_strategy.update_barrier(self.evaluator, self.state)

            # Let the line search object determine if a reset is appropriate based on the barrier parameter
            # update. For instance, this call may reset the filter
            line_search.reset_on_new_barrier(self.state, barrier_info)

            # Factor the KKT system considering the inertia
            # TODO: Implement a inertia info class
            factor_ok = inertia_corrector.factor_for_inertia(
                self.solver, self.evaluator, self.state
            )

            do_feasible_resto = True
            if factor_ok:
                # Compute the direction and store in self.state.step. This should be a descent direction
                # because of the inertia check
                newton_step.compute_step(self.solver, self.evaluator, self.state)

                # Using the same factorization and solver, assess whether a correction step is required
                # and compute it
                barrier_strategy.add_step_correction(
                    self.solver, self.evaluator, self.state
                )

                # Perform a line search along the step direction
                line_search_info = line_search.line_search(
                    self.solver, self.evaluator, self.state
                )

                # Assess what happened after the line search
                barrier_strategy.update_after_line_search(
                    line_search_info, self.evaluator, self.state
                )

                if line_search_info.success:
                    do_feasible_resto = False

                    # kappa_Sigma safeguard: clip bound duals toward mu/gap
                    kappa_sigma = options["kappa_sigma"]
                    if kappa_sigma > 0.0:
                        self.optimizer.correct_bound_multipliers(
                            self.state.mu, kappa_sigma, self.state.current
                        )

                    # Refresh multipliers near feasibility to prevent dual divergence
                    multiplier_init.recompute_multipliers(
                        self.evaluator, self.solver, self.state
                    )

                    # Steps accepted at near-zero alpha while still infeasible
                    if check.test_small_alpha_stall(self.state):
                        do_feasible_resto = True

            # On failure, filter the point and restore feasibility, else terminate
            if do_feasible_resto:
                resto_info = attempt_restoration(augment_filter=True)
                if resto_info is not None and resto_info.success:
                    continue
                # Keep an acceptable point when restoration fails
                if check.point_acceptable(self.evaluator, self.state):
                    status = CONVERGED_ACCEPTABLE
                elif resto_info is not None and resto_info.infeasible:
                    status = LOCALLY_INFEASIBLE
                else:
                    status = RESTORATION_NEEDED
                logger.log_iteration(status, self.state)
                break

        else:
            # The optimization for loop completed normally, so we did not converge
            # Check the convergence status
            self.state.iter = max_iters
            status = check.test_convergence(self.evaluator, self.state)

            # Log the iteration
            logger.log_iteration(status, self.state)

        return logger.get_data()

    def compute_output(self, output: ModelVector | Vector | None = None):
        """Evaluate model outputs at the final iterate."""

        if output is None:
            if self.model is not None:
                output = self.model.create_output_vector()
            else:
                output = self.problem.create_output_vector()

        if isinstance(output, ModelVector):
            out_vec = output.get_vector()
        elif isinstance(output, Vector):
            out_vec = output
        else:
            raise RuntimeError(
                "output vector is not an instance of Vector or ModelVector"
            )

        self.problem.compute_output(self.x, out_vec)
        return output

    def compute_post_opt_derivatives(
        self,
        of: str | list[str] | None = None,
        wrt: str | list[str] | None = None,
        method="adjoint",
    ):
        """
        Compute the post-optimality derivatives of the outputs.

        Parameters
        ----------
        of, wrt : list of str
            Output and input variable names.
        method : {"adjoint", "direct"}
            Use adjoint when len(of) < len(wrt), direct otherwise.
        """

        if self.model is None:
            raise RuntimeError("Model cannot be None for post-optimiality derivatives")
        if self.state is None:
            raise RuntimeError("Call optimize() before compute_post_opt_derivatives")

        if of is None:
            _, _, _, of = self.model.get_names()
        if wrt is None:
            _, _, wrt, _ = self.model.get_names()

        of_indices, of_map = self.model.get_indices_and_map(of)
        wrt_indices, wrt_map = self.model.get_indices_and_map(wrt)

        dfdx = np.zeros((len(of_indices), len(wrt_indices)))

        out_wrt_input = self.problem.create_output_jacobian_wrt_input()
        self.problem.output_jacobian_wrt_input(self.x, out_wrt_input)
        out_wrt_input = tocsr(out_wrt_input)

        out_wrt_data = self.problem.create_output_jacobian_wrt_data()
        self.problem.output_jacobian_wrt_data(self.x, out_wrt_data)
        out_wrt_data = tocsr(out_wrt_data)

        grad_wrt_data = self.problem.create_gradient_jacobian_wrt_data()
        self.problem.gradient_jacobian_wrt_data(self.x, grad_wrt_data)
        grad_wrt_data = tocsr(grad_wrt_data)

        self.evaluator.evaluate_hessian(self.state)
        self.evaluator.evaluate_diagonal(self.state)

        self.solver.factor(self.state.hessian, self.state.diagonal)

        psi = self.problem.create_vector()
        res = self.problem.create_vector()

        if method == "adjoint":
            for i in range(len(of_indices)):
                idx = of_indices[i]
                res.get_array()[:] = -out_wrt_input[idx, :].toarray()
                self.problem.zero_fixed_entries(res)
                self.solver.solve(res, psi)
                adjx = grad_wrt_data.T @ psi.get_array()
                dfdx[i, :] = out_wrt_data[idx, wrt_indices] + adjx[wrt_indices]
        elif method == "direct":
            grad_wrt_data = grad_wrt_data.tocsc()
            for i in range(len(wrt_indices)):
                idx = wrt_indices[i]
                res.get_array()[:] = -grad_wrt_data[:, idx].toarray().flatten()
                self.problem.zero_fixed_entries(res)
                self.solver.solve(res, psi)
                dirx = out_wrt_input @ psi.get_array()
                dfdx[:, i] = out_wrt_data[of_indices, idx] + dirx[of_indices]

        return dfdx, of_map, wrt_map
