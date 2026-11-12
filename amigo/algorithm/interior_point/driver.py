"""Primal-dual interior-point optimizer.

The Optimizer class composes the sibling modules into the main loop.
Each iteration checks convergence, updates the barrier, and takes a Newton step.
A line search accepts the step or hands the iterate to restoration.
"""

import time
import warnings
import numpy as np

# Raw pybind11 classes
from ...amigo import InteriorPointOptimizer, Vector

# Import from amigo classes
from ...modeling.model import ModelVector
from ...utils import tocsr

# Optimizer imports from algorithm classes
from ..barrier_strategy import barrier_minimum, make_barrier_strategy
from .convergence_check import (
    ConvergenceCheck,
    CONTINUE,
    CONVERGED,
    CONVERGED_ACCEPTABLE,
    CONVERGED_OBJECTIVE,
    DEGENERATE_DUALS,
    INVALID_NUMBER,
    MAX_ITERATIONS,
    RESTORATION_NEEDED,
    LOCALLY_INFEASIBLE,
    TIME_LIMIT,
    TINY_STEP,
)
from ..default_options import get_default_options
from .evaluator import Evaluator
from ..globalization import make_line_search, make_restoration
from ..initialization import (
    FeasibilityPresolve,
    IterateCenterer,
    MultiplierInitializer,
    NLPScaling,
    hessian_perturbation_rows,
)
from ..initialization.multiplier_initialization import FUTILE_FRACTION
from ..monitoring.iteration_logger import OptimizationLogger
from .step_contract import StepContract
from .polish import ConstraintPolish
from .state import InteriorPointState
from .newton_direction import NewtonStep
from ..solvers import InertiaCorrector, make_solver


def _as_vector(x):
    """Return the Vector behind a ModelVector or Vector, or None for None."""
    if x is None:
        return None
    if isinstance(x, ModelVector):
        return x.get_vector()
    if isinstance(x, Vector):
        return x
    raise TypeError(f"Expected a ModelVector or Vector, got {type(x).__name__}")


def _apply_assigned(src, target):
    """Copy src into target, only the entries set in it for a blank ModelVector."""
    vec = _as_vector(src)
    if vec is None or _same_storage(vec, target):
        return
    mask = src.get_assigned() if isinstance(src, ModelVector) else None
    if mask is None:
        target.copy(vec)
    else:
        target.get_array()[mask] = vec.get_array()[mask]


def _same_storage(a, b):
    """Return True when two vectors share their host storage."""
    return a.get_array().__array_interface__["data"][0] == (
        b.get_array().__array_interface__["data"][0]
    )


# Barrier a warm start resumes at when it carries none and the options name none
WARM_START_MU = 1e-4


class WarmStart:
    """A point with its multipliers and the bound duals that a later solve starts from.

    The vectors are laid out like the point of the model they belong to, so
    a warm start serves any model built the same way. The barrier is the one
    the solve that produced the point ended at, or None when it is unknown.
    """

    def __init__(self, point, zl, zu, mu=None):
        self.point = point
        self.zl = zl
        self.zu = zu
        self.mu = mu


class Optimizer:
    """Primal-dual interior-point optimizer."""

    def __init__(
        self,
        model=None,
        x=None,
        problem=None,
        comm=None,
        lower=None,
        upper=None,
        warm_start=None,
        **kwargs,
    ):
        """Initialize the optimizer.

        Every solve starts from the model initial point and writes the solution into x.
        The entries set in x, lower, and upper write into the model start and bounds.
        A warm start replaces the model start and the bound duals before x goes on top.

        Parameters
        ----------
        model : Model
            The amigo model to optimize
        x : ModelVector or Vector, optional
            Starting guess for the entries set in it, then the solution
        problem : OptimizationProblem, optional
            The problem to optimize when no model is given
        comm : MPI communicator, optional
            For distributed optimization
        lower, upper : ModelVector or Vector, optional
            Bounds for the entries set in them
        warm_start : WarmStart, optional
            Point, multipliers, and bound duals of an earlier solve of a model
            laid out the same way
        """

        if "solver" in kwargs:
            warnings.warn("Set the solver through the options")
            kwargs.pop("solver")
        if kwargs:
            raise TypeError(f"Unexpected Optimizer arguments {sorted(kwargs)}")

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
        self.x = _as_vector(x)
        if self.x is None:
            self.x = self.problem.create_vector()

        self.lower = self.problem.get_lower()
        self.upper = self.problem.get_upper()
        _apply_assigned(lower, self.lower)
        _apply_assigned(upper, self.upper)

        # A warm start replaces the model start and the guesses set in x go on top of it
        self.x_init = self.problem.get_initial_point()
        if warm_start is not None:
            self.x_init.copy(_as_vector(warm_start.point))
        _apply_assigned(x, self.x_init)

        # Keep the start separate from the solution vector
        if _same_storage(self.x, self.x_init):
            self.x_init = self.problem.create_vector()
            self.x_init.copy(self.x)

        # Variables with equal bounds become fixed before the backend reads them
        self.problem.update_fixed_variables()

        # The MPI communicator
        self.comm = comm

        # Set up the vectors
        self._create_interior_point_backend()

        # Objects created during optimization that are saved
        self.evaluator = None
        self.state = None
        self.solver = None

        # Unscaled bound duals of the last solve and the barrier it ended at, for a warm start
        self.last_duals = None
        self.last_mu = None

        # A given warm start supplies the bound duals and the barrier of the first solve
        self.warm = warm_start is not None
        if self.warm:
            self.set_bound_duals(warm_start.zl, warm_start.zu)
            self.last_mu = warm_start.mu

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

    def get_bound_duals(self):
        """Lower and upper bound duals of the last solve, laid out like the point and zero away from the primals."""
        if self.last_duals is None:
            raise RuntimeError("bound duals exist only after a solve")
        primal = self.problem.get_primal_indices()
        duals = []
        for z in (self.last_duals.get_zl(), self.last_duals.get_zu()):
            full = self.problem.create_vector()
            full.zero()
            full.set_values_at(primal, z)
            full.copy_device_to_host()
            duals.append(
                full if self.model is None else ModelVector(self.model, x=full)
            )
        return tuple(duals)

    def get_warm_start(self):
        """Copies of the point with its multipliers, of the bound duals, and the barrier of the last solve."""
        zl, zu = self.get_bound_duals()
        self.x.copy_device_to_host()
        point = self.problem.create_vector()
        point.copy(self.x)
        if self.model is not None:
            point = ModelVector(self.model, x=point)
        return WarmStart(point, zl, zu, self.last_mu)

    def set_bound_duals(self, zl, zu):
        """Bound duals that the next warm start begins from, laid out like the point."""
        primal = self.problem.get_primal_indices()
        if self.last_duals is None:
            self.last_duals = self.optimizer.create_opt_vector()
        for full, z in ((zl, self.last_duals.get_zl()), (zu, self.last_duals.get_zu())):
            vec = _as_vector(full)
            vec.copy_host_to_device()
            vec.get_values_at(primal, z)

    def optimize(self, options={}):
        """Run the interior point method from the initial point and return the log data.

        The InteriorPointState holds the current point with its gradient and
        Hessian, the Evaluator fills them for the current or a trial point,
        and each algorithm keeps the state consistent when it moves the point.
        """

        # Check and normalize the options dictionary for internal use
        options = self.get_options(options=options)

        # A given warm start holds for every part of the solve that reads the option
        if self.warm:
            options["warm_start"] = True

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

        # A warm start resumes at the barrier the options name, else the one it carries
        warm_mu = options["warm_start_mu_init"]
        if warm_mu is None:
            warm_mu = WARM_START_MU if self.last_mu is None else self.last_mu

        # The interior point state object contains information about the design point, the
        # gradient and the Hessian of the Lagrangian
        self.state = InteriorPointState(
            self.x, options, self.problem, self.optimizer, warm_mu=warm_mu
        )

        # The wall time budget runs from the start of the call
        self.state.deadline = _t_optimize_start + options["max_wall_time"]

        # The solver keeps its symbolic analysis across solves with the same options
        _t_solver_build = time.perf_counter()
        if self.solver is None or not self.solver.reset(options, self.state):
            # Release the previous factorization before the new symbolic analysis
            self.solver = None
            self.solver = make_solver(options, self.state, self.problem, self.optimizer)
        _solver_build_time = time.perf_counter() - _t_solver_build

        # The free variables of the model carry the inertia shift when the option names them
        corrector_options = options
        if isinstance(options["hessian_perturbation_indices"], str):
            if self.model is None:
                raise ValueError(
                    "The free variables need the model that built the problem"
                )
            corrector_options = dict(options)
            corrector_options["hessian_perturbation_indices"] = (
                hessian_perturbation_rows(self.model)
            )

        # The inertia correction
        inertia_corrector = InertiaCorrector(
            corrector_options, self.problem, self.optimizer
        )

        # Initialize the line search algorithm
        line_search = make_line_search(options, self.problem, self.optimizer)

        # Allocate the Newton step
        newton_step = NewtonStep(options, self.problem, self.optimizer)

        # Feasibility restoration phase algorithm
        feasible_resto = make_restoration(
            corrector_options, self.problem, self.optimizer
        )

        # Initialize the barrier strategy correction algorithm
        barrier_strategy = make_barrier_strategy(options, self.problem, self.optimizer)

        # Initialize the convergence check
        check = ConvergenceCheck(options, self.problem, self.optimizer)

        # The dual step contract that doses the stabilization after a breach
        contract = StepContract(options, self.problem)
        polish = ConstraintPolish(options)

        # Initialize the logger. The logger takes in additional objects that may
        # provide logging info via "obj.get_log_info()"
        objs = [line_search, inertia_corrector, feasible_resto, contract]
        if hasattr(self.solver, "add_log_info"):
            objs.append(self.solver)
        logger = OptimizationLogger(objs, options, self.problem, self.optimizer)

        # pre_loop_time precedes the logger clock, solver_build_time is within it
        logger.opt_data["solver_build_time"] = _solver_build_time
        logger.opt_data["pre_loop_time"] = time.perf_counter() - _t_optimize_start

        # Set the initial point with the fixed variables at their bounds
        self.x.copy(self.x_init)
        self.optimizer.set_fixed_values(self.x)

        # The elastic parts and their weights, set from the option before the scaling reads the gradient
        if self.model is not None:
            elastic_parts, elastic_weights = self.model.get_elastic()
        else:
            elastic_parts = elastic_weights = np.zeros(0, dtype=int)
        if elastic_parts.size > 0:
            data = self.problem.get_data_vector()
            data.copy_device_to_host()
            data.get_array()[elastic_weights] = options["elastic_weight"]
            data.copy_host_to_device()

        # The optimality scaling leaves out the duals of the elastic parts
        self.evaluator.set_elastic_parts(self._dual_positions(elastic_parts))

        # Scale at the start point so the rest of the algorithm runs scaled
        self.nlp_scaling = NLPScaling(options, self.problem, self.optimizer)
        self.nlp_scaling.compute(self.evaluator, self.state)

        # A warm start resumes from the bound duals of the previous solve
        warm = options["warm_start"]
        keep_bound_duals = warm and self.last_duals is not None
        if keep_bound_duals:
            self.state.current.get_zl().copy(self.last_duals.get_zl())
            self.state.current.get_zu().copy(self.last_duals.get_zu())
        self.nlp_scaling.scale_point(self.state)

        # Center the iterate on the central path at the initial barrier
        centerer = IterateCenterer(options, self.model, self.problem, self.optimizer)
        centerer.center(
            self.evaluator,
            self.state,
            warm=warm,
            keep_bound_duals=keep_bound_duals,
        )

        # Initialize the multipliers
        multiplier_init = MultiplierInitializer(
            options, self.model, self.problem, self.optimizer
        )
        multiplier_init.initialize_multipliers(self.evaluator, self.solver, self.state)

        # Set the initial status
        status = CONTINUE

        def attempt_restoration():
            """Run the restoration phase after a failed step and return its info, or None when it is disabled."""
            if not options["feasibility_restoration"]:
                return None

            # The barrier value must match the current mu before the failure point enters the filter
            self.evaluator.evaluate_objective_and_infeasibility(self.state)
            line_search.augment_filter(
                self.state.barrier_objective,
                self.state.con_infeasibility,
            )
            resto_info = feasible_resto.restore(
                self.solver, self.evaluator, self.state, line_search
            )
            if resto_info.success:
                line_search.reset_after_restoration()
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

        elastic_updates = 0

        max_iters = options["max_iterations"]
        tiny_steps = 0
        for counter in range(max_iters):
            # Update the iteration counter
            self.state.iter = counter

            # Evaluate the objective and barrier function
            self.evaluator.evaluate_objective_and_infeasibility(self.state)

            # Evaluate the residuals for the convergence check
            self.evaluator.evaluate_residual(self.state)

            # Check for convergence based on the initial point
            status = check.test_convergence(self.evaluator, self.state)

            # An acceptable stop with the constraints above the tolerance polishes them with normal steps first
            self.state.normal_step = False
            if status == CONVERGED_ACCEPTABLE and polish.claim_step(self.state):
                self.state.normal_step = True
                status = CONTINUE

            # Past the wall time budget the solve stops at the current point
            if status == CONTINUE and time.perf_counter() > self.state.deadline:
                status = TIME_LIMIT

            # The iterations of the restoration count against the budget of the solve
            spent = counter + feasible_resto.num_iterations
            if status == CONTINUE and spent >= max_iters:
                status = MAX_ITERATIONS

            # Log the information about the iteration and the status of all the
            # internal objects within the optimizer
            logger.log_iteration(status, self.state)

            # The hook sees every logged iterate with its count
            if options["iteration_callback"] is not None:
                point = self.x if self.model is None else self.get_optimized_point()
                options["iteration_callback"](counter, point)

            # Convergence, divergence, and invalid numbers all end the solve
            if status != CONTINUE:
                # A converged point with elastic parts left grows their weight and goes on
                grown = False
                if (
                    status in (CONVERGED, CONVERGED_ACCEPTABLE)
                    and elastic_parts.size > 0
                ):
                    if elastic_updates < options["elastic_max_updates"]:
                        grown = self._grow_elastic_weights(
                            options,
                            line_search,
                            elastic_parts,
                            elastic_weights,
                            force=True,
                        )

                    # Parts left at the last weight leave their rows open, so the point is infeasible
                    if not grown and self._elastic_parts_left(options, elastic_parts):
                        status = LOCALLY_INFEASIBLE
                        logger.mark_status(status)

                if grown:
                    elastic_updates += 1
                else:
                    if status == INVALID_NUMBER and self.state.comm_rank == 0:
                        names = self._invalid_entries()
                        if names:
                            print(f"  Non-finite values at {', '.join(names)}")
                        for line in self._invalid_derivatives():
                            print(f"  {line}")
                    break

            # Callback for the continuation control
            if continuation_control is not None:
                continuation_control(self.state)

            # Perform an update of the barrier parameter prior to any factorization or step
            barrier_info = barrier_strategy.update_barrier(self.evaluator, self.state)

            # Let the line search object determine if a reset is appropriate based on the barrier parameter
            # update. For instance, this call may reset the filter
            line_search.reset_on_new_barrier(self.state, barrier_info)

            # A new barrier level asks the duals of the elastic parts whether their weight is exact
            if (
                elastic_parts.size > 0
                and barrier_info.new_barrier
                and elastic_updates < options["elastic_max_updates"]
                and self._grow_elastic_weights(
                    options, line_search, elastic_parts, elastic_weights, force=False
                )
            ):
                elastic_updates += 1

            # Factor the KKT system considering the inertia
            # TODO: Implement a inertia info class
            factor_ok = inertia_corrector.factor_for_inertia(
                self.solver, self.evaluator, self.state
            )

            # A failed factorization names the entries of the KKT matrix that are not finite
            if inertia_corrector.failure and self.state.comm_rank == 0:
                for line in self._invalid_kkt():
                    print(f"  {line}")

            # Test whether the duals alone block the barrier after a step that did not cut them
            dual_blocked = False
            dual_gate = 0.0
            if (
                options["dual_complete_schedule"]
                and not barrier_info.new_barrier
                and contract.pre_step_dual is not None
                and self.state.dual_infeas > FUTILE_FRACTION * contract.pre_step_dual
            ):
                kappa_eps = options["barrier_tol_factor"]
                dual, primal, comp = self.optimizer.compute_kkt_error(
                    self.state.mu, self.state.current, self.state.gradient
                )
                s_d, s_c = self.evaluator.compute_optimality_scaling(self.state)
                dual_gate = kappa_eps * self.state.mu
                dual_blocked = (
                    dual / s_d > dual_gate
                    and primal <= dual_gate
                    and comp / s_c <= dual_gate
                )

            do_feasible_resto = True
            if factor_ok:
                # A polish step moves onto the constraints, else the barrier strategy or the Newton step gives a descent direction
                if self.state.normal_step:
                    newton_step.compute_normal_step(
                        self.solver, self.evaluator, self.state
                    )
                elif barrier_strategy.computes_step():
                    barrier_strategy.compute_step(
                        self.solver, self.evaluator, self.state
                    )
                else:
                    newton_step.compute_step(self.solver, self.evaluator, self.state)

                # The objective change the full step predicts, in the objective's own units
                self.state.pred_obj = (
                    self.evaluator.objective_directional_derivative(self.state)
                    / self.state.obj_scale
                )

                # A feasible centered iterate whose full step cannot move the objective ends the solve
                if check.objective_converged(self.state):
                    status = CONVERGED_OBJECTIVE
                    logger.mark_status(status)
                    grown = False
                    if elastic_parts.size > 0:
                        if elastic_updates < options["elastic_max_updates"]:
                            grown = self._grow_elastic_weights(
                                options,
                                line_search,
                                elastic_parts,
                                elastic_weights,
                                force=True,
                            )

                        # Parts left at the last weight leave their rows open, so the point is infeasible
                        if not grown and self._elastic_parts_left(
                            options, elastic_parts
                        ):
                            status = LOCALLY_INFEASIBLE
                            logger.mark_status(status)

                    if grown:
                        elastic_updates += 1
                        continue
                    break

                # Perform a line search along the step direction
                contract.before_step(self.state)
                line_search_info = line_search.line_search(
                    self.solver, self.evaluator, self.state
                )

                # Assess what happened after the line search
                barrier_strategy.update_after_line_search(
                    line_search_info, self.evaluator, self.state
                )

                # A second tiny step in a row with mu at its floor cannot make progress
                tiny_steps = tiny_steps + 1 if line_search_info.tiny_step else 0
                if tiny_steps >= 2 and self.state.mu <= barrier_minimum(
                    options, self.state
                ):
                    if check.point_acceptable(self.evaluator, self.state):
                        status = CONVERGED_ACCEPTABLE
                    else:
                        status = TINY_STEP
                    logger.log_iteration(status, self.state)
                    break
                if line_search_info.tiny_step:
                    barrier_strategy.force_decrease = True

                if line_search_info.success:
                    do_feasible_resto = False

                    # kappa_Sigma safeguard: clip bound duals toward mu/gap
                    kappa_sigma = options["kappa_sigma"]
                    if kappa_sigma > 0.0:
                        self.optimizer.correct_bound_multipliers(
                            self.state.mu, kappa_sigma, self.state.current
                        )

                    # Judge the dual promise of the step before any refresh repairs it
                    contract.after_step(
                        self.evaluator,
                        self.state,
                        line_search_info.alpha_primal,
                        inertia_corrector.last_delta_w,
                        inertia_corrector.last_delta_c,
                    )

                    # Refresh multipliers near feasibility to prevent dual divergence
                    helped = multiplier_init.recompute_multipliers(
                        self.evaluator,
                        self.solver,
                        self.state,
                        dual_prev=contract.pre_step_dual,
                        force=dual_blocked,
                    )
                    if dual_blocked and helped is False:
                        # The reduced gradient the multipliers of the refresh leave without their cap and their size
                        residual = multiplier_init.estimate_residual
                        size = multiplier_init.estimate_size
                        s_d, _ = self.evaluator.compute_optimality_scaling(self.state)

                        # A futile refresh at a converged primal certifies degenerate duals when only multipliers past the cap close them
                        if (
                            self.state.kkt_error > options["convergence_tolerance"]
                            and self.state.mu <= barrier_minimum(options, self.state)
                            and check.primal_converged(self.state)
                        ):
                            stationary = (
                                residual is not None
                                and residual / s_d <= options["convergence_tolerance"]
                            )
                            if stationary:
                                logger.log_iteration(DEGENERATE_DUALS, self.state)
                                status = DEGENERATE_DUALS
                                break
                            if self.state.comm_rank == 0:
                                print(
                                    f"  Certificate refused: multipliers up to {size:.2e} leave "
                                    f"a dual residual of {residual:.2e}, the point is not stationary"
                                )

                        # The duals are complete when multipliers past the cap close them to the gate or the step leaves the objective within it
                        closed = residual is not None and residual / s_d <= dual_gate
                        moved = abs(self.state.pred_obj) * self.state.obj_scale
                        if closed or moved <= dual_gate:
                            barrier_strategy.force_decrease = True
                            if self.state.comm_rank == 0:
                                print(
                                    f"  Dual complete: forcing the barrier below "
                                    f"mu={self.state.mu:.2e}"
                                )
                        elif self.state.comm_rank == 0 and options["verbose_barrier"]:
                            left = float("nan") if residual is None else residual
                            print(
                                f"  Dual open: a reduced gradient of {left:.2e} remains "
                                f"and the step moves the objective by {moved:.2e}"
                            )

                    # Steps accepted at near-zero alpha while still infeasible
                    if check.test_small_alpha_stall(
                        self.state, line_search_info.alpha_primal
                    ):
                        do_feasible_resto = True

            # A failed step stops at an acceptable point, otherwise restoration takes over
            if do_feasible_resto:
                if check.point_acceptable(self.evaluator, self.state):
                    status = CONVERGED_ACCEPTABLE
                    logger.log_iteration(status, self.state)
                    break
                resto_info = attempt_restoration()
                if resto_info is not None and resto_info.success:
                    continue

                # A call that fails ends the solve, under the limit that stopped it when one did
                status = RESTORATION_NEEDED
                if resto_info is not None:
                    if resto_info.infeasible:
                        status = LOCALLY_INFEASIBLE
                    elif resto_info.out_of_time:
                        status = TIME_LIMIT
                    elif resto_info.out_of_iterations:
                        status = MAX_ITERATIONS
                logger.log_iteration(status, self.state)
                break

        else:
            # The optimization for loop completed normally, so we did not converge
            # Check the convergence status
            self.state.iter = max_iters
            status = check.test_convergence(self.evaluator, self.state)

            # Running out of iterations short of convergence is its own outcome
            if status == CONTINUE:
                status = MAX_ITERATIONS

            # Log the iteration
            logger.log_iteration(status, self.state)

        # The summary and the returned data carry the final status on every exit
        logger.finish(status, self.state)

        # Return the unscaled solution and keep its bound duals and barrier for a warm start
        self.nlp_scaling.unscale_point(self.state)
        if self.last_duals is None:
            self.last_duals = self.optimizer.create_opt_vector()
        self.last_duals.get_zl().copy(self.state.current.get_zl())
        self.last_duals.get_zu().copy(self.state.current.get_zu())
        self.last_mu = self.state.mu

        return logger.get_data()

    def _dual_positions(self, parts):
        """Positions of the bound duals of the given variables"""
        primal = np.asarray(self.problem.get_primal_indices().get_array())
        size = self.state.current.get_solution().get_array().size
        position = np.full(size, -1)
        position[primal] = np.arange(primal.size)
        return position[parts]

    def _elastic_parts_sum(self, parts):
        """Sum of the elastic parts, the violation their rows still carry"""
        x = self.state.current.get_solution()
        x.copy_device_to_host()
        values = np.asarray(x.get_array())
        return float(np.abs(values[parts]).sum())

    def _elastic_parts_left(self, options, parts):
        """True when the elastic parts carry more than the violation tolerance"""
        return self._elastic_parts_sum(parts) > options["constr_viol_tol"]

    def _grow_elastic_weights(self, options, line_search, parts, weights, force):
        """Grow the weight of the elastic parts when their bound duals say the penalty is not exact.

        The scaled bound dual of a part is the scaled weight minus the
        multiplier of its row, so a part whose dual falls below the floor
        times the weight sits on a row whose multiplier has reached the
        weight and the part stays nonzero at the solution. With force the
        parts left at a converged point decide alone. Returns True when the
        weight grew.
        """
        total = self._elastic_parts_sum(parts)
        if total <= options["constr_viol_tol"]:
            return False

        data = self.problem.get_data_vector()
        data.copy_device_to_host()
        entries = data.get_array()
        weight = float(np.max(entries[weights]))

        # The duals of the parts against their scaled weight
        zl = self.state.current.get_zl()
        zl.copy_device_to_host()
        positions = self._dual_positions(parts)
        scaled_weight = weight * self.state.obj_scale
        floor = float(np.min(np.asarray(zl.get_array())[positions])) / scaled_weight
        if not force and floor >= options["elastic_dual_floor"]:
            return False

        # The duals mean nothing while the dual error is still of the order of the weight
        if (
            not force
            and self.state.dual_infeas > options["elastic_dual_floor"] * scaled_weight
        ):
            return False

        growth = options["elastic_weight_growth"]
        entries[weights] *= growth
        data.copy_host_to_device()
        self.state.invalidate()
        line_search.reset_after_restoration()
        if self.state.comm_rank == 0:
            print(
                f"  Elastic: parts sum {total:.2e}, smallest dual {floor:.2e} of the weight "
                f"{weight:.1e}, raising the weight to {weight * growth:.1e}"
            )
        return True

    def _invalid_entries(self, limit=5):
        """Names of the first variables and constraints whose value or derivative is not finite."""
        bad = set()
        for vec in (self.state.current.get_solution(), self.state.gradient):
            vec.copy_device_to_host()
            bad.update(np.flatnonzero(~np.isfinite(vec.get_array())).tolist())
        if not bad:
            return []
        names = self._index_names(bad)
        return [names[i] for i in sorted(bad)[:limit]]

    def _invalid_derivatives(self, limit=5):
        """Constraints and variable pairs whose derivatives are not finite at a finite point.

        The multiplier rows of the Hessian of the Lagrangian hold the constraint
        Jacobian, so a non-finite entry there names the constraint and the
        variable, and one between two variables is a second derivative.
        """
        x = self.state.current.get_solution()
        x.copy_device_to_host()
        if self.model is None or not np.all(np.isfinite(x.get_array())):
            return []

        self.evaluator.evaluate_hessian(self.state)
        mat = self.state.hessian
        mat.copy_data_device_to_host()
        _, _, _, rowp, cols = mat.get_nonzero_structure()
        rows = np.repeat(np.arange(len(rowp) - 1), np.diff(rowp))
        bad = np.flatnonzero(~np.isfinite(mat.get_data()))
        if bad.size == 0:
            return []

        # A Jacobian entry pairs a constraint with a variable and a second derivative pairs two variables
        constraints = set(np.asarray(self.model.constraint_indices).tolist())
        jacobian, hessian = {}, set()
        for k in bad:
            r, c = int(rows[k]), int(cols[k])
            if c in constraints:
                r, c = c, r
            if r in constraints and c not in constraints:
                jacobian.setdefault(r, set()).add(c)
            elif r not in constraints:
                hessian.update((r, c))

        names = self._index_names(
            set(jacobian) | set().union(*jacobian.values()) | hessian
        )
        lines = []
        for r in sorted(jacobian)[:limit]:
            wrt = ", ".join(names[c] for c in sorted(jacobian[r])[:limit])
            lines.append(f"Non-finite derivative of {names[r]} with respect to {wrt}")
        if hessian:
            wrt = ", ".join(names[i] for i in sorted(hessian)[:limit])
            lines.append(f"Non-finite second derivatives in {wrt}")
        return lines

    def _invalid_kkt(self, limit=5):
        """Lines that name the non-finite values, barrier diagonal entries, and derivatives of the KKT matrix."""
        lines = []
        names = self._invalid_entries(limit)
        if names:
            lines.append(f"Non-finite values at {', '.join(names)}")
        diagonal = self.state.diagonal
        diagonal.copy_device_to_host()
        bad = np.flatnonzero(~np.isfinite(diagonal.get_array())).tolist()
        if bad:
            where = self._index_names(set(bad))
            lines.append(
                f"Non-finite barrier diagonal at {', '.join(where[i] for i in sorted(where)[:limit])}"
            )
        return lines + self._invalid_derivatives(limit)

    def _index_names(self, indices):
        """Scoped names with the instance and element of each listed index."""
        names = {}
        if self.model is not None:
            inputs, cons, _, _ = self.model.get_names()
            for name in inputs + cons:
                found = np.asarray(self.model.get_indices(name))
                for pos in np.ndindex(found.shape):
                    index = int(found[pos])
                    if index in indices and index not in names:
                        names[index] = f"{name}[{', '.join(map(str, pos))}]"
        return {i: names.get(i, f"index {i}") for i in indices}

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

        # The unscaled solution gives a KKT matrix consistent with the data derivatives
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
                res.copy_host_to_device()
                self.solver.solve(res, psi)
                adjx = grad_wrt_data.T @ psi.get_array()
                dfdx[i, :] = out_wrt_data[idx, wrt_indices] + adjx[wrt_indices]
        elif method == "direct":
            grad_wrt_data = grad_wrt_data.tocsc()
            for i in range(len(wrt_indices)):
                idx = wrt_indices[i]
                res.get_array()[:] = -grad_wrt_data[:, idx].toarray().flatten()
                self.problem.zero_fixed_entries(res)
                res.copy_host_to_device()
                self.solver.solve(res, psi)
                dirx = out_wrt_input @ psi.get_array()
                dfdx[:, i] = out_wrt_data[of_indices, idx] + dirx[of_indices]

        return dfdx, of_map, wrt_map
