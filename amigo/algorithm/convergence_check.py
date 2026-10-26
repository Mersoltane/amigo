"""Convergence, divergence, and stall tests for the interior point loop."""

import numpy as np

from ..amigo import OptVarType

# Return codes for convergence check
CONTINUE = 0
CONVERGED = 1
CONVERGED_ACCEPTABLE = 2
DIVERGED = 3
MAX_ITERATIONS = 4
INVALID_NUMBER = 5
RESTORATION_NEEDED = 6
LOCALLY_INFEASIBLE = 7
TIME_LIMIT = 8
TINY_STEP = 9
STALLED = 10
DEGENERATE_DUALS = 11

# Message reported with each final status
STATUS_MESSAGES = {
    CONTINUE: "Optimization did not finish",
    CONVERGED: "Converged to the requested tolerance",
    CONVERGED_ACCEPTABLE: "Converged to an acceptable point",
    DIVERGED: "Iterates diverging, the problem may be unbounded",
    MAX_ITERATIONS: "Maximum number of iterations reached",
    INVALID_NUMBER: "Invalid number in a function or derivative evaluation",
    RESTORATION_NEEDED: "Restoration phase failed to find a better point",
    LOCALLY_INFEASIBLE: "Converged to a locally infeasible point",
    TIME_LIMIT: "Maximum wall time reached",
    TINY_STEP: "Search direction became too small to make progress",
    STALLED: "The error stopped shrinking",
    DEGENERATE_DUALS: "No multipliers close the duals at this primal optimum, degenerate constraints",
}


class ConvergenceCheck:
    """Primary, acceptable, divergence, and small step checks."""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        # Set the acceptable counter
        self.acceptable_counter = 0

        # Count consecutive accepted steps that stalled at a tiny alpha
        self.small_alpha_count = 0

        # Error and iteration of the last time the error shrank by the stall factor
        self.stall_error = np.inf
        self.stall_iter = 0

        # A square system has no degrees of freedom, so feasibility alone decides convergence
        self.square = problem is not None and self._is_square(problem, optimizer)

    @staticmethod
    def _is_square(problem, optimizer):
        """True when the free primal variables and the constraints are equal in number."""
        n_primal, n_con, _ = optimizer.get_kkt_element_counts()
        fixed = problem.create_vector()
        types = np.asarray(problem.get_var_types().get_array())
        fixed.get_array()[:] = types == int(OptVarType.FIXED)
        fixed.copy_host_to_device()
        n_fixed = int(round(problem.dot(fixed, fixed)))
        return n_con > 0 and n_primal - n_fixed == n_con

    def test_small_alpha_stall(self, state, alpha):
        """Report whether resto_trigger_iters accepted steps in a row stalled while infeasible."""
        stalled = (
            alpha < self.options["resto_trigger_alpha"]
            and state.constr_viol > self.options["acceptable_constr_viol_tol"]
        )
        if not stalled:
            self.small_alpha_count = 0
            return False

        self.small_alpha_count += 1
        if self.small_alpha_count >= self.options["resto_trigger_iters"]:
            self.small_alpha_count = 0
            return True
        return False

    def point_acceptable(self, evaluator, state):
        """One shot acceptable test for the failure rescue paths."""
        evaluator.evaluate_residual(state)
        if not np.isfinite(state.kkt_error):
            return False
        return self._meets(
            state,
            self.options["acceptable_tol"],
            self.options["acceptable_dual_inf_tol"],
            self.options["acceptable_constr_viol_tol"],
            self.options["acceptable_compl_inf_tol"],
        )

    def primal_converged(self, state):
        """Primal feasibility and complementarity at the solve tolerance."""
        tol = self.options["convergence_tolerance"]
        return (
            state.constr_viol <= tol
            and state.complementarity / state.obj_scale <= tol
        )

    @staticmethod
    def _meets(state, tol, dual_inf_tol, constr_viol_tol, compl_inf_tol):
        """Test the scaled error against tol and the unscaled measures against the absolute tolerances."""
        return (
            state.kkt_error <= tol
            and state.dual_infeas / state.obj_scale <= dual_inf_tol
            and state.constr_viol <= constr_viol_tol
            and state.complementarity / state.obj_scale <= compl_inf_tol
        )

    def test_convergence(self, evaluator, state):
        """
        Check for convergence
        """
        options = self.options
        diverging_iterates_tol = options["diverging_iterates_tol"]
        acceptable_iter = options["acceptable_iter"]

        # Evaluate the residual to compute the KKT error metrics
        evaluator.evaluate_residual(state)

        # A non-finite iterate cannot recover since every test is False on NaN
        if not (np.isfinite(state.kkt_error) and np.isfinite(state.objective_value)):
            return INVALID_NUMBER

        # A square system converges once it is feasible
        if (
            self.square
            and state.primal_infeas <= options["convergence_tolerance"]
            and state.constr_viol <= options["constr_viol_tol"]
        ):
            return CONVERGED

        # Primary convergence needs the scaled error and all absolute tolerances
        if self._meets(
            state,
            options["convergence_tolerance"],
            options["dual_inf_tol"],
            options["constr_viol_tol"],
            options["compl_inf_tol"],
        ):
            return CONVERGED

        # Only primal entries count toward the iterate size
        if diverging_iterates_tol > 0.0:
            x_max = evaluator.primal_maxabs(state.current.get_solution())
            if x_max > diverging_iterates_tol:
                if state.comm_rank == 0:
                    print(f"  Diverging iterates: max |x| = {x_max:.2e}")
                return DIVERGED

        # Acceptable convergence
        is_acceptable = self._meets(
            state,
            options["acceptable_tol"],
            options["acceptable_dual_inf_tol"],
            options["acceptable_constr_viol_tol"],
            options["acceptable_compl_inf_tol"],
        )
        if acceptable_iter > 0 and is_acceptable:
            self.acceptable_counter += 1
            if self.acceptable_counter >= acceptable_iter:
                return CONVERGED_ACCEPTABLE
        else:
            self.acceptable_counter = 0

        # A solve whose error stopped shrinking ends here instead of at the iteration limit
        if options["stall_iterations"] > 0:
            if state.kkt_error <= options["stall_factor"] * self.stall_error:
                self.stall_error, self.stall_iter = state.kkt_error, state.iter
            elif state.iter - self.stall_iter >= options["stall_iterations"]:
                return STALLED

        return CONTINUE
