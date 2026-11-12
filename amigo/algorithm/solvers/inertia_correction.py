"""Inertia correction of the KKT system

Every factorization testifies through its counts of positive, negative, and
zero pivots. The corrector reads the testimony of each attempt, grows the
primal shift delta_w for too many negative pivots and turns on the dual shift
delta_c for too few, until the counts match the primal and dual row counts.
The memory across systems is the primal shift the last system accepted and
the dual shift a failed probe holds on, and both re-earn their value by probes.
"""

import numpy as np

from .linear_solver import SINGULAR_MATRIX

# First primal shift and the range it may reach
DW_INIT = 1e-4
DW_MIN = 1e-20
DW_MAX = 1e20

# Growth factors of the shift within a system
KW_INC = 8.0
KW_FIRST_INC = 100.0

# Decay of the shift between systems
KW_DEC = 1.0 / 3.0

# Attempts one system may take
MAX_CORRECTIONS = 10

# Systems a failed decay holds the shift before it probes again
HOLD_SYSTEMS = 10

# Names of the testimony of a factorization
HEALTHY = "healthy"
CURVATURE = "curvature"
RANK = "rank"


class InertiaCorrector:
    """Grows the primal and dual shifts until the factored KKT matrix has the right inertia"""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        # Row counts the inertia must match
        self.num_primal = self.optimizer.get_num_primals()
        self.num_dual = self.optimizer.get_num_constraints()

        # Diagonal of the factored matrix with the shifts applied
        self.perturbed_diagonal = self.problem.create_vector()

        # Dual shift as a value times the barrier to an exponent
        self._dc_val = options["jacobian_regularization_value"]
        self._dc_exp = options["jacobian_regularization_exponent"]
        self._perturb_always_cd = options["perturb_always_cd"]

        # Dual shift proportional to the infeasibility on every factorization
        self._dc_infeas_coeff = options["dual_reg_infeas_coeff"]
        self._dc_infeas = 0.0

        # Floor of the dual shift from the solver's static pivoting, set per system
        self._dc_floor = 0.0

        # Resume the decayed shift of the last system instead of starting at zero
        self._warm_start = options["hessian_perturbation_warm_start"]

        # Accept a wrong inertia once the shift reaches this cap
        self._dw_accept = options["max_regularization"]

        # Rows that carry the shift when the free variables are named, else every primal row
        named = options["hessian_perturbation_indices"]
        self._delta_x_weights = None
        if named is not None:
            self._delta_x_weights = self.problem.create_vector()
            self._delta_x_weights.get_array()[np.asarray(named, dtype=int)] = 1.0
            self._delta_x_weights.copy_host_to_device()

        # Curvature test of the step that accepts a wrong inertia
        self._curv_tol = options["neg_curv_test_tol"]
        self.curv_step = self.problem.create_vector()
        self.curv_wd = self.problem.create_vector()

        # Shifts of the last system and of the current attempt
        self._delta_x_last = 0.0
        self._delta_x_curr = 0.0
        self._delta_c_curr = 0.0
        self._applied_delta_c = 0.0

        # A failed decay holds the shift until a later probe succeeds
        self._hold = False
        self._since_probe = 0
        self._probe_mu = None
        self._first_try = None
        self._probe_value = 0.0
        self._mu_seen = None

        # A failed probe without the dual shift holds it on until a later probe passes
        self._dual_hold = False
        self._dual_since_probe = 0
        self._dual_probe_mu = None
        self._dual_probed = False

        # Values of the accepted factorization for the log and the step contract
        self.last_delta_w = 0.0
        self.last_delta_c = 0.0
        self.last_attempts = 0

        # Flag of a factorization that failed in the last call, zero otherwise
        self.failure = 0

        self.verbose = options["verbose_barrier"]

    def _delta_cd(self, state):
        """Dual shift at the current barrier floored at the static pivot epsilon"""
        return max(self._dc_val * state.mu**self._dc_exp, self._dc_floor)

    def _begin_system(self, state):
        """Save the shift of the last system and reset the shifts of this one"""
        self._dc_infeas = self._dc_infeas_coeff * max(state.primal_infeas, state.mu)

        if self._delta_x_curr > 0.0:
            self._delta_x_last = self._delta_x_curr
        self._delta_x_curr = 0.0

        # A new barrier or a lapsed hold probes without the dual shift
        lapsed = self._dual_since_probe >= HOLD_SYSTEMS
        held = self._dual_hold and state.mu == self._dual_probe_mu and not lapsed

        # The dual shift starts on when the option asks for it or while a failed probe holds it
        self._dual_probed = False
        if self._perturb_always_cd:
            self._delta_c_curr = self._delta_cd(state)
        elif held:
            self._delta_c_curr = self._delta_cd(state)
            self._dual_since_probe += 1
        else:
            self._delta_c_curr = 0.0
            self._dual_probed = True

    def _resume_shift(self, state):
        """Start the first attempt from the decayed shift of the last system or hold it after a failed decay"""
        self._first_try = None

        if self._warm_start and self.last_delta_w > 0.0 and self._delta_x_last > 0.0:
            # A new barrier or a lapsed hold probes the decayed shift and a hold repeats the last one
            lapsed = self._since_probe >= HOLD_SYSTEMS
            if not self._hold or state.mu != self._probe_mu or lapsed:
                decayed = KW_DEC * self._delta_x_last
                self._delta_x_curr = decayed if decayed >= DW_INIT else 0.0
                self._since_probe = 0
                self._first_try = "probe"
            else:
                self._delta_x_curr = self._delta_x_last
                self._since_probe += 1
                self._first_try = "hold"

        self._probe_value = self._delta_x_curr

    def _compute_perturbation(self, diagonal, perturb):
        """Diagonal of the factored matrix with the shifts on their rows"""
        primal_indices = self.problem.get_primal_indices()
        dual_indices = self.problem.get_constraint_indices()

        perturb.copy(diagonal)

        # The primal shift on the rows that carry it
        if self._delta_x_curr > 0.0:
            if self._delta_x_weights is None:
                perturb.add_scalar_at(primal_indices, self._delta_x_curr)
            else:
                perturb.axpy(self._delta_x_curr, self._delta_x_weights)

        # The dual shift is the larger of the current and the infeasibility shift
        delta_c = max(self._delta_c_curr, self._dc_infeas)
        if delta_c > 0.0:
            delta_c = max(delta_c, self._dc_floor)
            perturb.add_scalar_at(dual_indices, -delta_c)
        self._applied_delta_c = delta_c

    def _apply_and_factor(self, solver, diagonal, perturb, hessian):
        """Factor with the current shifts and return the counts or None on a failed factorization"""
        self._compute_perturbation(diagonal, perturb)
        flag = solver.factor(hessian, perturb)

        # A singular matrix without an inertia count takes the rank path
        if flag == SINGULAR_MATRIX:
            return 0, 0, True
        if flag:
            self.failure = flag
            return None

        # Pivots that are neither positive nor negative are zero pivots
        n_pos, n_neg = solver.get_inertia()
        n_zero = (self.num_primal + self.num_dual) - (n_pos + n_neg)
        return n_pos, n_neg, n_zero > 0

    def _inertia_ok(self, n_pos, n_neg):
        """Whether the counts match the row counts within the tolerated departure"""
        itol = self.options["inertia_tolerance"]
        num_total = self.num_primal + self.num_dual

        return (
            abs(n_pos - self.num_primal) <= itol
            and abs(n_neg - self.num_dual) <= itol
            and n_pos + n_neg >= num_total - itol
        )

    def _diagnose(self, n_pos, n_neg, singular):
        """Name the deficiency the counts testify"""
        if not singular and self._inertia_ok(n_pos, n_neg):
            return HEALTHY
        if self.num_dual > 0 and (singular or n_neg < self.num_dual):
            return RANK
        return CURVATURE

    def _grow_shift(self):
        """Grow the primal shift and return False when it passes its cap"""
        prev = self._delta_x_curr

        # The first growth of a system starts from the decayed last shift or the initial value
        if self._delta_x_curr == 0.0:
            if self._delta_x_last == 0.0:
                self._delta_x_curr = DW_INIT
            else:
                self._delta_x_curr = max(DW_MIN, self._delta_x_last * KW_DEC)
        else:
            far_above_last = 1e5 * self._delta_x_last < self._delta_x_curr
            if self._delta_x_last == 0.0 or far_above_last:
                self._delta_x_curr *= KW_FIRST_INC
            else:
                self._delta_x_curr *= KW_INC

        # A shift past the cap was never factored and the last shift is forgotten
        if self._delta_x_curr > DW_MAX:
            self._delta_x_curr = prev
            self._delta_x_last = 0.0
            return False
        return True

    def _shift_for_rank(self, state):
        """Too few negative pivots turn the dual shift on and grow the primal shift once it is on"""
        if self._delta_c_curr > 0.0:
            return self._grow_shift()
        self._delta_c_curr = self._delta_cd(state)

        # The probe without the dual shift failed at this barrier
        self._dual_hold = True
        self._dual_since_probe = 0
        self._dual_probe_mu = state.mu
        return True

    def _positive_curvature(self, solver, evaluator, state):
        """Curvature test of Chiang and Zavala on the primal step of the current factorization"""
        evaluator.evaluate_residual(state)
        solver.solve(state.residual, self.curv_step)
        self.curv_step.fill_at(self.problem.get_constraint_indices(), 0.0)
        self.curv_step.copy_device_to_host()
        state.hessian.mult(self.curv_step, self.curv_wd)

        d = self.curv_step.get_array()
        curvature = np.dot(d, self.curv_wd.get_array())
        curvature += np.dot(self.perturbed_diagonal.get_array() * d, d)
        return curvature >= self._curv_tol * np.dot(d, d)

    def _shift_for_curvature(self, state):
        """Too many negative pivots grow the primal shift and a shift past its cap restarts with the dual shift on"""
        if self._grow_shift():
            return True
        if self._delta_c_curr == 0.0:
            self._delta_c_curr = self._delta_cd(state)
            self._delta_x_curr = 0.0
            return self._grow_shift()
        return False

    def _accept(self, attempt):
        """Record the accepted shifts and judge the probe of this system"""
        self.last_delta_w = self._delta_x_curr
        self.last_delta_c = self._applied_delta_c
        self.last_attempts = attempt + 1

        # A probe whose shift serves lets the shift keep decaying and one whose shift had to grow holds it
        if self._first_try == "probe":
            self._hold = self._delta_x_curr > self._probe_value
        self._probe_mu = self._mu_seen

        # A probe accepted without the dual shift releases its hold
        if self._dual_probed and self._delta_c_curr == 0.0:
            self._dual_hold = False
        return True

    def factor_for_inertia(self, solver, evaluator, state):
        """Assemble, shift, and factor the KKT matrix until its inertia is right"""
        evaluator.evaluate_hessian(state)
        evaluator.evaluate_diagonal(state)

        # A solver without an inertia count factors once with the current shifts
        if not solver.inertia_enabled():
            self._compute_perturbation(state.diagonal, self.perturbed_diagonal)
            solver.factor(state.hessian, self.perturbed_diagonal)
            self.last_attempts = 1
            return True

        self._dc_floor = solver.static_pivot_floor()
        self.failure = 0
        self._mu_seen = state.mu

        self._begin_system(state)
        self._resume_shift(state)

        for attempt in range(MAX_CORRECTIONS + 1):
            result = self._apply_and_factor(
                solver, state.diagonal, self.perturbed_diagonal, state.hessian
            )

            # A failed factorization cannot be repaired by more shift
            if result is None:
                if state.comm_rank == 0:
                    print(
                        f"  Inertia: factorization failed because {solver.describe_failure(self.failure)}"
                    )
                break

            n_pos, n_neg, singular = result
            diagnosis = self._diagnose(n_pos, n_neg, singular)
            if state.comm_rank == 0 and self.verbose:
                print(
                    f"  Inertia: attempt {attempt + 1} at delta_w={self._delta_x_curr:.2e} "
                    f"delta_c={self._applied_delta_c:.2e} met {diagnosis} "
                    f"with ({n_pos}+, {n_neg}-) for ({self.num_primal}+, {self.num_dual}-)"
                )

            if diagnosis == HEALTHY:
                return self._accept(attempt)

            if diagnosis == RANK:
                if not self._shift_for_rank(state):
                    break
                continue

            # A step of positive curvature is usable despite the wrong inertia
            if (
                not singular
                and self._curv_tol > 0.0
                and self._positive_curvature(solver, evaluator, state)
            ):
                if state.comm_rank == 0 and self.verbose:
                    print("  Inertia: accepting wrong inertia on positive curvature")
                return self._accept(attempt)

            # Past the cap a mildly shifted factorization beats a heavily convexified one
            if (
                not singular
                and self._dw_accept > 0.0
                and self._delta_x_curr >= self._dw_accept
            ):
                if state.comm_rank == 0 and self.verbose:
                    print(
                        f"  Inertia: accepting wrong inertia at delta_w={self._delta_x_curr:.2e} (cap)"
                    )
                return self._accept(attempt)

            if not self._shift_for_curvature(state):
                if state.comm_rank == 0:
                    print(
                        f"  Inertia: delta_w={self._delta_x_curr:.2e} > max, aborting correction"
                    )
                break

        # The correction failed and the last applied values go to the log
        self.last_delta_w = self._delta_x_curr
        self.last_delta_c = self._applied_delta_c
        self.last_attempts = MAX_CORRECTIONS + 1
        return False

    def add_log_info(self, info):
        """Shifts and attempts of the last system for the iteration log"""
        info["inertia_delta"] = self.last_delta_w
        info["inertia_delta_c"] = self.last_delta_c
        info["inertia_attempts"] = self.last_attempts
