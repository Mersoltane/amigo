"""Inertia correction for the augmented KKT system.

Grows the primal (delta_w) and constraint (delta_c) regularization until the
factorized KKT matrix has the correct inertia, with a state machine that
detects structural degeneracy across iterations.
"""

import numpy as np

from .linear_solver import SINGULAR_MATRIX


class InertiaCorrector:
    """Grows primal and constraint regularization until the factorized
    KKT matrix has the correct inertia, tracking structural degeneracy
    across iterations.
    """

    # Degeneracy status
    _NOT_YET = 0
    _NOT_DEGEN = 1
    _DEGENERATE = 2

    # Test status for finalize_test
    _NO_TEST = 0
    _TEST_DC0_DX0 = 1
    _TEST_DC1_DX0 = 2
    _TEST_DC0_DX1 = 3
    _TEST_DC1_DX1 = 4

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        self.num_primal = self.optimizer.get_num_primals()
        self.num_dual = self.optimizer.get_num_constraints()

        # We're going to create our own diagonal entries. Create a vector
        # for those entries
        self.perturbed_diagonal = self.problem.create_vector()

        # Set some parameters. Should these be options?
        self.max_corrections = 10

        # Set internal values
        self.numerical_eps = 1e-12

        # Perturbation state
        self._delta_x_last = 0.0
        self._delta_c_last = 0.0
        self._delta_x_curr = 0.0
        self._delta_c_curr = 0.0

        # Exposed for iterative refinement
        self.last_delta_w = 0.0
        self.last_delta_c = 0.0
        self.last_attempts = 0

        # Flag of a factorization that failed in the last call, zero otherwise
        self.failure = 0

        # Perturbation schedule constants
        self._dw_init = 1e-4  # first_hessian_perturbation
        self._dw_min = 1e-20  # min_hessian_perturbation
        self._dw_max = 1e20  # max_hessian_perturbation
        self._kw_inc = 8.0  # perturb_inc_fact
        self._kw_first_inc = 100.0  # perturb_inc_fact_first
        self._kw_dec = 1.0 / 3  # perturb_dec_fact
        # jacobian_regularization: bar delta_c = val * mu^exp
        self._dc_val = options["jacobian_regularization_value"]
        self._dc_exp = options["jacobian_regularization_exponent"]

        # Cap on delta_x past which a non-singular factorization with wrong inertia is accepted
        self._dw_accept = options["max_regularization"]

        # delta_c as applied to the last factorization, for the log
        self._applied_delta_c = 0.0

        # Dual regularization proportional to the infeasibility on every factorization
        self._dc_infeas_coeff = options["dual_reg_infeas_coeff"]
        self._dc_infeas = 0.0
        self._perturb_always_cd = options["perturb_always_cd"]
        self._warm_start = options["hessian_perturbation_warm_start"]

        # Curvature test workspace for accepting a wrong inertia
        self._curv_tol = options["neg_curv_test_tol"]
        self.curv_step = self.problem.create_vector()
        self.curv_wd = self.problem.create_vector()

        # A failed decay of the resumed delta_x holds it until a later probe succeeds
        self._hold = False
        self._since_probe = 0
        self._probe_mu = None
        self._first_try = None
        self._mu_seen = None

        # delta_c floor from the solver's static pivoting, set per factor call
        self._dc_floor = 0.0

        # Structural degeneracy detection
        self._hess_degen = self._NOT_YET
        self._jac_degen = self._NOT_YET
        self._degen_iters = 0
        self._degen_iters_max = 3
        self._test_status = self._NO_TEST

        self.verbose = self.options["verbose_barrier"]

    def _delta_cd(self, state):
        """Constraint regularization: delta_c = delta_cd_val * mu^delta_cd_exp.

        Floored at the solver's static-pivot epsilon: below that scale the
        dual block is inside the pivot perturbation and static pivoting
        miscounts the inertia.
        """
        return max(self._dc_val * state.mu**self._dc_exp, self._dc_floor)

    def _get_deltas_for_wrong_inertia(self):
        """Grow delta_x geometrically. Returns False if delta_x exceeds max."""
        prev = self._delta_x_curr
        if self._delta_x_curr == 0.0:
            if self._delta_x_last == 0.0:
                self._delta_x_curr = self._dw_init
            else:
                self._delta_x_curr = max(
                    self._dw_min, self._delta_x_last * self._kw_dec
                )
        else:
            if (
                self._delta_x_last == 0.0
                or 1e5 * self._delta_x_last < self._delta_x_curr
            ):
                self._delta_x_curr *= self._kw_first_inc
            else:
                self._delta_x_curr *= self._kw_inc

        if self._delta_x_curr > self._dw_max:
            # Revert: the overflowed value was never used in a factorization
            self._delta_x_curr = prev
            self._delta_x_last = 0.0
            return False
        return True

    def _perturb_for_wrong_inertia(self, state):
        """Perturb for wrong inertia (too many negative eigenvalues).

        Calls finalize_test, then grows delta_x.
        On overflow with delta_c==0: add delta_c and restart delta_x.
        """
        self._finalize_test()
        if self._get_deltas_for_wrong_inertia():
            return True
        # delta_x overflow: if delta_c==0, add it and retry from scratch
        if self._delta_c_curr == 0.0:
            self._delta_c_curr = self._delta_cd(state)
            self._delta_x_curr = 0.0
            if self._hess_degen == self._DEGENERATE:
                self._hess_degen = self._NOT_YET
            self._test_status = self._NO_TEST
            return self._get_deltas_for_wrong_inertia()
        return False

    def _perturb_for_singularity(self, state):
        """Perturb for singular system (too few negative eigenvalues).

        Handles the degeneracy test state machine for singular systems.
        """
        if self._hess_degen == self._NOT_YET or self._jac_degen == self._NOT_YET:
            # Degeneracy test state machine
            ts = self._test_status

            if ts == self._TEST_DC0_DX0:
                # Haven't tried anything yet for this matrix
                if self._jac_degen == self._NOT_YET:
                    # Try adding delta_c only (test if jac is degenerate)
                    self._delta_c_curr = self._delta_cd(state)
                    self._test_status = self._TEST_DC1_DX0
                else:
                    # jac known, hess NOT_YET: try delta_x only
                    if not self._get_deltas_for_wrong_inertia():
                        return False
                    self._test_status = self._TEST_DC0_DX1

            elif ts == self._TEST_DC1_DX0:
                # Still singular after delta_c>0 with delta_x=0
                # Now try delta_x>0, delta_c=0
                self._delta_c_curr = 0.0
                if not self._get_deltas_for_wrong_inertia():
                    return False
                self._test_status = self._TEST_DC0_DX1

            elif ts == self._TEST_DC0_DX1:
                # Still singular after delta_x>0 with delta_c=0, so try both
                self._delta_c_curr = self._delta_cd(state)
                if not self._get_deltas_for_wrong_inertia():
                    return False
                self._test_status = self._TEST_DC1_DX1

            elif ts == self._TEST_DC1_DX1:
                # Both perturbations active, increase delta_x
                if not self._get_deltas_for_wrong_inertia():
                    return False

            # else: NO_TEST should not occur here

        else:
            # Both hess/jac degeneracy resolved
            if self._delta_c_curr > 0.0:
                # Already perturbing constraints: treat like wrong inertia
                if not self._get_deltas_for_wrong_inertia():
                    return False
            else:
                # First singular encounter: add constraint regularization
                self._delta_c_curr = self._delta_cd(state)

        return True

    def _finalize_test(self):
        """Conclude degeneracy test after successful factorization.

        After degen_iters_max consecutive iterations needing the same
        perturbation type, declare structural degeneracy.
        """
        ts = self._test_status
        if ts == self._NO_TEST:
            return

        if ts == self._TEST_DC0_DX0:
            if self._hess_degen == self._NOT_YET:
                self._hess_degen = self._NOT_DEGEN
            if self._jac_degen == self._NOT_YET:
                self._jac_degen = self._NOT_DEGEN
        elif ts == self._TEST_DC1_DX0:
            if self._hess_degen == self._NOT_YET:
                self._hess_degen = self._NOT_DEGEN
            if self._jac_degen == self._NOT_YET:
                self._degen_iters += 1
                if self._degen_iters >= self._degen_iters_max:
                    self._jac_degen = self._DEGENERATE
        elif ts == self._TEST_DC0_DX1:
            if self._jac_degen == self._NOT_YET:
                self._jac_degen = self._NOT_DEGEN
            if self._hess_degen == self._NOT_YET:
                self._degen_iters += 1
                if self._degen_iters >= self._degen_iters_max:
                    self._hess_degen = self._DEGENERATE
        elif ts == self._TEST_DC1_DX1:
            self._degen_iters += 1
            if self._degen_iters >= self._degen_iters_max:
                self._hess_degen = self._DEGENERATE
                self._jac_degen = self._DEGENERATE

        self._test_status = self._NO_TEST

    def _consider_new_system(self, state):
        """Prepare for a new KKT system.

        Save last perturbation, reset current to zero. Pre-apply delta_c
        if Jacobian is structurally degenerate, and pre-populate delta_x
        if Hessian is structurally degenerate.
        """
        self._finalize_test()

        # Dual regularization for this KKT system with a mu floor at feasible points
        self._dc_infeas = self._dc_infeas_coeff * max(state.primal_infeas, state.mu)

        # Save last perturbation
        if self._delta_x_curr > 0.0:
            self._delta_x_last = self._delta_x_curr
        if self._delta_c_curr > 0.0:
            self._delta_c_last = self._delta_c_curr

        # Degeneracy test for this iteration
        if self._hess_degen == self._NOT_YET or self._jac_degen == self._NOT_YET:
            self._test_status = (
                self._TEST_DC1_DX0 if self._perturb_always_cd else self._TEST_DC0_DX0
            )
        else:
            self._test_status = self._NO_TEST

        # delta_c is delta_cd() when degenerate or always-on, else 0
        if self._jac_degen == self._DEGENERATE or self._perturb_always_cd:
            self._delta_c_curr = self._delta_cd(state)
        else:
            self._delta_c_curr = 0.0

        # Pre-apply delta_x if Hessian structurally degenerate
        if self._hess_degen == self._DEGENERATE:
            self._delta_x_curr = 0.0
            self._get_deltas_for_wrong_inertia()
        else:
            self._delta_x_curr = 0.0

    # Names for the deficiency a factorization testifies
    _HEALTHY = "healthy"
    _CURVATURE = "curvature"
    _RANK = "rank"

    def _diagnose(self, n_pos, n_neg, singular):
        """Name the deficiency of the factored system from its testimony."""
        if not singular and self._is_inertia_ok(n_pos, n_neg):
            return self._HEALTHY
        if self.num_dual > 0 and (singular or n_neg < self.num_dual):
            return self._RANK
        return self._CURVATURE

    def _is_inertia_ok(self, np, nn):
        num_total = self.num_primal + self.num_dual

        itol = self.options["inertia_tolerance"]

        return (
            abs(np - self.num_primal) <= itol
            and abs(nn - self.num_dual) <= itol
            and np + nn >= num_total - itol
        )

    def _compute_perturbation(self, diagonal, perturb):
        primal_indices = self.problem.get_primal_indices()
        dual_indices = self.problem.get_constraint_indices()

        perturb.copy(diagonal)

        delta_x = self.numerical_eps + self._delta_x_curr
        if delta_x > 0:
            perturb.add_scalar_at(primal_indices, delta_x)

        delta_c = max(self._delta_c_curr, self._dc_infeas)
        if delta_c > 0:
            delta_c = max(delta_c, self._dc_floor)
            perturb.add_scalar_at(dual_indices, -delta_c)
        self._applied_delta_c = delta_c

        return

    def _apply_and_factor(self, solver, diagonal, perturb, hessian):
        """Apply perturbation, factorize. Returns (n_pos, n_neg, singular), or None on a failed factorization."""

        self._compute_perturbation(diagonal, perturb)
        flag = solver.factor(hessian, perturb)
        if flag == SINGULAR_MATRIX:
            # A singular matrix without an inertia count takes the singular perturbations
            return 0, 0, True
        if flag:
            self.failure = flag
            return None

        npos, nneg = solver.get_inertia()
        # npos + nneg < n means structurally zero pivots, route to delta_c
        nzero = (self.num_primal + self.num_dual) - (npos + nneg)
        return npos, nneg, nzero > 0

    def factor_for_inertia(self, solver, evaluator, state):
        """Assemble, regularize, and factorize the KKT matrix."""

        # Evaluate the Hessian at the current design point
        evaluator.evaluate_hessian(state)

        # Evaluate the diagonal at the current point
        evaluator.evaluate_diagonal(state)

        # Return without modifying the diagonal
        if not solver.inertia_enabled():
            self._compute_perturbation(state.diagonal, self.perturbed_diagonal)
            solver.factor(state.hessian, self.perturbed_diagonal)
            self.last_attempts = 1
            return True

        self._dc_floor = solver.static_pivot_floor()
        self.failure = 0
        self._mu_seen = state.mu
        self._first_try = None

        # Prepare new system: save last perturbation, reset current
        self._consider_new_system(state)

        # Resume from the decayed delta_x whenever the last system was regularized
        if (
            self._warm_start
            and self.last_delta_w > 0.0
            and self._test_status == self._NO_TEST
            and self._hess_degen != self._DEGENERATE
            and self._delta_x_curr == 0.0
            and self._delta_x_last > 0.0
        ):
            # After a failed decay the perturbation holds at its last value, probing down again on a new barrier
            if not self._hold or state.mu != self._probe_mu or self._since_probe >= 10:
                decayed = self._kw_dec * self._delta_x_last
                self._delta_x_curr = decayed if decayed >= self._dw_init else 0.0
                self._since_probe = 0
                self._first_try = "probe"
            else:
                self._delta_x_curr = self._delta_x_last
                self._since_probe += 1
                self._first_try = "hold"

        # Main retry loop
        for attempt in range(self.max_corrections + 1):
            result = self._apply_and_factor(
                solver, state.diagonal, self.perturbed_diagonal, state.hessian
            )

            # A failed factorization cannot be repaired by more regularization
            if result is None:
                if state.comm_rank == 0:
                    print(f"  Inertia: factorization failed because {solver.describe_failure(self.failure)}")
                break

            n_pos, n_neg, singular = result
            diagnosis = self._diagnose(n_pos, n_neg, singular)

            if diagnosis == self._HEALTHY:
                if self._delta_x_curr > 0 and state.comm_rank == 0 and self.verbose:
                    print(
                        f"  Inertia correction: "
                        f"delta_w={self._delta_x_curr:.2e}, "
                        f"delta_c={self._delta_c_curr:.2e}, "
                        f"attempts={attempt + 1}"
                    )
                return self._accept(attempt)

            if state.comm_rank == 0 and not singular and self.verbose:
                print(
                    f"  Inertia: expected ({self.num_primal}+, {self.num_dual}-), "
                    f"got ({n_pos}+, {n_neg}-), dw={self._delta_x_curr:.1e}"
                )

            # A rank deficiency calls for delta_c first
            if diagnosis == self._RANK:
                if not self._perturb_for_singularity(state):
                    break
            else:
                # Too many negative pivots or a singular matrix without constraints grow delta_w
                # A step of positive curvature is usable despite the wrong inertia
                if (
                    not singular
                    and self._curv_tol > 0.0
                    and self._positive_curvature(solver, evaluator, state)
                ):
                    if state.comm_rank == 0 and self.verbose:
                        print("  Inertia: accepting wrong inertia on positive curvature")
                    return self._accept(attempt)
                # Past the cap a mildly perturbed factorization beats a heavily convexified one
                if (
                    not singular
                    and self._dw_accept > 0.0
                    and self._delta_x_curr >= self._dw_accept
                ):
                    if state.comm_rank == 0 and self.verbose:
                        print(
                            f"  Inertia: accepting wrong inertia at "
                            f"delta_w={self._delta_x_curr:.2e} (cap)"
                        )
                    return self._accept(attempt)
                if not self._perturb_for_wrong_inertia(state):
                    if state.comm_rank == 0:
                        print(
                            f"  Inertia: delta_w={self._delta_x_curr:.2e} "
                            f"> max, aborting correction"
                        )
                    break

        # Inertia correction failed, store the last applied values
        self.last_delta_w = self._delta_x_curr
        self.last_delta_c = self._applied_delta_c
        self.last_attempts = self.max_corrections + 1

        return False

    def _accept(self, attempt):
        self.last_delta_w = self._delta_x_curr
        self.last_delta_c = self._applied_delta_c
        self.last_attempts = attempt + 1

        # A probe that serves at its first try lets the perturbation keep decaying, and one that fails holds it
        if self._first_try == "probe":
            self._hold = attempt > 0
        self._probe_mu = self._mu_seen
        return True

    def _positive_curvature(self, solver, evaluator, state):
        """Curvature test of Chiang and Zavala on the primal step of the current factorization."""
        evaluator.evaluate_residual(state)
        solver.solve(state.residual, self.curv_step)
        self.curv_step.fill_at(self.problem.get_constraint_indices(), 0.0)
        self.curv_step.copy_device_to_host()
        state.hessian.mult(self.curv_step, self.curv_wd)

        d = self.curv_step.get_array()
        curvature = np.dot(d, self.curv_wd.get_array())
        curvature += np.dot(self.perturbed_diagonal.get_array() * d, d)
        return curvature >= self._curv_tol * np.dot(d, d)

    def add_log_info(self, info):
        """Add information to the logger"""
        info["inertia_delta"] = self.last_delta_w
        info["inertia_delta_c"] = self.last_delta_c
        info["inertia_attempts"] = self.last_attempts
