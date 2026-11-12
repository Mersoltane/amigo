"""
Inertia correction
------------------

- The schedule of the Hessian perturbation in the inertia corrector
- The resumed perturbation and its hold when the option asks for them
- The hold of the dual shift after a probe without it fails
- The diagonal the corrector hands to the factorization
- The reason a failed factorization reports
"""

from types import SimpleNamespace

import numpy as np
import pytest

from amigo.algorithm.default_options import get_default_options
from amigo.algorithm.solvers.amigo_solver import (
    FRONT_OVERFLOW,
    NONFINITE_MATRIX,
    AmigoSolver,
)
from amigo.algorithm.solvers.inertia_correction import HOLD_SYSTEMS, InertiaCorrector


class ScriptedCorrector(InertiaCorrector):
    """Corrector whose factorization has the right inertia once delta_w reaches the needed value"""

    needed = 0.0

    def _apply_and_factor(self, solver, diagonal, perturb, hessian):
        self.tried.append(self._delta_x_curr)

        if self._delta_x_curr >= self.needed:
            return self.num_primal, self.num_dual, False

        return self.num_primal - 1, self.num_dual + 1, False


class Optimizer:
    """Optimizer with four primal variables and two constraints"""

    def get_num_primals(self):
        return 4

    def get_num_constraints(self):
        return 2


class VectorProblem:
    """Problem whose vectors the corrector never touches"""

    def create_vector(self):
        return None


class InertiaSolver:
    """Solver that reports the inertia of every factorization"""

    def inertia_enabled(self):
        return True

    def static_pivot_floor(self):
        return 0.0


class QuietEvaluator:
    """Evaluator whose Hessian and diagonal are already in place"""

    def evaluate_hessian(self, state):
        pass

    def evaluate_diagonal(self, state):
        pass


def make_corrector(warm_start=False):
    """Scripted corrector on the optimizer above"""
    options = get_default_options({"hessian_perturbation_warm_start": warm_start})

    return ScriptedCorrector(options, VectorProblem(), Optimizer())


def run_iteration(corrector, needed, mu=0.1):
    """Values of delta_w tried by one factorization call that needs at least needed"""
    state = SimpleNamespace(
        primal_infeas=0.0, mu=mu, comm_rank=0, diagonal=None, hessian=None
    )

    corrector.needed = needed
    corrector.tried = []

    assert corrector.factor_for_inertia(InertiaSolver(), QuietEvaluator(), state)

    return corrector.tried


def test_every_system_starts_without_a_perturbation():
    corrector = make_corrector()

    first = run_iteration(corrector, needed=10.0)

    assert first[0] == 0.0 and first[-1] >= 10.0

    # The last system needed a correction and the next one still tries zero first
    second = run_iteration(corrector, needed=10.0)

    assert second[0] == 0.0
    assert second[1] == pytest.approx(first[-1] / 3.0)


def test_the_perturbation_falls_by_a_third_every_system_while_zero_fails():
    corrector = make_corrector()

    first = run_iteration(corrector, needed=10.0)
    second = run_iteration(corrector, needed=1.0)
    third = run_iteration(corrector, needed=1.0)

    assert second == [0.0, pytest.approx(first[-1] / 3.0)]
    assert third == [0.0, pytest.approx(second[-1] / 3.0)]


def test_a_system_that_needs_no_perturbation_is_factored_once_after_a_correction():
    corrector = make_corrector()

    run_iteration(corrector, needed=10.0)

    assert run_iteration(corrector, needed=0.0) == [0.0]


def test_warm_start_resumes_after_a_correction():
    corrector = make_corrector(warm_start=True)

    first = run_iteration(corrector, needed=10.0)

    assert first[0] == 0.0 and first[-1] >= 10.0

    # The last system needed a correction, so the decayed value is tried first
    second = run_iteration(corrector, needed=10.0)

    assert second[0] == pytest.approx(first[-1] / 3.0)


def test_decay_continues_after_a_first_try_success():
    corrector = make_corrector(warm_start=True)

    run_iteration(corrector, needed=10.0)
    second = run_iteration(corrector, needed=0.0)

    # A regularized system keeps decaying delta_w instead of retrying zero
    assert run_iteration(corrector, needed=0.0) == [pytest.approx(second[-1] / 3.0)]


def test_a_failed_decay_holds_the_perturbation_at_one_factorization_per_system():
    corrector = make_corrector(warm_start=True)

    run_iteration(corrector, needed=10.0)
    run_iteration(corrector, needed=10.0)
    run_iteration(corrector, needed=10.0)

    # The decayed probe fails, so the next systems factor once at the held value
    grown = run_iteration(corrector, needed=10.0)

    assert len(grown) == 2

    held = run_iteration(corrector, needed=10.0)

    assert held == [pytest.approx(grown[-1])]

    # The hold goes on at one factorization per system until the barrier moves
    for k in range(5):
        assert len(run_iteration(corrector, needed=10.0)) == 1


def test_a_new_barrier_probes_down_again_and_a_served_probe_ends_the_hold():
    corrector = make_corrector(warm_start=True)

    for k in range(3):
        run_iteration(corrector, needed=10.0)

    grown = run_iteration(corrector, needed=10.0)

    assert len(run_iteration(corrector, needed=10.0)) == 1

    # The barrier moved, so the held perturbation is probed down and keeps decaying after it serves
    probed = run_iteration(corrector, needed=1.0, mu=0.02)

    assert probed == [pytest.approx(grown[-1] / 3.0)]
    assert run_iteration(corrector, needed=1.0, mu=0.02) == [
        pytest.approx(probed[-1] / 3.0)
    ]


def test_zero_is_tried_once_the_decayed_value_is_small():
    corrector = make_corrector(warm_start=True)

    assert run_iteration(corrector, needed=1e-4) == [0.0, 1e-4]

    # The decayed value falls below the first perturbation so the system is factored unperturbed
    assert run_iteration(corrector, needed=0.0) == [0.0]


class RankCorrector(InertiaCorrector):
    """Corrector whose factorization meets too few negative pivots without the dual shift while deficient"""

    deficient = True

    def _apply_and_factor(self, solver, diagonal, perturb, hessian):
        self.tried.append(self._delta_c_curr > 0.0)

        if self.deficient and self._delta_c_curr == 0.0:
            return self.num_primal, self.num_dual - 1, False

        return self.num_primal, self.num_dual, False


def run_rank_iteration(corrector, deficient, mu=0.1):
    """Whether each attempt of one factorization call carried the dual shift"""
    state = SimpleNamespace(
        primal_infeas=0.0, mu=mu, comm_rank=0, diagonal=None, hessian=None
    )

    corrector.deficient = deficient
    corrector.tried = []

    assert corrector.factor_for_inertia(InertiaSolver(), QuietEvaluator(), state)

    return corrector.tried


def test_a_failed_probe_holds_the_dual_shift_at_one_factorization_per_system():
    corrector = RankCorrector(get_default_options(), VectorProblem(), Optimizer())

    # The first system finds the deficiency by a failed attempt
    assert run_rank_iteration(corrector, deficient=True) == [False, True]

    # The held systems start with the dual shift on
    for k in range(HOLD_SYSTEMS):
        assert run_rank_iteration(corrector, deficient=True) == [True]


def test_a_lapsed_hold_probes_without_the_dual_shift_and_holds_again():
    corrector = RankCorrector(get_default_options(), VectorProblem(), Optimizer())

    run_rank_iteration(corrector, deficient=True)
    for k in range(HOLD_SYSTEMS):
        run_rank_iteration(corrector, deficient=True)

    # The hold lapsed, so the next system pays one failed attempt and the hold starts again
    assert run_rank_iteration(corrector, deficient=True) == [False, True]
    assert run_rank_iteration(corrector, deficient=True) == [True]


def test_a_new_barrier_probes_without_the_dual_shift():
    corrector = RankCorrector(get_default_options(), VectorProblem(), Optimizer())

    run_rank_iteration(corrector, deficient=True)

    assert run_rank_iteration(corrector, deficient=True) == [True]
    assert run_rank_iteration(corrector, deficient=True, mu=0.02) == [False, True]


def test_a_probe_that_passes_releases_the_dual_shift():
    corrector = RankCorrector(get_default_options(), VectorProblem(), Optimizer())

    run_rank_iteration(corrector, deficient=True)

    assert run_rank_iteration(corrector, deficient=True) == [True]

    # The barrier moved and the deficiency is gone, so the probe passes and the hold ends
    assert run_rank_iteration(corrector, deficient=False, mu=0.02) == [False]
    assert run_rank_iteration(corrector, deficient=False, mu=0.02) == [False]


class DiagonalVector:
    """Vector that lives on the host and takes the shifts of the corrector"""

    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def copy(self, other):
        self.values[:] = other.values

    def add_scalar_at(self, indices, value):
        self.values[indices] += value

    def get_array(self):
        return self.values


class RowProblem(VectorProblem):
    """Problem with four primal rows followed by two constraint rows"""

    def get_primal_indices(self):
        return np.arange(4)

    def get_constraint_indices(self):
        return np.arange(4, 6)


def shifted_diagonal(delta_w, delta_c):
    """Diagonal the corrector hands to the factorization under the given shifts"""
    corrector = InertiaCorrector(get_default_options(), RowProblem(), Optimizer())
    corrector._delta_x_curr = delta_w
    corrector._delta_c_curr = delta_c

    # A flat direction, a slack far from its bound, a stiff bound, and a free row
    diagonal = DiagonalVector([0.0, 1e-21, 3.0, 0.0, 0.0, 0.0])
    shifted = DiagonalVector(np.zeros(6))
    corrector._compute_perturbation(diagonal, shifted)

    return shifted.get_array()


def test_a_system_without_a_shift_is_factored_with_its_own_diagonal():
    assert shifted_diagonal(0.0, 0.0).tolist() == [0.0, 1e-21, 3.0, 0.0, 0.0, 0.0]


def test_each_shift_lands_on_its_own_rows():
    shifted = shifted_diagonal(1e-4, 1e-9)

    assert shifted[:4].tolist() == pytest.approx([1e-4, 1e-4, 3.0001, 1e-4])
    assert shifted[4:].tolist() == pytest.approx([-1e-9, -1e-9])


class UnperturbedCorrector(InertiaCorrector):
    """Corrector that leaves the diagonal alone so a scripted solver decides every factorization"""

    def _compute_perturbation(self, diagonal, perturb):
        pass


class FlaggedSolver(InertiaSolver):
    """Solver whose factorization returns a set flag with the right inertia"""

    def __init__(self, flag):
        self.flag = flag

    def factor(self, hessian, perturb):
        return self.flag

    def get_inertia(self):
        return (4, 2)

    def describe_failure(self, flag):
        return AmigoSolver.describe_failure(None, flag)


def test_a_failed_factorization_reports_its_reason_and_clears_on_the_next_call(capsys):
    corrector = UnperturbedCorrector(
        get_default_options(), VectorProblem(), Optimizer()
    )
    solver = FlaggedSolver(NONFINITE_MATRIX)
    state = SimpleNamespace(
        primal_infeas=0.0, mu=0.1, comm_rank=0, diagonal=None, hessian=None
    )

    # No regularization repairs the failure, so a single factorization is tried
    assert not corrector.factor_for_inertia(solver, QuietEvaluator(), state)
    assert corrector.failure == NONFINITE_MATRIX
    assert (
        "factorization failed because the matrix has a NaN or infinite entry"
        in capsys.readouterr().out
    )

    # A good factorization clears the failure
    solver.flag = 0

    assert corrector.factor_for_inertia(solver, QuietEvaluator(), state)
    assert corrector.failure == 0


def test_the_amigo_solver_names_each_failure():
    assert "NaN or infinite" in AmigoSolver.describe_failure(None, NONFINITE_MATRIX)
    assert "past the memory cap" in AmigoSolver.describe_failure(None, FRONT_OVERFLOW)
    assert "flag 7" in AmigoSolver.describe_failure(None, 7)
