"""
Inertia correction
------------------

- The schedule of the Hessian perturbation in the inertia corrector
- The reason a failed factorization reports
"""

from types import SimpleNamespace

import pytest

from amigo.algorithm.default_options import get_default_options
from amigo.algorithm.solvers.amigo_solver import FRONT_OVERFLOW, NONFINITE_MATRIX, AmigoSolver
from amigo.algorithm.solvers.inertia_correction import InertiaCorrector


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


def make_corrector():
    """Scripted corrector on the optimizer above"""
    return ScriptedCorrector(get_default_options(), VectorProblem(), Optimizer())


def run_iteration(corrector, needed, mu=0.1):
    """Values of delta_w tried by one factorization call that needs at least needed"""
    state = SimpleNamespace(primal_infeas=0.0, mu=mu, comm_rank=0, diagonal=None, hessian=None)

    corrector.needed = needed
    corrector.tried = []

    assert corrector.factor_for_inertia(InertiaSolver(), QuietEvaluator(), state)

    return corrector.tried


def test_warm_start_resumes_after_a_correction():
    corrector = make_corrector()

    first = run_iteration(corrector, needed=10.0)

    assert first[0] == 0.0 and first[-1] >= 10.0

    # The last system needed a correction, so the decayed value is tried first
    second = run_iteration(corrector, needed=10.0)

    assert second[0] == pytest.approx(first[-1] / 3.0)


def test_decay_continues_after_a_first_try_success():
    corrector = make_corrector()

    run_iteration(corrector, needed=10.0)
    second = run_iteration(corrector, needed=0.0)

    # A regularized system keeps decaying delta_w instead of retrying zero
    assert run_iteration(corrector, needed=0.0) == [pytest.approx(second[-1] / 3.0)]


def test_a_failed_decay_holds_the_perturbation_at_one_factorization_per_system():
    corrector = make_corrector()

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
    corrector = make_corrector()

    for k in range(3):
        run_iteration(corrector, needed=10.0)

    grown = run_iteration(corrector, needed=10.0)

    assert len(run_iteration(corrector, needed=10.0)) == 1

    # The barrier moved, so the held perturbation is probed down and keeps decaying after it serves
    probed = run_iteration(corrector, needed=1.0, mu=0.02)

    assert probed == [pytest.approx(grown[-1] / 3.0)]
    assert run_iteration(corrector, needed=1.0, mu=0.02) == [pytest.approx(probed[-1] / 3.0)]


def test_zero_is_tried_once_the_decayed_value_is_small():
    corrector = make_corrector()

    assert run_iteration(corrector, needed=1e-4) == [0.0, 1e-4]

    # The decayed value falls below the first perturbation so the system is factored unperturbed
    assert run_iteration(corrector, needed=0.0) == [0.0]


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
    corrector = UnperturbedCorrector(get_default_options(), VectorProblem(), Optimizer())
    solver = FlaggedSolver(NONFINITE_MATRIX)
    state = SimpleNamespace(primal_infeas=0.0, mu=0.1, comm_rank=0, diagonal=None, hessian=None)

    # No regularization repairs the failure, so a single factorization is tried
    assert not corrector.factor_for_inertia(solver, QuietEvaluator(), state)
    assert corrector.failure == NONFINITE_MATRIX
    assert "factorization failed because the matrix has a NaN or infinite entry" in capsys.readouterr().out

    # A good factorization clears the failure
    solver.flag = 0

    assert corrector.factor_for_inertia(solver, QuietEvaluator(), state)
    assert corrector.failure == 0


def test_the_amigo_solver_names_each_failure():
    assert "NaN or infinite" in AmigoSolver.describe_failure(None, NONFINITE_MATRIX)
    assert "past the memory cap" in AmigoSolver.describe_failure(None, FRONT_OVERFLOW)
    assert "flag 7" in AmigoSolver.describe_failure(None, 7)
