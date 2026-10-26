"""Tests for the scheduling of the Hessian perturbation in the inertia corrector."""

from types import SimpleNamespace

import pytest

from amigo.algorithm.default_options import get_default_options
from amigo.algorithm.solvers.amigo_solver import FRONT_OVERFLOW, NONFINITE_MATRIX, AmigoSolver
from amigo.algorithm.solvers.inertia_correction import InertiaCorrector


class ScriptedCorrector(InertiaCorrector):
    """Corrector whose factorization has the right inertia once delta_w reaches the needed value."""

    needed = 0.0

    def _apply_and_factor(self, solver, diagonal, perturb, hessian):
        self.tried.append(self._delta_x_curr)
        if self._delta_x_curr >= self.needed:
            return self.num_primal, self.num_dual, False
        return self.num_primal - 1, self.num_dual + 1, False


def make_corrector():
    optimizer = SimpleNamespace(get_num_primals=lambda: 4, get_num_constraints=lambda: 2)
    problem = SimpleNamespace(create_vector=lambda: None)
    return ScriptedCorrector(get_default_options(), problem, optimizer)


def run_iteration(corrector, needed, mu=0.1):
    """Delta_w values tried by one factorization call that needs at least needed."""
    solver = SimpleNamespace(
        inertia_enabled=lambda: True,
        static_pivot_floor=lambda: 0.0,
    )
    evaluator = SimpleNamespace(
        evaluate_hessian=lambda state: None, evaluate_diagonal=lambda state: None
    )
    state = SimpleNamespace(primal_infeas=0.0, mu=mu, comm_rank=0, diagonal=None, hessian=None)
    corrector.needed = needed
    corrector.tried = []
    assert corrector.factor_for_inertia(solver, evaluator, state)
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
    for _ in range(5):
        assert len(run_iteration(corrector, needed=10.0)) == 1


def test_a_new_barrier_probes_down_again_and_a_served_probe_ends_the_hold():
    corrector = make_corrector()
    for _ in range(3):
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
    """Corrector that leaves the diagonal alone so a scripted solver decides every factorization."""

    def _compute_perturbation(self, diagonal, perturb):
        pass


def test_a_failed_factorization_reports_its_reason_and_clears_on_the_next_call(capsys):
    optimizer = SimpleNamespace(get_num_primals=lambda: 4, get_num_constraints=lambda: 2)
    problem = SimpleNamespace(create_vector=lambda: None)
    corrector = UnperturbedCorrector(get_default_options(), problem, optimizer)
    flags = [NONFINITE_MATRIX]
    solver = SimpleNamespace(
        inertia_enabled=lambda: True,
        static_pivot_floor=lambda: 0.0,
        factor=lambda hessian, perturb: flags[0],
        get_inertia=lambda: (4, 2),
        describe_failure=lambda flag: AmigoSolver.describe_failure(None, flag),
    )
    evaluator = SimpleNamespace(
        evaluate_hessian=lambda state: None, evaluate_diagonal=lambda state: None
    )
    state = SimpleNamespace(primal_infeas=0.0, mu=0.1, comm_rank=0, diagonal=None, hessian=None)

    # No regularization repairs the failure, so a single factorization is tried
    assert not corrector.factor_for_inertia(solver, evaluator, state)
    assert corrector.failure == NONFINITE_MATRIX
    assert "factorization failed because the matrix has a NaN or infinite entry" in capsys.readouterr().out

    flags[0] = 0
    assert corrector.factor_for_inertia(solver, evaluator, state)
    assert corrector.failure == 0


def test_the_amigo_solver_names_each_failure():
    assert "NaN or infinite" in AmigoSolver.describe_failure(None, NONFINITE_MATRIX)
    assert "pivoting broke down" in AmigoSolver.describe_failure(None, FRONT_OVERFLOW)
    assert "flag 7" in AmigoSolver.describe_failure(None, 7)


