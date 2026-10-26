"""Tests of the regime labels read from the diagnostics of each iteration."""

from amigo.algorithm.regime import EASY, HARD, JAM, STUCK, RegimeMonitor


def row(error=1e-3, theta=1e-4, dual=1e-3, alpha=1.0, trials=1, delta_w=0.0):
    return {"nlp_error": error, "inf_pr": theta, "inf_du": dual, "alpha_x": alpha,
            "line_iters": trials, "inertia_delta": delta_w}


def feed(monitor, entries):
    return [monitor.update(entry) for entry in entries]


def test_full_steps_without_perturbation_stay_easy():
    monitor = RegimeMonitor()
    labels = feed(monitor, [row(error=10.0 ** -k) for k in range(8)])
    assert set(labels) == {EASY}


def test_perturbed_steps_that_move_label_hard_after_three_iterations():
    monitor = RegimeMonitor()
    labels = feed(monitor, [row(error=10.0 ** -k, alpha=0.2, delta_w=1e-2) for k in range(6)])
    assert labels[:2] == [EASY, EASY]
    assert labels[-1] == HARD


def test_tiny_backtracked_steps_without_progress_label_a_jam():
    monitor = RegimeMonitor()
    quiet = [row(error=1.0, theta=3e-3) for _ in range(5)]
    jammed = [row(error=1.0, theta=3e-3, alpha=2e-3, trials=10, delta_w=1e-1) for _ in range(4)]
    labels = feed(monitor, quiet + jammed)
    assert labels[-2] == JAM and labels[-1] == JAM

    # Progress with full steps releases the jam after three agreeing iterations
    eased = [row(error=10.0 ** -k, theta=1e-10) for k in range(4)]
    labels = feed(monitor, eased)
    assert labels[-1] == EASY


def test_a_standing_violation_with_small_dual_error_labels_stuck():
    monitor = RegimeMonitor()
    quiet = [row(error=1e-2, theta=5e-3, dual=1e-4, alpha=0.9) for _ in range(9)]
    labels = feed(monitor, quiet)
    assert labels[-1] == STUCK


def test_two_agreeing_iterations_do_not_change_the_label():
    monitor = RegimeMonitor()
    feed(monitor, [row() for _ in range(5)])
    labels = feed(monitor, [row(alpha=0.2, delta_w=1e-2), row(alpha=0.2, delta_w=1e-2), row()])
    assert set(labels) == {EASY}
