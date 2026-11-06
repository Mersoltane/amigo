"""
Regime labels
-------------

- The easy, hard, jam, or stuck label read from the diagnostics of each iteration
"""

from amigo.algorithm.monitoring.regime import EASY, HARD, JAM, STUCK, RegimeMonitor


def row(error=1e-3, theta=1e-4, dual=1e-3, alpha=1.0, trials=1, delta_w=0.0):
    """The diagnostics of one iteration"""
    return {
        "nlp_error": error,
        "inf_pr": theta,
        "inf_du": dual,
        "alpha_x": alpha,
        "line_iters": trials,
        "inertia_delta": delta_w,
    }


def feed(monitor, entries):
    """Labels of the monitor after every entry in order"""
    labels = []

    for entry in entries:
        labels.append(monitor.update(entry))

    return labels


def test_full_steps_without_perturbation_stay_easy():
    monitor = RegimeMonitor()

    # Every iteration lowers the error tenfold with a full step
    entries = []

    for k in range(8):
        entries.append(row(error=10.0**-k))

    labels = feed(monitor, entries)

    assert set(labels) == {EASY}


def test_perturbed_steps_that_move_label_hard_after_three_iterations():
    monitor = RegimeMonitor()

    # Short perturbed steps that still lower the error
    entries = []

    for k in range(6):
        entries.append(row(error=10.0**-k, alpha=0.2, delta_w=1e-2))

    labels = feed(monitor, entries)

    assert labels[:2] == [EASY, EASY]
    assert labels[-1] == HARD


def test_tiny_backtracked_steps_without_progress_label_a_jam():
    monitor = RegimeMonitor()

    # Quiet iterations, then tiny backtracked steps that make no progress
    entries = []

    for k in range(5):
        entries.append(row(error=1.0, theta=3e-3))

    for k in range(4):
        entries.append(row(error=1.0, theta=3e-3, alpha=2e-3, trials=10, delta_w=1e-1))

    labels = feed(monitor, entries)

    assert labels[-2] == JAM and labels[-1] == JAM

    # Progress with full steps releases the jam after three agreeing iterations
    eased = []

    for k in range(4):
        eased.append(row(error=10.0**-k, theta=1e-10))

    labels = feed(monitor, eased)

    assert labels[-1] == EASY


def test_a_standing_violation_with_small_dual_error_labels_stuck():
    monitor = RegimeMonitor()

    entries = []

    for k in range(9):
        entries.append(row(error=1e-2, theta=5e-3, dual=1e-4, alpha=0.9))

    labels = feed(monitor, entries)

    assert labels[-1] == STUCK


def test_two_agreeing_iterations_do_not_change_the_label():
    monitor = RegimeMonitor()

    # Settle on easy first
    settled = []

    for k in range(5):
        settled.append(row())

    feed(monitor, settled)

    # Two hard looking iterations followed by an easy one
    labels = feed(monitor, [row(alpha=0.2, delta_w=1e-2), row(alpha=0.2, delta_w=1e-2), row()])

    assert set(labels) == {EASY}
