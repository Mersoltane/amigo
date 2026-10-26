"""Classification of a solve into regimes from the signals of each iteration.

The monitor reads the quantities the logger assembles and labels every
iteration easy, hard, jam, or stuck, changing its label only after three
iterations of agreement. An easy iteration takes a nearly full step with no
Hessian perturbation. A hard iteration needs the perturbation but still
moves. A jam iteration backtracks to a tiny step along a large direction
while the error stands still, the signature of a KKT matrix whose inertia
is correct but whose constraints are nearly singular. A stuck iteration
leaves the constraint violation in place although the dual error is small,
which leans toward infeasibility. The monitor only observes, so the label
is free to read and free to ignore.
"""

from collections import deque

EASY = "easy"
HARD = "hard"
JAM = "jam"
STUCK = "stuck"

# Iterations of agreement before the label changes
AGREE = 3

# History the progress tests look across
WINDOW = 5


class RegimeMonitor:
    """The regime of a solve, updated from the diagnostics of each iteration."""

    def __init__(self):
        self.regime = EASY
        self._candidate = None
        self._agreed = 0
        self._errors = deque(maxlen=WINDOW)
        self._thetas = deque(maxlen=WINDOW)

    def classify(self, error, theta, dual, alpha, trials, delta_w):
        """The label of one iteration alone, from its step and the recent progress."""
        full = len(self._errors) == WINDOW
        improving = not full or error < 0.9 * max(self._errors)
        settling = not full or theta < 0.99 * max(self._thetas) or theta <= 1e-9
        if alpha < 1e-2 and trials >= 8 and not improving:
            return JAM
        if not settling and theta > 1e-9 and dual < theta:
            return STUCK
        if delta_w == 0.0 and alpha >= 0.5 and trials <= 2:
            return EASY
        return HARD

    def update(self, info):
        """The regime after one iteration, read from the assembled diagnostics of the logger."""
        label = self.classify(
            info.get("nlp_error", 0.0),
            info.get("inf_pr", 0.0),
            info.get("inf_du", 0.0),
            info.get("alpha_x", 0.0),
            info.get("line_iters", 0),
            info.get("inertia_delta", 0.0),
        )
        self._errors.append(info.get("nlp_error", 0.0))
        self._thetas.append(info.get("inf_pr", 0.0))
        if label == self.regime:
            self._candidate, self._agreed = None, 0
        elif label == self._candidate:
            self._agreed += 1
            if self._agreed >= AGREE:
                self.regime = label
                self._candidate, self._agreed = None, 0
        else:
            self._candidate, self._agreed = label, 1
        return self.regime
