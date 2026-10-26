"""The homotopy loop that carries a solution along one parameter of a problem from its first value to its last.

Each step solves the problem at the next value from the solution of the last
value that converged, so every solve starts close to its answer. A step that
fails is halved toward the last converged value, geometrically when both ends
share a sign so that a parameter shrinking toward zero keeps its ratio, and the
path stops once a step fails after its halvings. The values and the solve
belong to the caller, which keeps any warm start of its own solver.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HomotopyOptions:
    """Options of a homotopy path."""

    # Most halvings in a row of one step before the path stops
    max_bisections: int = 6


@dataclass
class HomotopyStep:
    """One solve along the path at one value of the parameter."""

    value: float
    solution: object
    converged: bool


def midpoint(a, b):
    """The value halfway between two values, geometric when both share a sign and arithmetic otherwise."""
    if a * b > 0.0:
        return float(np.sign(a) * np.sqrt(a * b))
    return 0.5 * (a + b)


class HomotopyHistory(list):
    """The solves along a path in order."""

    @property
    def last(self):
        """The last step that converged, or None."""
        for step in reversed(self):
            if step.converged:
                return step
        return None

    def reached(self, value):
        """Whether a solve converged at a value."""
        return any(step.converged and step.value == value for step in self)


class Homotopy:
    """Path following in one parameter around a user supplied solve."""

    def __init__(self, options=None):
        self.options = HomotopyOptions() if options is None else options

    def run(self, values, solve, start=None):
        """Solve at every value in order, each from the last converged solution, with the steps that fail halved.

        solve(value, previous) returns a solution with a converged flag and
        starts from previous, the solution at the last value that converged,
        which is start before the first. A first value that fails stops the
        path, since no converged value lies behind it.
        """
        history = HomotopyHistory()
        targets = list(values)
        previous, at, halvings, i = start, None, 0, 0
        while i < len(targets):
            solution = solve(targets[i], previous)
            history.append(HomotopyStep(targets[i], solution, bool(solution.converged)))
            if solution.converged:
                previous, at, halvings = solution, targets[i], 0
                i += 1
            elif at is None or halvings == self.options.max_bisections:
                break
            else:
                # The failed value stays a target once the value halfway to it converges
                targets.insert(i, midpoint(at, targets[i]))
                halvings += 1
        return history
