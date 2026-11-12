"""Polish of the constraints at an acceptable stop

An acceptable stop ends a solve whose constraints may still sit above the
convergence tolerance. The polish takes normal and centering steps, which
move onto the constraints and the central path without descending the
objective, while each step halves the violation and the acceptable
iteration budget lasts.
"""


class ConstraintPolish:
    """Decides whether an acceptable stop takes another normal step"""

    def __init__(self, options):
        self.tolerance = options["convergence_tolerance"]
        self.budget = options["acceptable_iter"]

        # Normal steps taken and the violation before the last one
        self.steps = 0
        self.last_violation = None

    def claim_step(self, state):
        """Claim a normal step while the violation is above the tolerance and the last step halved it"""
        if state.constr_viol <= self.tolerance:
            return False

        # The budget of the acceptable stop bounds the polish
        if self.steps >= self.budget:
            return False

        # A step that did not halve the violation ends the polish
        if self.last_violation is not None and state.constr_viol > 0.5 * self.last_violation:
            return False

        self.last_violation = state.constr_viol
        self.steps += 1
        return True
