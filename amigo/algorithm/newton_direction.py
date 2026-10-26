"""Newton step from an already-factorized KKT system.

Solves against the factorization prepared by InertiaCorrector, forms the
full primal-dual update, and records the fraction-to-boundary step lengths
on the state.
"""


def set_step_lengths(optimizer, state):
    """Record the fraction to the boundary step lengths of state.step."""
    alpha_x, _, alpha_z, _ = optimizer.compute_max_step(
        state.tau, state.current, state.step
    )
    state.max_alpha_primal = alpha_x
    state.max_alpha_dual = alpha_z
    state.step_current = True


class NewtonStep:
    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        self.update = self.problem.create_vector()

    def compute_step(self, solver, evaluator, state):
        # Evalute the residual (may be required since the barrier may have changed)
        evaluator.evaluate_residual(state)

        # Solve the linear system to obtain the new step
        solver.solve(state.residual, self.update)

        # Compute the full step and its lengths
        self.optimizer.compute_update(state.mu, state.current, self.update, state.step)
        set_step_lengths(self.optimizer, state)
