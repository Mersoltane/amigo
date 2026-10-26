"""Gradient-based NLP scaling, computed once at the start point.

Scales the objective through state.obj_scale and each constraint at the
problem boundary, so the derivatives run in scaled space. The solver works
with scaled multipliers lambda_s = obj_scale * lambda / dc and scaled bound
duals z_s = obj_scale * z, while every point passed in or returned holds the
unscaled values.
"""

import numpy as np


class NLPScaling:
    def __init__(self, options, problem, optimizer):
        self.problem = problem
        self.optimizer = optimizer
        self.enabled = options["nlp_scaling"]
        self.g_max = options["nlp_scaling_max_gradient"]
        self.min_value = options["nlp_scaling_min_value"]
        self.obj_factor = float(options["objective_scaling_factor"])
        if not self.obj_factor > 0.0:
            raise ValueError("objective_scaling_factor must be positive")

        self.obj_scale = 1.0
        self.min_con_scale = 1.0
        self.con_scale = None
        self.temp_primal = problem.create_primal_vector()
        self.temp_con = problem.create_constraint_vector()

    def compute(self, evaluator, state):
        """Compute the scaling factors at the current point and apply them."""
        # Measure the unscaled problem so repeated solves find the same scaling
        self.optimizer.set_var_scale(None)
        self.obj_scale = 1.0
        self.min_con_scale = 1.0
        self.con_scale = None
        state.obj_scale = 1.0
        state.scaled_constraints = False
        state.invalidate()
        if self.enabled:
            self.obj_scale = self._objective_scale(evaluator, state)
            self._set_constraint_scale(evaluator, state)

        # The user factor applies with or without the gradient based scaling
        self.obj_scale *= self.obj_factor
        state.obj_scale = self.obj_scale
        state.invalidate()
        if self.obj_scale != 1.0 or self.min_con_scale != 1.0:
            print(
                f"NLP scaling: obj_scale = {self.obj_scale:.3e}, "
                f"min constraint scale = {self.min_con_scale:.3e}"
            )

    def scale_point(self, state):
        """Convert the multipliers and bound duals of the current point to scaled values."""
        self._rescale_point(state, self.obj_scale, True)

    def unscale_point(self, state):
        """Convert the multipliers and bound duals back, then restore the unscaled problem."""
        self._rescale_point(state, 1.0 / self.obj_scale, False)
        self.optimizer.set_var_scale(None)
        state.obj_scale = 1.0
        state.scaled_constraints = False
        state.invalidate()

    def _rescale_point(self, state, dual_factor, divide_by_con_scale):
        """Multiply the bound duals by dual_factor and the multipliers by dual_factor / dc or dual_factor * dc."""
        if dual_factor != 1.0:
            state.current.get_zl().scale(dual_factor)
            state.current.get_zu().scale(dual_factor)

        if dual_factor == 1.0 and self.con_scale is None:
            return

        x = state.get_current_point()
        con_indices = self.problem.get_constraint_indices()
        x.get_values_at(con_indices, self.temp_con)
        self.temp_con.copy_device_to_host()
        lam = self.temp_con.get_array()
        lam *= dual_factor
        if self.con_scale is not None:
            if divide_by_con_scale:
                lam /= self.con_scale
            else:
                lam *= self.con_scale
        self.temp_con.copy_host_to_device()
        x.set_values_at(con_indices, self.temp_con)
        state.invalidate()

    def _objective_scale(self, evaluator, state):
        """df from the inf-norm of the objective gradient at x0."""
        x = state.get_current_point()
        con_indices = self.problem.get_constraint_indices()
        primal_indices = self.problem.get_primal_indices()

        # Objective gradient alone: zero multipliers, obj_scale = 1
        x.get_values_at(con_indices, self.temp_con)
        x.fill_at(con_indices, 0.0)
        obj_scale_store = state.obj_scale
        state.obj_scale = 1.0
        state.invalidate()
        evaluator.evaluate_gradient(state)
        state.obj_scale = obj_scale_store

        state.gradient.get_values_at(primal_indices, self.temp_primal)
        max_grad_f = self.problem.maxabs(self.temp_primal)

        # Restore the multipliers and mark the state stale
        x.set_values_at(con_indices, self.temp_con)
        state.invalidate()

        if max_grad_f > self.g_max:
            return max(self.g_max / max_grad_f, self.min_value)
        return 1.0

    def _set_constraint_scale(self, evaluator, state):
        """dc[i] from the constraint Jacobian row inf-norms at x0.

        The assembled KKT has a structurally zero dual diagonal, so the dual
        row max-abs is exactly ||J_i||_inf.
        """
        con_indices = self.problem.get_constraint_indices()

        evaluator.evaluate_hessian(state)
        rowmax = self.problem.create_vector()
        state.hessian.row_maxabs(rowmax)
        rowmax.get_values_at(con_indices, self.temp_con)
        self.temp_con.copy_device_to_host()

        r = self.temp_con.get_array()
        dc = np.minimum(1.0, self.g_max / np.maximum(r, 1e-300))
        dc = np.maximum(dc, self.min_value)
        self.min_con_scale = float(dc.min()) if dc.size else 1.0

        # Unit constraint scales leave the problem unscaled
        if self.min_con_scale < 1.0:
            self.con_scale = dc.copy()
            state.scaled_constraints = True

            # dc goes on the constraint duals, primals and slacks stay unscaled
            self.temp_con.get_array()[:] = dc
            self.temp_con.copy_host_to_device()
            var_scale = self.problem.create_vector()
            var_scale.get_array()[:] = 1.0
            var_scale.copy_host_to_device()
            var_scale.set_values_at(con_indices, self.temp_con)
            self.optimizer.set_var_scale(var_scale)

        state.invalidate()

    def get_log_info(self):
        return {"obj_scale": self.obj_scale, "min_con_scale": self.min_con_scale}
