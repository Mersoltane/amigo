"""Iteration-data assembly and progress-table printing.

Builds the per-iteration record (objective, NLP error, step sizes,
step norm, filter size, etc.) and prints the IPM progress table.
"""

import math
import sys
import time

from .convergence_check import (
    CONVERGED,
    CONVERGED_ACCEPTABLE,
    STATUS_MESSAGES,
)
from .regime import RegimeMonitor


class OptimizationLogger:
    def __init__(self, log_objs, options, problem, optimizer):
        self.options = options
        self.log_objs = log_objs
        self.problem = problem
        self.optimizer = optimizer

        # Set the start time on creation of the optimization logger
        self.start_time = time.perf_counter()

        # Store the optimization data
        self.opt_data = {"options": self.options, "converged": False, "iterations": []}

        # The regime of the solve, read from the diagnostics of each iteration
        self.regime = RegimeMonitor()
        return

    def get_data(self):
        return self.opt_data

    def log_iteration(self, status, state):
        """Assemble the dict of per-iteration diagnostics for logging/history."""
        elapsed_time = time.perf_counter() - self.start_time

        iter_data = {
            "status": status,
            "iteration": state.iter,
            "time": elapsed_time,
            "residual": state.residual_norm,
            "barrier_param": state.mu,
        }

        # Add additional iteration-dependent data
        for obj in self.log_objs:
            obj.add_log_info(iter_data)

        # The unscaled objective and violation with the scaled dual and compl errors
        iter_data["inf_pr"] = state.constr_viol
        iter_data["inf_du"] = state.dual_infeas
        iter_data["compl"] = state.complementarity
        iter_data["nlp_error"] = state.kkt_error
        iter_data["objective"] = state.objective_value / state.obj_scale
        px = state.step.get_solution()
        iter_data["step_norm"] = self.problem.maxabs(px)

        # The breach counts come from the step contract
        iter_data.setdefault("dual_viol_total", 0)
        iter_data["regime"] = self.regime.update(iter_data)

        # Write out the values
        self._write_log(status, state, iter_data)

        # Append the iteration data
        self.opt_data["iterations"].append(iter_data)
        return

    def finish(self, status, state):
        """Record the final status in the returned data and print the summary."""
        self.opt_data["status"] = status
        self.opt_data["status_message"] = STATUS_MESSAGES[status]
        self.opt_data["converged"] = status in (CONVERGED, CONVERGED_ACCEPTABLE)
        self.opt_data["acceptable"] = status == CONVERGED_ACCEPTABLE

        if state.comm_rank != 0 or not self.opt_data["iterations"]:
            return
        iter_data = self.opt_data["iterations"][-1]
        iteration = iter_data["iteration"]
        if status == CONVERGED:
            title = f"Amigo converged in {iteration} iterations"
        elif status == CONVERGED_ACCEPTABLE:
            title = f"Amigo converged to acceptable point in {iteration} iterations"
        else:
            title = f"Amigo stopped after {iteration} iterations: {STATUS_MESSAGES[status]}"
        self._print_summary(title, state, iter_data)
        sys.stdout.flush()

    def _write_log(self, status, state, iter_data):
        """Print the IPM iteration table row for this iteration."""
        iteration = iter_data["iteration"]
        if iteration % 20 == 0:
            print(
                f"{'iter':>4s}  {'nlp_error':>9s}  {'objective':>12s}  "
                f"{'inf_pr':>9s}  {'inf_du':>9s}  {'compl':>9s}  "
                f"{'mu':>9s}  {'||d||':>9s}  {'delta_w':>8s}  {'delta_c':>8s}  {'piv':>7s}  {'ic':>2s}  "
                f"{'alpha_x':>8s}  {'alpha_z':>8s}  "
                f"{'ls':>2s}  {'filt':>4s}  {'rej':>4s}  {'frej':>4s}  {'dv':>3s}  {'why':>8s}  {'soc':>4s}  {'regime':>6s}"
            )

        mu = iter_data.get("barrier_param", 1.0)
        delta_w = iter_data.get("inertia_delta", 0.0)
        nlp_err = iter_data.get("nlp_error", 0.0)
        obj = iter_data.get("objective", 0.0)
        inf_pr = iter_data.get("inf_pr", 0.0)
        inf_du = iter_data.get("inf_du", 0.0)
        compl = iter_data.get("compl", 0.0)
        step_norm = iter_data.get("step_norm", 0.0)
        ax = iter_data.get("alpha_x", 0.0)
        az = iter_data.get("alpha_z", 0.0)
        ls = iter_data.get("line_iters", 0)
        fsize = iter_data.get("filter_size", 0)
        rej = iter_data.get("rejections", 0)
        frej = iter_data.get("filter_rejections", 0)
        # Factorizations spent on inertia correction, the main GPU cost
        ic = iter_data.get("inertia_attempts", 0)

        dw_str = f"{delta_w:8.1e}" if delta_w > 0 else f"{'---':>8s}"

        # The applied delta_c and the smallest pivot of the accepted system
        delta_c = iter_data.get("inertia_delta_c", 0.0)
        dc_str = f"{delta_c:8.1e}" if delta_c > 0 else f"{'---':>8s}"
        pivot = iter_data.get("kkt_min_pivot")
        piv_str = f"{pivot:7.0e}" if pivot is not None and math.isfinite(pivot) else f"{'---':>7s}"

        dv = iter_data.get("dual_viol_total", 0)
        why = iter_data.get("ls_why") or "---"
        soc = iter_data.get("soc") or "---"
        regime = iter_data.get("regime", "")
        print(
            f"{iteration:4d}  {nlp_err:9.2e}  {obj:12.5e}  "
            f"{inf_pr:9.2e}  {inf_du:9.2e}  {compl:9.2e}  "
            f"{mu:9.2e}  {step_norm:9.2e}  {dw_str}  {dc_str}  {piv_str}  {ic:2d}  "
            f"{ax:8.2e}  {az:8.2e}  "
            f"{ls:2d}  {fsize:4d}  {rej:4d}  {frej:4d}  {dv:3d}  {why:>8s}  {soc:>4s}  {regime:>6s}"
        )
        sys.stdout.flush()

    def _print_summary(self, title, state, iter_data):
        """Print the final summary with the unscaled measures of the solution."""
        rows = [
            ("Objective value", state.objective_value / state.obj_scale),
            ("NLP error", state.kkt_error),
            ("Constraint violation", state.constr_viol),
            ("Dual infeasibility", state.dual_infeas / state.obj_scale),
            ("Complementarity", state.complementarity / state.obj_scale),
            ("Barrier parameter", state.mu),
        ]
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        for label, value in rows:
            print(f"  {label:<24s}{value:>20.10e}")
        print(f"  {'Total iterations':<24s}{iter_data['iteration']:>20d}")
        resto_calls = iter_data.get("resto_calls", 0)
        if resto_calls > 0:
            resto_iters = iter_data["resto_iterations"]
            print(f"  {'Restoration iterations':<24s}{resto_iters:>20d}")
            print(f"  {'Restoration calls':<24s}{resto_calls:>20d}")
        total_rej = iter_data.get("total_rejections")
        if total_rej is not None:
            frej_total = iter_data.get("total_filter_rejections", 0)
            print(f"  {'Trial rejections':<24s}{total_rej:>20d}")
            print(f"  {'Filter vetoes':<24s}{frej_total:>20d}")
        print(f"{'='*70}")
