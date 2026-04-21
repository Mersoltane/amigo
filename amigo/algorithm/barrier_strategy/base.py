"""Base class for barrier-parameter strategies.

A BarrierStrategy owns the per-iteration barrier update:
  - decide the new mu (heuristic rule or QF oracle)
  - factorize the KKT system at the new mu
  - compute the Newton direction

Every concrete strategy reads and writes self.opt.barrier_param and uses
self.opt.{vars, grad, res, px, update, temp, optimizer, solver} plus
helpers on self.opt (_factorize_kkt, _find_direction, etc.).
"""

from abc import ABC, abstractmethod


class BarrierStrategy(ABC):
    """Abstract base for barrier-parameter update strategies."""

    def __init__(self, opt, options):
        self.opt = opt
        self.options = options

    @abstractmethod
    def step(self, ctx):
        """Update barrier, factorize KKT, compute Newton direction.

        Parameters
        ----------
        ctx : StepContext
            Per-iteration context (see ipm_driver).

        Returns
        -------
        factorize_ok : bool
            False if inertia correction failed.
        """

    def initialize(self, ctx):
        """One-shot init at the start of optimize().  Override if needed."""

    def on_step_rejected(self, ctx):
        """Run after the line search rejects the step."""

    def on_barrier_increased(self):
        """Run after increase_barrier_on_rejections() raised mu."""

    def handle_zero_step_recovery(
        self, i, alpha_x_prev, alpha_z_prev, zero_step_count, comm_rank
    ):
        """Escape stuck iterates by bumping mu when tiny steps repeat.

        Only active on the non-inertia-corrector path.  After three
        consecutive iterations with max(alpha_x, alpha_z) < 1e-10,
        scale mu up by 10x (capped at 1.0).
        """
        if i > 0 and max(alpha_x_prev, alpha_z_prev) < 1e-10:
            zero_step_count += 1
            if zero_step_count >= 3:
                old = self.opt.barrier_param
                self.opt.barrier_param = min(old * 10.0, 1.0)
                if comm_rank == 0 and self.opt.barrier_param != old:
                    print(
                        f"  Zero step recovery: barrier "
                        f"{old:.2e} -> {self.opt.barrier_param:.2e}"
                    )
                zero_step_count = 0
        else:
            zero_step_count = 0
        return zero_step_count

    def increase_barrier_on_rejections(
        self,
        consecutive_rejections,
        max_rejections,
        barrier_inc,
        initial_barrier,
        comm_rank,
    ):
        """Increase mu after max_rejections consecutive rejections."""
        if consecutive_rejections >= max_rejections:
            new_barrier = min(self.opt.barrier_param * barrier_inc, initial_barrier)
            if new_barrier > self.opt.barrier_param:
                if comm_rank == 0:
                    print(
                        f"  Barrier increased: {self.opt.barrier_param:.2e} -> "
                        f"{new_barrier:.2e}"
                    )
                self.opt.barrier_param = new_barrier
            elif comm_rank == 0:
                print(
                    f"  Barrier at max ({self.opt.barrier_param:.2e}), "
                    f"cannot increase further"
                )
            return 0
        return consecutive_rejections
