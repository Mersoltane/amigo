"""Classical barrier strategy: LOQO-style heuristic or Ipopt monotone rule."""

from .base import BarrierStrategy


class HeuristicBarrier(BarrierStrategy):
    """Classical barrier update: heuristic reduction or monotone rule."""

    def step(self, ctx):
        """Run one classical barrier-update iteration."""
        opt = self.opt
        options = self.options
        tol = ctx.tol
        compl_inf_tol = ctx.compl_inf_tol
        comm_rank = ctx.comm_rank
        res_norm = ctx.res_norm
        i = ctx.i
        mult_ind = ctx.mult_ind

        # Filter-monotone overrides both heuristic and standard monotone
        if ctx.filter_monotone_mode:
            if res_norm <= options["barrier_progress_tol"] * ctx.filter_monotone_mu:
                new_mu = max(
                    ctx.filter_monotone_mu * options["monotone_barrier_fraction"],
                    tol,
                )
                if comm_rank == 0:
                    print(
                        f"  Filter monotone: mu "
                        f"{ctx.filter_monotone_mu:.2e}->{new_mu:.2e}"
                    )
                ctx.filter_monotone_mu = new_mu
            opt.barrier_param = ctx.filter_monotone_mu

        elif i > 0 and opt.barrier_param > min(tol, compl_inf_tol) / (
            options["barrier_tol_factor"] + 1.0
        ):
            self._maybe_reduce_barrier(mult_ind, res_norm, tol, compl_inf_tol)

        if ctx.inertia_corrector:
            ctx.inertia_corrector.update_barrier(opt.barrier_param)
        return opt._find_direction(
            ctx.x,
            ctx.diag_base,
            ctx.inertia_corrector,
            mult_ind,
            options,
            ctx.zero_hessian_indices,
            ctx.zero_hessian_eps,
            comm_rank,
        )

    def _maybe_reduce_barrier(self, mult_ind, res_norm, tol, compl_inf_tol):
        """Reduce mu via heuristic or monotone rule if subproblem solved."""
        opt = self.opt
        options = self.options
        kappa_eps = options["barrier_tol_factor"]
        kappa_mu = options["mu_linear_decrease_factor"]
        theta_mu = options["mu_superlinear_decrease_power"]

        heuristic = options["barrier_strategy"] == "heuristic"
        if heuristic:
            comp_h, xi_h = opt.optimizer.compute_complementarity(opt.vars)
            should_reduce = True
        elif options["progress_based_barrier"]:
            should_reduce = res_norm <= kappa_eps * opt.barrier_param
        else:
            should_reduce = res_norm <= 0.1 * opt.barrier_param

        if not should_reduce:
            return

        if heuristic:
            mu_floor = min(tol, compl_inf_tol) / (kappa_eps + 1.0)
            opt.barrier_param, _ = self._loqo_heuristic(
                xi_h,
                comp_h,
                options["heuristic_barrier_gamma"],
                options["heuristic_barrier_r"],
                mu_floor,
            )
        else:
            while True:
                mu = opt.barrier_param
                e_mu = opt._compute_scaled_barrier_error(mu, mult_ind)
                if e_mu > kappa_eps * mu:
                    break
                old_mu = mu
                new_mu = min(kappa_mu * mu, mu**theta_mu)
                mu_fl = min(tol, compl_inf_tol) / (kappa_eps + 1.0)
                new_mu = max(new_mu, mu_fl)
                if new_mu >= old_mu:
                    break
                opt.barrier_param = new_mu

    @staticmethod
    def _loqo_heuristic(xi, complementarity, gamma, r, tol=1e-12):
        """LOQO-style barrier parameter: mu = gamma * heuristic_factor * comp."""
        if xi > 1e-10:
            term = (1 - r) * (1 - xi) / xi
            heuristic_factor = min(term, 2.0) ** 3
        else:
            heuristic_factor = 2.0**3
        mu_new = gamma * heuristic_factor * complementarity
        return max(mu_new, tol), heuristic_factor
