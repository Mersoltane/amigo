"""Quality-function (adaptive) barrier strategy.

Picks the next mu by either a Mehrotra predictor-corrector or a golden-
section search on q_L(sigma).  Wraps this oracle in the adaptive-mu
globalization from Nocedal & Wachter (2006): if the QF-picked mu stops
making progress (by KKT error or obj-constr filter), fall back to a
monotone decrease of mu until the subproblem is solved.
"""

import numpy as np

from .base import BarrierStrategy


class QualityFunctionBarrier(BarrierStrategy):
    """Mehrotra PC / golden-section QF oracle with adaptive-mu globalization."""

    def __init__(self, opt, options):
        super().__init__(opt, options)

        # Scaling factors for the QF (set in initialize())
        self.sd = 1.0
        self.sp = 1.0
        self.sc = 1.0
        self.mult_ind = None

        # Globalization state
        self.free_mode = True
        self.monotone_mu = None
        self.refs = []  # kkt-error reference values
        self.glob_filter = []  # (f, theta) filter for obj-constr-filter
        self.init_dual_inf = -1.0
        self.init_primal_inf = -1.0

        # Affine step snapshot (for diagnostics/restoration)
        self.delta_aff = None

        # One-shot mu bounds (populated on first step)
        self.mu_min = options["mu_min"]
        self.mu_max = -1.0
        self._mu_min_default = True

    def initialize(self, ctx):
        """Scaling factors, mult_ind, initial QF reference value."""
        opt = self.opt
        options = self.options
        self.mult_ind = ctx.mult_ind

        if options["quality_function_norm_scaling"]:
            n_d, n_p, n_c = opt.optimizer.get_kkt_element_counts()
            self.sd = 1.0 / max(n_d, 1)
            self.sp = 1.0 / max(n_p, 1)
            self.sc = 1.0 / max(n_c, 1)

        # Seed the kkt-error reference with the initial error
        d0, p0, c0 = opt.optimizer.compute_kkt_error(opt.vars, opt.grad)
        init_qf = d0 * self.sd + p0 * self.sp + c0 * self.sc
        self.refs.append(init_qf)

    def step(self, ctx):
        """Run one QF barrier-update iteration."""
        opt = self.opt
        options = self.options
        comm_rank = ctx.comm_rank

        self._initialize_bounds_once(ctx)

        # Mode switching before direction
        if not self.free_mode:
            if self._sufficient_progress():
                if comm_rank == 0 and options.get("verbose_barrier"):
                    print("  QF: switching back to free mode")
                self.free_mode = True
                self._remember_point()
            else:
                self._monotone_reduce(ctx)

        if self.free_mode:
            glob = options["adaptive_mu_globalization"]
            if glob == "never-monotone" or self._sufficient_progress():
                self._remember_point()
            else:
                self._enter_monotone_mode(ctx)

        factorize_ok = opt._factorize_kkt(
            ctx.x,
            ctx.diag_base,
            ctx.inertia_corrector,
            ctx.mult_ind,
            options,
            ctx.zero_hessian_indices,
            ctx.zero_hessian_eps,
            comm_rank,
        )
        if not factorize_ok:
            return False

        if self.free_mode:
            result = self._quality_function_mu(comm_rank)
            if result is not None:
                _, mu_qf = result
                mu_qf = max(mu_qf, self.mu_min, self._lower_safeguard())
                mu_qf = min(mu_qf, self.mu_max)
                opt.barrier_param = mu_qf
        else:
            opt._solve_with_mu(opt.barrier_param)
        return True

    def on_step_rejected(self, ctx):
        """If free mode rejects, fall back to monotone (unless never-monotone)."""
        options = self.options
        comm_rank = ctx.comm_rank
        opt = self.opt

        if not self.free_mode:
            return
        if options["adaptive_mu_globalization"] == "never-monotone":
            return

        comp, _ = opt.optimizer.compute_complementarity(opt.vars)
        cand = options["adaptive_mu_monotone_init_factor"] * comp
        cand = max(cand, self._lower_safeguard(), self.mu_min)
        cand = min(cand, self.mu_max)
        opt.barrier_param = cand
        self.free_mode = False
        self.monotone_mu = cand
        if comm_rank == 0:
            print(f"  QF -> monotone (step rejected): mu_bar={cand:.3e}")

    def on_barrier_increased(self):
        """If in monotone mode, keep monotone_mu in sync with the new mu."""
        if not self.free_mode:
            self.monotone_mu = self.opt.barrier_param

    def _initialize_bounds_once(self, ctx):
        """Populate mu bounds and initial-infeasibility refs on first call."""
        opt = self.opt
        options = self.options
        if self._mu_min_default:
            self.mu_min = min(options["mu_min"], 0.5 * min(ctx.tol, ctx.compl_inf_tol))
            self._mu_min_default = False
        if self.mu_max < 0:
            avg_comp_init, _ = opt.optimizer.compute_complementarity(opt.vars)
            self.mu_max = options["mu_max_fact"] * max(avg_comp_init, 1.0)
        if self.init_dual_inf < 0:
            d0, p0, _ = opt.optimizer.compute_kkt_error_mu(0.0, opt.vars, opt.grad)
            self.init_dual_inf = max(1.0, d0)
            self.init_primal_inf = max(1.0, p0)

    def _monotone_reduce(self, ctx):
        """Monotone mu reduction when subproblem is solved."""
        opt = self.opt
        options = self.options
        btf = options["barrier_tol_factor"]
        barrier_err = opt._compute_scaled_barrier_error(opt.barrier_param, ctx.mult_ind)
        if barrier_err > btf * opt.barrier_param:
            return

        kmu = options["mu_linear_decrease_factor"]
        tmu = options["mu_superlinear_decrease_power"]
        new_mu = min(kmu * opt.barrier_param, opt.barrier_param**tmu)
        floor = min(ctx.tol, ctx.compl_inf_tol) / (btf + 1.0)
        new_mu = max(new_mu, floor, self.mu_min)
        new_mu = min(new_mu, self.mu_max)

        if ctx.comm_rank == 0 and options.get("verbose_barrier"):
            print(f"  QF monotone: {opt.barrier_param:.3e} -> {new_mu:.3e}")
        opt.barrier_param = new_mu
        self.monotone_mu = new_mu

    def _enter_monotone_mode(self, ctx):
        """Free mode lost progress: start monotone phase."""
        opt = self.opt
        options = self.options
        self.free_mode = False
        avg_c, _ = opt.optimizer.compute_complementarity(opt.vars)
        new_mu = options["adaptive_mu_monotone_init_factor"] * avg_c
        new_mu = max(new_mu, self._lower_safeguard(), self.mu_min)
        new_mu = min(new_mu, self.mu_max)
        opt.barrier_param = new_mu
        self.monotone_mu = new_mu
        if ctx.comm_rank == 0:
            print(f"  QF -> monotone: mu_bar={new_mu:.3e} (avg_comp={avg_c:.3e})")

    def _sufficient_progress(self):
        """Is free mode still making progress w.r.t. the chosen globalization?"""
        opt = self.opt
        options = self.options
        glob = options["adaptive_mu_globalization"]

        if glob == "never-monotone":
            return True

        if glob == "kkt-error":
            num_refs_max = options["adaptive_mu_kkterror_red_iters"]
            if len(self.refs) < num_refs_max:
                return True
            curr = self._kkt_quality()
            red_fact = options["adaptive_mu_kkterror_red_fact"]
            return any(curr <= red_fact * ref for ref in self.refs)

        if glob == "obj-constr-filter":
            f_curr = opt._compute_barrier_objective(opt.vars)
            theta_curr = opt._compute_filter_theta()
            margin = options.get("filter_margin_fact", 1e-5) * min(
                options.get("filter_max_margin", 1.0),
                max(f_curr, theta_curr, 1e-30),
            )
            for f_filt, theta_filt in self.glob_filter:
                if f_curr + margin < f_filt or theta_curr + margin < theta_filt:
                    return True
            return len(self.glob_filter) == 0

        return True

    def _remember_point(self):
        """Record the current point as an accepted reference."""
        opt = self.opt
        options = self.options
        glob = options["adaptive_mu_globalization"]

        if glob == "kkt-error":
            curr = self._kkt_quality()
            num_refs_max = options["adaptive_mu_kkterror_red_iters"]
            if len(self.refs) >= num_refs_max:
                self.refs.pop(0)
            self.refs.append(curr)
        elif glob == "obj-constr-filter":
            f_curr = opt._compute_barrier_objective(opt.vars)
            theta_curr = opt._compute_filter_theta()
            self.glob_filter.append((f_curr, theta_curr))

    def _lower_safeguard(self):
        """Lower mu safeguard based on infeasibility progress."""
        options = self.options
        opt = self.opt
        factor = options["adaptive_mu_safeguard_factor"]
        if factor == 0.0:
            return 0.0
        d_inf, p_inf, _ = opt.optimizer.compute_kkt_error_mu(0.0, opt.vars, opt.grad)
        safe = max(
            factor * d_inf / max(self.init_dual_inf, 1.0),
            factor * p_inf / max(self.init_primal_inf, 1.0),
        )
        if options["adaptive_mu_globalization"] == "kkt-error" and self.refs:
            safe = min(safe, min(self.refs))
        return safe

    def _kkt_quality(self):
        """Scalar KKT quality used for kkt-error globalization."""
        # TODO: move to backend - combine scaling/centrality/balancing into
        # one backend.kkt_quality(options) call.
        opt = self.opt
        options = self.options
        dual_sq, primal_sq, comp_sq = opt.optimizer.compute_kkt_error(
            opt.vars, opt.grad
        )
        qf = dual_sq * self.sd + primal_sq * self.sp + comp_sq * self.sc

        centrality = options["quality_function_centrality"]
        if centrality != "none" and comp_sq > 0:
            _, xi = opt.optimizer.compute_complementarity(opt.vars)
            xi = max(xi, 1e-30)
            c_term = comp_sq * self.sc
            if centrality == "log":
                qf -= c_term * np.log(xi)
            elif centrality == "reciprocal":
                qf += c_term / xi
            elif centrality == "cubed-reciprocal":
                qf += c_term / xi**3

        if options["quality_function_balancing_term"] == "cubic":
            d_term = dual_sq * self.sd
            p_term = primal_sq * self.sp
            c_term = comp_sq * self.sc
            qf += max(0.0, max(d_term, p_term) - c_term) ** 3

        return qf

    def _quality_function_mu(self, comm_rank):
        """Pick new mu via Mehrotra PC or golden-section QF search.

        Sets self.px and self.update at the chosen mu for the caller.
        Returns (sigma, new_mu) or None on degenerate complementarity.
        """
        opt = self.opt
        options = self.options
        avg_comp, _ = opt.optimizer.compute_complementarity(opt.vars)
        if avg_comp < 1e-30:
            return 1.0, opt.barrier_param
        mu_nat = avg_comp

        # Affine (mu=0) and centering (mu=avg_comp) solves -> one factor
        opt.optimizer.compute_residual(0.0, opt.vars, opt.grad, opt.res)
        dual_inf, primal_inf, _ = opt.optimizer.compute_kkt_error(opt.vars, opt.grad)
        opt.solver.solve(opt.res, opt.px)
        px0 = opt.px.get_array().copy()

        opt.optimizer.compute_residual(mu_nat, opt.vars, opt.grad, opt.res)
        opt.solver.solve(opt.res, opt.px)
        px1 = opt.px.get_array().copy()
        dpx = px1 - px0

        if options["quality_function_predictor_corrector"]:
            return self._mehrotra(px0, dpx, mu_nat, avg_comp, comm_rank)

        return self._golden_search(
            px0, dpx, mu_nat, dual_inf, primal_inf, avg_comp, comm_rank
        )

    def _mehrotra(self, px0, dpx, mu_nat, avg_comp, comm_rank):
        opt = self.opt
        options = self.options
        opt.px.get_array()[:] = px0
        opt.px.copy_host_to_device()
        opt.optimizer.compute_update(0.0, opt.vars, opt.px, opt.update)
        alpha_aff_x, _, alpha_aff_z, _ = opt.optimizer.compute_max_step(
            1.0, opt.vars, opt.update
        )
        opt.optimizer.apply_step_update(
            alpha_aff_x, alpha_aff_z, opt.vars, opt.update, opt.temp
        )
        mu_aff, _ = opt.optimizer.compute_complementarity(opt.temp)

        sigma_max = options["quality_function_sigma_max"]
        sigma = min((mu_aff / mu_nat) ** 3, sigma_max)
        new_mu = sigma * mu_nat
        new_mu = max(new_mu, self.mu_min)
        new_mu = min(new_mu, self.mu_max)

        if comm_rank == 0 and options.get("verbose_barrier"):
            print(
                f"  PC: sigma={sigma:.4f}, mu={new_mu:.3e} "
                f"(comp={avg_comp:.3e}, mu_aff={mu_aff:.3e}, "
                f"a_aff=[{alpha_aff_x:.3f},{alpha_aff_z:.3f}])"
            )

        sigma_eff = new_mu / mu_nat if mu_nat > 0 else sigma
        opt.px.get_array()[:] = px0 + sigma_eff * dpx
        opt.px.copy_host_to_device()
        opt.optimizer.compute_update(new_mu, opt.vars, opt.px, opt.update)

        self.delta_aff = px0.copy()
        return sigma, new_mu

    def _golden_search(
        self, px0, dpx, mu_nat, dual_inf, primal_inf, avg_comp, comm_rank
    ):
        opt = self.opt
        options = self.options

        # Set up tau for trial-step probes
        d_inf_qf, p_inf_qf, c_inf_qf = opt.optimizer.compute_kkt_error_mu(
            0.0, opt.vars, opt.grad
        )
        s_d_qf, s_c_qf = opt._compute_optimality_scaling(self.mult_ind)
        nlp_error_qf = max(d_inf_qf / s_d_qf, p_inf_qf, c_inf_qf / s_c_qf)
        tau_qf = max(options["tau_min"], 1.0 - nlp_error_qf)

        sigma_lo_opt = max(options["quality_function_sigma_min"], self.mu_min / mu_nat)
        sigma_up_opt = min(options["quality_function_sigma_max"], self.mu_max / mu_nat)
        n_gs = options["quality_function_golden_iters"]
        sigma_tol = options["quality_function_section_sigma_tol"]
        qf_tol = options["quality_function_section_qf_tol"]
        centrality = options["quality_function_centrality"]
        balancing = options["quality_function_balancing_term"]

        def _eval(sigma):
            return self._evaluate_qf(
                sigma,
                px0,
                dpx,
                mu_nat,
                tau_qf,
                dual_inf,
                primal_inf,
                0.0,
                centrality,
                balancing,
            )

        tol_probe = max(1e-4, sigma_tol)
        sigma_1m = 1.0 - tol_probe
        qf_1 = _eval(1.0)
        qf_1m = _eval(sigma_1m)

        if comm_rank == 0 and options.get("verbose_barrier"):
            print(
                f"  QF slope: qf(1-)={qf_1m:.4e}, qf(1)={qf_1:.4e}, "
                f"search={'>' if qf_1m > qf_1 else '<'}1, "
                f"tau={tau_qf:.6f}, nlp_err={nlp_error_qf:.2e}"
            )

        if qf_1m > qf_1:
            if 1.0 >= sigma_up_opt:
                sigma_star = sigma_up_opt
            else:
                sigma_star, _ = _golden_section(
                    _eval, 1.0, sigma_up_opt, sigma_tol, qf_tol, n_gs
                )
        else:
            gs_up = min(max(sigma_lo_opt, sigma_1m), self.mu_max / mu_nat)
            if sigma_lo_opt >= gs_up:
                sigma_star = sigma_lo_opt
            else:
                sigma_star, _ = _golden_section(
                    _eval, sigma_lo_opt, gs_up, sigma_tol, qf_tol, n_gs
                )

        new_mu = sigma_star * mu_nat
        new_mu = max(new_mu, self.mu_min)
        new_mu = min(new_mu, self.mu_max)

        if comm_rank == 0 and options.get("verbose_barrier"):
            print(
                f"  QF: sigma={sigma_star:.4f}, mu={new_mu:.3e} "
                f"(comp={avg_comp:.3e})"
            )

        sigma_eff = new_mu / mu_nat if mu_nat > 0 else sigma_star
        opt.px.get_array()[:] = px0 + sigma_eff * dpx
        opt.px.copy_host_to_device()
        opt.optimizer.compute_update(new_mu, opt.vars, opt.px, opt.update)
        return sigma_star, new_mu

    def _evaluate_qf(
        self,
        sigma,
        px0,
        dpx,
        mu_nat,
        tau,
        dual_inf,
        primal_inf,
        comp_inf,
        centrality,
        balancing,
    ):
        """q_L(sigma) at the combined step px0 + sigma * dpx."""
        opt = self.opt
        mu_s = sigma * mu_nat
        opt.px.get_array()[:] = px0 + sigma * dpx
        opt.px.copy_host_to_device()
        opt.optimizer.compute_update(mu_s, opt.vars, opt.px, opt.update)
        alpha_x, _, alpha_z, _ = opt.optimizer.compute_max_step(
            tau, opt.vars, opt.update
        )
        opt.optimizer.apply_step_update(
            alpha_x, alpha_z, opt.vars, opt.update, opt.temp
        )
        trial_comp_sq = opt.optimizer.compute_complementarity_sq(opt.temp)

        qf = (
            (1.0 - alpha_z) ** 2 * dual_inf * self.sd
            + (1.0 - alpha_x) ** 2 * primal_inf * self.sp
            + trial_comp_sq * self.sc
        )

        if centrality != "none" and trial_comp_sq > 0:
            _, trial_xi = opt.optimizer.compute_complementarity(opt.temp)
            trial_xi = max(trial_xi, 1e-30)
            if centrality == "log":
                qf -= trial_comp_sq * self.sc * np.log(trial_xi)
            elif centrality == "reciprocal":
                qf += trial_comp_sq * self.sc / trial_xi
            elif centrality == "cubed-reciprocal":
                qf += trial_comp_sq * self.sc / trial_xi**3

        if balancing == "cubic":
            d_term = (1.0 - alpha_z) ** 2 * dual_inf * self.sd
            p_term = (1.0 - alpha_x) ** 2 * primal_inf * self.sp
            c_term = trial_comp_sq * self.sc
            qf += max(0.0, max(d_term, p_term) - c_term) ** 3

        return qf


def _golden_section(f, a, b, sigma_tol, qf_tol, max_iters):
    """Golden-section search for minimum of f on [a, b]."""
    gfac = (3.0 - np.sqrt(5.0)) / 2.0  # ~ 0.382
    lo, up = a, b
    m1 = lo + gfac * (up - lo)
    m2 = lo + (1.0 - gfac) * (up - lo)
    q_lo = f(lo)
    q_up = f(up)
    q_m1 = f(m1)
    q_m2 = f(m2)

    for _ in range(max_iters):
        width = up - lo
        if width < sigma_tol * up:
            break
        q_all = (q_lo, q_up, q_m1, q_m2)
        qmin, qmax = min(q_all), max(q_all)
        if qmax > 0 and 1.0 - qmin / qmax < qf_tol:
            break
        if q_m1 > q_m2:
            lo, q_lo = m1, q_m1
            m1, q_m1 = m2, q_m2
            m2 = lo + (1.0 - gfac) * (up - lo)
            q_m2 = f(m2)
        else:
            up, q_up = m2, q_m2
            m2, q_m2 = m1, q_m1
            m1 = lo + gfac * (up - lo)
            q_m1 = f(m1)

    best_sigma, best_q = lo, q_lo
    for s, q in ((m1, q_m1), (m2, q_m2), (up, q_up)):
        if q < best_q:
            best_sigma, best_q = s, q
    return best_sigma, best_q
