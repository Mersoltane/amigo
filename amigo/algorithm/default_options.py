"""Default option values for the interior-point optimizer.

The options are split into one group per stage of the algorithm, in the
order a solve runs through them.
"""

import copy
import numbers

import numpy as np

# Iteration limit, convergence tolerances, and the divergence check
CONVERGENCE = {
    "max_iterations": 100,
    "convergence_tolerance": 1e-8,
    "dual_inf_tol": 1.0,
    "constr_viol_tol": 1e-4,
    "compl_inf_tol": 1e-4,
    # Relative objective change the full step predicts below which an acceptable iterate finishes on the constraints alone, and 0 disables
    "objective_tolerance": 0.0,
    # Acceptable convergence, a weaker solution held for several iterations
    "acceptable_tol": 1e-6,
    "acceptable_iter": 15,
    "acceptable_dual_inf_tol": 1e10,
    "acceptable_constr_viol_tol": 1e-2,
    "acceptable_compl_inf_tol": 1e-2,
    # Largest primal magnitude before the iterates count as diverging and 0 disables
    "diverging_iterates_tol": 1e20,
    # Wall time in seconds after which the solve stops at the current point
    "max_wall_time": 1e20,
    # Iterations without the scaled error shrinking by stall_factor after which the solve stops, and 0 disables
    "stall_iterations": 0,
    "stall_factor": 0.5,
}

# Starting point, multiplier estimate, warm start, and NLP scaling
STARTING_POINT = {
    "init_least_squares_multipliers": True,
    # Cap on the initial multiplier estimate magnitude
    "constr_mult_init_max": 1e3,
    # Least squares multiplier refresh after untrusted steps below this violation
    "recompute_multipliers": True,
    "recompute_multiplier_tol": 1e-1,
    # Begin from a restored feasible point with centered duals
    "feasibility_presolve": False,
    # Relax bounds by factor * max(1, |b|), capped at constr_viol_tol
    "bound_relax_factor": 1e-8,
    # Warm start keeping the supplied multipliers and the previous bound duals
    "warm_start": False,
    # Cap on the centered bound duals of a warm start, which holds every gap at least mu over the cap
    "warm_start_mult_init_max": 1e3,
    # Barrier a warm start resumes at, None for the barrier the warm start carries or 1e-4 without one
    "warm_start_mu_init": None,
    # Gradient based NLP scaling and a user factor on the objective that multiplies it
    "nlp_scaling": True,
    "nlp_scaling_max_gradient": 100.0,
    "nlp_scaling_min_value": 1e-8,
    "objective_scaling_factor": 1.0,
}

# Barrier parameter and the monotone, heuristic, and quality function strategies
BARRIER = {
    "barrier_strategy": "monotone",
    "initial_barrier_param": 1.0,
    # Floor of the fraction to the boundary, raised to 1 - mu when adaptive
    "fraction_to_boundary": 0.99,
    "adaptive_tau": True,
    "verbose_barrier": False,
    "continuation_control": None,
    # Subproblem accuracy kappa_eps * mu that ends each barrier subproblem
    "barrier_tol_factor": 10.0,
    # Monotone strategy, initial mu is "fixed" or "complementarity"
    "mu_init_strategy": "fixed",
    "mu_linear_decrease_factor": 0.2,
    "mu_superlinear_decrease_power": 1.5,
    "mu_min": 1e-11,
    # Heuristic strategy
    "heuristic_barrier_gamma": 0.1,
    "heuristic_barrier_r": 0.95,
    # Quality function strategy capped at the initial mu with its adaptive globalization
    "adaptive_mu_globalization": "obj-constr-filter",
    "adaptive_mu_kkterror_red_iters": 4,
    "adaptive_mu_kkterror_red_fact": 0.9999,
    "adaptive_mu_monotone_init_factor": 0.8,
    # Progress margin of the obj-constr-filter globalization
    "filter_margin_fact": 1e-5,
    "filter_max_margin": 1.0,
    # Lower safeguard, mu >= factor * scaled infeasibility
    "adaptive_mu_safeguard_factor": 0.0,
    "quality_function_sigma_max": 100.0,
    "quality_function_sigma_min": 1e-6,
    "quality_function_section_sigma_tol": 1e-2,
    "quality_function_section_qf_tol": 0.0,
    "quality_function_golden_iters": 8,
    "quality_function_norm_scaling": True,
    # "none", "log", "reciprocal", or "cubed-reciprocal"
    "quality_function_centrality": "none",
    # "none" or "cubic"
    "quality_function_balancing_term": "none",
    "quality_function_predictor_corrector": False,
}

# Linear solver, regularization, and inertia correction
LINEAR_SOLVER = {
    "solver": "amigo",
    "amigo_pivot_tol": 1e-14,
    # Threshold pivoting parameter where larger values favor stability over sparsity
    "amigo_pivot_threshold": 1e-4,
    # Symmetric equilibration sweeps before each factorization and 0 disables
    "amigo_scaling_sweeps": 2,
    # Memory in GB one frontal matrix may take as delayed pivots grow it
    "amigo_max_front_gb": 4.0,
    # Iterative refinement until the solve backward error meets the tolerance
    "amigo_refinement_steps": 2,
    "amigo_residual_tol": 1e-10,
    # MUMPS workspace above its estimate in percent, doubled on a workspace error
    "mumps_mem_percent": 1000,
    # GPU solver, pivots below the epsilon are statically perturbed
    "cuda_pivot_eps": 1e-8,
    "cuda_ir_steps": 2,
    "cuda_check_residual": True,
    "cuda_residual_rtol": 1e-4,
    # Constraint regularization delta_c = value * mu^exponent
    "jacobian_regularization_value": 1e-8,
    "jacobian_regularization_exponent": 0.25,
    # Apply delta_c on every factorization, for persistently singular KKTs
    "perturb_always_cd": False,
    # Resume from the decayed previous delta_x instead of restarting at 0, holding it after a failed decay
    "hessian_perturbation_warm_start": False,
    # Primal rows that carry delta_x, None for all, "free" for the variables the equalities leave free, or an index list
    "hessian_perturbation_indices": None,
    # Accept a wrong inertia when the step curvature exceeds this fraction of its norm and 0 disables
    "neg_curv_test_tol": 0.0,
    # Always-on dual regularization delta_c = coeff * max(||c||, mu) and 0 disables
    "dual_reg_infeas_coeff": 0.0,
    # Accept a non-singular factorization with wrong inertia once delta_w reaches this cap
    "max_regularization": 0.0,
    # Tolerated departure from the expected positive and negative counts
    "inertia_tolerance": 0,
    # Force the barrier down after a futile refresh when only the duals block it
    "dual_complete_schedule": True,
}

# Line search with the filter, merit, and funnel acceptance rules
LINE_SEARCH = {
    "line_search": "filter",
    "max_line_search_iterations": 40,
    "backtracking_factor": 0.5,
    "second_order_correction": True,
    # A feasible step below this relative size is taken whole and forces a barrier decrease
    "tiny_step_tol": 2.2204460492503131e-15,
    "tiny_step_y_tol": 1e-2,
    # Clip the bound duals into a band around mu / gap and 0 disables
    "kappa_sigma": 1e10,
    # Filter line search
    # Violation ceiling theta_max and floor theta_min as factors of max(1, theta0)
    "filter_theta_max_fact": 1e4,
    "filter_theta_min_fact": 1e-4,
    "filter_gamma_theta": 1e-5,
    "filter_gamma_phi": 1e-8,
    "filter_delta": 1.0,
    "filter_s_theta": 1.1,
    "filter_s_phi": 2.3,
    "filter_eta_phi": 1e-8,
    "filter_max_soc": 4,
    "filter_kappa_soc": 0.99,
    "filter_reset_trigger": 5,
    "max_filter_resets": 5,
    # Merit line search with a fixed penalty on the squared violation
    "merit_penalty": 1.0,
    # Funnel line search sharing the filter margins and Armijo constant
    "funnel_ubd": 1.0,
    "funnel_fact": 1.5,
    "funnel_beta": 0.9999,
    "funnel_kappa": 0.5,
    "funnel_update_strategy": 1,
    "funnel_switching_delta": 0.999,
    "funnel_switching_infeasibility_exponent": 2,
    "funnel_min_step_length": 1e-12,
}

# Feasibility restoration
RESTORATION = {
    "feasibility_restoration": True,
    # Weight of the l1 violation in the restoration objective
    "resto_penalty": 1000.0,
    # Weight of the elastic parts of the constraints the model marks elastic at a cold start
    "elastic_weight": 1000.0,
    # Growth of the weight of the elastic parts when their bound duals fall below the floor times the weight
    "elastic_weight_growth": 10.0,
    "elastic_dual_floor": 0.1,
    "elastic_max_updates": 5,
    # Fraction of its entry value the largest violation must reach before the phase returns
    "resto_required_reduction": 0.9,
    # Trigger after this many accepted steps below the alpha threshold
    "resto_trigger_alpha": 1e-3,
    "resto_trigger_iters": 5,
}

# Hook called with the iteration count and the current point after every logged iteration
OUTPUT = {
    "iteration_callback": None,
}

GROUPS = (
    CONVERGENCE,
    STARTING_POINT,
    BARRIER,
    LINEAR_SOLVER,
    LINE_SEARCH,
    RESTORATION,
    OUTPUT,
)

# Allowed values of the string options that select a variant
CHOICES = {
    "barrier_strategy": ("monotone", "heuristic", "quality_function"),
    "mu_init_strategy": ("fixed", "complementarity"),
    "adaptive_mu_globalization": ("obj-constr-filter", "kkt-error", "never-monotone"),
    "quality_function_centrality": ("none", "log", "reciprocal", "cubed-reciprocal"),
    "quality_function_balancing_term": ("none", "cubic"),
    "solver": ("amigo", "mumps", "cuda", "cuda_bordered"),
    "line_search": ("filter", "merit", "funnel"),
}


def _check_value(name, value, default):
    """Raise when an option value cannot be one the solver reads."""
    if name in CHOICES and isinstance(value, str) and value not in CHOICES[name]:
        allowed = ", ".join(CHOICES[name])
        raise ValueError(f"Option {name} is {value!r}, expected one of {allowed}")
    if isinstance(default, bool):
        if not isinstance(value, (bool, np.bool_)) and value not in (0, 1):
            raise TypeError(f"Option {name} must be True or False, got {value!r}")
    elif isinstance(default, numbers.Real):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"Option {name} must be a number, got {value!r}")
        if np.isnan(value):
            raise ValueError(f"Option {name} is NaN")


def get_default_options(options={}):
    """Return the merged options dict with all defaults filled in."""
    default = {}
    for group in GROUPS:
        overlap = default.keys() & group.keys()
        if overlap:
            raise ValueError(f"Option defined in two groups: {sorted(overlap)}")
        default.update(copy.deepcopy(group))

    options = dict(options)

    for name in options:
        if name not in default:
            raise ValueError(f"Unrecognized option {name}")
        _check_value(name, options[name], default[name])
        default[name] = options[name]

    return default
