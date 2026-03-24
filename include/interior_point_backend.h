#ifndef AMIGO_INTERIOR_POINT_BACKEND_H
#define AMIGO_INTERIOR_POINT_BACKEND_H

/*
  Primal-dual interior-point backend for the 2x2 augmented system.

  Problem formulation after slack introduction:

    min  f(x)
    s.t. c(x) = 0          (all constraints are equalities)
         xL <= x <= xU      (bounds on all primals: design vars + slacks)

  Primal-dual equations (KKT conditions for the barrier subproblem):

    (4a) grad_x L + Sigma*x - zl + zu = 0      (stationarity)
    (4b) c(x) = 0                                (primal feasibility)
    (4c) (x - xL)*zl = mu*e                      (complementarity, lower)
         (xU - x)*zu = mu*e                      (complementarity, upper)

  The 2x2 augmented system solved at each iteration:

    [ W + Sigma + dw*I    A^T   ] [ dx   ]   [ r_d ]
    [       A           -dc*I   ] [ dlam ] = [ r_p ]

  where Sigma = diag(zl/(x-xL) + zu/(xU-x)), A = constraint Jacobian.

  Bound duals are recovered via back-substitution:

    dzl = (-(gap_l * zl - mu) - zl * dx) / gap_l
    dzu = (-(gap_u * zu - mu) + zu * dx) / gap_u

  Step sizes are chosen by the fraction-to-the-boundary rule:

    alpha_x = max{ a in (0,1] : x + a*dx >= (1-tau)*x }
    alpha_z = max{ a in (0,1] : z + a*dz >= (1-tau)*z }
*/

#include <cmath>
#include <memory>
#include "a2dcore.h"
#include "amigo.h"

namespace amigo {

template <typename T>
class OptVector;

namespace ipm {

// Two categories: bounded primals and equality constraints.
template <typename T>
struct ProblemInfo {
  int n_primal = 0;
  int n_constraints = 0;
  const int* primal_indices = nullptr;
  const int* constraint_indices = nullptr;
  const T *lbx = nullptr, *ubx = nullptr;
  const T* lbh = nullptr;
};

// Pointers into OptVector storage. T may be const-qualified for read-only access.
template <typename T>
struct State {
  T* xlam = nullptr;
  T* zl = nullptr;
  T* zu = nullptr;

  template <ExecPolicy policy, typename R>
  static State make(std::shared_ptr<OptVector<R>> vars);
};

// Utilities

template <typename T>
void set_primal_value(const ProblemInfo<T>& p, T val, T* xlam) {
  for (int i = 0; i < p.n_primal; i++) xlam[p.primal_indices[i]] = val;
}

template <typename T>
void set_constraint_value(const ProblemInfo<T>& p, T val, T* xlam) {
  for (int i = 0; i < p.n_constraints; i++) xlam[p.constraint_indices[i]] = val;
}

template <typename T>
void copy_primals(const ProblemInfo<T>& p, const T* src, T* dst) {
  for (int i = 0; i < p.n_primal; i++) {
    int k = p.primal_indices[i];
    dst[k] = src[k];
  }
}

template <typename T>
void copy_constraints(const ProblemInfo<T>& p, const T* src, T* dst) {
  for (int i = 0; i < p.n_constraints; i++) {
    int k = p.constraint_indices[i];
    dst[k] = src[k];
  }
}

// Relax bounds by a small factor to avoid numerical issues at exact bounds.
//   x_L -= min(constr_viol_tol, factor * max(1, |x_L|))
//   x_U += min(constr_viol_tol, factor * max(1, |x_U|))
// Default: bound_relax_factor = 1e-8, constr_viol_tol = 1e-4.
template <typename T>
void relax_bounds(ProblemInfo<T>& p, T* lbx_buf, T* ubx_buf,
                  T factor = 1e-8, T constr_viol_tol = 1e-4) {
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i])) {
      T delta = A2D::min2(constr_viol_tol, factor * A2D::max2(T(1), std::abs(p.lbx[i])));
      lbx_buf[i] = p.lbx[i] - delta;
    } else {
      lbx_buf[i] = p.lbx[i];
    }
    if (!std::isinf(p.ubx[i])) {
      T delta = A2D::min2(constr_viol_tol, factor * A2D::max2(T(1), std::abs(p.ubx[i])));
      ubx_buf[i] = p.ubx[i] + delta;
    } else {
      ubx_buf[i] = p.ubx[i];
    }
  }
  p.lbx = lbx_buf;
  p.ubx = ubx_buf;
}

// Project all primals into the strict interior of their bounds (Section 3.6).
// For one-sided bounds: x <- max(x, lb + kappa1 * max(1, |lb|))  or similar.
// For two-sided bounds: x is projected into [lb + p_l, ub - p_u] where
//   p_l = min(kappa1 * max(1, |lb|), kappa2 * (ub - lb))
//   p_u = min(kappa1 * max(1, |ub|), kappa2 * (ub - lb))
// Defaults: kappa1 = 0.01, kappa2 = 0.01.
// Must be called before initialize_bound_duals.
template <typename T>
void project_primals_into_interior(const ProblemInfo<T>& p, T* xlam,
                                   T kappa1 = 1e-2, T kappa2 = 1e-2) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = xlam[idx];
    T lb = p.lbx[i];
    T ub = p.ubx[i];
    bool has_lb = !std::isinf(lb);
    bool has_ub = !std::isinf(ub);

    if (has_lb && has_ub) {
      T range = ub - lb;
      T pl = A2D::min2(kappa1 * A2D::max2(T(1), std::abs(lb)), kappa2 * range);
      T pu = A2D::min2(kappa1 * A2D::max2(T(1), std::abs(ub)), kappa2 * range);
      xlam[idx] = A2D::max2(A2D::min2(x, ub - pu), lb + pl);
    } else if (has_lb) {
      xlam[idx] = A2D::max2(x, lb + kappa1 * A2D::max2(T(1), std::abs(lb)));
    } else if (has_ub) {
      xlam[idx] = A2D::min2(x, ub - kappa1 * A2D::max2(T(1), std::abs(ub)));
    }
  }
}

// Initialize bound duals to 1.0 for all finite bounds (Section 3.6).
// Must be called after project_primals_into_interior.
template <typename T>
void initialize_bound_duals(T mu, const ProblemInfo<T>& p,
                            const T* xlam, T* zl, T* zu) {
  for (int i = 0; i < p.n_primal; i++) {
    zl[i] = std::isinf(p.lbx[i]) ? T(0) : T(1);
    zu[i] = std::isinf(p.ubx[i]) ? T(0) : T(1);
  }
}

// Condensed residual for the 2x2 augmented system.
//
// Primal rows (right-hand side of the first block row):
//   r[i] = -(grad[i] - zl[i] + zu[i])                    stationarity residual
//         + (-(gap_l * zl - mu)) / gap_l                  lower complementarity condensation
//         - (-(gap_u * zu - mu)) / gap_u                  upper complementarity condensation
//
// Constraint rows (right-hand side of the second block row):
//   r[j] = -(constraint_value[j] - target[j])             primal feasibility
template <typename T>
void compute_residual(T mu, const ProblemInfo<T>& p,
                      State<const T>& s, const T* grad, T* res) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = s.xlam[idx];
    T r = -(grad[idx] - s.zl[i] + s.zu[i]);

    if (!std::isinf(p.lbx[i])) {
      T gap = x - p.lbx[i];
      r += -(gap * s.zl[i] - mu) / gap;
    }
    if (!std::isinf(p.ubx[i])) {
      T gap = p.ubx[i] - x;
      r -= -(gap * s.zu[i] - mu) / gap;
    }
    res[idx] = r;
  }

  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    res[idx] = -(grad[idx] - p.lbh[j]);
  }
}

// Same as compute_residual, but also accumulates squared norms of the
// unscaled dual and primal infeasibility for convergence monitoring.
template <typename T>
void compute_residual_and_infeasibility(
    T mu, const ProblemInfo<T>& p, State<const T>& s,
    const T* grad, T* res, T& dual_sq, T& primal_sq) {
  dual_sq = 0.0;
  primal_sq = 0.0;

  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = s.xlam[idx];
    T rd = grad[idx] - s.zl[i] + s.zu[i];
    dual_sq += rd * rd;

    T r = -rd;
    if (!std::isinf(p.lbx[i])) {
      T gap = x - p.lbx[i];
      r += -(gap * s.zl[i] - mu) / gap;
    }
    if (!std::isinf(p.ubx[i])) {
      T gap = p.ubx[i] - x;
      r -= -(gap * s.zu[i] - mu) / gap;
    }
    res[idx] = r;
  }

  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    T rp = grad[idx] - p.lbh[j];
    primal_sq += rp * rp;
    res[idx] = -rp;
  }
}

// Barrier diagonal Sigma for the 2x2 augmented system (eq. 13).
//
//   Primal rows:     Sigma_i = zl_i/(x_i - lb_i) + zu_i/(ub_i - x_i)
//   Constraint rows: 0  (inertia correction adds -delta_c separately)
//
// Each entry is written, not accumulated: the function fully owns the
// primal and constraint diagonal entries it touches.
template <typename T>
void compute_diagonal(const ProblemInfo<T>& p, State<const T>& s, T* diag) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = s.xlam[idx];
    T sigma = T(0);
    if (!std::isinf(p.lbx[i])) sigma += s.zl[i] / (x - p.lbx[i]);
    if (!std::isinf(p.ubx[i])) sigma += s.zu[i] / (p.ubx[i] - x);
    diag[idx] = sigma;
  }
  for (int j = 0; j < p.n_constraints; j++) {
    diag[p.constraint_indices[j]] = T(0);
  }
}

// Bound dual back-substitution. After solving the 2x2 system for (dx, dlam),
// recover the bound dual steps from the complementarity equations:
//   dzl = (-(gap_l * zl - mu) - zl * dx) / gap_l
//   dzu = (-(gap_u * zu - mu) + zu * dx) / gap_u
template <typename T>
void compute_bound_dual_step(T mu, const ProblemInfo<T>& p,
                             State<const T>& s, const T* px,
                             T* dzl, T* dzu) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = s.xlam[idx], dx = px[idx];
    dzl[i] = dzu[i] = 0.0;

    if (!std::isinf(p.lbx[i])) {
      T gap = x - p.lbx[i];
      dzl[i] = (-(gap * s.zl[i] - mu) - s.zl[i] * dx) / gap;
    }
    if (!std::isinf(p.ubx[i])) {
      T gap = p.ubx[i] - x;
      dzu[i] = (-(gap * s.zu[i] - mu) + s.zu[i] * dx) / gap;
    }
  }
}

// Fraction-to-the-boundary rule. Finds the largest step alpha in (0,1]
// such that all primals stay within bounds and all duals stay positive:
//   x + alpha*dx >= (1-tau)*(x - lb)  for each finite lower bound
//   ub - (x + alpha*dx) >= (1-tau)*(ub - x)  for each finite upper bound
//   zl + alpha*dzl >= (1-tau)*zl, zu + alpha*dzu >= (1-tau)*zu
template <typename T>
void compute_max_step(T tau, const ProblemInfo<T>& p,
                      State<const T>& s, const T* px,
                      const T* dzl, const T* dzu,
                      T& ax, int& xi, T& az, int& zi) {
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = s.xlam[idx], dx = px[idx];

    if (!std::isinf(p.lbx[i])) {
      if (dx < 0.0) {
        T a = -tau * (x - p.lbx[i]) / dx;
        if (a < ax) { ax = a; xi = idx; }
      }
      if (dzl[i] < 0.0) {
        T a = -tau * s.zl[i] / dzl[i];
        if (a < az) { az = a; zi = idx; }
      }
    }
    if (!std::isinf(p.ubx[i])) {
      if (dx > 0.0) {
        T a = tau * (p.ubx[i] - x) / dx;
        if (a < ax) { ax = a; xi = idx; }
      }
      if (dzu[i] < 0.0) {
        T a = -tau * s.zu[i] / dzu[i];
        if (a < az) { az = a; zi = idx; }
      }
    }
  }
}

// Apply the full primal-dual-bound trial step (eq. 14-15).
//   xlam_new = xlam + alpha_x * dxlam   (primals + multipliers)
//   zl_new   = zl   + alpha_z * dzl     (lower bound duals)
//   zu_new   = zu   + alpha_z * dzu     (upper bound duals)
template <typename T>
void apply_step(T ax, T az, const ProblemInfo<T>& p,
                State<const T>& s, const T* dxlam,
                const T* dzl, const T* dzu,
                T* xlam_new, int n_xlam,
                T* zl_new, T* zu_new) {
  for (int i = 0; i < n_xlam; i++) {
    xlam_new[i] = s.xlam[i] + ax * dxlam[i];
  }
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i])) zl_new[i] = s.zl[i] + az * dzl[i];
    if (!std::isinf(p.ubx[i])) zu_new[i] = s.zu[i] + az * dzu[i];
  }
}

// Average complementarity mu_avg = sum(gap*z) / n_bounds, and
// minimum complementarity product (for uniformity measure xi).
template <typename T>
void compute_complementarity(const ProblemInfo<T>& p, State<const T>& s,
                             T partial_sum[], T& local_min) {
  for (int i = 0; i < p.n_primal; i++) {
    T x = s.xlam[p.primal_indices[i]];
    if (!std::isinf(p.lbx[i])) {
      T c = (x - p.lbx[i]) * s.zl[i];
      partial_sum[0] += c; partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, c);
    }
    if (!std::isinf(p.ubx[i])) {
      T c = (p.ubx[i] - x) * s.zu[i];
      partial_sum[0] += c; partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, c);
    }
  }
}

// Maximum deviation of individual complementarity products from mu.
template <typename T>
void compute_max_comp_deviation(const ProblemInfo<T>& p, State<const T>& s,
                                T mu, T& max_dev) {
  max_dev = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    T x = s.xlam[p.primal_indices[i]];
    if (!std::isinf(p.lbx[i]))
      max_dev = A2D::max2(max_dev, std::abs((x - p.lbx[i]) * s.zl[i] - mu));
    if (!std::isinf(p.ubx[i]))
      max_dev = A2D::max2(max_dev, std::abs((p.ubx[i] - x) * s.zu[i] - mu));
  }
}

// Sum of squared complementarity products (for quality function evaluation).
template <typename T>
void compute_complementarity_sq(const ProblemInfo<T>& p, State<const T>& s,
                                T mu, T& sq) {
  sq = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    T x = s.xlam[p.primal_indices[i]];
    if (!std::isinf(p.lbx[i])) {
      T r = (x - p.lbx[i]) * s.zl[i] - mu; sq += r * r;
    }
    if (!std::isinf(p.ubx[i])) {
      T r = (p.ubx[i] - x) * s.zu[i] - mu; sq += r * r;
    }
  }
}

// Optimality error E_mu with three components (infinity norms):
//   dual    = max |grad_i - zl_i + zu_i|           (stationarity)
//   primal  = max |c_j(x) - target_j|              (feasibility)
//   comp    = max |gap_i * z_i - mu|                (complementarity)
template <typename T>
void compute_kkt_error(T mu, const ProblemInfo<T>& p, State<const T>& s,
                       const T* grad, T& dual, T& primal, T& comp) {
  dual = primal = comp = 0.0;

  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = s.xlam[idx];
    dual = A2D::max2(dual, std::abs(grad[idx] - s.zl[i] + s.zu[i]));
    if (!std::isinf(p.lbx[i]))
      comp = A2D::max2(comp, std::abs((x - p.lbx[i]) * s.zl[i] - mu));
    if (!std::isinf(p.ubx[i]))
      comp = A2D::max2(comp, std::abs((p.ubx[i] - x) * s.zu[i] - mu));
  }
  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    primal = A2D::max2(primal, std::abs(grad[idx] - p.lbh[j]));
  }
}

// Barrier log-sum: -mu * sum_i ln(x_i - lb_i) - mu * sum_i ln(ub_i - x_i).
// Added to the objective f(x) to form the barrier objective phi_mu(x).
template <typename T>
T compute_barrier_log_sum(T mu, const ProblemInfo<T>& p, State<const T>& s) {
  T b = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    T x = s.xlam[p.primal_indices[i]];
    if (!std::isinf(p.lbx[i])) { T g = x - p.lbx[i]; if (g > 0) b -= mu * std::log(g); }
    if (!std::isinf(p.ubx[i])) { T g = p.ubx[i] - x; if (g > 0) b -= mu * std::log(g); }
  }
  return b;
}

// Directional derivative of the barrier objective along the search direction:
//   dphi = sum_i (grad_i * dx_i - mu * dx_i / gap_l_i + mu * dx_i / gap_u_i)
// Used in the Armijo condition and switching condition of the filter line search.
template <typename T>
T compute_barrier_dphi(T mu, const ProblemInfo<T>& p, State<const T>& s,
                       const T* grad, const T* px) {
  T d = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = s.xlam[idx], dx = px[idx];
    d += grad[idx] * dx;
    if (!std::isinf(p.lbx[i])) d -= mu * dx / (x - p.lbx[i]);
    if (!std::isinf(p.ubx[i])) d += mu * dx / (p.ubx[i] - x);
  }
  return d;
}

// Bound multiplier reset: clamp each z_i to [mu/(kappa*gap), kappa*mu/gap].
// Prevents the ratio Sigma_i = z_i/gap from deviating too far from mu/gap^2,
// which is needed for the convergence proof of primal-dual methods.
template <typename T>
void reset_bound_multipliers(T mu, T kappa, const ProblemInfo<T>& p,
                             State<const T>& s, T* zl_out, T* zu_out) {
  for (int i = 0; i < p.n_primal; i++) {
    T x = s.xlam[p.primal_indices[i]];
    if (!std::isinf(p.lbx[i])) {
      T g = x - p.lbx[i];
      zl_out[i] = A2D::max2(A2D::min2(s.zl[i], kappa * mu / g), mu / (kappa * g));
    }
    if (!std::isinf(p.ubx[i])) {
      T g = p.ubx[i] - x;
      zu_out[i] = A2D::max2(A2D::min2(s.zu[i], kappa * mu / g), mu / (kappa * g));
    }
  }
}

// Project bound duals to at least beta_min after the affine scaling step.
template <typename T>
void compute_affine_start_point(T beta_min, const ProblemInfo<T>& p,
                                State<const T>& s,
                                const T* dzl, const T* dzu,
                                T* zl_out, T* zu_out) {
  for (int i = 0; i < p.n_primal; i++) {
    if (!std::isinf(p.lbx[i])) zl_out[i] = A2D::max2(s.zl[i] + dzl[i], beta_min);
    if (!std::isinf(p.ubx[i])) zu_out[i] = A2D::max2(s.zu[i] + dzu[i], beta_min);
  }
}

// Squared KKT error norms (sum-of-squares, for quality function / convergence).
template <typename T>
void compute_kkt_error_sq(const ProblemInfo<T>& p, State<const T>& s,
                          const T* grad, T& dual_sq, T& primal_sq, T& comp_sq) {
  dual_sq = primal_sq = comp_sq = 0.0;

  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    T x = s.xlam[idx];
    T rd = grad[idx] - s.zl[i] + s.zu[i];
    dual_sq += rd * rd;
    if (!std::isinf(p.lbx[i])) {
      T c = (x - p.lbx[i]) * s.zl[i];
      comp_sq += c * c;
    }
    if (!std::isinf(p.ubx[i])) {
      T c = (p.ubx[i] - x) * s.zu[i];
      comp_sq += c * c;
    }
  }
  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    T rp = grad[idx] - p.lbh[j];
    primal_sq += rp * rp;
  }
}

// Dual residual vector: r_d[i] = grad[i] - zl[i] + zu[i] for primals, 0 elsewhere.
// Used by the quality function to compute the cross term r_d^T * (Hessian_mod * dx).
template <typename T>
void compute_dual_residual_vector(const ProblemInfo<T>& p, State<const T>& s,
                                  const T* grad, T* out, int size) {
  for (int i = 0; i < size; i++) out[i] = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    out[idx] = grad[idx] - s.zl[i] + s.zu[i];
  }
}

// Barrier directional derivative computed from the KKT residual and solution.
// dphi = -r_primal^T * px - lam^T * (J * dx)
// where J*dx is reconstructed from constraint rows: J*dx = r_j - diag_j * px_j.
// This avoids recomputing the barrier gradient explicitly and handles
// the regularization terms (delta_w, delta_c) correctly.
template <typename T>
T compute_barrier_dphi_from_kkt(const ProblemInfo<T>& p, State<const T>& s,
                                const T* res, const T* px, const T* diag) {
  T dphi = 0.0;
  for (int i = 0; i < p.n_primal; i++) {
    int idx = p.primal_indices[i];
    dphi -= res[idx] * px[idx];
  }
  for (int j = 0; j < p.n_constraints; j++) {
    int idx = p.constraint_indices[j];
    T lam = s.xlam[idx];
    T jdx = res[idx] - diag[idx] * px[idx];
    dphi -= lam * jdx;
  }
  return dphi;
}

}  // namespace ipm
}  // namespace amigo

#endif  // AMIGO_INTERIOR_POINT_BACKEND_H
