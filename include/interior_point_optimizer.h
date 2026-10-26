#ifndef AMIGO_INTERIOR_POINT_OPTIMIZER_H
#define AMIGO_INTERIOR_POINT_OPTIMIZER_H

#include <mpi.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "amigo.h"
#include "interior_point_backend.h"
#include "optimization_problem.h"

namespace amigo {

/**
 * Optimization state: solution vector + bound duals.
 *
 * Storage:
 *   x_     = xlam vector (primals + multipliers)
 *   duals = [zl(n_primal) | zu(n_primal)]
 */
template <typename T>
class OptVector {
 public:
  OptVector(int num_primals, int num_constraints, std::shared_ptr<Vector<T>> x)
      : num_primals(num_primals), num_constraints(num_constraints), x(x) {
    // Create the slack and dual variable vectors for the upper and lower bounds
    MemoryLocation loc = x->get_memory_location();

    zl = std::make_shared<Vector<T>>(num_primals, 0, loc);
    zu = std::make_shared<Vector<T>>(num_primals, 0, loc);
  }

  void zero() {
    zl->zero();
    zu->zero();
  }

  void copy(std::shared_ptr<OptVector<T>> src) {
    x->copy(src->x);
    zl->copy(src->zl);
    zu->copy(src->zu);
  }

  int get_num_primals() const { return num_primals; }
  int get_num_constraints() const { return num_constraints; }

  std::shared_ptr<Vector<T>> get_solution() { return x; }
  std::shared_ptr<Vector<T>> get_zl() { return zl; }
  std::shared_ptr<Vector<T>> get_zu() { return zu; }

  const std::shared_ptr<Vector<T>> get_solution() const { return x; }

  template <ExecPolicy policy>
  void get_bound_duals(T** zl_, T** zu_) {
    if (zl_) {
      *zl_ = zl->template get_array<policy>();
    }
    if (zu_) {
      *zu_ = zu->template get_array<policy>();
    }
  }
  template <ExecPolicy policy>
  void get_bound_duals(const T** zl_, const T** zu_) const {
    if (zl_) {
      *zl_ = zl->template get_array<policy>();
    }
    if (zu_) {
      *zu_ = zu->template get_array<policy>();
    }
  }

  template <ExecPolicy policy>
  T* get_solution_array() {
    return x->template get_array<policy>();
  }
  template <ExecPolicy policy>
  const T* get_solution_array() const {
    return x->template get_array<policy>();
  }

 private:
  int num_primals, num_constraints;

  // The primal/dual vector
  std::shared_ptr<Vector<T>> x;

  // Duals associated with the lower/upper bounds
  std::shared_ptr<Vector<T>> zl, zu;
};

/**
 * Interior-point optimizer for the 2x2 augmented system.
 *
 * Every variable is either a bounded primal or an equality constraint.
 * Wraps detail:: backend functions for the Python/pybind interface.
 */
template <typename T, ExecPolicy policy>
class InteriorPointOptimizer {
 public:
  InteriorPointOptimizer(
      std::shared_ptr<OptimizationProblem<T, policy>> problem)
      : problem(problem) {
    comm = problem->get_mpi_comm();

    int size = problem->get_num_variables();
    const Vector<int>& vtypes = *problem->get_var_types();
    const Vector<T>& lb = *problem->get_lower();
    const Vector<T>& ub = *problem->get_upper();

    primal_indices = problem->get_primal_indices();
    num_primals = primal_indices->get_local_size();

    constraint_indices = problem->get_constraint_indices();
    num_constraints = constraint_indices->get_local_size();

    // Set the memory location depending on the execution policy
    MemoryLocation loc = MemoryLocation::HOST_ONLY;
    if (policy == ExecPolicy::CUDA) {
      loc = MemoryLocation::HOST_AND_DEVICE;
    }

    // Fixed variables stay out of the barrier and sit at equal bounds
    const T inf = std::numeric_limits<T>::infinity();
    std::vector<int> fixed_idx;
    std::vector<T> fixed_val;
    lbx = std::make_shared<Vector<T>>(num_primals, 0, loc);
    ubx = std::make_shared<Vector<T>>(num_primals, 0, loc);
    for (int i = 0; i < num_primals; i++) {
      int idx = (*primal_indices)[i];

      // Bounds at or beyond the infinity threshold are absent, as in IPOPT
      (*lbx)[i] = lb[idx] <= -bound_infinity ? -inf : lb[idx];
      (*ubx)[i] = ub[idx] >= bound_infinity ? inf : ub[idx];
      if (vtypes[idx] == static_cast<int>(OptVarType::FIXED)) {
        (*lbx)[i] = -inf;
        (*ubx)[i] = inf;
        if (std::isfinite(lb[idx]) && lb[idx] == ub[idx]) {
          fixed_idx.push_back(idx);
          fixed_val.push_back(lb[idx]);
        }
      }
    }
    lbx->copy_host_to_device();
    ubx->copy_host_to_device();

    int num_fixed = static_cast<int>(fixed_idx.size());
    if (num_fixed > 0) {
      fixed_indices = std::make_shared<Vector<int>>(num_fixed, 0, loc);
      fixed_values = std::make_shared<Vector<T>>(num_fixed, 0, loc);
      for (int i = 0; i < num_fixed; i++) {
        (*fixed_indices)[i] = fixed_idx[i];
        (*fixed_values)[i] = fixed_val[i];
      }
      fixed_indices->copy_host_to_device();
      fixed_values->copy_host_to_device();
    }

    // Set the constraint targets, kept unscaled in lbh0
    lbh0 = std::make_shared<Vector<T>>(num_constraints, 0, loc);
    lbh = std::make_shared<Vector<T>>(num_constraints, 0, loc);
    for (int i = 0; i < num_constraints; i++) {
      int idx = (*constraint_indices)[i];
      (*lbh0)[i] = lb[idx];
      (*lbh)[i] = lb[idx];
    }
    lbh0->copy_host_to_device();
    lbh->copy_host_to_device();
    con_scale = std::make_shared<Vector<T>>(num_constraints, 0, loc);

    // Set the host/device pointers into the info
    info.num_primals = num_primals;
    info.num_constraints = num_constraints;
    info.primal_indices = primal_indices->template get_array<policy>();
    info.constraint_indices = constraint_indices->template get_array<policy>();
    info.lbx = lbx->template get_array<policy>();
    info.ubx = ubx->template get_array<policy>();
    info.lbh = lbh->template get_array<policy>();

    // Add up the total number of constraints
    MPI_Allreduce(&num_primals, &num_global_primals, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&num_constraints, &num_global_constraints, 1, MPI_INT,
                  MPI_SUM, comm);
  }

  /**
   * @brief Get the number of primal variables
   *
   * @return int
   */
  int get_num_primals() const { return num_global_primals; }

  /**
   * @brief Get the number of duals/constraints
   *
   * @return int
   */
  int get_num_constraints() const { return num_global_constraints; }

  /**
   * @brief Create an instance of the optimization state vector
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_opt_vector() const {
    return std::make_shared<OptVector<T>>(num_primals, num_constraints,
                                          problem->create_vector());
  }

  /**
   * @brief Create an instance of an optimization state vector with the provided
   * design vector
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_opt_vector(
      std::shared_ptr<Vector<T>> x) const {
    return std::make_shared<OptVector<T>>(num_primals, num_constraints, x);
  }

  /**
   * @brief Initialize the dual and slack variables in the problem
   *
   * @param vars All of the optimization variables
   */
  void initialize_duals(T mu, std::shared_ptr<OptVector<T>> vars) const {
    // Project all primals into strict interior of bounds (Section 3.6),
    // then initialize bound duals and slacks from the projected values.
    T* xlam = vars->template get_solution_array<policy>();
    T *zl, *zu;
    vars->template get_bound_duals<policy>(&zl, &zu);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::project_primals_into_interior(info, xlam);
      detail::initialize_bound_duals(mu, info, xlam, zl, zu);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::project_primals_into_interior_cuda(info, xlam);
      AMIGO_CHECK_CUDA(cudaGetLastError());
      AMIGO_CHECK_CUDA(cudaDeviceSynchronize());

      detail::initialize_bound_duals_cuda(mu, info, xlam, zl, zu);
      AMIGO_CHECK_CUDA(cudaGetLastError());
      AMIGO_CHECK_CUDA(cudaDeviceSynchronize());
    }
#endif
  }

  /**
   * @brief Warm-start initialization: project the primals with a small push
   * and set mu-consistent bound duals z = min(mu/gap, zmax). Multipliers and
   * slacks in vars are kept as given.
   *
   * @param mu The initial barrier parameter
   * @param push The bound push fraction for the interior projection
   * @param zmax The bound dual cap
   * @param vars All of the optimization variables
   */
  void initialize_duals_warm(T mu, T push, T zmax,
                             std::shared_ptr<OptVector<T>> vars) const {
    T* xlam = vars->template get_solution_array<policy>();
    T *zl, *zu;
    vars->template get_bound_duals<policy>(&zl, &zu);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::project_primals_into_interior(info, xlam, push, push);
      detail::initialize_bound_duals_warm(mu, zmax, info, xlam, zl, zu);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::project_primals_into_interior_cuda(info, xlam, push, push);
      AMIGO_CHECK_CUDA(cudaGetLastError());
      AMIGO_CHECK_CUDA(cudaDeviceSynchronize());

      detail::initialize_bound_duals_warm_cuda(mu, zmax, info, xlam, zl, zu);
      AMIGO_CHECK_CUDA(cudaGetLastError());
      AMIGO_CHECK_CUDA(cudaDeviceSynchronize());
    }
#endif
  }

  /**
   * @brief Place the fixed variables with equal bounds at their bound value
   *
   * @param x The primal-dual vector
   */
  void set_fixed_values(std::shared_ptr<Vector<T>> x) const {
    if (fixed_indices) {
      x->template set_values_at<policy>(fixed_indices, fixed_values);
    }
  }

  /**
   * @brief Warm start that keeps the given bound duals, projecting the
   * primals with a small push and raising each bound dual to at least zmin
   *
   * @param push The bound push fraction for the interior projection
   * @param zmin The bound dual floor
   * @param vars All of the optimization variables
   */
  void initialize_duals_keep(T push, T zmin,
                             std::shared_ptr<OptVector<T>> vars) const {
    T* xlam = vars->template get_solution_array<policy>();
    T *zl, *zu;
    vars->template get_bound_duals<policy>(&zl, &zu);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::project_primals_into_interior(info, xlam, push, push);
      detail::push_bound_duals(zmin, info, zl, zu);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::project_primals_into_interior_cuda(info, xlam, push, push);
      AMIGO_CHECK_CUDA(cudaGetLastError());
      detail::push_bound_duals_cuda(zmin, info, zl, zu);
      AMIGO_CHECK_CUDA(cudaGetLastError());
      AMIGO_CHECK_CUDA(cudaDeviceSynchronize());
    }
#endif
  }

  /**
   * @brief Compute the negative of the primal-dual residuals based on the value
   * of the gradient and the optimizer state variables
   *
   * This function computes the condensed augmented system RHS (8-block to
   * 4-block).
   *
   * @param mu The barrier parameter for the residual
   * @param vars The optimization variables
   * @param grad The gradient computed from the problem
   * @param res The full KKT residual
   * @return T Returns L2 norm of the condensed residual.
   */
  T compute_residual(T mu, const std::shared_ptr<OptVector<T>> vars,
                     const std::shared_ptr<Vector<T>> grad,
                     std::shared_ptr<Vector<T>> res) const {
    detail::OptState<const T> pt =
        detail::OptState<const T>::template make<policy>(vars);
    T* g = grad->template get_array<policy>();
    T* r = res->template get_array<policy>();

    res->zero();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_residual(mu, info, pt, g, r);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_residual_cuda(mu, info, pt, g, r);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif

    // Compute the local contributions to the residual norm
    T local = res->template dot<policy>(res);
    T total;
    MPI_Allreduce(&local, &total, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return std::sqrt(total);
  }

  /**
   * @brief Compute the diagonal contribution to the KKT matrix
   *
   * @param vars The values of the optimization variables
   * @param diag The vector containing the diagonal components of the matrix
   */
  void compute_diagonal(const std::shared_ptr<OptVector<T>> vars,
                        std::shared_ptr<Vector<T>> diagonal) const {
    // Zero the diagonal
    diagonal->zero();

    detail::OptState<const T> pt =
        detail::OptState<const T>::template make<policy>(vars);
    T* diag = diagonal->template get_array<policy>();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_diagonal(info, pt, diag);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_diagonal_cuda(info, pt, diag);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif
  }

  /**
   * @brief Compute the step in the primal, dual and bound dual variables based
   * on the computed primal dual step. This is equvalent to a block
   * back-substitution for the KKT system.
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   * @param px The primal dual update
   * @param update The full primal/dual step
   */
  void compute_update(T mu, const std::shared_ptr<OptVector<T>> vars,
                      const std::shared_ptr<Vector<T>> px,
                      std::shared_ptr<OptVector<T>> update) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    std::shared_ptr<Vector<T>> px_update = update->get_solution();
    if (px_update != px) {
      px_update->copy(px);
    }

    // Set the updates for the dual variables
    T *dzl, *dzu;
    update->template get_bound_duals<policy>(&dzl, &dzu);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_bound_dual_step(
          mu, info, current, px->template get_array<policy>(), dzl, dzu);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_bound_dual_step_cuda(
          mu, info, current, px->template get_array<policy>(), dzl, dzu);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif
  }

  /**
   * @brief Compute the max primal/dual step lengths based on the
   * fraction-to-the-boundary rule.
   *
   * @param tau The fraction to the boundary
   * @param vars The values of the optimization variables
   * @param update The full step (often the result of compute_update)
   * @param ax The max primal step
   * @param xi The index of the constraining variable
   * @param az The max dual step
   * @param zi The index of the constraining dual variable
   */
  void compute_max_step(T tau, const std::shared_ptr<OptVector<T>> vars,
                        const std::shared_ptr<OptVector<T>> update, T& ax,
                        int& xi, T& az, int& zi) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    detail::OptState<const T> step =
        detail::OptState<const T>::template make<policy>(update);

    const T *dzl, *dzu;
    update->template get_bound_duals<policy>(&dzl, &dzu);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_max_step(tau, info, current, step, ax, xi, az, zi);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_max_step_cuda(tau, info, current, step, ax, xi, az, zi);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif
  }

  /**
   * @brief Apply the primal dual step to the variables
   *
   * @param ax The max primal step
   * @param az The max dual step
   * @param vars The values of the optimization variables
   * @param update The full step
   * @param result The variables at the full step
   */
  void apply_step_update(T ax, T az, const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<OptVector<T>> update,
                         std::shared_ptr<OptVector<T>> result) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    detail::OptState<const T> step =
        detail::OptState<const T>::template make<policy>(update);
    detail::OptState<T> result_state =
        detail::OptState<T>::template make<policy>(result);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::apply_step(ax, az, info, current, step, result_state);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::apply_step_cuda(ax, az, info, current, step, result_state);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif
  }

  /**
   * @brief Clip the bound duals into [mu/(kappa*gap), kappa*mu/gap]
   *
   * @param mu The barrier parameter
   * @param kappa The kappa_Sigma safeguard factor
   * @param vars The values of the optimization variables
   */
  void correct_bound_multipliers(T mu, T kappa,
                                 std::shared_ptr<OptVector<T>> vars) const {
    detail::OptState<T> current =
        detail::OptState<T>::template make<policy>(vars);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::correct_bound_multipliers(mu, kappa, info, current);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::correct_bound_multipliers_cuda(mu, kappa, info, current);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif
  }

  /**
   * @brief Compute the average complementarity and the xi parameter from LOQO
   *
   * @param vars The values of the optimization variables
   * @param avg The average complementarity
   * @param xi The parameter xi
   */
  void compute_complementarity(const std::shared_ptr<OptVector<T>> vars, T& avg,
                               T& xi) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T ps[2] = {0, 0};
    T lm = std::numeric_limits<T>::max();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_complementarity(info, s, ps, lm);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_complementarity_cuda(info, s, ps, lm);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif

    T gps[2];
    T gm;
    MPI_Allreduce(ps, gps, 2, get_mpi_type<T>(), MPI_SUM, comm);
    MPI_Allreduce(&lm, &gm, 1, get_mpi_type<T>(), MPI_MIN, comm);

    // Compute the average complementarity
    avg = 0.0;
    if (gps[1] > 0.0) {
      avg = gps[0] / gps[1];
    }

    // Compute the xi vactor
    xi = 1.0;
    if (avg > 0.0) {
      xi = A2D::max2(0.0, A2D::min2(T(1.0), gm / avg));
    }
  }

  /**
   * @brief Compute the KKT errors for the dual, primal and complementarity
   * equations in the infinity norm
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   * @param grad The gradient
   * @param d_inf The output dual error
   * @param p_inf The output primal error
   * @param c_inf The output complementarity error
   */
  void compute_kkt_error(T mu, const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<Vector<T>> grad, T& d_inf,
                         T& p_inf, T& c_inf) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T ld = 0, lp = 0, lc = 0;

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_kkt_error(mu, info, s, grad->template get_array<policy>(),
                                ld, lp, lc);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_kkt_error_cuda(
          mu, info, s, grad->template get_array<policy>(), ld, lp, lc);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif

    T lv[3] = {ld, lp, lc}, gv[3];
    MPI_Allreduce(lv, gv, 3, get_mpi_type<T>(), MPI_MAX, comm);
    d_inf = gv[0];
    p_inf = gv[1];
    c_inf = gv[2];
  }

  /**
   * @brief Compute the log-barrier term
   *
   * Returns the value of
   *
   * - mu * log(x - lb) - mu * log(ub - x)
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   * @return The value of the log-barrier term
   */
  T compute_log_barrier(T mu, const std::shared_ptr<OptVector<T>> vars) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    T local = 0.0;

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      local = detail::compute_log_barrier(mu, info, current);
    }
#ifdef AMIGO_USE_CUDA
    else {
      local = detail::compute_log_barrier_cuda(mu, info, current);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif

    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  /**
   * @brief Compute the directional derivative of the log-barrier term
   *
   * Returns the value of
   *
   * - mu * e^{T} * (X - LB)^{-1} * px + mu * e^{T} * (UB - X)^{-1} * px
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   * @param update The update for the optimization variables
   * @return The value of the directional derivative of the log-barrier term
   */
  T compute_log_barrier_derivative(
      T mu, const std::shared_ptr<OptVector<T>> vars,
      const std::shared_ptr<OptVector<T>> update) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    detail::OptState<const T> step =
        detail::OptState<const T>::template make<policy>(update);

    T local = 0.0;
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      local = detail::compute_log_barrier_derivative(mu, info, current, step);
    }
#ifdef AMIGO_USE_CUDA
    else {
      local =
          detail::compute_log_barrier_derivative_cuda(mu, info, current, step);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif

    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  /**
   * @brief Compute the squared sum of the complementarity
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   */
  T compute_sum_squared_complementarity(
      T mu, const std::shared_ptr<OptVector<T>> vars) {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    T local = 0.0;

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      local = detail::compute_sum_squared_complementarity(mu, info, current);
    }
#ifdef AMIGO_USE_CUDA
    else {
      local =
          detail::compute_sum_squared_complementarity_cuda(mu, info, current);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif

    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  /**
   * @brief Step lengths and trial complementarity of a quality function step
   *
   * Replaces compute_update, compute_max_step, apply_step_update and
   * compute_sum_squared_complementarity for the step px0 + sigma * dpx at mu =
   * sigma * mu_nat without forming the step or the trial point.
   *
   * @param sigma The centering parameter
   * @param mu_nat The average complementarity
   * @param tau The fraction to the boundary
   * @param vars The values of the optimization variables
   * @param px0 The primal dual step at mu = 0
   * @param dpx The change of the primal dual step from mu = 0 to mu_nat
   * @param ax The max primal step
   * @param az The max dual step
   * @param comp_sq The squared complementarity at the trial point
   */
  void compute_quality_function_trial(T sigma, T mu_nat, T tau,
                                      const std::shared_ptr<OptVector<T>> vars,
                                      const std::shared_ptr<Vector<T>> px0,
                                      const std::shared_ptr<Vector<T>> dpx,
                                      T& ax, T& az, T& comp_sq) const {
    static_assert(policy != ExecPolicy::CUDA,
                  "The quality function trial has no CUDA kernel");
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    const T* p0 = px0->template get_array<policy>();
    const T* dp = dpx->template get_array<policy>();

    // Every rank evaluates the trial point at the global step lengths
    T local[2] = {1.0, 1.0}, global[2];
    detail::compute_quality_function_steps(sigma, mu_nat, tau, info, current, p0,
                                           dp, local[0], local[1]);
    MPI_Allreduce(local, global, 2, get_mpi_type<T>(), MPI_MIN, comm);
    ax = global[0];
    az = global[1];

    T local_sq = detail::compute_quality_function_complementarity(
        sigma, mu_nat, ax, az, info, current, p0, dp);
    MPI_Allreduce(&local_sq, &comp_sq, 1, get_mpi_type<T>(), MPI_SUM, comm);
  }

  /**
   * @brief Compute the infeasibility in the l1 norm based on the gradient
   *
   * @param gradient The gradient at a point
   */
  T compute_infeasibility(const std::shared_ptr<Vector<T>> gradient) {
    const T* grad = gradient->template get_array<policy>();
    T local = 0.0;

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      local = detail::compute_infeasibility(info, grad);
    }
#ifdef AMIGO_USE_CUDA
    else {
      local = detail::compute_infeasibility_cuda(info, grad);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif

    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  /**
   * @brief Compute the dual residual = grad(f) - zl + zu
   *
   * This is used to find a least-squares estiamte of the initial multipliers
   *
   * @param vars The values of the optimization variables
   * @param gradient The gradient at a point
   * @param out The result = grad(f) - zl + zu
   */
  void compute_dual_residual(const std::shared_ptr<OptVector<T>> vars,
                             const std::shared_ptr<Vector<T>> grad,
                             std::shared_ptr<Vector<T>> out) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);

    out->zero();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_dual_residual(info, s, grad->template get_array<policy>(),
                                    out->template get_array<policy>(),
                                    out->get_size());
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_dual_residual_cuda(
          info, s, grad->template get_array<policy>(),
          out->template get_array<policy>(), out->get_size());
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif
  }

  // Global counts of the dual, primal, and complementarity components
  void get_kkt_element_counts(int& n_d, int& n_p, int& n_c) const {
    n_d = num_global_primals;
    n_p = num_global_constraints;
    int local = 0;
    for (int i = 0; i < num_primals; i++) {
      if (!std::isinf((*lbx)[i])) local++;
      if (!std::isinf((*ubx)[i])) local++;
    }
    MPI_Allreduce(&local, &n_c, 1, MPI_INT, MPI_SUM, comm);
  }

  /**
   * @brief Set the NLP scale factors on the problem and match the constraint
   * targets to them
   *
   * The scale is one on the primals and dc on each constraint row, so a
   * scaled constraint dc * c(x) = dc * b keeps the solution of c(x) = b. A
   * null scale restores the unscaled problem.
   *
   * @param scale The scale factors over the full vector, or null
   */
  void set_var_scale(std::shared_ptr<Vector<T>> scale) {
    problem->set_var_scale(scale);
    info.con_scale = nullptr;
    lbh->copy(lbh0);
    if (scale) {
      scale->copy_device_to_host();
      for (int i = 0; i < num_constraints; i++) {
        T dc = (*scale)[(*constraint_indices)[i]];
        (*con_scale)[i] = dc;
        (*lbh)[i] = dc * (*lbh0)[i];
      }
      con_scale->copy_host_to_device();
      info.con_scale = con_scale->template get_array<policy>();
    }
    lbh->copy_host_to_device();
  }

  /**
   * @brief Squared 2-norms of the dual and primal residuals
   *
   * @param vars The values of the optimization variables
   * @param gradient The gradient at the point
   * @param dual_sq The output sum of squares of the dual residual
   * @param primal_sq The output sum of squares of the primal residual
   */
  void compute_kkt_error_squares(const std::shared_ptr<OptVector<T>> vars,
                                 const std::shared_ptr<Vector<T>> gradient,
                                 T& dual_sq, T& primal_sq) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    const T* grad = gradient->template get_array<policy>();
    T local[2] = {0, 0}, global[2];

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_kkt_error_squares(info, s, grad, local[0], local[1]);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_kkt_error_squares_cuda(info, s, grad, local[0], local[1]);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif

    MPI_Allreduce(local, global, 2, get_mpi_type<T>(), MPI_SUM, comm);
    dual_sq = global[0];
    primal_sq = global[1];
  }

  /**
   * @brief Largest constraint violation in the units of the unscaled problem
   *
   * @param gradient The gradient whose constraint rows hold the residuals
   */
  T compute_constraint_violation(const std::shared_ptr<Vector<T>> gradient) {
    const T* grad = gradient->template get_array<policy>();
    T local = 0.0;

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      local = detail::compute_constraint_violation(info, grad);
    }
#ifdef AMIGO_USE_CUDA
    else {
      local = detail::compute_constraint_violation_cuda(info, grad);
      AMIGO_CHECK_CUDA(cudaGetLastError());
    }
#endif

    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_MAX, comm);
    return result;
  }

  // Relax bounds by bound_relax_factor from the original bounds
  // A factor of zero restores the original bounds
  void relax_bounds(T factor = 1e-8, T constr_viol_tol = 1e-4) {
    if (factor <= 0) {
      lbx_relaxed = nullptr;
      ubx_relaxed = nullptr;
      info.lbx = lbx->template get_array<policy>();
      info.ubx = ubx->template get_array<policy>();
      return;
    }
    lbx_relaxed =
        std::make_shared<Vector<T>>(num_primals, 0, lbx->get_memory_location());
    ubx_relaxed =
        std::make_shared<Vector<T>>(num_primals, 0, ubx->get_memory_location());

    detail::OptProblemInfo<T> host_info = info;
    host_info.lbx = lbx->template get_array<ExecPolicy::SERIAL>();
    host_info.ubx = ubx->template get_array<ExecPolicy::SERIAL>();
    T* lb_buf = lbx_relaxed->template get_array<ExecPolicy::SERIAL>();
    T* ub_buf = ubx_relaxed->template get_array<ExecPolicy::SERIAL>();
    detail::relax_bounds(host_info, lb_buf, ub_buf, factor, constr_viol_tol);

    lbx_relaxed->copy_host_to_device();
    ubx_relaxed->copy_host_to_device();
    info.lbx = lbx_relaxed->template get_array<policy>();
    info.ubx = ubx_relaxed->template get_array<policy>();
  }

 private:
  // Bound magnitude treated as infinite
  static constexpr double bound_infinity = 1e19;

  std::shared_ptr<OptimizationProblem<T, policy>> problem;
  MPI_Comm comm;

  int num_primals;
  int num_constraints;
  std::shared_ptr<Vector<int>> primal_indices;
  std::shared_ptr<Vector<int>> constraint_indices;
  std::shared_ptr<Vector<T>> lbx, ubx;
  std::shared_ptr<Vector<T>> lbx_relaxed, ubx_relaxed;

  // Unscaled constraint targets, the scaled targets, and the constraint scales
  std::shared_ptr<Vector<T>> lbh0, lbh, con_scale;

  // Fixed variables with equal bounds and their values
  std::shared_ptr<Vector<int>> fixed_indices;
  std::shared_ptr<Vector<T>> fixed_values;

  // Object for storing informabout about the problem
  detail::OptProblemInfo<T> info;

  // Number of global constraints
  int num_global_primals;
  int num_global_constraints;
};

}  // namespace amigo

#endif  // AMIGO_INTERIOR_POINT_OPTIMIZER_H
