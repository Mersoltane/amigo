#ifndef AMIGO_SLACK_COUPLING_H
#define AMIGO_SLACK_COUPLING_H

#include "component_group_base.h"
#include "csr_matrix.h"
#include "vector.h"

namespace amigo {

/*
  Couples slack variables to inequality constraints.

  Each inequality  lbc <= c(x) <= ubc  is reformulated as the equality
  c(x) - s = 0  with  lbc <= s <= ubc.  This component declares the
  Jacobian sparsity (-I coupling), adds gradient contributions
  (dL/ds = -lambda, primal feasibility c(x) - s), and writes the
  constant -1 entries into the KKT matrix.
*/
template <typename T, ExecPolicy policy>
class SlackCouplingGroup : public ComponentGroupBase<T, policy> {
 public:
  SlackCouplingGroup(int num_slacks, const int slack_indices[],
                     const int ineq_indices[])
      : n_(num_slacks) {
    si_ = std::make_shared<Vector<int>>(n_);
    ci_ = std::make_shared<Vector<int>>(n_);
    si_->copy(slack_indices);
    ci_->copy(ineq_indices);

    // Jacobian CSR: identity pattern in local indexing.
    // Row k has one entry at column k, mapping to (ineq_k, slack_k).
    jac_rowp_ = std::make_shared<Vector<int>>(n_ + 1);
    jac_cols_ = std::make_shared<Vector<int>>(n_);
    int* rp = jac_rowp_->get_array();
    int* cl = jac_cols_->get_array();
    for (int k = 0; k < n_; k++) { rp[k] = k; cl[k] = k; }
    rp[n_] = n_;

    // Data positions in the CSR matrix, filled once during initialize_hessian_pattern
    loc_si_ = std::make_shared<Vector<int>>(n_);
    loc_ci_ = std::make_shared<Vector<int>>(n_);
  }

  std::shared_ptr<ComponentGroupBase<T, policy>> clone(
      int, std::shared_ptr<Vector<int>>, std::shared_ptr<Vector<int>>,
      std::shared_ptr<Vector<int>>) const override {
    return nullptr;
  }

  // Called once after create_matrix(). Finds the data[] positions of the
  // -1 entries so add_hessian is O(n) with direct array writes.
  void initialize_hessian_pattern(const NodeOwners& owners,
                                  CSRMat<T>& mat) override {
    const int* si = si_->get_array();
    const int* ci = ci_->get_array();
    int* lsi = loc_si_->get_array();
    int* lci = loc_ci_->get_array();
    for (int k = 0; k < n_; k++) {
      mat.get_sorted_locations(si[k], 1, &ci[k], &lsi[k]);
      mat.get_sorted_locations(ci[k], 1, &si[k], &lci[k]);
    }
  }

  // grad[slack_k] += -lambda_k   (slack stationarity: dL/ds = -lambda)
  // grad[ineq_k]  += -s_k        (primal feasibility: c(x) - s)
  void add_gradient(T alpha, const Vector<T>& data, const Vector<T>& x,
                    Vector<T>& g) const override {
    const int* si = si_->template get_array<policy>();
    const int* ci = ci_->template get_array<policy>();
    const T* xlam = x.template get_array<policy>();
    T* grad = g.template get_array<policy>();

    for (int k = 0; k < n_; k++) {
      grad[si[k]] += -xlam[ci[k]];
      grad[ci[k]] += -xlam[si[k]];
    }
  }

  // Write -1 at precomputed positions: H[slack_k, ineq_k] and H[ineq_k, slack_k].
  void add_hessian(T alpha, const Vector<T>& data, const Vector<T>& x,
                   const NodeOwners& owners, CSRMat<T>& mat) const override {
    const int* lsi = loc_si_->get_array();
    const int* lci = loc_ci_->get_array();
    T* d = mat.get_data_ptr();
    for (int k = 0; k < n_; k++) {
      d[lsi[k]] += T(-1);
      d[lci[k]] += T(-1);
    }
  }

  void get_csr_data(int* nvars, const int* vars[], int* ncon,
                    const int* cons[], const int* jac_rowp[],
                    const int* jac_cols[], const int* hess_rowp[],
                    const int* hess_cols[]) const override {
    if (nvars) *nvars = n_;
    if (vars) *vars = si_->get_array();
    if (ncon) *ncon = n_;
    if (cons) *cons = ci_->get_array();
    if (jac_rowp) *jac_rowp = jac_rowp_->get_array();
    if (jac_cols) *jac_cols = jac_cols_->get_array();
    if (hess_rowp) *hess_rowp = nullptr;
    if (hess_cols) *hess_cols = nullptr;
  }

 private:
  int n_;
  std::shared_ptr<Vector<int>> si_, ci_;
  std::shared_ptr<Vector<int>> jac_rowp_, jac_cols_;
  std::shared_ptr<Vector<int>> loc_si_, loc_ci_;
};

}  // namespace amigo

#endif  // AMIGO_SLACK_COUPLING_H
