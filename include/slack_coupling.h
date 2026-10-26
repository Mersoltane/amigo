#ifndef AMIGO_SLACK_COUPLING_H
#define AMIGO_SLACK_COUPLING_H

#include "component_group_base.h"

namespace amigo {

/**
 * Slack component generated from the following python class:
 *
 * class SlackComponent(am.Component):
 *     def __init__(self):
 *         super().__init__()
 *         self.add_input("s")
 *         self.add_constraint("res")
 *
 *     def compute(self):
 *         self.constraints["res"] = -self.inputs["s"]
 */
template <typename T__>
class SlackComponent__ {
 public:
  template <typename R__>
  using Input = A2D::VarTuple<R__, R__, R__>;
  static constexpr int ncomp = Input<T__>::ncomp;
  static constexpr int nconstraints = 1;
  static constexpr bool is_linear = false;
  template <typename R__>
  using Data = typename A2D::VarTuple<R__, R__>;
  static constexpr int ndata = 0;
  static constexpr bool is_compute_empty = false;
  static constexpr bool is_continuation_component = false;
  static constexpr bool is_output_empty = true;
  template <typename R__>
  using Output = typename A2D::VarTuple<R__, R__>;
  static constexpr int noutputs = 0;
  template <typename R__>
  AMIGO_HOST_DEVICE static R__ lagrange(R__ alpha__, Data<R__>& data__,
                                        Input<R__>& input__) {
    R__& s = A2D::get<0>(input__);
    R__& lam_res__ = A2D::get<1>(input__);
    R__ lagrangian__;
    lagrangian__ = (-(s)*lam_res__);
    return lagrangian__;
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void gradient(R__ alpha__, Data<R__>& data__,
                                         Input<R__>& input__,
                                         Input<R__>& boutput__) {
    A2D::ADObj<R__&> s(A2D::get<0>(input__), A2D::get<0>(boutput__));
    A2D::ADObj<R__&> lam_res__(A2D::get<1>(input__), A2D::get<1>(boutput__));
    A2D::ADObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(A2D::Eval((-(s)*lam_res__), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void hessian(R__ alpha__, Data<R__>& data__,
                                        Input<R__>& input__,
                                        Input<R__>& pinput__,
                                        Input<R__>& boutput__,
                                        Input<R__>& houtput__) {
    A2D::A2DObj<R__&> s(A2D::get<0>(input__), A2D::get<0>(boutput__),
                        A2D::get<0>(pinput__), A2D::get<0>(houtput__));
    A2D::A2DObj<R__&> lam_res__(A2D::get<1>(input__), A2D::get<1>(boutput__),
                                A2D::get<1>(pinput__), A2D::get<1>(houtput__));
    A2D::A2DObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(A2D::Eval((-(s)*lam_res__), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
    stack__.hforward();
    stack__.hreverse();
  }
  static constexpr int hessian_num_dirs = 1;
  static constexpr int hessian_seed_ptr[] = {0, 2};
  static constexpr int hessian_seeds[] = {0, 1};
  static constexpr int hessian_entry_ptr[] = {0, 2};
  static constexpr int hessian_entry_row[] = {0, 1};
  static constexpr int hessian_entry_col[] = {1, 0};
  static constexpr int hessian_entry_src[] = {0, 1};
  template <typename R__, class Emit__>
  static void hessian_entries(R__ alpha__, Data<R__>& data__,
                              Input<R__>& input__, const Emit__& emit__) {
    Input<R__> boutput__, pinput__, houtput__;
    boutput__.zero();
    pinput__.zero();
    houtput__.zero();
    A2D::A2DObj<R__&> s(A2D::get<0>(input__), A2D::get<0>(boutput__),
                        A2D::get<0>(pinput__), A2D::get<0>(houtput__));
    A2D::A2DObj<R__&> lam_res__(A2D::get<1>(input__), A2D::get<1>(boutput__),
                                A2D::get<1>(pinput__), A2D::get<1>(houtput__));
    A2D::A2DObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(A2D::Eval((-(s)*lam_res__), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
    for (int d__ = 0; d__ < hessian_num_dirs; d__++) {
      pinput__.zero();
      houtput__.zero();
      stack__.hzero();
      for (int k__ = hessian_seed_ptr[d__]; k__ < hessian_seed_ptr[d__ + 1];
           k__++) {
        pinput__[hessian_seeds[k__]] = 1.0;
      }
      stack__.hforward();
      stack__.hreverse();
      for (int e__ = hessian_entry_ptr[d__]; e__ < hessian_entry_ptr[d__ + 1];
           e__++) {
        emit__(hessian_entry_row[e__], hessian_entry_col[e__],
               houtput__[hessian_entry_src[e__]]);
      }
    }
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void compute_output(Data<R__>& data__,
                                               Input<R__>& input__,
                                               Output<R__>& output__) {}
};

}  // namespace amigo

#endif  // AMIGO_SLACK_COUPLING_H
