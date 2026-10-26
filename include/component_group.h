#ifndef AMIGO_COMPONENT_GROUP_H
#define AMIGO_COMPONENT_GROUP_H

#include <algorithm>
#include <vector>

#include "component_group_base.h"
#include "csr_matrix.h"
#include "layout.h"
#include "node_owners.h"
#include "ordering_utils.h"
#include "scalar_ad.h"
#include "vector.h"

#ifdef AMIGO_USE_CUDA
#include "cuda/component_group_backend.cuh"
#endif

namespace amigo {

namespace detail {

template <typename T, class Data, class Input, class Component, class... Remain>
T compute_lagrangian(T alpha, Data& data, Input& input) {
  T value = 0.0;

  if constexpr (!Component::is_compute_empty) {
    value = Component::lagrange(alpha, data, input);
  }

  if constexpr (sizeof...(Remain) > 0) {
    return value +
           compute_lagrangian<T, Data, Input, Remain...>(alpha, data, input);
  } else {
    return value;
  }
}

template <typename T, class Data, class Input, class Component, class... Remain>
void compute_gradient(T alpha, Data& data, Input& input, Input& grad) {
  if constexpr (!Component::is_compute_empty) {
    Component::gradient(alpha, data, input, grad);
  }

  if constexpr (sizeof...(Remain) > 0) {
    compute_gradient<T, Data, Input, Remain...>(alpha, data, input, grad);
  }
}

template <typename T, class Data, class Input, class Component, class... Remain>
void compute_hessian(T alpha, Data& data, Input& input, Input& dir, Input& grad,
                     Input& h) {
  if constexpr (!Component::is_compute_empty) {
    Component::hessian(alpha, data, input, dir, grad, h);
  }

  if constexpr (sizeof...(Remain) > 0) {
    compute_hessian<T, Data, Input, Remain...>(alpha, data, input, dir, grad,
                                               h);
  }
}

// Add the structurally nonzero Hessian entries of each component to the matrix
template <typename T, class Data, class Input, class Component, class... Remain>
void add_hessian_entries(T alpha, Data& data, Input& input, const int index[],
                         const int index_global[], CSRMat<T>& jac) {
  if constexpr (!Component::is_compute_empty) {
    Component::hessian_entries(alpha, data, input,
                               [&](int row, int col, T value) {
                                 jac.add_row(index[row], 1, &index_global[col],
                                             &value);
                               });
  }

  if constexpr (sizeof...(Remain) > 0) {
    add_hessian_entries<T, Data, Input, Remain...>(alpha, data, input, index,
                                                   index_global, jac);
  }
}

// Number of Hessian entries the components emit for one element
template <class Component, class... Remain>
constexpr int count_hessian_entries() {
  int count = 0;
  if constexpr (!Component::is_compute_empty) {
    count = Component::hessian_entry_ptr[Component::hessian_num_dirs];
  }
  if constexpr (sizeof...(Remain) > 0) {
    return count + count_hessian_entries<Remain...>();
  } else {
    return count;
  }
}

// Store the Hessian entries of each component in the order they are emitted
template <typename T, class Data, class Input, class Component, class... Remain>
void compute_hessian_entries(T alpha, Data& data, Input& input, T values[]) {
  if constexpr (!Component::is_compute_empty) {
    int k = 0;
    Component::hessian_entries(
        alpha, data, input, [&](int row, int col, T value) { values[k++] = value; });
    values += Component::hessian_entry_ptr[Component::hessian_num_dirs];
  }
  if constexpr (sizeof...(Remain) > 0) {
    compute_hessian_entries<T, Data, Input, Remain...>(alpha, data, input,
                                                       values);
  }
}

// Matrix data position of each Hessian entry of one element or -1 when absent
template <class Component, class... Remain>
void find_hessian_positions(const int rowp[], const int cols[],
                            const int index[], const int index_global[],
                            int positions[]) {
  if constexpr (!Component::is_compute_empty) {
    int count = Component::hessian_entry_ptr[Component::hessian_num_dirs];
    for (int e = 0; e < count; e++) {
      int row = index[Component::hessian_entry_row[e]];
      int col = index_global[Component::hessian_entry_col[e]];
      const int* start = &cols[rowp[row]];
      const int* end = &cols[rowp[row + 1]];
      const int* it = std::lower_bound(start, end, col);
      positions[e] = (it != end && *it == col) ? static_cast<int>(it - cols) : -1;
    }
    positions += count;
  }
  if constexpr (sizeof...(Remain) > 0) {
    find_hessian_positions<Remain...>(rowp, cols, index, index_global,
                                      positions);
  }
}

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
class SerialGroupBackend {
 public:
  template <int noutputs>
  SerialGroupBackend(IndexLayout<ndata>& data_layout,
                     IndexLayout<ncomp>& layout,
                     IndexLayout<noutputs>& outputs) {}

  T lagrangian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                      const IndexLayout<ncomp>& layout,
                      const Vector<T>& data_vec, const Vector<T>& vec) const {
    T value = 0.0;
    if constexpr (ncomp > 0) {
      int num_elems = layout.get_num_elements();
      for (int i = 0; i < num_elems; i++) {
        Data data;
        Input input;
        data_layout.get_values(i, data_vec, data);
        layout.get_values(i, vec, input);
        value += compute_lagrangian<T, Data, Input, Components...>(alpha, data,
                                                                   input);
      }
    }
    return value;
  }

  void add_gradient_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                           const IndexLayout<ncomp>& layout,
                           const Vector<T>& data_vec, const Vector<T>& vec,
                           Vector<T>& res) const {
    if constexpr (ncomp > 0) {
      int num_elems = layout.get_num_elements();
      for (int i = 0; i < num_elems; i++) {
        Data data;
        Input input, gradient;
        data_layout.get_values(i, data_vec, data);
        layout.get_values(i, vec, input);
        gradient.zero();
        compute_gradient<T, Data, Input, Components...>(alpha, data, input,
                                                        gradient);
        layout.add_values(i, gradient, res);
      }
    }
  }

  void add_hessian_product_kernel(T alpha,
                                  const IndexLayout<ndata>& data_layout,
                                  const IndexLayout<ncomp>& layout,
                                  const Vector<T>& data_vec,
                                  const Vector<T>& vec, const Vector<T>& dir,
                                  Vector<T>& res) const {
    if constexpr (ncomp > 0) {
      int num_elems = layout.get_num_elements();
      for (int elem = 0; elem < num_elems; elem++) {
        Data data;
        Input input, gradient, direction, result;
        data_layout.get_values(elem, data_vec, data);
        layout.get_values(elem, vec, input);
        layout.get_values(elem, dir, direction);
        gradient.zero();
        result.zero();
        compute_hessian<T, Data, Input, Components...>(
            alpha, data, input, direction, gradient, result);
        layout.add_values(elem, result, res);
      }
    }
  }

  void initialize_hessian_pattern(const IndexLayout<ncomp>& layout,
                                  const NodeOwners& owners,
                                  CSRMat<T>& mat) const {}

  void add_hessian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                          const IndexLayout<ncomp>& layout,
                          const Vector<T>& data_vec, const Vector<T>& vec,
                          const NodeOwners& owners, CSRMat<T>& jac) const {
    if constexpr (ncomp > 0) {
      int num_elems = layout.get_num_elements();
      Data data;
      Input input;
      for (int elem = 0; elem < num_elems; elem++) {
        int index[ncomp], index_global[ncomp];
        layout.get_indices(elem, index);
        owners.local_to_global(ncomp, index, index_global);

        data_layout.get_values(elem, data_vec, data);
        layout.get_values(elem, vec, input);

        add_hessian_entries<T, Data, Input, Components...>(
            alpha, data, input, index, index_global, jac);
      }
    }
  }

  void add_grad_jac_product_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_tproduct_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_wrt_data_kernel(const IndexLayout<ndata>& data_layout,
                                    const IndexLayout<ncomp>& layout,
                                    const Vector<T>& data_vec,
                                    const Vector<T>& vec,
                                    const NodeOwners& owners,
                                    CSRMat<T>& jac) const {
    using ADtype = A2D::ADScalar<T, 1>;
    using DataAD =
        typename __get_collection_data_type<ADtype, Components...>::type;
    using InputAD =
        typename __get_collection_input_type<ADtype, Components...>::type;

    if constexpr (ncomp > 0 && ndata > 0) {
      int num_elems = layout.get_num_elements();

      for (int i = 0; i < num_elems; i++) {
        InputAD input;
        layout.get_values(i, vec, input);

        T jac_elem[ncomp * ndata];
        for (int j = 0; j < ndata; j++) {
          DataAD data;
          data_layout.get_values(i, data_vec, data);
          data[j].deriv[0] = 1.0;

          InputAD gradient;
          gradient.zero();

          ADtype alpha(1.0);
          compute_gradient<ADtype, DataAD, InputAD, Components...>(
              alpha, data, input, gradient);

          for (int k = 0; k < ncomp; k++) {
            jac_elem[ndata * k + j] = gradient[k].deriv[0];
          }
        }

        int input_index[ncomp];
        layout.get_indices(i, input_index);

        int data_indices[ndata], data_indices_global[ndata];
        data_layout.get_indices(i, data_indices);
        owners.local_to_global(ndata, data_indices, data_indices_global);

        for (int j = 0; j < ncomp; j++) {
          jac.add_row(input_index[j], ndata, data_indices_global,
                      &jac_elem[ndata * j]);
        }
      }
    }
  }
};

template <typename T, int ncomp, int ndata, int noutputs, class... Components>
class SerialOutputBackend {
 public:
  SerialOutputBackend() {}

  void add_output_kernel(const IndexLayout<ndata>& data_layout,
                         const IndexLayout<ncomp>& layout,
                         const IndexLayout<noutputs>& output_layout,
                         const Vector<T>& data_vec, const Vector<T>& vec,
                         Vector<T>& out) const {
    add_output_kernel<Components...>(data_layout, layout, output_layout,
                                     data_vec, vec, out);
  }

  template <class Component, class... Remain>
  void add_output_kernel(const IndexLayout<ndata>& data_layout,
                         const IndexLayout<ncomp>& layout,
                         const IndexLayout<noutputs>& output_layout,
                         const Vector<T>& data_vec, const Vector<T>& vec,
                         Vector<T>& out) const {
    if constexpr (Component::noutputs > 0) {
      typename Component::template Data<T> data;
      typename Component::template Input<T> input;
      typename Component::template Output<T> output;
      int num_elems = layout.get_num_elements();

      for (int i = 0; i < num_elems; i++) {
        data_layout.get_values(i, data_vec, data);
        layout.get_values(i, vec, input);
        Component::compute_output(data, input, output);
        output_layout.add_values(i, output, out);
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_output_kernel<Remain...>(data_layout, layout, output_layout, data_vec,
                                   vec, out);
    }
  }

  void add_output_jac_wrt_input_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const IndexLayout<noutputs>& output_layout, const Vector<T>& data_vec,
      const Vector<T>& vec, CSRMat<T>& jac) const {
    add_output_jac_wrt_input_kernel<Components...>(
        data_layout, layout, output_layout, data_vec, vec, jac);
  }

  template <class Component, class... Remain>
  void add_output_jac_wrt_input_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const IndexLayout<noutputs>& output_layout, const Vector<T>& data_vec,
      const Vector<T>& vec, CSRMat<T>& jac) const {
    if constexpr (Component::noutputs > 0 && Component::ncomp > 0) {
      typename Component::template Data<A2D::ADScalar<T, 1>> data;
      typename Component::template Input<A2D::ADScalar<T, 1>> input;
      typename Component::template Output<A2D::ADScalar<T, 1>> output;
      int num_elems = layout.get_num_elements();

      for (int i = 0; i < num_elems; i++) {
        data_layout.get_values(i, data_vec, data);

        T jac_elem[noutputs * ncomp];
        for (int j = 0; j < ncomp; j++) {
          layout.get_values(i, vec, input);
          input[j].deriv[0] = 1.0;
          Component::compute_output(data, input, output);

          for (int k = 0; k < noutputs; k++) {
            jac_elem[ncomp * k + j] = output[k].deriv[0];
          }
        }

        int input_index[ncomp];
        layout.get_indices(i, input_index);

        int output_index[noutputs];
        output_layout.get_indices(i, output_index);

        for (int j = 0; j < noutputs; j++) {
          jac.add_row(output_index[j], ncomp, input_index,
                      &jac_elem[ncomp * j]);
        }
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_output_jac_wrt_input_kernel<Remain...>(
          data_layout, layout, output_layout, data_vec, vec, jac);
    }
  }

  void add_output_jac_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const IndexLayout<noutputs>& output_layout, const Vector<T>& data_vec,
      const Vector<T>& vec, CSRMat<T>& jac) const {
    add_output_jac_wrt_data_kernel<Components...>(
        data_layout, layout, output_layout, data_vec, vec, jac);
  }

  template <class Component, class... Remain>
  void add_output_jac_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const IndexLayout<noutputs>& output_layout, const Vector<T>& data_vec,
      const Vector<T>& vec, CSRMat<T>& jac) const {
    if constexpr (Component::noutputs > 0 && Component::ndata > 0) {
      typename Component::template Data<A2D::ADScalar<T, 1>> data;
      typename Component::template Input<A2D::ADScalar<T, 1>> input;
      typename Component::template Output<A2D::ADScalar<T, 1>> output;
      int num_elems = layout.get_num_elements();

      for (int i = 0; i < num_elems; i++) {
        layout.get_values(i, vec, input);

        T jac_elem[noutputs * ndata];
        for (int j = 0; j < ndata; j++) {
          data_layout.get_values(i, data_vec, data);
          data[j].deriv[0] = 1.0;
          Component::compute_output(data, input, output);

          for (int k = 0; k < noutputs; k++) {
            jac_elem[ndata * k + j] = output[k].deriv[0];
          }
        }

        int data_index[ndata];
        data_layout.get_indices(i, data_index);

        int output_index[noutputs];
        output_layout.get_indices(i, output_index);

        for (int j = 0; j < noutputs; j++) {
          jac.add_row(output_index[j], ndata, data_index, &jac_elem[ndata * j]);
        }
      }
    }

    if constexpr (sizeof...(Remain) > 0) {
      add_output_jac_wrt_data_kernel<Remain...>(
          data_layout, layout, output_layout, data_vec, vec, jac);
    }
  }
};

template <typename T, int ncomp, int ndata, int noutput, class... Components>
using DefaultOutputBackend =
    SerialOutputBackend<T, ncomp, ndata, noutput, Components...>;

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
class OmpGroupBackend {
 public:
  // Smaller loops run serially since a parallel region costs more than their work
  static constexpr int min_parallel_work = 1 << 14;

  // Values buffered per chunk of elements
  static constexpr int max_buffer_entries = 1 << 20;

  static constexpr int entries_per_element =
      count_hessian_entries<Components...>();

  template <int noutputs>
  OmpGroupBackend(IndexLayout<ndata>& data_layout, IndexLayout<ncomp>& layout,
                  IndexLayout<noutputs>& output_layout) {}

  // Elements are evaluated in parallel and accumulated serially in element order
  T lagrangian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                      const IndexLayout<ncomp>& layout,
                      const Vector<T>& data_vec, const Vector<T>& vec) const {
    T value = 0.0;
    if constexpr (ncomp > 0) {
      const int num_elems = layout.get_num_elements();
      const int chunk = chunk_size(1);
      values.resize(std::min(chunk, num_elems));
      for (int start = 0; start < num_elems; start += chunk) {
        const int end = std::min(start + chunk, num_elems);
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for schedule(dynamic, 32) if ((end - start) * ncomp >= min_parallel_work)
#endif
        for (int elem = start; elem < end; elem++) {
          Data data;
          Input input;
          data_layout.get_values(elem, data_vec, data);
          layout.get_values(elem, vec, input);
          values[elem - start] = compute_lagrangian<T, Data, Input, Components...>(
              alpha, data, input);
        }
        for (int elem = start; elem < end; elem++) {
          value += values[elem - start];
        }
      }
    }

    return value;
  }

  void add_gradient_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                           const IndexLayout<ncomp>& layout,
                           const Vector<T>& data_vec, const Vector<T>& vec,
                           Vector<T>& res) const {
    if constexpr (ncomp > 0) {
      const int num_elems = layout.get_num_elements();
      const int chunk = chunk_size(ncomp);
      inputs.resize(std::min(chunk, num_elems));
      for (int start = 0; start < num_elems; start += chunk) {
        const int end = std::min(start + chunk, num_elems);
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for schedule(dynamic, 32) if ((end - start) * ncomp >= min_parallel_work)
#endif
        for (int elem = start; elem < end; elem++) {
          Data data;
          Input input;
          Input& gradient = inputs[elem - start];
          data_layout.get_values(elem, data_vec, data);
          layout.get_values(elem, vec, input);
          gradient.zero();
          compute_gradient<T, Data, Input, Components...>(alpha, data, input,
                                                          gradient);
        }
        for (int elem = start; elem < end; elem++) {
          layout.add_values(elem, inputs[elem - start], res);
        }
      }
    }
  }

  void add_hessian_product_kernel(T alpha,
                                  const IndexLayout<ndata>& data_layout,
                                  const IndexLayout<ncomp>& layout,
                                  const Vector<T>& data_vec,
                                  const Vector<T>& vec, const Vector<T>& dir,
                                  Vector<T>& res) const {
    if constexpr (ncomp > 0) {
      const int num_elems = layout.get_num_elements();
      const int chunk = chunk_size(ncomp);
      inputs.resize(std::min(chunk, num_elems));
      for (int start = 0; start < num_elems; start += chunk) {
        const int end = std::min(start + chunk, num_elems);
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for schedule(dynamic, 32) if ((end - start) * ncomp >= min_parallel_work)
#endif
        for (int elem = start; elem < end; elem++) {
          Data data;
          Input input, gradient, direction;
          Input& result = inputs[elem - start];
          data_layout.get_values(elem, data_vec, data);
          layout.get_values(elem, vec, input);
          layout.get_values(elem, dir, direction);
          gradient.zero();
          result.zero();
          compute_hessian<T, Data, Input, Components...>(
              alpha, data, input, direction, gradient, result);
        }
        for (int elem = start; elem < end; elem++) {
          layout.add_values(elem, inputs[elem - start], res);
        }
      }
    }
  }

  void initialize_hessian_pattern(const IndexLayout<ncomp>& layout,
                                  const NodeOwners& owners, CSRMat<T>& mat) {
    find_positions(layout, owners, mat);
  }

  void add_hessian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                          const IndexLayout<ncomp>& layout,
                          const Vector<T>& data_vec, const Vector<T>& vec,
                          const NodeOwners& owners, CSRMat<T>& jac) const {
    if constexpr (ncomp > 0 && entries_per_element > 0) {
      int nnz;
      const int *rowp, *cols;
      T* matrix;
      jac.get_data(nullptr, nullptr, &nnz, &rowp, &cols, &matrix);

      // Positions found for another pattern are found again for this one
      if (cols != pattern_cols || nnz != pattern_nnz) {
        find_positions(layout, owners, jac);
      }

      const int num_elems = layout.get_num_elements();
      const int chunk = chunk_size(entries_per_element);
      values.resize(static_cast<std::size_t>(std::min(chunk, num_elems)) *
                    entries_per_element);
      for (int start = 0; start < num_elems; start += chunk) {
        const int end = std::min(start + chunk, num_elems);
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for schedule(dynamic, 32) if ((end - start) * entries_per_element >= min_parallel_work)
#endif
        for (int elem = start; elem < end; elem++) {
          Data data;
          Input input;
          data_layout.get_values(elem, data_vec, data);
          layout.get_values(elem, vec, input);
          compute_hessian_entries<T, Data, Input, Components...>(
              alpha, data, input,
              &values[static_cast<std::size_t>(elem - start) *
                      entries_per_element]);
        }
        for (int elem = start; elem < end; elem++) {
          const int* pos =
              &positions[static_cast<std::size_t>(elem) * entries_per_element];
          const T* value =
              &values[static_cast<std::size_t>(elem - start) * entries_per_element];
          for (int k = 0; k < entries_per_element; k++) {
            if (pos[k] >= 0) {
              matrix[pos[k]] += value[k];
            }
          }
        }
      }
    }
  }

  void add_grad_jac_product_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_tproduct_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_wrt_data_kernel(const IndexLayout<ndata>& data_layout,
                                    const IndexLayout<ncomp>& layout,
                                    const Vector<T>& data_vec,
                                    const Vector<T>& vec,
                                    const NodeOwners& owners,
                                    CSRMat<T>& jac) const {
    using ADtype = A2D::ADScalar<T, 1>;
    using DataAD =
        typename __get_collection_data_type<ADtype, Components...>::type;
    using InputAD =
        typename __get_collection_input_type<ADtype, Components...>::type;

    if constexpr (ncomp > 0 && ndata > 0) {
      // Element Jacobians are computed in parallel and added serially in element order
      constexpr int per_element = ncomp * ndata;
      const int num_elems = layout.get_num_elements();
      const int chunk = chunk_size(per_element);
      values.resize(static_cast<std::size_t>(std::min(chunk, num_elems)) *
                    per_element);
      for (int start = 0; start < num_elems; start += chunk) {
        const int end = std::min(start + chunk, num_elems);
#ifdef AMIGO_USE_OPENMP
#pragma omp parallel for schedule(dynamic, 32) if ((end - start) * per_element >= min_parallel_work)
#endif
        for (int elem = start; elem < end; elem++) {
          InputAD input;
          layout.get_values(elem, vec, input);

          T* jac_elem =
              &values[static_cast<std::size_t>(elem - start) * per_element];
          for (int i = 0; i < ndata; i++) {
            DataAD data;
            data_layout.get_values(elem, data_vec, data);
            data[i].deriv[0] = 1.0;

            InputAD gradient;
            gradient.zero();

            ADtype alpha(1.0);
            compute_gradient<ADtype, DataAD, InputAD, Components...>(
                alpha, data, input, gradient);

            for (int k = 0; k < ncomp; k++) {
              jac_elem[ndata * k + i] = gradient[k].deriv[0];
            }
          }
        }
        for (int elem = start; elem < end; elem++) {
          int input_index[ncomp];
          layout.get_indices(elem, input_index);

          int data_indices[ndata], data_indices_global[ndata];
          data_layout.get_indices(elem, data_indices);
          owners.local_to_global(ndata, data_indices, data_indices_global);

          const T* jac_elem =
              &values[static_cast<std::size_t>(elem - start) * per_element];
          for (int i = 0; i < ncomp; i++) {
            jac.add_row(input_index[i], ndata, data_indices_global,
                        &jac_elem[ndata * i]);
          }
        }
      }
    }
  }

 private:
  // Elements per buffered chunk for the given number of values per element
  static int chunk_size(int per_element) {
    return std::max(1, max_buffer_entries / std::max(1, per_element));
  }

  // Find the matrix position of every Hessian entry of every element
  void find_positions(const IndexLayout<ncomp>& layout, const NodeOwners& owners,
                      CSRMat<T>& mat) const {
    if constexpr (ncomp > 0 && entries_per_element > 0) {
      const int *rowp, *cols;
      T* data;
      mat.get_data(nullptr, nullptr, &pattern_nnz, &rowp, &cols, &data);
      pattern_cols = cols;

      const int num_elems = layout.get_num_elements();
      positions.resize(static_cast<std::size_t>(num_elems) * entries_per_element);
      for (int elem = 0; elem < num_elems; elem++) {
        int index[ncomp], index_global[ncomp];
        layout.get_indices(elem, index);
        owners.local_to_global(ncomp, index, index_global);
        find_hessian_positions<Components...>(
            rowp, cols, index, index_global,
            &positions[static_cast<std::size_t>(elem) * entries_per_element]);
      }
    }
  }

  // Matrix positions of the Hessian entries and the pattern they index
  mutable std::vector<int> positions;
  mutable int pattern_nnz = -1;
  mutable const int* pattern_cols = nullptr;

  // Buffers reused by the kernels
  mutable std::vector<T> values;
  mutable std::vector<Input> inputs;
};

// Set up a struct to select the backend type depending on policy
template <ExecPolicy policy, typename T, int ncomp, typename Input, int ndata,
          typename Data, typename... Components>
struct GroupBackendSelector;

// SERIAL specialization
template <typename T, int ncomp, typename Input, int ndata, typename Data,
          typename... Components>
struct GroupBackendSelector<ExecPolicy::SERIAL, T, ncomp, Input, ndata, Data,
                            Components...> {
  using type = SerialGroupBackend<T, ncomp, Input, ndata, Data, Components...>;
};

// OPENMP specialization
template <typename T, int ncomp, typename Input, int ndata, typename Data,
          typename... Components>
struct GroupBackendSelector<ExecPolicy::OPENMP, T, ncomp, Input, ndata, Data,
                            Components...> {
  using type = OmpGroupBackend<T, ncomp, Input, ndata, Data, Components...>;
};

#ifdef AMIGO_USE_CUDA
// CUDA specialization only when enabled
template <typename T, int ncomp, typename Input, int ndata, typename Data,
          typename... Components>
struct GroupBackendSelector<ExecPolicy::CUDA, T, ncomp, Input, ndata, Data,
                            Components...> {
  using type = CudaGroupBackend<T, ncomp, Input, ndata, Data, Components...>;
};
#endif

}  // namespace detail

template <typename T, ExecPolicy policy, class... Components>
class ComponentGroup : public ComponentGroupBase<T, policy> {
 public:
  static constexpr int num_components = sizeof...(Components);

  using Input = typename __get_collection_input_type<T, Components...>::type;
  using Data = typename __get_collection_data_type<T, Components...>::type;

  static constexpr int ncomp = __get_collection_ncomp<Components...>::value;
  static constexpr int ndata = __get_collection_ndata<Components...>::value;
  static constexpr int nconstraints =
      __get_collection_nconstraints<Components...>::value;
  static constexpr int noutputs =
      __get_collection_noutputs<Components...>::value;
  static constexpr bool continuation =
      __get_collection_continuation<Components...>::value;
  static constexpr bool linear = __get_collection_linear<Components...>::value;

  // Use whatever class is defined as the default backend
  using Backend =
      typename detail::GroupBackendSelector<policy, T, ncomp, Input, ndata,
                                            Data, Components...>::type;

  using OutputBackend =
      detail::DefaultOutputBackend<T, ncomp, ndata, noutputs, Components...>;

  ComponentGroup(int num_elements, std::shared_ptr<Vector<int>> data_indices,
                 std::shared_ptr<Vector<int>> indices,
                 std::shared_ptr<Vector<int>> output_indices)
      : data_layout(num_elements, data_indices),
        layout(num_elements, indices),
        output_layout(num_elements, output_indices),
        backend(data_layout, layout, output_layout) {
    // Union of the structural Hessian patterns of the components by local row
    std::vector<std::vector<int>> rows(ncomp);
    (add_component_pattern<Components>(rows), ...);

    pattern_rowp.assign(1, 0);
    for (auto& row : rows) {
      std::sort(row.begin(), row.end());
      row.erase(std::unique(row.begin(), row.end()), row.end());
      pattern_cols.insert(pattern_cols.end(), row.begin(), row.end());
      pattern_rowp.push_back(static_cast<int>(pattern_cols.size()));
    }
  }

  std::shared_ptr<ComponentGroupBase<T, policy>> clone(
      int num_elements, std::shared_ptr<Vector<int>> data_idx,
      std::shared_ptr<Vector<int>> layout_idx,
      std::shared_ptr<Vector<int>> output_idx) const {
    return std::make_shared<ComponentGroup<T, policy, Components...>>(
        num_elements, data_idx, layout_idx, output_idx);
  }

  bool is_continuation() const { return continuation; }
  bool is_linear() const { return linear; }
  int get_num_component_constraints() const { return nconstraints; }

  // Group compute functions
  T lagrangian(T alpha, const Vector<T>& data_vec, const Vector<T>& vec) const {
    return backend.lagrangian_kernel(alpha, data_layout, layout, data_vec, vec);
  }
  void add_gradient(T alpha, const Vector<T>& data_vec, const Vector<T>& vec,
                    Vector<T>& res) const {
    backend.add_gradient_kernel(alpha, data_layout, layout, data_vec, vec, res);
  }
  void add_hessian_product(T alpha, const Vector<T>& data_vec,
                           const Vector<T>& vec, const Vector<T>& dir,
                           Vector<T>& res) const {
    backend.add_hessian_product_kernel(alpha, data_layout, layout, data_vec,
                                       vec, dir, res);
  }
  void initialize_hessian_pattern(const NodeOwners& owners, CSRMat<T>& mat) {
    backend.initialize_hessian_pattern(layout, owners, mat);
  }
  void get_element_hessian_pattern(const int* rowp[],
                                   const int* cols[]) const {
    // The CUDA backend writes every element pair and keeps the full pattern
    if constexpr (policy == ExecPolicy::CUDA) {
      *rowp = nullptr;
      *cols = nullptr;
    } else {
      *rowp = pattern_rowp.data();
      *cols = pattern_cols.data();
    }
  }
  void add_hessian(T alpha, const Vector<T>& data_vec, const Vector<T>& vec,
                   const NodeOwners& owners, CSRMat<T>& jac) const {
    backend.add_hessian_kernel(alpha, data_layout, layout, data_vec, vec,
                               owners, jac);
  }

  // Gradient coupling function
  void add_grad_jac_product_wrt_data(const Vector<T>& data_vec,
                                     const Vector<T>& vec,
                                     const Vector<T>& pdata,
                                     Vector<T>& grad) const {
    backend.add_grad_jac_product_wrt_data_kernel(data_layout, layout, data_vec,
                                                 vec, pdata, grad);
  }
  void add_grad_jac_tproduct_wrt_data(const Vector<T>& data_vec,
                                      const Vector<T>& vec,
                                      const Vector<T>& pvec,
                                      Vector<T>& pdata) const {
    backend.add_grad_jac_tproduct_wrt_data_kernel(data_layout, layout, data_vec,
                                                  vec, pvec, pdata);
  }
  void add_grad_jac_wrt_data(const Vector<T>& data_vec, const Vector<T>& vec,
                             const NodeOwners& owners, CSRMat<T>& jac) const {
    backend.add_grad_jac_wrt_data_kernel(data_layout, layout, data_vec, vec,
                                         owners, jac);
  }

  // Compute the output as a function of the input and data
  void add_output(const Vector<T>& data_vec, const Vector<T>& vec,
                  Vector<T>& output) const {
    output_backend.add_output_kernel(data_layout, layout, output_layout,
                                     data_vec, vec, output);
  }
  void add_output_jac_wrt_input(const Vector<T>& data_vec, const Vector<T>& vec,
                                CSRMat<T>& jac) const {
    output_backend.add_output_jac_wrt_input_kernel(
        data_layout, layout, output_layout, data_vec, vec, jac);
  }
  void add_output_jac_wrt_data(const Vector<T>& data_vec, const Vector<T>& vec,
                               CSRMat<T>& jac) const {
    output_backend.add_output_jac_wrt_data_kernel(
        data_layout, layout, output_layout, data_vec, vec, jac);
  }

  // Get the ordering information
  void get_data_layout_data(int* num_elements, int* nodes_per_elem,
                            const int** array) const {
    data_layout.get_data(num_elements, nodes_per_elem, array);
  }
  void get_layout_data(int* num_elements, int* nodes_per_elem,
                       const int** array) const {
    layout.get_data(num_elements, nodes_per_elem, array);
  }
  void get_output_layout_data(int* num_elements, int* outputs_per_elem,
                              const int** array) const {
    output_layout.get_data(num_elements, outputs_per_elem, array);
  }

 private:
  template <class Component>
  static void add_component_pattern(std::vector<std::vector<int>>& rows) {
    if constexpr (!Component::is_compute_empty) {
      int nnz = Component::hessian_entry_ptr[Component::hessian_num_dirs];
      for (int e = 0; e < nnz; e++) {
        rows[Component::hessian_entry_row[e]].push_back(
            Component::hessian_entry_col[e]);
      }
    }
  }

  IndexLayout<ndata> data_layout;
  IndexLayout<ncomp> layout;
  IndexLayout<noutputs> output_layout;

  Backend backend;
  OutputBackend output_backend;

  // Element-local Hessian pattern shared by all elements of the group
  std::vector<int> pattern_rowp;
  std::vector<int> pattern_cols;
};

}  // namespace amigo

#endif  // AMIGO_COMPONENT_GROUP_H
