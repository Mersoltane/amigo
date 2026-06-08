#ifndef AMIGO_CUDA_VECTOR_BACKEND_H
#define AMIGO_CUDA_VECTOR_BACKEND_H

#include <cuda_runtime.h>

#include "amigo.h"
#include "cuda/csr_matrix_backend.cuh"

namespace amigo {

namespace detail {

template <typename T>
struct CublasVecOps;  // primary template left undefined on purpose

template <>
struct CublasVecOps<float> {
  static cublasStatus_t dot(cublasHandle_t h, int n, const float* x, int incx,
                            const float* y, int incy, float* result) {
    return cublasSdot(h, n, x, incx, y, incy, result);
  }

  static cublasStatus_t axpy(cublasHandle_t h, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy) {
    return cublasSaxpy(h, n, alpha, x, incx, y, incy);
  }

  static cublasStatus_t scal(cublasHandle_t h, int n, const float* alpha,
                             float* x, int incx) {
    return cublasSscal(h, n, alpha, x, incx);
  }
};

template <>
struct CublasVecOps<double> {
  static cublasStatus_t dot(cublasHandle_t h, int n, const double* x, int incx,
                            const double* y, int incy, double* result) {
    return cublasDdot(h, n, x, incx, y, incy, result);
  }

  static cublasStatus_t axpy(cublasHandle_t h, int n, const double* alpha,
                             const double* x, int incx, double* y, int incy) {
    return cublasDaxpy(h, n, alpha, x, incx, y, incy);
  }

  static cublasStatus_t scal(cublasHandle_t h, int n, const double* alpha,
                             double* x, int incx) {
    return cublasDscal(h, n, alpha, x, incx);
  }
};

}  // namespace detail

template <typename T>
class CudaVecBackend {
 public:
  CudaVecBackend() : size(0), d_ptr(nullptr), handle(nullptr) {
    AMIGO_CHECK_CUBLAS(cublasCreate(&handle));
  }
  ~CudaVecBackend() {
    if (d_ptr) {
      cudaFree(d_ptr);
    }
    if (handle) {
      cublasDestroy(handle);
    }
  }

  void allocate(int size_) {
    if (d_ptr) {
      cudaFree(d_ptr);
    }
    size = size_;
    AMIGO_CHECK_CUDA(cudaMalloc(&d_ptr, size * sizeof(T)));
  }

  void copy_host_to_device(const T* h_ptr) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copy_device_to_host(T* h_ptr) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void copy(const T* d_src) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_ptr, d_src, size * sizeof(T), cudaMemcpyDeviceToDevice));
  }

  void zero() { AMIGO_CHECK_CUDA(cudaMemset(d_ptr, 0, size * sizeof(T))); }

  void fill(T scalar) {
    // Need to implement this
    // for (i = 0; i < size; i++ ){ d_ptr[i] = scalar; }
  }

  void add_scalar(T scalar) {
    // Need to implement this
    // for (i = 0; i < size; i++ ){ d_ptr[i] += scalar; }
  }
  void scale(T alpha) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      AMIGO_CHECK_CUBLAS(
          detail::CublasVecOps<T>::scal(handle, size, &alpha, d_ptr, 1));
    }
  }

  void axpy(T alpha, const T* d_x) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      AMIGO_CHECK_CUBLAS(detail::CublasVecOps<T>::axpy(handle, size, &alpha,
                                                       d_x, 1, d_ptr, 1));
    }
  }

  T dot(const T* d_src) const {
    T result{};
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      AMIGO_CHECK_CUBLAS(detail::CublasVecOps<T>::dot(handle, size, d_ptr, 1,
                                                      d_src, 1, &result));
    }
    return result;  // host scalar
  }

  T maxabs(int& index) {
    // Need to implement this
    // T value = 0.0;
    // for (int i = 0; i < size; i++ ){ if (value > |d_ptr[i]|){ index = i;
    // value = |d_ptr[i]|;}
    return T(0);
  }

  T abssum() {
    // Need to implement this
    // T value = 0.0;
    // for (int i = 0; i < size; i++ ){ value += |d_ptr[i]|;}
    return T(0);
  }

  void copy_at(int n, const int d_idx[], const T d_src[]) {
    // Need to implement
    // for (int i = 0; i < n; i++ ){ d_ptr[d_idx[i]] = d_src[i]; }
  }
  void fill_at(int n, const int d_idx[], T value) {
    // Need to implement
    // for (int i = 0; i < n; i++ ){ d_ptr[d_idx[i]] = value; }
  }
  void add_scalar_at(int n, const int d_idx[], T scalar) {
    // Need to implement
    // for (int i = 0; i < n; i++ ){ d_ptr[d_idx[i]] += scalar; }
  }
  void scale_at(int n, const int d_idx[], T scalar) {
    // Need to implement
    // for (int i = 0; i < n; i++ ){ d_ptr[d_idx[i]] *= scalar; }
  }
  void axpy_at(int n, const int d_idx[], const T d_x[]) {
    // Need to implement
    // for (int i = 0; i < n; i++ ){ d_ptr[d_idx[i]] *= d_x[i]; }
  }
  void get_values_at(int n, const int d_idx[], T d_vals[]) {
    // Need to implement
    // for (int i = 0; i < n; i++ ){ d_vals[i] = d_ptr[d_idx[i]]; }
  }
  void set_values_at(int n, const int d_idx[], const T d_vals[]) {
    // Need to implement
    // for (int i = 0; i < n; i++ ){ d_ptr[d_idx[i]] = d_vals[i]; }
  }

  T* get_device_ptr() { return d_ptr; }
  const T* get_device_ptr() const { return d_ptr; }

 private:
  int size;
  T* d_ptr;
  cublasHandle_t handle;
};

}  // namespace amigo

#endif  // AMIGO_CUDA_VECTOR_BACKEND_H