// Reference kernels in plain C++/CUDA, one output unit per call, fixed sequential order.
//
// Every function here is __host__ __device__: the CPU build (ref_cpu.cpp) and the CUDA
// build (ref_cuda.cu) run the same arithmetic, so the CPU build, which can be tested
// without a GPU, stands for the CUDA one up to the device's exp/sqrt.
//
// Inputs arrive as float (the Python side widens fp16 inputs exactly), the arithmetic is
// done in Acc (float or double) in index order, and the result is returned as Acc; the
// Python side rounds to the output buffer's dtype. Sums are sequential left to right,
// not the tree order of tl.sum or a tensor-core MMA: this is an independent reference, not
// a bit-for-bit model of the Triton kernels.
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define REF_HD __host__ __device__ inline
#else
#define REF_HD inline
#endif

namespace ref {

REF_HD float r_exp(float x) { return expf(x); }
REF_HD double r_exp(double x) { return exp(x); }
REF_HD float r_sqrt(float x) { return sqrtf(x); }
REF_HD double r_sqrt(double x) { return sqrt(x); }

template <class Acc>
REF_HD Acc vector_add(const float* x, const float* y, long i) {
  return Acc(x[i]) + Acc(y[i]);
}

// row_sum and row_sum_loop: out[row] = sum_j x[row * n_cols + j]
template <class Acc>
REF_HD Acc row_sum(const float* x, long n_cols, long row) {
  Acc s = 0;
  for (long j = 0; j < n_cols; ++j) s += Acc(x[row * n_cols + j]);
  return s;
}

// matmul and matmul_fp16: c[i, j] = sum_k a[i, k] * b[k, j], row-major
template <class Acc>
REF_HD Acc matmul(const float* a, const float* b, long N, long K, long i, long j) {
  Acc s = 0;
  for (long k = 0; k < K; ++k) s += Acc(a[i * K + k]) * Acc(b[k * N + j]);
  return s;
}

// softmax over one row; writes n_cols values to out
template <class Acc>
REF_HD void softmax_row(const float* x, long n_cols, long row, Acc* out) {
  const float* r = x + row * n_cols;
  Acc m = Acc(r[0]);
  for (long j = 1; j < n_cols; ++j) m = Acc(r[j]) > m ? Acc(r[j]) : m;
  Acc den = 0;
  for (long j = 0; j < n_cols; ++j) den += r_exp(Acc(r[j]) - m);
  for (long j = 0; j < n_cols; ++j) out[j] = r_exp(Acc(r[j]) - m) / den;
}

// layernorm over one row, no affine: (x - mean) / sqrt(var + eps), biased variance
template <class Acc>
REF_HD void layernorm_row(const float* x, long n_cols, Acc eps, long row, Acc* out) {
  const float* r = x + row * n_cols;
  Acc mean = 0;
  for (long j = 0; j < n_cols; ++j) mean += Acc(r[j]);
  mean /= Acc(n_cols);
  Acc var = 0;
  for (long j = 0; j < n_cols; ++j) var += (Acc(r[j]) - mean) * (Acc(r[j]) - mean);
  var /= Acc(n_cols);
  Acc rstd = Acc(1) / r_sqrt(var + eps);
  for (long j = 0; j < n_cols; ++j) out[j] = (Acc(r[j]) - mean) * rstd;
}

// single-head attention for query row i, no scaling: softmax(q_i K^T) V. scratch holds N.
template <class Acc>
REF_HD void attention_row(const float* q, const float* k, const float* v, long N, long D, long i,
                          Acc* scratch, Acc* out) {
  Acc m = 0;
  for (long n = 0; n < N; ++n) {
    Acc s = 0;
    for (long d = 0; d < D; ++d) s += Acc(q[i * D + d]) * Acc(k[n * D + d]);
    scratch[n] = s;
    m = (n == 0 || s > m) ? s : m;
  }
  Acc l = 0;
  for (long n = 0; n < N; ++n) {
    scratch[n] = r_exp(scratch[n] - m);
    l += scratch[n];
  }
  for (long d = 0; d < D; ++d) {
    Acc o = 0;
    for (long n = 0; n < N; ++n) o += scratch[n] * Acc(v[n * D + d]);
    out[d] = o / l;
  }
}

}  // namespace ref
