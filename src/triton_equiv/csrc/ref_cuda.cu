// CUDA build of the reference kernels: one thread per output unit (an element for add and
// matmul, a row for the row-wise ops and attention). Deliberately naive: no tiling, no
// shared memory, no tensor cores, so it shares no code path with the Triton kernels.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <string>
#include <vector>

#include "ref_kernels.h"

namespace {

constexpr int THREADS = 128;

template <class Acc>
__global__ void k_add(const float* a, const float* b, Acc* o, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) o[i] = ref::vector_add<Acc>(a, b, i);
}

template <class Acc>
__global__ void k_rowsum(const float* a, Acc* o, long R, long C) {
  long r = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (r < R) o[r] = ref::row_sum<Acc>(a, C, r);
}

template <class Acc>
__global__ void k_matmul(const float* a, const float* b, Acc* o, long M, long N, long K) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t < M * N) o[t] = ref::matmul<Acc>(a, b, N, K, t / N, t % N);
}

template <class Acc>
__global__ void k_rowwise(const float* a, Acc* o, long R, long C, Acc eps, bool layernorm) {
  long r = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (r >= R) return;
  if (layernorm) ref::layernorm_row<Acc>(a, C, eps, r, o + r * C);
  else ref::softmax_row<Acc>(a, C, r, o + r * C);
}

template <class Acc>
__global__ void k_attention(const float* q, const float* k, const float* v, Acc* scratch, Acc* o, long M, long N,
                            long D) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < M) ref::attention_row<Acc>(q, k, v, N, D, i, scratch + i * N, o + i * D);
}

long blocks(long n) { return (n + THREADS - 1) / THREADS; }

template <class Acc>
at::Tensor run_t(const std::string& op, const std::vector<at::Tensor>& t, const std::vector<int64_t>& n,
                 double eps) {
  for (const auto& x : t)
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.is_contiguous(),
                "ref_cuda: inputs must be contiguous float32 CUDA tensors");
  auto opt = t.at(0).options().dtype(sizeof(Acc) == 8 ? at::kDouble : at::kFloat);
  const float* a = t.at(0).data_ptr<float>();
  at::Tensor out;
  if (op == "add") {
    out = at::empty({n.at(0)}, opt);
    k_add<Acc><<<blocks(n[0]), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(a, t.at(1).data_ptr<float>(), out.data_ptr<Acc>(), n[0]);
  } else if (op == "rowsum") {
    out = at::empty({n.at(0)}, opt);
    k_rowsum<Acc><<<blocks(n[0]), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(a, out.data_ptr<Acc>(), n[0], n.at(1));
  } else if (op == "matmul") {
    long M = n.at(0), N = n.at(1), K = n.at(2);
    out = at::empty({M * N}, opt);
    k_matmul<Acc><<<blocks(M * N), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(a, t.at(1).data_ptr<float>(), out.data_ptr<Acc>(), M, N, K);
  } else if (op == "softmax" || op == "layernorm") {
    long R = n.at(0), C = n.at(1);
    out = at::empty({R * C}, opt);
    k_rowwise<Acc><<<blocks(R), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(a, out.data_ptr<Acc>(), R, C, Acc(eps), op == "layernorm");
  } else if (op == "attention") {
    long M = n.at(0), N = n.at(1), D = n.at(2);
    out = at::empty({M * D}, opt);
    auto scratch = at::empty({M * N}, opt);
    k_attention<Acc><<<blocks(M), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(a, t.at(1).data_ptr<float>(), t.at(2).data_ptr<float>(),
                                             scratch.data_ptr<Acc>(), out.data_ptr<Acc>(), M, N, D);
  } else {
    TORCH_CHECK(false, "ref_cuda: unknown op ", op);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

at::Tensor run(const std::string& op, const std::vector<at::Tensor>& t, const std::vector<int64_t>& n, double eps,
               bool use_double) {
  return use_double ? run_t<double>(op, t, n, eps) : run_t<float>(op, t, n, eps);
}
