// CPU build of the reference kernels: one loop iteration per output unit.
#include <torch/extension.h>

#include <string>
#include <vector>

#include "ref_kernels.h"

namespace {

template <class Acc>
at::Tensor run_t(const std::string& op, const std::vector<at::Tensor>& t, const std::vector<int64_t>& n,
                 double eps) {
  for (const auto& x : t)
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kFloat && x.is_contiguous(),
                "ref_cpu: inputs must be contiguous float32 CPU tensors");
  auto opt = t.at(0).options().dtype(sizeof(Acc) == 8 ? at::kDouble : at::kFloat);
  const float* a = t.at(0).data_ptr<float>();
  if (op == "add") {
    auto out = at::empty({n.at(0)}, opt);
    Acc* o = out.data_ptr<Acc>();
    const float* b = t.at(1).data_ptr<float>();
    for (long i = 0; i < n[0]; ++i) o[i] = ref::vector_add<Acc>(a, b, i);
    return out;
  }
  if (op == "rowsum") {
    auto out = at::empty({n.at(0)}, opt);
    Acc* o = out.data_ptr<Acc>();
    for (long r = 0; r < n[0]; ++r) o[r] = ref::row_sum<Acc>(a, n.at(1), r);
    return out;
  }
  if (op == "matmul") {
    long M = n.at(0), N = n.at(1), K = n.at(2);
    auto out = at::empty({M * N}, opt);
    Acc* o = out.data_ptr<Acc>();
    const float* b = t.at(1).data_ptr<float>();
    for (long i = 0; i < M; ++i)
      for (long j = 0; j < N; ++j) o[i * N + j] = ref::matmul<Acc>(a, b, N, K, i, j);
    return out;
  }
  if (op == "softmax" || op == "layernorm") {
    long R = n.at(0), C = n.at(1);
    auto out = at::empty({R * C}, opt);
    Acc* o = out.data_ptr<Acc>();
    for (long r = 0; r < R; ++r) {
      if (op == "softmax") ref::softmax_row<Acc>(a, C, r, o + r * C);
      else ref::layernorm_row<Acc>(a, C, Acc(eps), r, o + r * C);
    }
    return out;
  }
  if (op == "attention") {
    long M = n.at(0), N = n.at(1), D = n.at(2);
    auto out = at::empty({M * D}, opt);
    auto scratch = at::empty({N}, opt);
    Acc* o = out.data_ptr<Acc>();
    const float* k = t.at(1).data_ptr<float>();
    const float* v = t.at(2).data_ptr<float>();
    for (long i = 0; i < M; ++i) ref::attention_row<Acc>(a, k, v, N, D, i, scratch.data_ptr<Acc>(), o + i * D);
    return out;
  }
  TORCH_CHECK(false, "ref_cpu: unknown op ", op);
}

}  // namespace

at::Tensor run(const std::string& op, const std::vector<at::Tensor>& t, const std::vector<int64_t>& n, double eps,
               bool use_double) {
  return use_double ? run_t<double>(op, t, n, eps) : run_t<float>(op, t, n, eps);
}
