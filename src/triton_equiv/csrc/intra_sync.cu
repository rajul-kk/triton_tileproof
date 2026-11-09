// Single-kernel synchronisation mutants: block-level shared-memory kernels whose barriers
// can be switched off one at a time, with injected delays. See msync/intra.py.
//
// Every barrier is `if (mask >> k & 1) __syncthreads()` (or __syncwarp): the mask is a
// kernel argument, uniform across the block, so the conditional barrier is legal and one
// compiled kernel serves every mutant. Delays are per thread: d[p] nanoseconds at delay
// point p, spun on %globaltimer with __nanosleep. Lanes of one warp can get different
// delays, which makes them diverge under independent thread scheduling (sm_70+).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

#define NT 256
#define NP 8
#define BAR(k) do { if ((mask >> (k)) & 1u) __syncthreads(); } while (0)
#define WBAR(k) do { if ((mask >> (k)) & 1u) __syncwarp(); } while (0)

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void dly(const long long* d, int p) {
  long long ns = d[p];
  if (ns > 0) {
    unsigned long long t0 = gtime();
    while ((long long)(gtime() - t0) < ns) __nanosleep(64);
  }
}

// Sum of 256 values per block: a shared-memory tree down to 64, then a warp tail with
// __syncwarp between reads and writes (the CUDA programming guide's pattern).
// Bits: 0 before the load (redundant), 1 after the load, 2 after the 128 level,
// 3 after the 64 level, then per warp step off = 32, 16, 8, 4, 2, 1:
// 4 + 2i between its read and its write (WAR), 5 + 2i after its write (RAW).
__global__ void k_reduce(const float* __restrict__ x, float* out, unsigned mask, const long long* dl) {
  __shared__ float s[NT];
  const int t = threadIdx.x;
  const long long* d = dl + ((size_t)blockIdx.x * NT + t) * NP;
  BAR(0);
  dly(d, 0);
  s[t] = x[(size_t)blockIdx.x * NT + t];
  BAR(1);
  dly(d, 1);
  if (t < 128) s[t] += s[t + 128];
  BAR(2);
  dly(d, 2);
  if (t < 64) s[t] += s[t + 64];
  BAR(3);
  if (t < 32) {
    dly(d, 3);
    int bit = 4;
#pragma unroll
    for (int off = 32; off >= 1; off >>= 1, bit += 2) {
      float v = s[t] + s[t + off];
      dly(d, 4);
      WBAR(bit);
      s[t] = v;
      dly(d, 5);
      WBAR(bit + 1);
    }
    if (t == 0) out[blockIdx.x] = s[0];
  }
}

// C = A @ B with 16x16 shared tiles, one 16x16 output tile per block (256 threads).
// Bits: 0 after the tile loads (RAW), 1 after the tile products (WAR, before the next
// load), 2 after the loop (redundant: the store is from registers).
__global__ void k_matmul(const float* __restrict__ A, const float* __restrict__ B, float* C, unsigned mask,
                         const long long* dl, int N, int K) {
  __shared__ float As[16][16], Bs[16][16];
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
  const int tiles_n = N / 16, bx = blockIdx.x % tiles_n, by = blockIdx.x / tiles_n;
  const long long* d = dl + ((size_t)blockIdx.x * NT + t) * NP;
  const int row = by * 16 + ty, col = bx * 16 + tx;
  float acc = 0.f;
  for (int kt = 0; kt < K / 16; ++kt) {
    dly(d, 0);
    As[ty][tx] = A[(size_t)row * K + kt * 16 + tx];
    Bs[ty][tx] = B[(size_t)(kt * 16 + ty) * N + col];
    BAR(0);
    dly(d, 1);
#pragma unroll
    for (int k = 0; k < 16; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
    dly(d, 2);
    BAR(1);
  }
  BAR(2);
  C[(size_t)row * N + col] = acc;
}

// 3-point stencil with a halo: out[g] = x[g-1] + x[g] + x[g+1] (zero outside).
// Bits: 0 after the loads (RAW), 1 before the store (redundant).
__global__ void k_stencil(const float* __restrict__ x, float* out, unsigned mask, const long long* dl, int n) {
  __shared__ float s[NT + 2];
  const int t = threadIdx.x, g = blockIdx.x * NT + t;
  const long long* d = dl + ((size_t)blockIdx.x * NT + t) * NP;
  dly(d, 0);
  s[t + 1] = x[g];
  if (t == 0) s[0] = g > 0 ? x[g - 1] : 0.f;
  if (t == NT - 1) s[NT + 1] = g + 1 < n ? x[g + 1] : 0.f;
  BAR(0);
  dly(d, 1);
  float v = (s[t] + s[t + 1]) + s[t + 2];
  BAR(1);
  out[g] = v;
}

// Inclusive Hillis-Steele scan of 256 values per block, in place.
// Bits: 16 after the load (RAW); per step off = 1, 2, ..., 128 (i = 0..7):
// 2i between its reads and its write (WAR), 2i + 1 after its write (RAW; redundant for the
// last step, whose value each thread reads back itself).
__global__ void k_scan(const float* __restrict__ x, float* out, unsigned mask, const long long* dl) {
  __shared__ float s[NT];
  const int t = threadIdx.x, g = blockIdx.x * NT + t;
  const long long* d = dl + ((size_t)blockIdx.x * NT + t) * NP;
  dly(d, 0);
  s[t] = x[g];
  BAR(16);
  int i = 0;
#pragma unroll
  for (int off = 1; off < NT; off <<= 1, ++i) {
    dly(d, 1);
    float v = s[t] + (t >= off ? s[t - off] : 0.f);
    BAR(2 * i);
    dly(d, 2);
    s[t] = v;
    BAR(2 * i + 1);
  }
  out[g] = s[t];
}

void intra_run(int64_t kernel, at::Tensor x, at::Tensor y, at::Tensor out, int64_t mask, at::Tensor delays,
               int64_t nblocks, int64_t n, int64_t k) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned m = (unsigned)(mask & 0xffffffffLL);
  const long long* d = reinterpret_cast<const long long*>(delays.data_ptr<int64_t>());
  switch (kernel) {
    case 0: k_reduce<<<nblocks, NT, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), m, d); break;
    case 1: k_matmul<<<nblocks, NT, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), m, d,
                                                 (int)n, (int)k); break;
    case 2: k_stencil<<<nblocks, NT, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), m, d, (int)n); break;
    case 3: k_scan<<<nblocks, NT, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), m, d); break;
    default: TORCH_CHECK(false, "unknown kernel ", kernel);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
