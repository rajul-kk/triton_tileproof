// CUDA build of msync_vm.h: one block per program. Needs sm_70+ (ld.acquire, atom.release,
// __nanosleep, %globaltimer); the T4 is sm_75. Every program must be resident at once for
// cross-program waits to make progress, which holds for the handful of programs used here.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <vector>

#include "msync_vm.h"

namespace {

struct DeviceHw {
  int* sh;
  __device__ static float sqrt(float x) { return sqrtf(x); }
  __device__ int tid() const { return threadIdx.x; }
  __device__ int nthreads() const { return blockDim.x; }
  __device__ void sync() { __syncthreads(); }
  __device__ void syncwarp() { __syncwarp(); }
  __device__ int load_acquire(int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
  }
  __device__ void add_release(int* p, int v) {
    int old;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  }
  __device__ int fetch_add(int* p, int v) { return atomicAdd(p, v); }
  __device__ void exchange(int* p, int v) { atomicExch(p, v); }
  __device__ int64_t now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return (int64_t)t;
  }
  __device__ void sleep_ns(int64_t ns) {
    if (ns <= 0) return;
    const int64_t end = now_ns() + ns;
    while (now_ns() < end) __nanosleep(256);
  }
  __device__ void pause() { __nanosleep(64); }
  __device__ int bcast(int v) {
    if (threadIdx.x == 0) *sh = v;
    __syncthreads();
    int r = *sh;
    __syncthreads();
    return r;
  }
};

__global__ void vm_kernel(msync::Args a) {
  extern __shared__ float smem[];
  __shared__ int sh;
  DeviceHw hw{&sh};
  msync::run_program(hw, a, blockIdx.x, smem);
}

}  // namespace

void vm_run(at::Tensor arena, at::Tensor cnt, at::Tensor tasks, at::Tensor qtask, at::Tensor qstart, at::Tensor qlen,
            at::Tensor delay_ns, at::Tensor no_barrier, at::Tensor status, at::Tensor qhead, at::Tensor t_task,
            int64_t arena_size, int64_t n_dyn, int64_t timeout_ns, bool dynamic, int64_t n_programs, int64_t threads,
            int64_t smem_floats) {
  msync::Args a{arena.data_ptr<float>(), cnt.data_ptr<int>(), tasks.data_ptr<int>(), qtask.data_ptr<int>(),
                qstart.data_ptr<int>(), qlen.data_ptr<int>(), delay_ns.data_ptr<int64_t>(),
                no_barrier.data_ptr<int>(), status.data_ptr<int>(), qhead.data_ptr<int>(),
                t_task.data_ptr<int64_t>(), (int)arena_size, (int)n_dyn, timeout_ns, dynamic ? 1 : 0};
  size_t smem = sizeof(float) * (smem_floats > 0 ? smem_floats : 1);
  vm_kernel<<<(int)n_programs, (int)threads, smem, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
