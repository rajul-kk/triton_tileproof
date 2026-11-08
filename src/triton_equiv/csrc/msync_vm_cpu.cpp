// CPU build of msync_vm.h: one thread per program, programs run one after another. Only
// meaningful for one program (serial mode) or one task per launch (per_task mode); with
// several programs a cross-program wait times out, as in the Triton interpreter.
#include <torch/extension.h>

#include <chrono>
#include <cmath>
#include <vector>

#include "msync_vm.h"

namespace {

struct HostHw {
  static float sqrt(float x) { return std::sqrt(x); }
  int tid() const { return 0; }
  int nthreads() const { return 1; }
  void sync() {}
  void syncwarp() {}
  int load_acquire(int* p) { return *p; }
  void add_release(int* p, int v) { *p += v; }
  int fetch_add(int* p, int v) { int o = *p; *p += v; return o; }
  void exchange(int* p, int v) { *p = v; }
  int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void sleep_ns(int64_t) {}  // no other thread to give way to
  void pause() {}
  int bcast(int v) { return v; }
};

}  // namespace

void vm_run(at::Tensor arena, at::Tensor cnt, at::Tensor tasks, at::Tensor qtask, at::Tensor qstart, at::Tensor qlen,
            at::Tensor delay_ns, at::Tensor no_barrier, at::Tensor status, at::Tensor qhead, at::Tensor t_task,
            int64_t arena_size, int64_t n_dyn, int64_t timeout_ns, bool dynamic, int64_t n_programs, int64_t threads,
            int64_t smem_floats) {
  (void)threads;
  msync::Args a{arena.data_ptr<float>(), cnt.data_ptr<int>(), tasks.data_ptr<int>(), qtask.data_ptr<int>(),
                qstart.data_ptr<int>(), qlen.data_ptr<int>(), delay_ns.data_ptr<int64_t>(),
                no_barrier.data_ptr<int>(), status.data_ptr<int>(), qhead.data_ptr<int>(),
                t_task.data_ptr<int64_t>(), (int)arena_size, (int)n_dyn, timeout_ns, dynamic ? 1 : 0};
  std::vector<float> smem(smem_floats > 0 ? smem_floats : 1);
  HostHw hw;
  for (int p = 0; p < n_programs; ++p) msync::run_program(hw, a, p, smem.data());
}

std::vector<int64_t> layout() {
  using namespace msync;
  return {RUNNING, DONE, TIMED_OUT, FAULTED, STEP_WAIT, STEP_READ, STEP_WRITE, OP_GEMV, OP_RMS, OP_SUM, OP_SCALE,
          OP_TICKET, OP_RESET, F_OP, F_OC, F_SIGFIRST, F_NW, F_W0C, F_W0T, F_R0B, F_R0L, F_R1B, F_R1L, F_WB, F_WL,
          F_K, F_CAP, F_AUX, NF, D_PRE, D_MID, D_POST};
}
