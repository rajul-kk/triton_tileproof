// The schedule-program VM of msync/gpu.py in CUDA C++, written once over a hardware policy.
//
// One block plays one SM and walks its queue. Task tables are gpu.encode()'s rows. The
// per-program state machine is the same as the Triton kernel's (gpu.ProgState):
//
//   RUNNING --queue finished----------------------------> DONE
//   RUNNING --a wait not met before the deadline---------> TIMED_OUT
//   RUNNING --a read or write region outside the arena---> FAULTED
//
// Within a task: [D_PRE delay] waits (thread 0, ld.acquire polls against a %globaltimer
// deadline) -> barrier -> all reads into shared memory -> barrier -> [ticket slot] ->
// [D_MID delay] -> write -> barrier -> [D_POST delay] -> signal (thread 0, atom.release).
// signal_first moves the signal before the write, as in the Triton kernel.
//
// What Triton cannot express, added here: the no_barrier mutant (per task). Output
// elements are spread over the block's warps (owner()); with no_barrier, thread 0 signals
// as soon as warp 0 has written, without the block barrier, and the writers in the other
// warps first spend the D_POST delay: a slow warp still writing after the signal.
//
// Hw provides tid/nthreads/sync/syncwarp, load_acquire/add_release/fetch_add/exchange on
// int*, now_ns/sleep_ns/pause and an int broadcast from thread 0. DeviceHw is in
// msync_vm.cu; HostHw (one thread, no real concurrency) is in msync_vm_cpu.cpp and is used
// to test this file without a GPU.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define VM_HD __host__ __device__ inline
#else
#define VM_HD inline
#endif

namespace msync {

// Values shared with the Python side; tests compare layout() with gpu.KERNEL_LITERALS.
enum State : int { RUNNING = 0, DONE = 1, TIMED_OUT = 2, FAULTED = 3 };
enum Step : int { STEP_WAIT = 0, STEP_READ = 1, STEP_WRITE = 2 };
enum Op : int { OP_GEMV = 0, OP_RMS = 1, OP_SUM = 2, OP_SCALE = 3, OP_TICKET = 4, OP_RESET = 5 };
enum Field : int {
  F_OP = 0, F_OC, F_SIGFIRST, F_NW, F_W0C, F_W0T, F_W1C, F_W1T, F_W2C, F_W2T,
  F_R0B, F_R0L, F_R1B, F_R1L, F_WB, F_WL, F_K, F_CAP, F_AUX, NF
};
enum Delay : int { D_PRE = 0, D_MID = 1, D_POST = 2 };

struct Args {
  float* arena;
  int* cnt;
  const int* tasks;           // n_tasks * NF
  const int* qtask;
  const int* qstart;
  const int* qlen;
  const int64_t* delay_ns;  // n_tasks * 3
  const int* no_barrier;      // n_tasks, 0 or 1
  int* status;                // 3 per program: state, stop task, stop step
  int* qhead;                 // dynamic mode: next index into qtask
  int64_t* t_task;            // 2 per task: %globaltimer after the waits, after the signal
  int arena_size;
  int n_dyn;
  int64_t timeout_ns;         // per wait
  int dynamic;
};

// Thread that writes output element e: elements are dealt round robin over the warps so
// that a task with more than one element has writers outside warp 0.
VM_HD int owner(int e, int nthreads) {
  if (nthreads < 64) return e % nthreads;
  int warps = nthreads / 32;
  return (e % warps) * 32 + (e / warps) % 32;
}

template <class Hw>
VM_HD float element(int op, const float* r0, int r0l, const float* r1, int K, int e) {
  (void)r1;
  if (op == OP_GEMV) {
    float s = 0.f;
    for (int k = 0; k < K; ++k) s += r0[e * K + k] * r1[k];
    return s;
  }
  if (op == OP_RMS) {
    float ms = 0.f;
    for (int j = 0; j < r0l; ++j) ms += r0[j] * r0[j];
    ms /= float(r0l > 0 ? r0l : 1);
    return r0[e] / Hw::sqrt(ms + 1e-6f);
  }
  if (op == OP_SUM) {
    float s = 0.f;
    for (int j = 0; j < r0l; ++j) s += r0[j];
    return s;
  }
  if (op == OP_SCALE) return 2.f * r0[e];
  return r0[0];  // ticket
}

template <class Hw>
VM_HD void run_program(Hw& hw, const Args& a, int pid, float* smem) {
  int state = RUNNING, stop_task = -1, stop_step = 0;
  const int qs = a.qstart[pid], ql = a.qlen[pid];
  const int tid = hw.tid(), nt = hw.nthreads();
  for (int qi = 0; qi < ql && state == RUNNING; ++qi) {
    int ti;
    if (a.dynamic) {
      int j = hw.bcast(tid == 0 ? hw.fetch_add(a.qhead, 1) : 0);
      if (j >= a.n_dyn) break;
      ti = a.qtask[j];
    } else {
      ti = a.qtask[qs + qi];
    }
    const int* row = a.tasks + ti * NF;
    const int op = row[F_OP], oc = row[F_OC], sigfirst = row[F_SIGFIRST], nw = row[F_NW];
    const int r0b = row[F_R0B], r0l = row[F_R0L], r1b = row[F_R1B], r1l = row[F_R1L];
    const int wb = row[F_WB], wl = row[F_WL], K = row[F_K];
    const bool nobar = a.no_barrier[ti] != 0;
    hw.sleep_ns(a.delay_ns[ti * 3 + D_PRE]);

    // ---- acquire: waits (thread 0), then everyone learns the result
    int st = RUNNING;
    if (tid == 0) {
      for (int w = 0; w < nw && st == RUNNING; ++w) {
        int* c = a.cnt + row[F_W0C + 2 * w];
        const int th = row[F_W0T + 2 * w];
        const int64_t deadline = hw.now_ns() + a.timeout_ns;
        while (hw.load_acquire(c) < th) {
          if (hw.now_ns() > deadline) { st = TIMED_OUT; break; }
          hw.pause();
        }
      }
      if (st == RUNNING) a.t_task[2 * ti] = hw.now_ns();
    }
    st = hw.bcast(st);
    if (st != RUNNING) { state = st; stop_task = ti; stop_step = STEP_WAIT; break; }

    // ---- acquire: every read into shared memory before any write
    if (r0b < 0 || r0b + r0l > a.arena_size || r1b < 0 || r1b + r1l > a.arena_size) {
      state = FAULTED; stop_task = ti; stop_step = STEP_READ; break;
    }
    float* s0 = smem;
    float* s1 = smem + r0l;
    for (int j = tid; j < r0l; j += nt) s0[j] = a.arena[r0b + j];
    for (int j = tid; j < r1l; j += nt) s1[j] = a.arena[r1b + j];
    hw.sync();
    int slot = 0;
    if (op == OP_TICKET) slot = hw.bcast(tid == 0 ? hw.fetch_add(a.cnt + row[F_AUX], 1) : 0);
    hw.sleep_ns(a.delay_ns[ti * 3 + D_MID]);

    if (sigfirst) {
      hw.sync();
      if (tid == 0 && oc >= 0) hw.add_release(a.cnt + oc, 1);
      hw.sleep_ns(a.delay_ns[ti * 3 + D_POST]);
    }

    // ---- write
    if (wl > 0 && (wb < 0 || wb + wl > a.arena_size)) {
      state = FAULTED; stop_task = ti; stop_step = STEP_WRITE; break;
    }
    int n_out = 0, base = wb;
    if (op == OP_GEMV) n_out = r0l / (K > 0 ? K : 1);
    else if (op == OP_RMS || op == OP_SCALE) n_out = r0l;
    else if (op == OP_SUM) n_out = 1;
    else if (op == OP_TICKET) { n_out = slot < row[F_CAP] ? 1 : 0; base = wb + slot; }
    if (op != OP_TICKET && n_out > wl) n_out = wl;
    const bool slow_warp = nobar && !sigfirst && tid >= 32;
    for (int e = 0; e < n_out; ++e) {
      if (owner(e, nt) != tid) continue;
      if (slow_warp) hw.sleep_ns(a.delay_ns[ti * 3 + D_POST]);
      a.arena[base + e] = element<Hw>(op, s0, r0l, s1, K, e);
    }
    if (op == OP_RESET && tid == 0) hw.exchange(a.cnt + row[F_AUX], 0);

    // ---- signal
    if (!sigfirst) {
      if (nobar) {
        hw.syncwarp();  // warp 0's own writes only
      } else {
        hw.sleep_ns(a.delay_ns[ti * 3 + D_POST]);
        hw.sync();
      }
      if (tid == 0 && oc >= 0) hw.add_release(a.cnt + oc, 1);
    }
    if (tid == 0) a.t_task[2 * ti + 1] = hw.now_ns();
    hw.sync();  // the shared read buffers are reused by the next task
  }
  if (state == RUNNING) state = DONE;
  if (tid == 0) {
    a.status[3 * pid] = state;
    a.status[3 * pid + 1] = stop_task;
    a.status[3 * pid + 2] = stop_step;
  }
}

}  // namespace msync
