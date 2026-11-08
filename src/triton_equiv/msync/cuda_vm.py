"""The schedule-program VM in CUDA C++ (``csrc/msync_vm.h``), a second backend to ``gpu``.

Same task tables (``gpu.encode``), queues (``gpu.queues``), modes and outcome enums
(``gpu.ProgState``, ``gpu.StopStep``, ``gpu.RunOutcome``) as the Triton kernel, so the two
backends can be swapped in ``races`` and ``latency``. Differences, all deliberate:

* delays are nanoseconds of ``__nanosleep`` against ``%globaltimer`` (Triton: spin counts);
* a wait gives up after ``timeout_ns`` of wall time (Triton: a number of polls);
* waits are ``ld.acquire.gpu`` and signals ``atom.add.release.gpu``, written in PTX;
* every task's start (after its waits) and end (after its signal) is time-stamped, so
  per-task latency can be studied without launch overhead (:attr:`Run.task_ns`);
* ``no_barrier`` tasks (``ir.Task.no_barrier``) signal without the block barrier while
  their writers in other warps are delayed by the task's D_POST delay (see the header).
  The step machine models this with one step per warp (``machine``), so these mutants
  have oracle labels; ``mutate.warp_mutants`` plants them.

``Backend.CPU`` builds the same header for one host thread, to test it without a GPU
(``serial`` and ``per_task`` modes only, as with the Triton interpreter).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import cuda_ref
from ..cuda_ref import Backend
from . import gpu
from .ir import Program

THREADS = 128  # 4 warps, so a task's output elements have writers in 3 warps besides warp 0

_DECL = ("void vm_run(at::Tensor arena, at::Tensor cnt, at::Tensor tasks, at::Tensor qtask, at::Tensor qstart, "
         "at::Tensor qlen, at::Tensor delay_ns, at::Tensor no_barrier, at::Tensor status, at::Tensor qhead, "
         "at::Tensor t_task, int64_t arena_size, int64_t n_dyn, int64_t timeout_ns, bool dynamic, "
         "int64_t n_programs, int64_t threads, int64_t smem_floats);")

# order of csrc/msync_vm_cpu.cpp layout()
LAYOUT_NAMES = ("RUNNING", "DONE", "TIMED_OUT", "FAULTED", "WAIT", "READ", "WRITE", "gemv", "rms", "sum", "scale",
                "ticket", "reset", "F_OP", "F_OC", "F_SIGFIRST", "F_NW", "F_W0C", "F_W0T", "F_R0B", "F_R0L",
                "F_R1B", "F_R1L", "F_WB", "F_WL", "F_K", "F_CAP", "F_AUX", "NF", "D_PRE", "D_MID", "D_POST")


@functools.lru_cache(maxsize=None)
def module(backend: Backend):
    from torch.utils.cpp_extension import load_inline
    csrc = cuda_ref.CSRC
    common = dict(extra_include_paths=[str(csrc)], extra_cflags=["-O2"], verbose=False)
    cuda_ref.ensure_ninja_on_path()
    if backend is Backend.CPU:
        return load_inline("tileproof_msync_vm_cpu", cpp_sources=[(csrc / "msync_vm_cpu.cpp").read_text()],
                           functions=["vm_run", "layout"], **common)
    return load_inline("tileproof_msync_vm_cuda", cpp_sources=[_DECL], cuda_sources=[(csrc / "msync_vm.cu").read_text()],
                       functions=["vm_run"], extra_cuda_cflags=["-O2", "-arch=sm_75"], **common)


def build_error(backend: Backend) -> str | None:
    try:
        module(backend)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:300]}"


def layout() -> dict:
    """The C++ enum values (CPU build), keyed like ``gpu.KERNEL_LITERALS``'s entries."""
    return dict(zip(LAYOUT_NAMES, module(Backend.CPU).layout()))


def no_barrier_sites(prog: Program) -> list:
    """See ``mutate.no_barrier_sites``."""
    from .mutate import no_barrier_sites as sites
    return sites(prog)


def no_barrier_tasks(prog: Program) -> list:
    """Indices of the tasks with ``no_barrier`` set."""
    return [i for i, t in enumerate(prog.tasks) if t.no_barrier]


@dataclass
class Run:
    launch: gpu.Launch
    task_ns: list   # per task (start, end) from %globaltimer, or None if it never started


class CudaRunner:
    """Like ``gpu.Runner``: buffers for one program, mode and initial arena; every launch
    starts from the same arena and counters. ``no_barrier``: task indices to plant the
    warp-level mutant in; None (the default) takes them from the program's tasks."""

    def __init__(self, enc: gpu.Encoded, arena: list, mode: str = "parallel", n_programs: int | None = None,
                 backend: Backend = Backend.CUDA, threads: int = THREADS, no_barrier=None):
        import torch
        if mode not in gpu.Runner.MODES:
            raise ValueError(mode)
        self.torch, self.enc, self.mode, self.backend = torch, enc, mode, backend
        dev = "cuda" if backend is Backend.CUDA else "cpu"
        self.dev = dev
        i32 = dict(dtype=torch.int32, device=dev)
        n = enc.n_tasks
        self.tasks = torch.tensor(enc.rows, **i32).reshape(-1)
        self.arena0 = torch.tensor(arena, dtype=torch.float32, device=dev)
        self.cnt0 = torch.tensor(enc.counter_init or [0], **i32)
        self.arena, self.cnt = self.arena0.clone(), self.cnt0.clone()
        self.qhead = torch.zeros(1, **i32)
        self.no_barrier = torch.zeros(max(1, n), **i32)
        for i in (no_barrier_tasks(enc.prog) if no_barrier is None else no_barrier):
            self.no_barrier[i] = 1
        self.t_task = torch.zeros(max(1, n) * 2, dtype=torch.int64, device=dev)
        self.zero_delays = torch.zeros(max(1, n) * 3, dtype=torch.int64, device=dev)
        self.threads = threads
        self.smem = max([r[gpu.F_R0L] + r[gpu.F_R1L] for r in enc.rows] or [1])
        launches = ([([t], [0], [1], 1) for t in gpu.topo_order(enc.prog)] if mode == "per_task"
                    else [gpu.queues(enc.prog, mode, n_programs)])
        self.args = [(torch.tensor(q or [0], **i32), torch.tensor(s, **i32), torch.tensor(ln, **i32), k, len(q))
                     for q, s, ln, k in launches]
        self.status = torch.full((max(a[3] for a in self.args) * 3,), -1, **i32)

    def _reset(self):
        self.arena.copy_(self.arena0)
        self.cnt.copy_(self.cnt0)
        self.qhead.zero_()
        self.status.fill_(-1)
        self.t_task.zero_()

    def _delays(self, delays_ns):
        if delays_ns is None:
            return self.zero_delays
        d = self.torch.as_tensor(delays_ns, dtype=self.torch.int64).reshape(-1)
        if d.numel() != self.enc.n_tasks * 3:
            raise ValueError(f"delays: need {self.enc.n_tasks} x 3 values, got {d.numel()}")
        return d.to(self.dev)

    def _launch(self, a, dl, timeout_ns):
        qtask, qstart, qlen, n, n_dyn = a
        module(self.backend).vm_run(self.arena, self.cnt, self.tasks, qtask, qstart, qlen, dl, self.no_barrier,
                                    self.status, self.qhead, self.t_task, self.enc.prog.arena_size, n_dyn,
                                    int(timeout_ns), self.mode == "dynamic", n, self.threads, self.smem)

    def run(self, delays_ns=None, timeout_ns: int = 20_000_000) -> Run:
        dl = self._delays(delays_ns)
        self._reset()
        states, stops = [], []
        for a in self.args:
            if self.mode == "per_task":
                self.status.fill_(-1)
            self._launch(a, dl, timeout_ns)
            st = self.status.cpu().tolist()
            n = a[3]
            states_a = [gpu.Runner._state(st[3 * p]) for p in range(n)]
            stops_a = [gpu.Runner._stop(states_a[p], st[3 * p + 1], st[3 * p + 2]) for p in range(n)]
            if self.mode == "per_task":
                states += states_a
                stops += stops_a
                if states_a[0] is not gpu.ProgState.DONE:
                    break
            else:
                states, stops = states_a, stops_a
        t = self.t_task.cpu().tolist()
        task_ns = [(t[2 * i], t[2 * i + 1]) if t[2 * i] else None for i in range(self.enc.n_tasks)]
        return Run(gpu.Launch(gpu.outcome_of(states), states, stops, self.output()), task_ns)

    def output(self) -> list:
        out = self.arena.cpu().tolist()
        cells = []
        for b in self.enc.prog.buffers:
            if b.kind == "output":
                base = self.enc.prog.layout[b.name]
                cells += out[base:base + b.size]
        return cells

    def time_one(self, delays_ns=None, timeout_ns: int = 20_000_000) -> float:
        """One launch (``per_task``: one sequence) in microseconds, CUDA events around the
        launches only, as ``gpu.Runner.time_one``."""
        torch = self.torch
        if self.backend is not Backend.CUDA:
            raise RuntimeError("time_one() needs the CUDA backend")
        if getattr(self, "_events", None) is None:
            self._events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        start, end = self._events
        dl = self._delays(delays_ns)
        self._reset()
        start.record()
        for a in self.args:
            self._launch(a, dl, timeout_ns)
        end.record()
        end.synchronize()
        st = self.status.cpu().tolist()
        if any(v != gpu.ProgState.DONE for v in st[0:3 * self.args[-1][3]:3]):
            raise RuntimeError(f"{self.enc.prog.name} {self.mode}: a launch did not finish: {st}")
        return start.elapsed_time(end) * 1000.0

    def time(self, n: int, delays_ns=None, timeout_ns: int = 20_000_000, warmup: int = 50) -> list:
        """``n`` back-to-back :meth:`time_one` samples after ``warmup`` untimed ones."""
        out = [self.time_one(delays_ns, timeout_ns) for _ in range(warmup + n)]
        return out[warmup:]
