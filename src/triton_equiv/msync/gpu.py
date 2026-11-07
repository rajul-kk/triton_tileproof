"""Run schedule programs on a GPU as one persistent Triton kernel, with injected delays.

One launched program plays one SM. Each walks its queue (task-list order, as in
``machine``), spin-waits on counters in global memory, reads its regions, writes and
signals, exactly like ``machine.Machine`` but on hardware: the interleaving is whatever the
GPU produces, optionally perturbed by delays (RESOLVE's idea, arXiv 2610.05683).

Per-program state machine (the kernel's ``state`` variable, decoded by :class:`ProgState`)::

    RUNNING --all queued tasks finished------------------------> DONE
    RUNNING --a wait not reached within max_spin polls---------> TIMED_OUT   (deadlock)
    RUNNING --a read or write region outside the arena---------> FAULTED
    DONE, TIMED_OUT and FAULTED are final; a program never leaves them.

Within one task the steps are the model's three (``machine`` docstring): acquire (waits,
then all reads into registers), write, signal; ``signal_first`` swaps the last two. Delays
are injected at three points per task, by spinning on a volatile load:

    D_PRE   before the waits          (the task starts late)
    D_MID   after the reads           (a slow producer: reads done, writes late)
    D_POST  between write and signal  (with signal_first: between signal and write)

A spin-wait gives up after ``max_spin`` polls instead of hanging the GPU, so a deadlock
shows up as TIMED_OUT. After a timeout or a fault the program stops; the others run on
until they finish or time out themselves.

Ordering: each wait is an acquire atomic followed by a block barrier, and each signal is a
block barrier followed by a release atomic, so a correct program is race-free under the
PTX memory model, the guarantee AMK's runtime relies on. ``signal_first`` signals before
the writes, which is exactly the missing fence the mutant models.

Modes (:func:`queues`):

``parallel``  one program per SM of the schedule (the real megakernel).
``serial``    one program runs every task in list order. Used to test the kernel against
              ``machine`` in the Triton CPU interpreter, which runs programs one after the
              other and so cannot run spin-waits between programs.
``dynamic``   ``n`` programs take the next task from a shared atomic head over a
              topological order (work stealing, as dynamic megakernel schedulers do).

The device code needs ``triton``; :func:`encode` and :func:`queues` are plain Python.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from .ir import Program

# ---------------------------------------------------------------- encoding

OPS = {"gemv": 0, "rms": 1, "sum": 2, "scale": 3, "ticket": 4, "reset": 5}
MAX_WAITS = 3
(F_OP, F_OC, F_SIGFIRST, F_NW, F_W0C, F_W0T, F_W1C, F_W1T, F_W2C, F_W2T,
 F_R0B, F_R0L, F_R1B, F_R1L, F_WB, F_WL, F_K, F_CAP, F_AUX) = range(19)
NF = 19
D_PRE, D_MID, D_POST = 0, 1, 2


class ProgState(enum.IntEnum):
    RUNNING = 0  # never observed after a launch returns: the kernel always ends in a final state
    DONE = 1
    TIMED_OUT = 2
    FAULTED = 3


class StopStep(enum.IntEnum):
    """Where a program that is not DONE stopped, within the task it stopped on."""
    WAIT = 0    # a wait was not met within max_spin polls
    READ = 1    # a read region lies outside the arena
    WRITE = 2   # the write region lies outside the arena


# The jit code cannot use these enums, so it writes their values as literals; KERNEL_LITERALS
# lists what it assumes and tests/test_msync_gpu.py checks it against the enums and F_*.
KERNEL_LITERALS = {
    "states": {"RUNNING": 0, "DONE": 1, "TIMED_OUT": 2, "FAULTED": 3},
    "steps": {"WAIT": 0, "READ": 1, "WRITE": 2},
    "fields": {"F_OP": 0, "F_OC": 1, "F_SIGFIRST": 2, "F_NW": 3, "F_W0C": 4, "F_W0T": 5,
               "F_R0B": 10, "F_R0L": 11, "F_R1B": 12, "F_R1L": 13, "F_WB": 14, "F_WL": 15,
               "F_K": 16, "F_CAP": 17, "F_AUX": 18},
    "ops": {"gemv": 0, "rms": 1, "sum": 2, "scale": 3, "ticket": 4, "reset": 5},
    "delays": {"D_PRE": 0, "D_MID": 1, "D_POST": 2},
}


class RunOutcome(enum.Enum):
    """One launch, from its programs' final states (first match wins)."""
    FAULTED = "faulted"      # some program faulted
    TIMED_OUT = "timed_out"  # no fault, some program timed out (a deadlock, or a too-small max_spin)
    COMPLETED = "completed"  # every program DONE


def outcome_of(states) -> RunOutcome:
    states = [ProgState(int(s)) for s in states]
    if ProgState.RUNNING in states:
        raise RuntimeError(f"kernel returned with a program still RUNNING: {states}")
    if ProgState.FAULTED in states:
        return RunOutcome.FAULTED
    if ProgState.TIMED_OUT in states:
        return RunOutcome.TIMED_OUT
    return RunOutcome.COMPLETED


@dataclass
class Encoded:
    prog: Program
    rows: list            # NF ints per task, task-list order
    counters: list        # counter names, index = counter id
    counter_init: list
    blocks: tuple         # (BR, BK, BV) tile sizes, powers of two

    @property
    def n_tasks(self):
        return len(self.rows)


def _p2(n):
    return 1 << max(0, math.ceil(math.log2(max(1, n))))


def encode(prog: Program) -> Encoded:
    cidx = {c: i for i, c in enumerate(sorted(prog.counters))}
    rows, br, bk, bv = [], 1, 1, 1

    def region(r):
        return prog.layout[r.buf] + r.lo, r.hi - r.lo

    for t in prog.tasks:
        if len(t.waits) > MAX_WAITS:
            raise ValueError(f"{t.id}: more than {MAX_WAITS} waits")
        row = [0] * NF
        row[F_OP] = OPS[t.op]
        row[F_OC] = cidx[t.out_counter] if t.out_counter is not None else -1
        row[F_SIGFIRST] = int(t.signal_first)
        row[F_NW] = len(t.waits)
        for j, w in enumerate(t.waits):
            row[F_W0C + 2 * j], row[F_W0T + 2 * j] = cidx[w.counter], w.threshold
        for j, r in enumerate(t.reads[:2]):
            row[F_R0B + 2 * j], row[F_R0L + 2 * j] = region(r)
        if t.writes:
            row[F_WB], row[F_WL] = region(t.writes[0])
        if t.op == "gemv":
            K = t.p("K")
            row[F_K] = K
            br, bk = max(br, row[F_R0L] // K), max(bk, K)
        if t.op == "ticket":
            row[F_CAP], row[F_AUX] = t.p("cap"), cidx[t.p("tick")]
        if t.op == "reset":
            row[F_AUX] = cidx[t.p("counter")]
        bv = max(bv, row[F_R0L], row[F_WL])
        rows.append(row)
    names = sorted(prog.counters)
    return Encoded(prog, rows, names, [prog.counters[c] for c in names], (_p2(br), _p2(bk), _p2(bv)))


def topo_order(prog: Program) -> list:
    """Task indices in an order where every wait can be met by earlier tasks (greedy
    sequential run of the counters, ignoring SMs). Raises if there is none."""
    cval, done, order = dict(prog.counters), [False] * len(prog.tasks), []
    progress = True
    while progress:
        progress = False
        for i, t in enumerate(prog.tasks):
            if not done[i] and all(cval[w.counter] >= w.threshold for w in t.waits):
                done[i] = True
                order.append(i)
                if t.out_counter is not None:
                    cval[t.out_counter] += 1
                progress = True
    if len(order) != len(prog.tasks):
        raise ValueError(f"{prog.name}: no sequential order meets every wait")
    return order


def queues(prog: Program, mode: str = "parallel", n_programs: int | None = None):
    """``(qtask, qstart, qlen, n_programs)`` for a mode (module docstring)."""
    if mode == "parallel":
        qtask, qstart, qlen = [], [], []
        for s in range(prog.n_sms):
            q = [i for i, t in enumerate(prog.tasks) if t.sm == s]
            qstart.append(len(qtask))
            qlen.append(len(q))
            qtask += q
        return qtask, qstart, qlen, prog.n_sms
    if mode == "serial":
        return list(range(len(prog.tasks))), [0], [len(prog.tasks)], 1
    if mode == "dynamic":
        n = n_programs or prog.n_sms
        order = topo_order(prog)
        return order, [0] * n, [len(order)] * n, n
    raise ValueError(mode)


# ---------------------------------------------------------------- device code

try:
    import triton
    import triton.language as tl
except ImportError:  # host-side helpers still work
    triton = None

if triton is not None:
    @triton.jit
    def _spin(dummy_ptr, d):
        acc = tl.load(dummy_ptr) * 0.0
        for _ in range(0, d):
            acc += tl.load(dummy_ptr, volatile=True)
        return acc

    @triton.jit
    def sched_kernel(arena_ptr, cnt_ptr, tasks_ptr, qtask_ptr, qstart_ptr, qlen_ptr, delay_ptr,
                     status_ptr, dummy_ptr, qhead_ptr, arena_size, n_dyn, max_spin,
                     NF: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                     DYNAMIC: tl.constexpr):
        pid = tl.program_id(0)
        qs = tl.load(qstart_ptr + pid)
        ql = tl.load(qlen_ptr + pid)
        state = pid * 0            # ProgState.RUNNING
        stop_task = pid * 0 - 1
        stop_step = pid * 0
        sink = tl.load(dummy_ptr) * 0.0
        rr = tl.arange(0, BR)
        kk = tl.arange(0, BK)
        vv = tl.arange(0, BV)
        for qi in range(0, ql):
            if state == 0:
                if DYNAMIC:
                    j = tl.atomic_add(qhead_ptr, 1)
                    ti = tl.load(qtask_ptr + tl.minimum(j, n_dyn - 1))
                    live = j < n_dyn
                else:
                    ti = tl.load(qtask_ptr + qs + qi)
                    live = qi >= 0
                if live:
                    row = tasks_ptr + ti * NF
                    op = tl.load(row + 0)
                    oc = tl.load(row + 1)
                    sigfirst = tl.load(row + 2)
                    nw = tl.load(row + 3)
                    sink += _spin(dummy_ptr, tl.load(delay_ptr + ti * 3 + 0))
                    # ---- acquire: waits
                    for w in range(0, nw):
                        if state == 0:
                            c = tl.load(row + 4 + 2 * w)
                            th = tl.load(row + 5 + 2 * w)
                            it = pid * 0
                            v = tl.atomic_add(cnt_ptr + c, 0, sem="acquire")
                            while (v < th) & (it < max_spin):
                                v = tl.atomic_add(cnt_ptr + c, 0, sem="acquire")
                                it += 1
                            if v < th:
                                state = pid * 0 + 2   # TIMED_OUT
                                stop_task = ti
                                stop_step = pid * 0
                    tl.debug_barrier()
                    r0b = tl.load(row + 10)
                    r0l = tl.load(row + 11)
                    r1b = tl.load(row + 12)
                    r1l = tl.load(row + 13)
                    wb = tl.load(row + 14)
                    wl = tl.load(row + 15)
                    K = tl.load(row + 16)
                    if state == 0:
                        bad = (r0b < 0) | (r0b + r0l > arena_size) | (r1b < 0) | (r1b + r1l > arena_size)
                        if bad:
                            state = pid * 0 + 3   # FAULTED
                            stop_task = ti
                            stop_step = pid * 0 + 1
                    if state == 0:
                        # ---- acquire: reads (all of them, before any write)
                        # masks are bounded by lengths that are 0 for the other ops (the interpreter
                        # rejects scalar booleans in a mask)
                        rows_n = tl.where(op == 0, r0l // tl.maximum(K, 1), 0)
                        kn = tl.where(op == 0, K, 0)
                        xn = tl.where(op == 0, r1l, 0)
                        vn = tl.where(op == 0, 0, r0l)
                        wt = tl.load(arena_ptr + r0b + rr[:, None] * K + kk[None, :],
                                     mask=(rr[:, None] < rows_n) & (kk[None, :] < kn), other=0.0)
                        xt = tl.load(arena_ptr + r1b + kk, mask=kk < xn, other=0.0)
                        vt = tl.load(arena_ptr + r0b + vv, mask=vv < vn, other=0.0)
                        slot = pid * 0
                        if op == 4:
                            slot = tl.atomic_add(cnt_ptr + tl.load(row + 18), 1)
                        sink += _spin(dummy_ptr, tl.load(delay_ptr + ti * 3 + 1))
                        # ---- compute
                        gem = tl.sum(wt * xt[None, :], axis=1)                       # [BR]
                        ms = tl.sum(vt * vt, axis=0) / tl.maximum(r0l, 1).to(tl.float32)
                        rms = vt / tl.sqrt(ms + 1e-6)
                        tot = tl.sum(vt, axis=0)
                        if sigfirst != 0:
                            tl.debug_barrier()
                            if oc >= 0:
                                tl.atomic_add(cnt_ptr + oc, 1, sem="release")
                            sink += _spin(dummy_ptr, tl.load(delay_ptr + ti * 3 + 2))
                        # ---- write
                        if (wl > 0) & ((wb < 0) | (wb + wl > arena_size)):
                            state = pid * 0 + 3   # FAULTED
                            stop_task = ti
                            stop_step = pid * 0 + 2
                        if state == 0:
                            if op == 0:
                                tl.store(arena_ptr + wb + rr, gem, mask=rr < tl.minimum(rows_n, wl))
                            if op == 1:
                                tl.store(arena_ptr + wb + vv, rms, mask=vv < tl.minimum(r0l, wl))
                            if op == 2:
                                tl.store(arena_ptr + wb + vv, tot + vt * 0.0, mask=vv < tl.minimum(wl, 1))
                            if op == 3:
                                tl.store(arena_ptr + wb + vv, 2.0 * vt, mask=vv < tl.minimum(r0l, wl))
                            if op == 4:
                                cap = tl.load(row + 17)
                                tl.store(arena_ptr + wb + slot + vv, vt, mask=vv < tl.where(slot < cap, 1, 0))
                            if op == 5:
                                tl.atomic_xchg(cnt_ptr + tl.load(row + 18), 0)
                            # ---- signal
                            if sigfirst == 0:
                                sink += _spin(dummy_ptr, tl.load(delay_ptr + ti * 3 + 2))
                                tl.debug_barrier()
                                if oc >= 0:
                                    tl.atomic_add(cnt_ptr + oc, 1, sem="release")
        if state == 0:
            state = pid * 0 + 1   # DONE
        tl.store(status_ptr + pid * 3 + 0, state)
        tl.store(status_ptr + pid * 3 + 1, stop_task)
        tl.store(status_ptr + pid * 3 + 2, stop_step)
        tl.store(dummy_ptr + 1 + pid, sink)


def kernel():
    if triton is None:
        raise RuntimeError("msync.gpu needs triton")
    return sched_kernel


# ---------------------------------------------------------------- host side

@dataclass
class Launch:
    outcome: RunOutcome
    states: list            # ProgState per program
    stops: list             # (task index, StopStep) per program; None for a DONE program
    output: list            # output buffer cells (floats)


class Runner:
    """Device buffers for one program, one mode and one initial arena.

    Every :meth:`run` and :meth:`time` starts from the same initial arena and counters
    (copied in before the launch), so launches are independent and no state carries from
    one to the next. ``"per_task"`` is not a :func:`queues` mode: it launches the kernel
    once per task, in :func:`topo_order`, each launch one program running one task (the
    one-kernel-per-op baseline; kernel boundaries order the tasks, the counters are still
    updated but every wait already holds)."""

    MODES = ("parallel", "serial", "dynamic", "per_task")

    def __init__(self, enc: Encoded, arena: list, mode: str = "parallel", n_programs: int | None = None,
                 device: str = "cuda"):
        import torch
        if mode not in self.MODES:
            raise ValueError(mode)
        self.torch, self.enc, self.mode, self.device = torch, enc, mode, device
        prog = enc.prog
        i32 = dict(dtype=torch.int32, device=device)
        self.i32 = i32
        self.tasks = torch.tensor(enc.rows, **i32).reshape(-1)
        self.arena0 = torch.tensor(arena, dtype=torch.float32, device=device)
        self.cnt0 = torch.tensor(enc.counter_init or [0], **i32)
        self.arena = self.arena0.clone()
        self.cnt = self.cnt0.clone()
        self.qhead = torch.zeros(1, **i32)
        self.zero_delays = torch.zeros(max(1, enc.n_tasks) * 3, **i32)
        if mode == "per_task":
            self.launches = [([t], [0], [1], 1) for t in topo_order(prog)]
        else:
            self.launches = [queues(prog, mode, n_programs)]
        self.args = []
        for qtask, qstart, qlen, n in self.launches:
            self.args.append((torch.tensor(qtask or [0], **i32), torch.tensor(qstart, **i32),
                              torch.tensor(qlen, **i32), n, len(qtask)))
        n_max = max(a[3] for a in self.args)
        self.status = torch.full((n_max * 3,), -1, **i32)
        self.dummy = torch.ones(1 + n_max, dtype=torch.float32, device=device)

    def _reset(self):
        self.arena.copy_(self.arena0)
        self.cnt.copy_(self.cnt0)
        self.qhead.zero_()
        self.status.fill_(-1)

    def _delays(self, delays):
        if delays is None:
            return self.zero_delays
        d = self.torch.as_tensor(delays, dtype=self.torch.int32).reshape(-1)
        if d.numel() != self.enc.n_tasks * 3:
            raise ValueError(f"delays: need {self.enc.n_tasks} x 3 values, got {d.numel()}")
        return d.to(self.device)

    def _launch(self, a, dl, max_spin):
        qtask, qstart, qlen, n, n_dyn = a
        BR, BK, BV = self.enc.blocks
        sched_kernel[(n,)](self.arena, self.cnt, self.tasks, qtask, qstart, qlen, dl, self.status,
                           self.dummy, self.qhead, self.enc.prog.arena_size, n_dyn, max_spin,
                           NF=NF, BR=BR, BK=BK, BV=BV, DYNAMIC=(self.mode == "dynamic"), num_warps=1)

    def _launch_all(self, dl, max_spin):
        for a in self.args:
            self._launch(a, dl, max_spin)

    def run(self, delays=None, max_spin: int = 100_000) -> Launch:
        """One launch (``per_task``: one sequence of launches). ``delays``: ``n_tasks x 3``
        spin counts indexed ``[task, D_PRE|D_MID|D_POST]``, or None for none."""
        kernel()
        dl = self._delays(delays)
        self._reset()
        if self.mode == "per_task":
            states, stops = [], []
            for a in self.args:
                self.status.fill_(-1)
                self._launch(a, dl, max_spin)
                st = self.status.cpu().tolist()
                states.append(self._state(st[0]))
                stops.append(self._stop(states[-1], st[1], st[2]))
                if states[-1] is not ProgState.DONE:
                    break
        else:
            self._launch_all(dl, max_spin)
            st = self.status.cpu().tolist()
            n = self.args[0][3]
            states = [self._state(st[3 * p]) for p in range(n)]
            stops = [self._stop(states[p], st[3 * p + 1], st[3 * p + 2]) for p in range(n)]
        return Launch(outcome_of(states), states, stops, self.output())

    @staticmethod
    def _stop(state, task, step):
        return None if state is ProgState.DONE else (task, StopStep(step))

    @staticmethod
    def _state(v):
        try:
            return ProgState(v)
        except ValueError:
            raise RuntimeError(f"program status {v}: the kernel did not record a final state") from None

    def output(self) -> list:
        out = self.arena.cpu().tolist()
        cells = []
        for b in self.enc.prog.buffers:
            if b.kind == "output":
                base = self.enc.prog.layout[b.name]
                cells += out[base:base + b.size]
        return cells

    def time(self, n: int, delays=None, max_spin: int = 100_000, warmup: int = 50) -> list:
        """``n`` launch times in microseconds, CUDA events around the launch(es) only (the
        arena copy-in happens before the start event). Raises if a launch does not finish."""
        kernel()
        torch = self.torch
        dl = self._delays(delays)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        n_last = self.args[-1][3]  # per_task: the status buffer holds the last launch only
        out = []
        for i in range(warmup + n):
            self._reset()
            start.record()
            self._launch_all(dl, max_spin)
            end.record()
            end.synchronize()
            st = self.status.cpu().tolist()  # after the end event: not in the timed window
            if any(v != ProgState.DONE for v in st[0:3 * n_last:3]):
                raise RuntimeError(f"{self.enc.prog.name} {self.mode}: launch {i} did not finish: {st}")
            if i >= warmup:
                out.append(start.elapsed_time(end) * 1000.0)
        return out


def launch(enc: Encoded, arena: list, mode: str = "parallel", delays=None, max_spin: int = 100_000,
           n_programs: int | None = None, device: str = "cuda") -> Launch:
    """One launch from a fresh :class:`Runner`."""
    return Runner(enc, arena, mode, n_programs, device).run(delays, max_spin)
