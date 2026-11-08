"""A small megakernel schedule IR and four reference programs.

The IR follows AutoMegaKernel's (RightNow-AI/AutoMegaKernel, ``docs/IR_SPEC.md`` and
``schedule/ir.py`` on ``main``, read 2026-11-07), cut down to what the synchronisation
model needs:

* A program is a list of **tasks**. Each task runs on one SM. An SM runs its tasks
  serially, in list order (AMK: "each SM runs a serial queue in task-list order").
* **Counters** are the only way tasks on different SMs talk. A task may ``wait`` until a
  counter reaches a static threshold, and increments exactly one ``out_counter`` by one
  after its writes (AMK puts a release fence before the increment).
* **Buffers** are flat arrays laid out back to back in one arena, as separate
  ``cudaMalloc`` allocations usually are. A read or write past the end of a buffer touches
  the next buffer, not an error. Only an access outside the whole arena (plus
  ``ARENA_PAD`` spare cells at the end) faults.

Two operations are not in AMK's IR and are here to model bugs RESOLVE (arXiv 2610.05683)
found in other megakernels: ``reset`` (a task sets a counter back to 0, as a reusable grid
barrier does) and ``ticket`` (a task takes a slot index from an atomic counter, as MoE
token dispatch does). ``Task.signal_first`` models a missing release fence: the counter is
incremented before the task's writes land.

``Task.no_barrier`` models a bug one level down, inside a task. A task's output elements
are written by the threads of one block, dealt over its :data:`WARPS` warps (element ``e``
by warp :func:`warp_of`, as ``owner()`` in ``csrc/msync_vm.h`` does). Normally a block
barrier makes every warp's writes land before thread 0 signals. Without it, thread 0
signals once warp 0 has written, and the other warps' writes may land later. The block
still waits for all of them before its next task.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

READONLY_KINDS = ("input", "weight")
# cudaMalloc rounds allocations up (to at least 256 bytes), so a few elements past the last
# buffer are still mapped memory, not a fault. The arena has this many spare cells at the
# end; only an access beyond them faults.
ARENA_PAD = 8
# Warps per block in the CUDA VM (cuda_vm.THREADS // 32). Only no_barrier tasks see them.
WARPS = 4


def warp_of(e: int) -> int:
    """The warp that writes output element ``e`` of a task."""
    return e % WARPS


@dataclass(frozen=True)
class Region:
    buf: str
    lo: int
    hi: int  # exclusive


@dataclass(frozen=True)
class Wait:
    counter: str
    threshold: int


@dataclass(frozen=True)
class Task:
    id: str
    sm: int
    op: str  # gemv | rms | sum | scale | ticket | reset
    reads: tuple = ()
    writes: tuple = ()
    out_counter: str | None = None
    waits: tuple = ()
    params: tuple = ()  # (key, value) pairs
    signal_first: bool = False
    no_barrier: bool = False

    def p(self, key, default=None):
        return dict(self.params).get(key, default)

    def with_params(self, **kw):
        d = dict(self.params)
        d.update(kw)
        return replace(self, params=tuple(sorted(d.items())))


@dataclass(frozen=True)
class Buffer:
    name: str
    size: int
    kind: str  # input | weight | act | scratch | output


@dataclass
class Program:
    name: str
    n_sms: int
    buffers: list
    counters: dict  # name -> initial value (the host zeroes counters before a launch)
    tasks: list
    size_kind: str = "aligned"
    notes: str = ""
    layout: dict = field(init=False)
    arena_size: int = field(init=False)

    def __post_init__(self):
        self.layout, off = {}, 0
        for b in self.buffers:
            self.layout[b.name] = off
            off += b.size
        self.data_size = off
        self.arena_size = off + ARENA_PAD

    def buf(self, name) -> Buffer:
        return next(b for b in self.buffers if b.name == name)

    def queue(self, sm) -> list:
        return [t for t in self.tasks if t.sm == sm]

    def producers(self, counter) -> list:
        return [t for t in self.tasks if t.out_counter == counter]

    def task(self, tid) -> Task:
        return next(t for t in self.tasks if t.id == tid)

    def copy(self, **kw) -> "Program":
        d = dict(name=self.name, n_sms=self.n_sms, buffers=list(self.buffers),
                 counters=dict(self.counters), tasks=list(self.tasks), size_kind=self.size_kind,
                 notes=self.notes)
        d.update(kw)
        return Program(**d)


# ---------------------------------------------------------------- task builders

def gemv(tid, sm, W, x, y, K, n0, n1, T, out_counter, waits=()):
    """``y[n0:n1] = W[n0:n1, :] @ x`` with ``W`` stored row-major, ``K`` columns."""
    return Task(tid, sm, "gemv", reads=(Region(W, n0 * K, n1 * K), Region(x, 0, K)),
                writes=(Region(y, n0, n1),), out_counter=out_counter,
                waits=tuple(Wait(*w) for w in waits),
                params=tuple(sorted(dict(W=W, x=x, y=y, K=K, n0=n0, n1=n1, T=T).items())))


def psum(tid, sm, x, lo, hi, T, scratch, slot, out_counter, waits=()):
    """``scratch[slot] = sum(x[lo:hi])``, summed left to right."""
    return Task(tid, sm, "sum", reads=(Region(x, lo, hi),), writes=(Region(scratch, slot, slot + 1),),
                out_counter=out_counter, waits=tuple(Wait(*w) for w in waits),
                params=tuple(sorted(dict(x=x, lo=lo, hi=hi, T=T, scratch=scratch, slot=slot).items())))


def retile(t: Task, lo: int, hi: int) -> Task:
    """The same gemv or sum task over a different row or element range."""
    p = dict(t.params)
    if t.op == "gemv":
        return replace(gemv(t.id, t.sm, p["W"], p["x"], p["y"], p["K"], lo, hi, p["T"], t.out_counter),
                       waits=t.waits, signal_first=t.signal_first, no_barrier=t.no_barrier)
    if t.op == "sum":
        return replace(psum(t.id, t.sm, p["x"], lo, hi, p["T"], p["scratch"], p["slot"], t.out_counter),
                       waits=t.waits, signal_first=t.signal_first, no_barrier=t.no_barrier, writes=t.writes)
    raise ValueError(t.op)


def _tiles(n, T):
    return [(n0, min(n0 + T, n)) for n0 in range(0, n, T)]


# ---------------------------------------------------------------- reference programs

def mlp_rms(N=4, T=2, K0=2) -> Program:
    """Two GEMV layers with an RMSNorm between them, the shape of one decoder MLP step.

    Layer-1 tiles are spread over two SMs and join on ``c_y`` (all-join: the norm needs all
    of ``y``). Layer-2 tiles wait for the norm on ``c_z``. Tiles are clamped to ``N``, so
    with ``N % T != 0`` the last tile is short (a tail).
    """
    tiles = _tiles(N, T)
    bufs = [Buffer("x", K0, "input"), Buffer("W1", N * K0, "weight"), Buffer("y", N, "act"),
            Buffer("z", N, "act"), Buffer("W2", N * N, "weight"), Buffer("out", N, "output")]
    tasks = [gemv(f"g{i}", i % 2, "W1", "x", "y", K0, a, b, T, "c_y") for i, (a, b) in enumerate(tiles)]
    tasks.append(Task("r", 0, "rms", reads=(Region("y", 0, N),), writes=(Region("z", 0, N),),
                      out_counter="c_z", waits=(Wait("c_y", len(tiles)),)))
    tasks += [gemv(f"h{i}", (i + 1) % 2, "W2", "z", "out", N, a, b, T, "c_out", waits=[("c_z", 1)])
              for i, (a, b) in enumerate(tiles)]
    kind = "aligned" if N % T == 0 else "odd"
    return Program(f"mlp_rms_{kind}", 2, bufs, {"c_y": 0, "c_z": 0, "c_out": 0}, tasks, kind)


def split_reduce(n=8, S=2) -> Program:
    """Split-K style reduction: ``S`` workers write partial sums to a scratch buffer, one
    task adds them. The scratch slot is the worker's index (the offset RESOLVE found
    missing in one megakernel's attention scratch buffer)."""
    T = -(-n // S)
    bufs = [Buffer("x", n, "input"), Buffer("scratch", S, "scratch"), Buffer("out", 1, "output")]
    tasks = [psum(f"p{i}", i % 2, "x", a, b, T, "scratch", i, "c_p") for i, (a, b) in enumerate(_tiles(n, T))]
    tasks.append(Task("red", 0, "sum", reads=(Region("scratch", 0, S),), writes=(Region("out", 0, 1),),
                      out_counter="c_out", waits=(Wait("c_p", S),)))
    kind = "aligned" if n % S == 0 else "odd"
    return Program(f"split_reduce_{kind}", 2, bufs, {"c_p": 0, "c_out": 0}, tasks, kind)


def barrier2(N=4, T=2, K0=2) -> Program:
    """Two layers separated by grid barriers, one counter per barrier (the AMK style).

    Each layer-2 tile reads all of layer 1's output, so it needs a barrier, not a
    per-tile dependency. A final norm on SM 0 waits for every layer-2 tile."""
    tiles = _tiles(N, T)
    bufs = [Buffer("x", K0, "input"), Buffer("W1", N * K0, "weight"), Buffer("y", N, "act"),
            Buffer("W2", N * N, "weight"), Buffer("o", N, "act"), Buffer("res", N, "output")]
    tasks = []
    for i, (a, b) in enumerate(tiles):
        tasks.append(gemv(f"a{i}", i % 2, "W1", "x", "y", K0, a, b, T, "bar1"))
    for i, (a, b) in enumerate(tiles):
        tasks.append(gemv(f"b{i}", i % 2, "W2", "y", "o", N, a, b, T, "bar2", waits=[("bar1", len(tiles))]))
    tasks.append(Task("fin", 0, "rms", reads=(Region("o", 0, N),), writes=(Region("res", 0, N),),
                      out_counter="c_fin", waits=(Wait("bar2", len(tiles)),)))
    return Program("barrier2", 2, bufs, {"bar1": 0, "bar2": 0, "c_fin": 0}, tasks, "aligned")


def moe_dispatch(n_tokens=3, cap=2) -> Program:
    """Capacity-limited token dispatch: each token task takes a slot from an atomic ticket
    counter and drops its token if the slot is past capacity. Which tokens survive depends
    on the order the tasks reach the atomic (the FlashMoE/FastMoE behaviour RESOLVE
    reports). This reference is nondeterministic by design."""
    bufs = [Buffer("x", n_tokens, "input"), Buffer("slots", cap, "act"), Buffer("out", cap, "output")]
    tasks = [Task(f"d{i}", i % 2, "ticket", reads=(Region("x", i, i + 1),), writes=(Region("slots", 0, cap),),
                  out_counter="c_d", params=(("cap", cap), ("tick", "tick"))) for i in range(n_tokens)]
    tasks.append(Task("e", 1, "scale", reads=(Region("slots", 0, cap),), writes=(Region("out", 0, cap),),
                      out_counter="c_e", waits=(Wait("c_d", n_tokens),)))
    return Program("moe_dispatch", 2, bufs, {"tick": 0, "c_d": 0, "c_e": 0}, tasks, "aligned")


def reference_programs() -> list:
    return [mlp_rms(4, 2), mlp_rms(3, 2), split_reduce(8), split_reduce(7), barrier2(), moe_dispatch()]
