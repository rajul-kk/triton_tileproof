"""Mutation operators on schedule programs: synchronisation, tiling and launch-state bugs.

Synchronisation:

``partial_wait``        a wait on a counter with several producers waits for one fewer
                        (first-k-of-N: the consumer may run before the last producer).
``threshold_plus_one``  a wait threshold one higher than the number of producers.
``drop_wait``           remove one wait.
``wrong_wait_counter``  wait on a different counter (the threshold is clamped to its
                        producer count, so the wait is satisfiable).
``signal_before_write`` increment the counter before the writes land (missing release
                        fence), on tasks whose counter is waited on.
``swap_same_sm``        swap two adjacent tasks of one SM's queue.

Tiling (the megakernel analogue of the Triton mask operators):

``tile_shift``          move one tile's range up by one (an overlap and a gap).
``tile_unclamped``      drop the clamp of a short last tile, so it runs a full tile past
                        the end of the buffer. Exists only at odd sizes.
``scratch_no_offset``   a worker writes its partial result to slot 0 instead of its own
                        slot (the missing per-block offset RESOLVE found).

Launch state:

``counter_not_zeroed``  a counter starts at the value the previous launch left in it
                        (the host forgot to zero it).

Grid barrier reuse (``barrier2`` only):

``barrier_reuse_reset``    both barriers use one counter; SM 0 resets it to 0 as soon as
                           it passes the first barrier (the barrier-reset race RESOLVE
                           found in one megakernel).
``barrier_reuse_no_reset`` both barriers use one counter with threshold 2 for both, so the
                           second barrier can be passed on the first barrier's arrivals.

Inside a task (:func:`warp_mutants`, kept apart from :func:`mutants` so the part-1 audit
is unchanged):

``no_barrier``          drop the block barrier before a task's signal, so warps other than
                        warp 0 may write after it (``ir.Task.no_barrier``). Sites: tasks
                        that signal a counter somebody waits on and write more than one
                        element.

Each operator yields one mutant per site.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .ir import Program, Task, Wait, retile

OPERATORS = ("partial_wait", "threshold_plus_one", "drop_wait", "wrong_wait_counter",
             "signal_before_write", "swap_same_sm", "tile_shift", "tile_unclamped",
             "scratch_no_offset", "counter_not_zeroed", "barrier_reuse_reset", "barrier_reuse_no_reset")
WARP_OPERATORS = ("no_barrier",)


@dataclass
class SyncMutant:
    id: str
    operator: str
    site: str
    program: Program


def _with_task(prog, i, t):
    tasks = list(prog.tasks)
    tasks[i] = t
    return prog.copy(tasks=tasks)


def _wait_sites(prog):
    for i, t in enumerate(prog.tasks):
        for j, w in enumerate(t.waits):
            yield i, t, j, w


def _set_wait(t, j, w):
    waits = list(t.waits)
    if w is None:
        del waits[j]
    else:
        waits[j] = w
    return replace(t, waits=tuple(waits))


def _sites(prog: Program):
    for i, t, j, w in _wait_sites(prog):
        n = len(prog.producers(w.counter))
        if w.threshold > 1:
            yield "partial_wait", f"{t.id}.{w.counter}", _with_task(prog, i, _set_wait(t, j, Wait(w.counter, w.threshold - 1)))
        yield "threshold_plus_one", f"{t.id}.{w.counter}", _with_task(prog, i, _set_wait(t, j, Wait(w.counter, n + 1)))
        yield "drop_wait", f"{t.id}.{w.counter}", _with_task(prog, i, _set_wait(t, j, None))
        others = sorted(c for c in prog.counters if c != w.counter and c != t.out_counter and prog.producers(c))
        if others:
            c = others[0]
            th = min(w.threshold, len(prog.producers(c)))
            yield "wrong_wait_counter", f"{t.id}.{w.counter}->{c}", _with_task(prog, i, _set_wait(t, j, Wait(c, th)))

    for i, t in enumerate(prog.tasks):
        if t.out_counter and any(w.counter == t.out_counter for u in prog.tasks for w in u.waits) \
                and t.op not in ("ticket", "reset"):
            yield "signal_before_write", t.id, _with_task(prog, i, replace(t, signal_first=True))

    for sm in range(prog.n_sms):
        q = [i for i, t in enumerate(prog.tasks) if t.sm == sm]
        for a, b in zip(q, q[1:]):
            tasks = list(prog.tasks)
            tasks[a], tasks[b] = tasks[b], tasks[a]
            yield "swap_same_sm", f"{prog.tasks[a].id}<->{prog.tasks[b].id}", prog.copy(tasks=tasks)

    for i, t in enumerate(prog.tasks):
        if t.op in ("gemv", "sum") and t.p("T") is not None:
            lo, hi = (t.p("n0"), t.p("n1")) if t.op == "gemv" else (t.p("lo"), t.p("hi"))
            yield "tile_shift", t.id, _with_task(prog, i, retile(t, lo + 1, hi + 1))
            if hi - lo < t.p("T"):
                yield "tile_unclamped", t.id, _with_task(prog, i, retile(t, lo, lo + t.p("T")))
        if t.op == "sum" and t.p("slot", 0) > 0:
            w = t.writes[0]
            yield "scratch_no_offset", t.id, _with_task(prog, i, replace(t, writes=(replace(w, lo=0, hi=1),)).with_params(slot=0))

    for c in sorted(prog.counters):
        n = len(prog.producers(c))
        if n:
            counters = dict(prog.counters)
            counters[c] = n
            yield "counter_not_zeroed", c, prog.copy(counters=counters)

    if prog.name == "barrier2":
        yield from _barrier_mutants(prog)


def _barrier_mutants(prog: Program):
    def merged(t: Task, threshold_second: int) -> Task:
        out = "bar" if t.out_counter in ("bar1", "bar2") else t.out_counter
        waits = tuple(Wait("bar", w.threshold if w.counter == "bar1" else threshold_second)
                      if w.counter in ("bar1", "bar2") else w for w in t.waits)
        return replace(t, out_counter=out, waits=waits)

    counters = {"bar": 0, "c_fin": 0}
    # Reuse with a reset: SM 0 zeroes the counter once it is through the first barrier.
    tasks = []
    for t in prog.tasks:
        if t.id == "b0":
            tasks.append(Task("reset", 0, "reset", waits=(Wait("bar", 2),), params=(("counter", "bar"),)))
            tasks.append(replace(merged(t, 2), waits=()))  # ordered after reset on the same SM
        else:
            tasks.append(merged(t, 2))
    yield "barrier_reuse_reset", "bar1+bar2", prog.copy(tasks=tasks, counters=counters)
    # Reuse without a reset and without generation counting: both barriers wait for 2.
    yield "barrier_reuse_no_reset", "bar1+bar2", prog.copy(tasks=[merged(t, 2) for t in prog.tasks],
                                                           counters=counters)


def mutants(prog: Program) -> list:
    out, seen = [], {}
    for op, site, p in _sites(prog):
        k = seen.get(op, 0)
        seen[op] = k + 1
        out.append(SyncMutant(f"{prog.name}:{op}:{k}", op, site, p.copy(name=prog.name)))
    return out


def no_barrier_sites(prog: Program) -> list:
    """Task indices where a missing block barrier can matter: the task signals a counter
    somebody waits on and writes more than one element (so some writer is outside warp 0)."""
    waited = {w.counter for t in prog.tasks for w in t.waits}
    return [i for i, t in enumerate(prog.tasks)
            if t.out_counter in waited and not t.signal_first and t.writes
            and t.writes[0].hi - t.writes[0].lo > 1]


def warp_mutants(prog: Program) -> list:
    """One ``no_barrier`` mutant per :func:`no_barrier_sites` site, ids
    ``<program>:no_barrier:<k>`` in site order (as the T4 run numbered them)."""
    return [SyncMutant(f"{prog.name}:no_barrier:{k}", "no_barrier", prog.tasks[i].id,
                       _with_task(prog, i, replace(prog.tasks[i], no_barrier=True)).copy(name=prog.name))
            for k, i in enumerate(no_barrier_sites(prog))]
