"""Step semantics for the schedule IR, in two value domains.

Each task runs in three steps on its SM:

1. **acquire**: enabled once every wait holds (``counter >= threshold``). The task reads
   all of its input regions at this step (a ``ticket`` task also takes its slot here, with
   an atomic fetch-and-add).
2. **write**: the task's outputs land in the arena (a ``reset`` task zeroes its counter).
3. **signal**: ``out_counter += 1``.

With ``signal_first`` the last two swap, which is what a missing release fence allows.

A ``no_barrier`` task splits its write over warps (``ir.warp_of``). The SM's main line is
acquire, write warp 0's elements, signal. Each other warp that owns an element writes it in
a step of its own, enabled any time after the acquire, before or after the signal. The task
ends (the SM moves to its next task) once the signal and every warp's write are done: the
block barrier that closes a task in the CUDA VM. An SM in this state is in phase 3,
"signalled, waiting for its slow warps".

A step is an action ``(sm, lane)``: lane 0 is the SM's main line, lanes ``1..WARPS-1`` the
pending warp writes of a ``no_barrier`` task. Traces record lane 0 as the bare SM number.
Steps of different SMs, and of an SM's lanes, interleave arbitrarily, and each step is atomic. That is sequential
consistency at step granularity: finer reorderings allowed by the PTX memory model (for
example a read of half a tile before and half after another SM's write) are not modelled.

Value domains:

* :class:`Sym` builds hash-consed terms with every operation uninterpreted
  (``dot(row, x)``, ``rms(i, v)``, ``sum(v)`` ...). Two runs whose outputs are the same
  terms compute the same values for **every** input, so equality of terms is a proof of
  equivalence. Different terms are only a candidate difference, confirmed numerically.
* :class:`Num` computes float64 values. Uninitialised memory can be NaN (poison), zero, or
  the arena left by a previous correct launch ("warm"), which is what a benchmark that
  launches the same kernel repeatedly sees.
"""

from __future__ import annotations

import math
import random

from .ir import WARPS, Program, warp_of


class Fault(Exception):
    """An access outside the whole arena."""


class Sym:
    def __init__(self):
        self.table = {}

    def mk(self, term):
        return self.table.setdefault(term, len(self.table))

    def input(self, buf, i, value=None):
        return self.mk(("in", buf, i))

    def uninit(self, cell):
        return self.mk(("uninit", cell))

    def dot(self, row, x):
        return self.mk(("dot", tuple(row), tuple(x)))

    def rms(self, i, v):
        return self.mk(("rms", i, tuple(v)))

    def sum(self, v):
        return self.mk(("sum", tuple(v)))

    def scale(self, a):
        return self.mk(("scale", a))


class Num:
    def input(self, buf, i, value=None):
        return float(value)

    def dot(self, row, x):
        return float(sum(a * b for a, b in zip(row, x)))

    def rms(self, i, v):
        return v[i] / math.sqrt(sum(a * a for a in v) / len(v) + 1e-6)

    def sum(self, v):
        s = 0.0
        for a in v:
            s += a
        return s

    def scale(self, a):
        return 2.0 * a


def compute(task, vals, dom):
    """Values to write, one list per write region (``ticket`` is handled by the machine)."""
    if task.op == "gemv":
        K = task.p("K")
        W, x = vals
        rows = len(W) // K
        return [[dom.dot(W[r * K:(r + 1) * K], x) for r in range(rows)]]
    if task.op == "rms":
        (v,) = vals
        return [[dom.rms(i, v) for i in range(len(v))]]
    if task.op == "sum":
        return [[dom.sum(vals[0])]]
    if task.op == "scale":
        return [[dom.scale(a) for a in vals[0]]]
    if task.op == "reset":
        return []
    raise ValueError(task.op)


class Machine:
    """One execution state. ``copy`` is cheap enough for exhaustive exploration of the
    small programs here (tens of tasks, arenas of tens of cells)."""

    def __init__(self, prog: Program, dom, arena, counters=None):
        self.prog, self.dom = prog, dom
        self.queues = [prog.queue(s) for s in range(prog.n_sms)]
        self.cnames = sorted(prog.counters)
        self.arena = list(arena)
        self.counters = dict(prog.counters) if counters is None else dict(counters)
        self.pc = [0] * prog.n_sms
        self.phase = [0] * prog.n_sms
        self.local = [None] * prog.n_sms
        self.lanes = [0] * prog.n_sms  # bit k: warp k of a no_barrier task still has to write

    def copy(self):
        m = Machine.__new__(Machine)
        m.prog, m.dom, m.queues, m.cnames = self.prog, self.dom, self.queues, self.cnames
        m.arena, m.counters = list(self.arena), dict(self.counters)
        m.pc, m.phase, m.local = list(self.pc), list(self.phase), list(self.local)
        m.lanes = list(self.lanes)
        return m

    def key(self):
        return (tuple(self.arena), tuple(self.counters[c] for c in self.cnames), tuple(self.pc),
                tuple(self.phase), tuple(self.local), tuple(self.lanes))

    def current(self, sm):
        q = self.queues[sm]
        return q[self.pc[sm]] if self.pc[sm] < len(q) else None

    def done(self):
        return all(self.current(s) is None for s in range(self.prog.n_sms))

    def _main_enabled(self, sm):
        t = self.current(sm)
        if t is None or self.phase[sm] == 3:
            return False
        if self.phase[sm] > 0:
            return True
        return all(self.counters[w.counter] >= w.threshold for w in t.waits)

    def actions(self, sm=None) -> list:
        """Enabled ``(sm, lane)`` pairs, of one SM or of all."""
        out = []
        for s in (range(self.prog.n_sms) if sm is None else (sm,)):
            if self._main_enabled(s):
                out.append((s, 0))
            out += [(s, k) for k in range(1, WARPS) if self.lanes[s] >> k & 1]
        return out

    def enabled(self, sm):
        return bool(self.actions(sm))

    @staticmethod
    def split(t) -> bool:
        """Whether ``t``'s write is split over warps."""
        return t.no_barrier and not t.signal_first and t.op in ("gemv", "rms", "scale")

    def _warp_mask(self, t, vals):
        mask = 0
        for region, out in zip(t.writes, compute(t, [list(v) for v in vals], self.dom)):
            for e in range(min(len(out), region.hi - region.lo)):
                mask |= 1 << warp_of(e)
        return mask & ~1

    def _cells(self, region):
        base = self.prog.layout[region.buf]
        cells = range(base + region.lo, base + region.hi)
        if region.lo < 0 and base + region.lo < 0 or base + region.hi > self.prog.arena_size:
            raise Fault(region)
        return cells

    def step(self, sm, lane=0):
        t, ph = self.current(sm), self.phase[sm]
        if lane:
            if not (self.lanes[sm] >> lane & 1):
                raise ValueError(f"SM {sm} lane {lane} has nothing to write")
            self._write(t, *self.local[sm], warp=lane)
            self.lanes[sm] &= ~(1 << lane)
            if ph == 3 and not self.lanes[sm]:
                self._finish(sm)
            return
        if ph == 0:
            vals = tuple(tuple(self.arena[c] for c in self._cells(r)) for r in t.reads)
            ticket = None
            if t.op == "ticket":
                ticket = self.counters[t.p("tick")]
                self.counters[t.p("tick")] += 1
            self.local[sm] = (vals, ticket)
            self.phase[sm] = 1
            if self.split(t):
                self.lanes[sm] = self._warp_mask(t, vals)
            return
        action = ("signal", "write") if t.signal_first else ("write", "signal")
        what = action[ph - 1]
        if what == "write":
            self._write(t, *self.local[sm], warp=0 if self.split(t) else None)
        elif t.out_counter is not None:
            self.counters[t.out_counter] += 1
        if ph == 2:
            if self.lanes[sm]:
                self.phase[sm] = 3
            else:
                self._finish(sm)
        else:
            self.phase[sm] = 2

    def _finish(self, sm):
        self.pc[sm] += 1
        self.phase[sm] = 0
        self.local[sm] = None

    def _write(self, t, vals, ticket, warp=None):
        """Land ``t``'s outputs; with ``warp`` set, only that warp's elements."""
        if t.op == "reset":
            self.counters[t.p("counter")] = 0
            return
        if t.op == "ticket":
            if ticket < t.p("cap"):
                cell = self._cells(t.writes[0])[ticket]
                self.arena[cell] = vals[0][0]
            return
        for region, out in zip(t.writes, compute(t, [list(v) for v in vals], self.dom)):
            cells = self._cells(region)
            for e, (c, v) in enumerate(zip(cells, out)):
                if warp is None or warp_of(e) == warp:
                    self.arena[c] = v

    def output(self):
        cells = []
        for b in self.prog.buffers:
            if b.kind == "output":
                base = self.prog.layout[b.name]
                cells += self.arena[base:base + b.size]
        return tuple(cells)


# ------------------------------------------------------------------ arenas

def sym_arena(prog: Program, sym: Sym):
    arena = []
    for b in prog.buffers:
        base = prog.layout[b.name]
        for i in range(b.size):
            arena.append(sym.input(b.name, i) if b.kind in ("input", "weight") else sym.uninit(base + i))
    return arena + [sym.uninit(c) for c in range(prog.data_size, prog.arena_size)]


def num_inputs(prog: Program, seed: int) -> dict:
    rng = random.Random(seed)
    return {b.name: [rng.gauss(0.0, 1.0) for _ in range(b.size)]
            for b in prog.buffers if b.kind in ("input", "weight")}


def num_arena(prog: Program, inputs: dict, fill="nan", warm=None):
    arena = []
    for b in prog.buffers:
        base = prog.layout[b.name]
        for i in range(b.size):
            if b.kind in ("input", "weight"):
                arena.append(float(inputs[b.name][i]))
            elif fill == "warm":
                arena.append(warm[base + i])
            elif fill == "zero":
                arena.append(0.0)
            else:
                arena.append(math.nan)
    pad = range(prog.data_size, prog.arena_size)
    if fill == "warm":
        return arena + [warm[c] for c in pad]
    return arena + [0.0 if fill == "zero" else math.nan for _ in pad]


def run(prog: Program, arena, dom=None, policy="rr", seed=0, max_steps=10_000):
    """Run one schedule to completion. ``policy`` is ``"rr"`` (round robin over the SMs
    that can step, the closest thing to "all SMs advance together") or ``"random"``.

    Round robin lets a ``no_barrier`` task's warps write before its main line moves on (all
    warps equally fast), so it never shows the missing barrier.

    Returns ``(status, machine)`` with status ``ok``, ``deadlock`` or ``fault``."""
    m = Machine(prog, dom or Num(), arena)
    rng, last = random.Random(seed), -1
    for _ in range(max_steps):
        if m.done():
            return "ok", m
        acts = m.actions()
        if not acts:
            return "deadlock", m
        if policy == "rr":
            sms = sorted({s for s, _ in acts})
            sm = next((s for s in sms if s > last), sms[0])
            act = min((a for a in acts if a[0] == sm), key=lambda a: (a[1] == 0, a[1]))
        else:
            act = rng.choice(acts)
        last = act[0]
        try:
            m.step(*act)
        except Fault:
            return "fault", m
    return "timeout", m


def replay(prog: Program, arena, trace, dom=None):
    """Replay a trace of SM numbers (lane 0) and ``(sm, lane)`` pairs."""
    m = Machine(prog, dom or Num(), arena)
    for a in trace:
        if isinstance(a, int):
            m.step(a)
        else:
            m.step(*a)
    return m
