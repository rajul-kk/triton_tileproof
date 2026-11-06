"""Step semantics for the schedule IR, in two value domains.

Each task runs in three steps on its SM:

1. **acquire**: enabled once every wait holds (``counter >= threshold``). The task reads
   all of its input regions at this step (a ``ticket`` task also takes its slot here, with
   an atomic fetch-and-add).
2. **write**: the task's outputs land in the arena (a ``reset`` task zeroes its counter).
3. **signal**: ``out_counter += 1``.

With ``signal_first`` the last two swap, which is what a missing release fence allows.
Steps of different SMs interleave arbitrarily, and each step is atomic. That is sequential
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

from .ir import Program


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

    def copy(self):
        m = Machine.__new__(Machine)
        m.prog, m.dom, m.queues, m.cnames = self.prog, self.dom, self.queues, self.cnames
        m.arena, m.counters = list(self.arena), dict(self.counters)
        m.pc, m.phase, m.local = list(self.pc), list(self.phase), list(self.local)
        return m

    def key(self):
        return (tuple(self.arena), tuple(self.counters[c] for c in self.cnames), tuple(self.pc),
                tuple(self.phase), tuple(self.local))

    def current(self, sm):
        q = self.queues[sm]
        return q[self.pc[sm]] if self.pc[sm] < len(q) else None

    def done(self):
        return all(self.current(s) is None for s in range(self.prog.n_sms))

    def enabled(self, sm):
        t = self.current(sm)
        if t is None:
            return False
        if self.phase[sm] > 0:
            return True
        return all(self.counters[w.counter] >= w.threshold for w in t.waits)

    def _cells(self, region):
        base = self.prog.layout[region.buf]
        cells = range(base + region.lo, base + region.hi)
        if region.lo < 0 and base + region.lo < 0 or base + region.hi > self.prog.arena_size:
            raise Fault(region)
        return cells

    def step(self, sm):
        t, ph = self.current(sm), self.phase[sm]
        if ph == 0:
            vals = tuple(tuple(self.arena[c] for c in self._cells(r)) for r in t.reads)
            ticket = None
            if t.op == "ticket":
                ticket = self.counters[t.p("tick")]
                self.counters[t.p("tick")] += 1
            self.local[sm] = (vals, ticket)
            self.phase[sm] = 1
            return
        action = ("signal", "write") if t.signal_first else ("write", "signal")
        what = action[ph - 1]
        if what == "write":
            self._write(t, *self.local[sm])
        elif t.out_counter is not None:
            self.counters[t.out_counter] += 1
        if ph == 2:
            self.pc[sm] += 1
            self.phase[sm] = 0
            self.local[sm] = None
        else:
            self.phase[sm] = 2

    def _write(self, t, vals, ticket):
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
            for c, v in zip(cells, out):
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

    Returns ``(status, machine)`` with status ``ok``, ``deadlock`` or ``fault``."""
    m = Machine(prog, dom or Num(), arena)
    rng, last = random.Random(seed), -1
    for _ in range(max_steps):
        if m.done():
            return "ok", m
        en = [s for s in range(prog.n_sms) if m.enabled(s)]
        if not en:
            return "deadlock", m
        if policy == "rr":
            sm = next((s for s in en if s > last), en[0])
        else:
            sm = rng.choice(en)
        last = sm
        try:
            m.step(sm)
        except Fault:
            return "fault", m
    return "timeout", m


def replay(prog: Program, arena, trace, dom=None):
    m = Machine(prog, dom or Num(), arena)
    for sm in trace:
        m.step(sm)
    return m
