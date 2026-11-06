"""Correctness checkers for schedule programs, and the audit against oracle labels.

Run-and-compare checkers (each compares against a float64 run of the reference program):

``warm_rr_allclose``   one launch, round-robin schedule, arena left by a previous correct
                       launch (what repeated benchmark launches see), allclose.
``zero_rr_allclose``   as above with a zeroed arena (``torch.zeros`` outputs).
``nan_rr_allclose``    as above with NaN in every non-input cell (poisoned memory).
``warm_rr_argmax``     warm arena, compares only ``argmax`` of the output: a greedy-token
                       check, as megakernel papers report token agreement. Not applicable
                       to programs with a single output value.
``random20_nan_exact`` 20 random interleavings on a poisoned arena, bitwise comparison:
                       the schedule-perturbation idea behind RESOLVE, without a GPU.

A deadlock or fault in any run is a rejection (a hang is caught by a timeout).

Static checker:

``amk_rules``          a re-implementation of the rules of AutoMegaKernel's ``validate()``
                       as described in its ``docs/IR_SPEC.md`` and ``schedule/ir.py``
                       (read 2026-11-07): wait thresholds in ``[1, #producers]`` and equal
                       to ``#producers`` when there are several (the all-join rule); no
                       self-wait; acyclic counter graph; every read of a writable buffer
                       ordered after all of its writers (happens-before over counter edges);
                       multiple writers either disjoint under one counter or totally
                       ordered; same-SM edges in queue order; every output written. It does
                       not check counter initial values or that tiles stay inside their
                       buffers, because AMK's validator does not. Programs using ``reset``,
                       ``ticket`` or ``signal_first`` cannot be written in AMK's IR and are
                       reported as not applicable. This is our reading of AMK's rules, not
                       its code; running the real validator is separate work.
"""

from __future__ import annotations

import math

from .ir import READONLY_KINDS, Program
from .machine import Num, num_arena, num_inputs, run

CHECKERS = ("warm_rr_allclose", "zero_rr_allclose", "nan_rr_allclose", "warm_rr_argmax",
            "random20_nan_exact", "amk_rules")
RTOL, ATOL = 1e-5, 1e-6


def _allclose(a, b):
    return all(abs(x - y) <= ATOL + RTOL * abs(y) for x, y in zip(a, b))  # False on NaN


def _argmax(v):
    best = 0
    for i, x in enumerate(v):
        if math.isnan(x):
            return i  # torch.argmax treats NaN as the maximum
        if x > v[best]:
            best = i
    return best


def reference_numeric(ref: Program, seed: int):
    inputs = num_inputs(ref, seed)
    status, m = run(ref, num_arena(ref, inputs, "nan"), Num())
    assert status == "ok", (ref.name, status)
    return inputs, m.output(), m.arena


def run_checker(name: str, prog: Program, ref: Program, seed: int = 0):
    """True = accepted, False = rejected, None = not applicable."""
    if name == "amk_rules":
        errs = amk_rules(prog)
        return None if errs is None else not errs
    inputs, ref_out, warm = reference_numeric(ref, seed)
    if name == "warm_rr_argmax" and len(ref_out) < 2:
        return None  # argmax of a single value always agrees
    if name == "random20_nan_exact":
        for s in range(20):
            status, m = run(prog, num_arena(prog, inputs, "nan"), Num(), policy="random", seed=s)
            if status != "ok" or tuple(m.output()) != tuple(ref_out):
                return False
        return True
    fill = {"warm_rr_allclose": "warm", "zero_rr_allclose": "zero", "nan_rr_allclose": "nan",
            "warm_rr_argmax": "warm"}[name]
    status, m = run(prog, num_arena(prog, inputs, fill, warm), Num())
    if status != "ok":
        return False
    if name == "warm_rr_argmax":
        return _argmax(m.output()) == _argmax(ref_out)
    return _allclose(m.output(), ref_out)


def amk_rules(prog: Program):
    """List of rule violations, or None if the program is outside AMK's IR."""
    if any(t.op in ("reset", "ticket") or t.signal_first for t in prog.tasks):
        return None
    errs = []
    prods = {c: prog.producers(c) for c in prog.counters}
    for t in prog.tasks:
        for w in t.waits:
            n = len(prods[w.counter])
            if w.counter == t.out_counter:
                errs.append(f"{t.id}: waits on its own counter")
            if n == 0 or w.threshold < 1 or w.threshold > n:
                errs.append(f"{t.id}: unsatisfiable wait {w.counter}>={w.threshold} ({n} producers)")
            elif n > 1 and w.threshold != n:
                errs.append(f"{t.id}: partial wait {w.counter}>={w.threshold} of {n} (race)")
    preds = {t.id: {a.id for w in t.waits for a in prods[w.counter]} for t in prog.tasks}
    # Kahn's algorithm over counter edges.
    indeg = {k: len(v) for k, v in preds.items()}
    succ = {t.id: [u for u, ps in preds.items() if t.id in ps] for t in prog.tasks}
    order, ready = [], [k for k, d in indeg.items() if d == 0]
    while ready:
        k = ready.pop()
        order.append(k)
        for u in succ[k]:
            indeg[u] -= 1
            if indeg[u] == 0:
                ready.append(u)
    if len(order) != len(prog.tasks):
        return errs + ["cycle in counter graph (deadlock)"]
    anc = {}
    for k in order:
        anc[k] = set(preds[k]).union(*[anc[p] for p in preds[k]]) if preds[k] else set()
    writers = {}
    for t in prog.tasks:
        for r in t.writes:
            writers.setdefault(r.buf, []).append((t, r))
    for t in prog.tasks:
        for r in t.reads:
            if prog.buf(r.buf).kind in READONLY_KINDS:
                continue
            ws = [w for w, _ in writers.get(r.buf, []) if w.id != t.id]
            if not ws:
                errs.append(f"{t.id}: reads {r.buf}, never produced")
            for w in ws:
                if w.id not in anc[t.id]:
                    errs.append(f"{t.id}: reads {r.buf} written by {w.id}, not ordered before it (race)")
    for buf, ws in writers.items():
        if len(ws) < 2:
            continue
        same = len({w.out_counter for w, _ in ws}) == 1
        disjoint = all(a.hi <= b.lo or b.hi <= a.lo for i, (_, a) in enumerate(ws) for _, b in ws[i + 1:])
        total = all(x.id in anc[y.id] or y.id in anc[x.id] for i, (x, _) in enumerate(ws) for y, _ in ws[i + 1:])
        if not ((same and disjoint) or total):
            errs.append(f"multi-writer race on {buf}")
    pos = {t.id: i for i, t in enumerate(prog.tasks)}
    for t in prog.tasks:
        for p in preds[t.id]:
            if prog.task(p).sm == t.sm and pos[p] > pos[t.id]:
                errs.append(f"{p}->{t.id} on SM {t.sm} out of queue order (deadlock)")
    for b in prog.buffers:
        if b.kind == "output" and b.name not in writers:
            errs.append(f"output {b.name} never written")
    return errs
