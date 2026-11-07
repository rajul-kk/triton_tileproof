"""Run AutoMegaKernel's own schedule checks on our schedule programs.

AutoMegaKernel (https://github.com/RightNow-AI/AutoMegaKernel) is not a dependency. Point
``AMK_ROOT`` at a clone; results in this repo used commit ``AMK_COMMIT`` below. Only
``schedule/ir.py`` is imported here, which is pure stdlib.

Translation (structure only; AMK's checks never look at values):

* buffers: ``input`` -> ``IO_INPUT``, ``weight`` -> ``WEIGHT``, ``act``/``scratch`` ->
  ``ACTIVATION``, ``output`` -> ``IO_OUTPUT``. Shapes are ``(1, size)``.
* ``gemv`` -> ``GEMV_TILE`` (inputs ``[x, W]``, ``n_off``/``N_tile`` from the tile);
  ``sum`` -> ``GEMV_TILE`` with ``N_tile=1`` and ``n_off`` = the scratch slot (a partial dot
  with a ones vector); ``rms`` -> ``RMSNORM`` with an added gain weight.
* a task with no ``out_counter`` gets a fresh counter nobody waits on (AMK requires one).
* ``Counter.init`` carries over, so ``counter_not_zeroed`` mutants reach AMK unchanged.
* ``reset``, ``ticket`` and ``signal_first`` have no AMK form: those programs are skipped.

The three checks:

``amk_validate``     ``schedule.ir.validate`` (what the VM requires before loading).
``amk_sim_counters`` ``simulate_counters()`` has no stuck task (AMK's deadlock oracle).
``amk_sim_adv``      ``simulate_adversarial()`` reports no read-before-write (AMK's race
                     oracle; ``seeds=96`` as in AMK's validator experiment).

Plus one check AMK does not have:

``queue_sim``        SM-queue-aware run: each SM may only start the head of its own queue.
                     Reports a deadlock where AMK's simulators, which let any ready task fire,
                     see none.
"""

from __future__ import annotations

import importlib
import os
import sys

from .ir import Program

AMK_COMMIT = "884534c4d2dbed4381576f81a79ae97feca53c34"


def load_amk_ir(root: str | None = None):
    root = root or os.environ.get("AMK_ROOT")
    if not root:
        raise RuntimeError("set AMK_ROOT to an AutoMegaKernel clone")
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("schedule.ir")


def expressible(prog: Program) -> bool:
    return not any(t.op in ("reset", "ticket") or t.signal_first for t in prog.tasks)


def to_amk(prog: Program, ir):
    p = ir.MegakernelProgram(meta={"model": prog.name})
    kind = {"input": ir.BufferKind.IO_INPUT, "weight": ir.BufferKind.WEIGHT,
            "act": ir.BufferKind.ACTIVATION, "scratch": ir.BufferKind.ACTIVATION,
            "output": ir.BufferKind.IO_OUTPUT}
    bid = {}
    for b in prog.buffers:
        src = b.name if b.kind == "weight" else None
        bid[b.name] = p.new_buffer(b.name, kind[b.kind], ir.DType.F32, (1, b.size), source=src).id
    cid = {}
    for name in sorted(prog.counters):
        c = p.new_counter(name)
        c.init = prog.counters[name]
        cid[name] = c.id
    for t in prog.tasks:
        if t.out_counter is None:
            out = p.new_counter(f"{t.id}.done").id
        else:
            out = cid[t.out_counter]
        waits = [ir.Wait(cid[w.counter], w.threshold) for w in t.waits]
        if t.op == "gemv":
            n0, n1 = t.p("n0"), t.p("n1")
            op, ins = ir.InstructionKind.GEMV_TILE, [bid[t.p("x")], bid[t.p("W")]]
            params = {"K": t.p("K"), "N_tile": n1 - n0, "n_off": n0}
        elif t.op == "sum":
            w = t.writes[0]
            ones = p.new_buffer(f"ones.{t.id}", ir.BufferKind.WEIGHT, ir.DType.F32,
                                (1, t.reads[0].hi - t.reads[0].lo), source=f"ones.{t.id}").id
            op, ins = ir.InstructionKind.GEMV_TILE, [bid[t.reads[0].buf], ones]
            params = {"K": t.reads[0].hi - t.reads[0].lo, "N_tile": w.hi - w.lo, "n_off": w.lo}
        elif t.op == "rms":
            g = p.new_buffer(f"gain.{t.id}", ir.BufferKind.WEIGHT, ir.DType.F32,
                             (1, t.reads[0].hi - t.reads[0].lo), source=f"gain.{t.id}").id
            op, ins = ir.InstructionKind.RMSNORM, [bid[t.reads[0].buf], g]
            params = {"eps": 1e-6, "hidden": t.reads[0].hi - t.reads[0].lo}
        else:
            raise ValueError(f"no AMK form for {t.op}")
        outs = [bid[r.buf] for r in t.writes]
        p.add_task(op, ins, outs, out_counter=out, waits=waits, params=params, label=t.id, sm=t.sm)
    return p


def queue_sim(p, order=None) -> list:
    """Stuck task ids when each SM runs its own queue to completion, firing only queue heads
    whose waits hold. Empty means no deadlock under any schedule: firing a ready head never
    disables another SM's head, so one greedy run decides it. That needs counters that only
    go up, which holds for every AMK program (AMK has no ``reset``; part-1 programs with
    ``reset`` or ``ticket`` are not ``expressible`` and never reach here).

    ``order`` is the sequence the queues are built from: task-list order by default (what
    ``validate()`` assumes and part 1 models), or ``p.topological_order()``, which is what
    AMK's CUDA loader (``vm/loader.py`` ``_assign_sms``) actually uses."""
    cval = {c.id: c.init for c in p.counters}
    queues = {}
    for tid in (range(len(p.tasks)) if order is None else order):
        t = p.tasks[tid]
        queues.setdefault(t.sm, []).append(t)
    progress = True
    while progress:
        progress = False
        for q in queues.values():
            while q and all(cval.get(w.counter, 0) >= w.threshold for w in q[0].waits):
                t = q.pop(0)
                cval[t.out_counter] = cval.get(t.out_counter, 0) + 1
                progress = True
    return [t.id for q in queues.values() for t in q]


def amk_checks(prog: Program, ir) -> dict:
    """``{check: accepted?}`` for the four checks above, None when not expressible."""
    names = ("amk_validate", "amk_sim_counters", "amk_sim_adv", "queue_sim")
    if not expressible(prog):
        return {n: None for n in names} | {"amk_errors": None}
    p = to_amk(prog, ir)
    res = ir.validate(p)
    _, stuck = p.simulate_counters()
    return {"amk_validate": bool(res.ok), "amk_sim_counters": not stuck,
            "amk_sim_adv": not p.simulate_adversarial(seeds=96), "queue_sim": not queue_sim(p),
            "amk_errors": list(res.errors)}
