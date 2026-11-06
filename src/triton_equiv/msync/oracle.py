"""Exhaustive-interleaving oracle for schedule programs.

For a program it explores **every** interleaving of the SMs' steps (memoised DFS over
machine states in the symbolic domain) and labels the program:

``oob``          a declared region lies outside its buffer, or some interleaving faults.
``deadlock``     some reachable state has unfinished tasks and none can step.
``racy``         interleavings reach more than one output.
``inequivalent`` every interleaving reaches the same output, but not the reference's.
``benign``       every interleaving reaches the reference's output. With uninterpreted
                 operations this holds for every input value (see ``machine``).

Priority is the order above; all flags are recorded. A ``racy`` or ``inequivalent`` label
comes with a witness interleaving, replayed in float64 with random inputs and NaN for
uninitialised memory to confirm the output really differs. The labels are proofs only at
the program's size and only under step-level sequential consistency.
"""

from __future__ import annotations

import math
import sys

from .ir import Program
from .machine import Fault, Machine, Num, Sym, num_arena, num_inputs, replay, sym_arena

MAX_STATES = 2_000_000


def static_oob(prog: Program) -> list:
    bad = []
    for t in prog.tasks:
        for r in t.reads + t.writes:
            if r.lo < 0 or r.hi > prog.buf(r.buf).size:
                bad.append(f"{t.id}:{r.buf}[{r.lo}:{r.hi}] (size {prog.buf(r.buf).size})")
    return bad


def explore(prog: Program, sym: Sym):
    """All outcomes of ``prog``: ``{output_terms: witness_trace}``, plus the first deadlock
    and fault traces found (or None) and the number of states visited."""
    sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))
    outcomes, visited = {}, set()
    first = {"deadlock": None, "fault": None}

    def dfs(m: Machine, trace):
        k = m.key()
        if k in visited:
            return
        visited.add(k)
        if len(visited) > MAX_STATES:
            raise RuntimeError(f"{prog.name}: more than {MAX_STATES} states")
        if m.done():
            outcomes.setdefault(m.output(), list(trace))
            return
        en = [s for s in range(prog.n_sms) if m.enabled(s)]
        if not en:
            if first["deadlock"] is None:
                first["deadlock"] = list(trace)
            return
        for s in en:
            m2 = m.copy()
            try:
                m2.step(s)
            except Fault:
                if first["fault"] is None:
                    first["fault"] = trace + [s]
                continue
            trace.append(s)
            dfs(m2, trace)
            trace.pop()

    dfs(Machine(prog, sym, sym_arena(prog, sym)), [])
    return outcomes, first["deadlock"], first["fault"], len(visited)


def _num_out(prog, trace, seed):
    m = replay(prog, num_arena(prog, num_inputs(prog, seed)), trace, Num())
    return m.output()


def _differs(a, b):
    return any(not (x == y) for x, y in zip(a, b))  # NaN never equals


def label(prog: Program, ref: Program, sym: Sym, ref_outcomes=None) -> dict:
    ref_outcomes = ref_outcomes if ref_outcomes is not None else explore(ref, sym)[0]
    outcomes, dead, fault, n = explore(prog, sym)
    oob = static_oob(prog)
    flags = {
        "oob": bool(oob) or fault is not None,
        "deadlock": dead is not None,
        "racy": len(outcomes) > 1,
        "differs": any(o not in ref_outcomes for o in outcomes),
    }
    if flags["oob"]:
        lab = "oob"
    elif flags["deadlock"]:
        lab = "deadlock"
    elif flags["racy"]:
        lab = "racy"
    elif flags["differs"]:
        lab = "inequivalent"
    else:
        lab = "benign"
    out = {"label": lab, "flags": flags, "states": n, "outcomes": len(outcomes), "oob_regions": oob,
           "deadlock_trace": dead, "fault_trace": fault}
    # Numeric confirmation: one witness per distinct symbolic outcome, against the reference.
    if lab in ("racy", "inequivalent") and len(ref_outcomes) == 1:
        ref_trace = next(iter(ref_outcomes.values()))
        ref_num = _num_out(ref, ref_trace, seed=1)
        wit = next((tr for o, tr in outcomes.items() if o not in ref_outcomes), None)
        if wit is not None:
            got = _num_out(prog, wit, seed=1)
            out["witness_trace"] = wit
            out["witness_confirmed"] = _differs(got, ref_num)
            out["witness_output"] = [None if math.isnan(v) else round(v, 6) for v in got]
            out["reference_output"] = [round(v, 6) for v in ref_num]
    return out
