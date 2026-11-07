"""Plant schedule bugs in AutoMegaKernel's own lowering of its toy Llama and run AMK's gate.

    AMK_ROOT=/path/to/AutoMegaKernel python -m triton_equiv.msync.amk_e2e  [out.json]

Needs ``torch`` and an AutoMegaKernel clone (commit ``amk_bridge.AMK_COMMIT``). CPU only.

For one decode step of ``make_toy(n_layers=1)`` lowered by ``schedule.lower.lower`` (the
path ``tests/test_lower.py`` exercises) it builds these mutants of the AMK program:

``counter_init_stale``  a counter with several producers starts at its producer count, as if
                        the host had not zeroed it after the previous launch.
``tile_oob``            the last GEMV tile of a projection gets ``N_tile + 4``: it now runs
                        past the end of its output buffer.
``tile_gap``            the last GEMV tile gets ``N_tile - 1``: one output column is never
                        written (a control the gate should catch).
``queue_inversion``     a chain ``a -> b -> c`` (counter edges, no direct ``a -> c`` edge) is
                        put on SMs ``s, t, s`` and ``c`` is moved ahead of ``a`` in the task
                        list. If SM queues followed task-list order, SM ``s`` would block on
                        ``c`` before running ``a``. AMK's CUDA loader builds queues in
                        topological order instead, so ``loader_queue_stuck`` should be empty.

and records, for the reference and every mutant: AMK's ``validate()``, its two simulators,
our SM-queue simulator in task-list order and in the loader's topological order, and
AMK's authoritative
correctness gate (``ReferenceVM`` on CPU against eager PyTorch through
``eval.oracle.logit_equivalence``, as in ``harness._evaluate_program``).
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

from .amk_bridge import AMK_COMMIT, load_amk_ir, queue_sim

N_EACH = 3  # mutants per operator
N_SMS = 4   # AMK's lowerer leaves Task.sm unset; we assign round robin over the topo order,
            # as vm/loader.py's sm_round_robin ablation does, so queue mutants have SMs to move


def _amk():
    ir = load_amk_ir()
    mods = {n: importlib.import_module(n) for n in
            ("models.toy", "schedule.graph", "schedule.lower", "vm.reference_vm", "eval.oracle")}
    return ir, mods


def _renumber(p):
    for i, t in enumerate(p.tasks):
        t.id = i
    return p


def counter_init_stale(prog, ir):
    prods = prog.producers_by_counter()
    waited = {w.counter for t in prog.tasks for w in t.waits}
    for c in prog.counters:
        if len(prods.get(c.id, [])) >= 2 and c.id in waited:
            p = copy.deepcopy(prog)
            p.counters[c.id].init = len(prods[c.id])
            yield f"counter {c.id} ({len(prods[c.id])} producers)", p


def _last_tiles(prog, ir):
    by_buf = {}
    for t in prog.tasks:
        if t.op == ir.InstructionKind.GEMV_TILE and t.outputs:
            by_buf.setdefault(t.outputs[0], []).append(t)
    for b, ts in by_buf.items():
        if len(ts) >= 2:
            yield b, max(ts, key=lambda t: t.params["n_off"])


def tile_oob(prog, ir):
    for b, last in _last_tiles(prog, ir):
        p = copy.deepcopy(prog)
        p.tasks[last.id].params["N_tile"] += 4
        yield f"{last.label} N_tile+4 (buffer {prog.buffers[b].name} {prog.buffers[b].shape})", p


def tile_gap(prog, ir):
    for b, last in _last_tiles(prog, ir):
        if last.params["N_tile"] > 1:
            p = copy.deepcopy(prog)
            p.tasks[last.id].params["N_tile"] -= 1
            yield f"{last.label} N_tile-1", p


def queue_inversion(prog, ir):
    edges = set(prog.dependency_edges())
    succ = {}
    for a, b in edges:
        succ.setdefault(a, []).append(b)
    for a in sorted(succ):
        for b in succ[a]:
            for c in succ.get(b, []):
                if (a, c) in edges or c <= a:
                    continue
                p = copy.deepcopy(prog)
                s = p.tasks[a].sm
                p.tasks[c].sm = s
                p.tasks[b].sm = (s + 1) % N_SMS
                tc = p.tasks.pop(c)
                p.tasks.insert(a, tc)
                _renumber(p)
                if ir.validate(p).ok:
                    yield f"{prog.tasks[a].label} -> {prog.tasks[b].label} -> {prog.tasks[c].label}", p
                    break
            else:
                continue
            break


OPERATORS = {"counter_init_stale": counter_init_stale, "tile_oob": tile_oob, "tile_gap": tile_gap,
             "queue_inversion": queue_inversion}


def gate(p, model, mods, torch, tok=7):
    """AMK's authoritative correctness verdict for one program.

    ``outcome`` is one of ``refused`` (the ReferenceVM's own ``validate()`` rejected it),
    ``crashed`` (it raised, e.g. ``DeadlockError``), ``correct`` or ``wrong``."""
    lower = mods["schedule.lower"]
    inputs = {lower.TOKEN_NAME: torch.tensor([tok], dtype=torch.int32),
              lower.POS_NAME: torch.tensor([0], dtype=torch.int32),
              lower.RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32)}
    try:
        out = mods["vm.reference_vm"].ReferenceVM(p, model.weights_dict(), device="cpu").run(inputs, kv={})
    except ValueError as e:
        return {"outcome": "refused", "detail": str(e)[:300]}
    except Exception as e:  # DeadlockError, index errors from out-of-range tiles, ...
        return {"outcome": "crashed", "detail": f"{type(e).__name__}: {str(e)[:300]}"}
    with torch.no_grad():
        eager = model.forward(torch.tensor([tok]))[-1].view(1, -1)
    v = mods["eval.oracle"].logit_equivalence(out["logits"], eager, dtype=torch.float32)
    return {"outcome": "correct" if v.correct else "wrong", "max_abs_err": float(v.max_abs_err),
            "top1_agreement": float(v.top1_agreement)}


def checks(p, ir):
    res = ir.validate(p)
    _, stuck = p.simulate_counters()
    return {"amk_validate": bool(res.ok), "amk_errors": list(res.errors)[:3],
            "amk_sim_counters": not stuck, "amk_sim_adv": not p.simulate_adversarial(seeds=96),
            "queue_sim_stuck": queue_sim(p)[:8],
            # None when validate() rejects: the loader refuses those programs (and a cyclic
            # one has no topological order to build queues from).
            "loader_queue_stuck": (queue_sim(p, p.topological_order()) if res.ok else None)}


def main(argv=None):
    import torch

    ir, mods = _amk()
    model = mods["models.toy"].make_toy(seed=0, dtype=torch.float32, n_layers=1)
    graph = mods["schedule.graph"].from_toy(model)
    ref = mods["schedule.lower"].lower(graph, target=ir.TARGETS["rtx5090"], pos=0, dtype=ir.DType.F32)
    for i, tid in enumerate(ref.topological_order()):
        ref.tasks[tid].sm = i % N_SMS
    rows = [{"id": "reference", "operator": None, "site": None, "tasks": len(ref.tasks),
             **checks(ref, ir), "gate": gate(ref, model, mods, torch)}]
    for op, fn in OPERATORS.items():
        for k, (site, p) in enumerate(fn(ref, ir)):
            if k >= N_EACH:
                break
            rows.append({"id": f"{op}:{k}", "operator": op, "site": site, **checks(p, ir),
                         "gate": gate(p, model, mods, torch)})
    res = {"command": "AMK_ROOT=... python -m triton_equiv.msync.amk_e2e", "amk_commit": AMK_COMMIT,
           "torch": torch.__version__, "model": "models.toy.make_toy(seed=0, n_layers=1), fp32, pos=0, token 7",
           "rows": rows}
    path = Path(argv[0] if argv else "results/amk_toy_mutants.json")
    path.write_text(json.dumps(res, indent=1))
    for r in rows:
        lq = r["loader_queue_stuck"]
        print(f"{r['id']:<22} validate={r['amk_validate']!s:<5} sim_counters={r['amk_sim_counters']!s:<5} "
              f"sim_adv={r['amk_sim_adv']!s:<5} queue_stuck={len(r['queue_sim_stuck'])} "
              f"loader_stuck={'-' if lq is None else len(lq)} gate={r['gate']['outcome']}  {r['site'] or ''}")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
