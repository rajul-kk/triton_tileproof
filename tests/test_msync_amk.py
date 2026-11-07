"""Checks against a real AutoMegaKernel clone. Skipped unless AMK_ROOT is set."""

import os

import pytest

from triton_equiv.msync import amk_bridge
from triton_equiv.msync.ir import reference_programs
from triton_equiv.msync.mutate import mutants

pytestmark = pytest.mark.skipif(not os.environ.get("AMK_ROOT"), reason="AMK_ROOT not set")


@pytest.fixture(scope="module")
def ir():
    return amk_bridge.load_amk_ir()


def test_references_pass_amk(ir):
    for prog in reference_programs():
        if not amk_bridge.expressible(prog):
            continue
        r = amk_bridge.amk_checks(prog, ir)
        assert r["amk_validate"], (prog.name, r["amk_errors"])
        assert r["queue_sim"] and r["amk_sim_counters"], prog.name


def test_transitive_inversion_passes_validate_but_hangs(ir):
    prog = next(p for p in reference_programs() if p.name == "mlp_rms_aligned")
    m = next(m for m in mutants(prog) if m.id.endswith("swap_same_sm:2"))
    r = amk_bridge.amk_checks(m.program, ir)
    assert r["amk_sim_counters"]          # AMK's simulator ignores SM queues
    assert not r["queue_sim"]             # task-list-order SM queues deadlock
    p = amk_bridge.to_amk(m.program, ir)
    assert amk_bridge.queue_sim(p, p.topological_order()) == []  # AMK's loader order does not


def test_queue_sim_on_amk_lowering(ir):
    torch = pytest.importorskip("torch")
    from triton_equiv.msync import amk_e2e

    _, mods = amk_e2e._amk()
    model = mods["models.toy"].make_toy(seed=0, dtype=torch.float32, n_layers=1)
    ref = mods["schedule.lower"].lower(mods["schedule.graph"].from_toy(model),
                                       target=ir.TARGETS["rtx5090"], pos=0, dtype=ir.DType.F32)
    for i, tid in enumerate(ref.topological_order()):
        ref.tasks[tid].sm = i % amk_e2e.N_SMS
    assert ir.validate(ref).ok
    assert amk_bridge.queue_sim(ref) == []
    g = amk_e2e.gate(ref, model, mods, torch)
    assert g["outcome"] == "correct", g
