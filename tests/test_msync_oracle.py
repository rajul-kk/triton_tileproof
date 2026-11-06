"""Schedule IR, interleaving oracle and mutants (pure Python, no GPU)."""

from triton_equiv.msync.ir import Wait, mlp_rms, reference_programs, split_reduce
from triton_equiv.msync.machine import Sym, num_arena, num_inputs, run
from triton_equiv.msync.mutate import mutants
from triton_equiv.msync.oracle import explore, label


def _mut(prog, mid):
    return next(m.program for m in mutants(prog) if m.id == mid)


def test_references_are_deterministic_except_moe():
    sym = Sym()
    for ref in reference_programs():
        outcomes, dead, fault, _ = explore(ref, sym)
        assert dead is None and fault is None, ref.name
        assert (len(outcomes) > 1) == (ref.name == "moe_dispatch"), ref.name


def test_threshold_too_high_deadlocks():
    ref = split_reduce(8)
    m = _mut(ref, "split_reduce_aligned:threshold_plus_one:0")
    assert label(m, ref, Sym())["label"] == "deadlock"
    assert run(m, num_arena(m, num_inputs(m, 0)))[0] == "deadlock"


def test_symbolic_equality_means_numeric_equality():
    ref = mlp_rms(4, 2)
    sym = Sym()
    ref_out = explore(ref, sym)[0]
    for mu in mutants(ref):
        lab = label(mu.program, ref, sym, ref_out)
        if lab["label"] == "benign":
            for seed in (0, 1):
                inputs = num_inputs(ref, seed)
                a = run(ref, num_arena(ref, inputs))[1].output()
                b = run(mu.program, num_arena(mu.program, inputs), policy="random", seed=seed)[1].output()
                assert a == b, mu.id


def test_wait_dataclass_roundtrip():
    assert Wait("c", 2).threshold == 2
