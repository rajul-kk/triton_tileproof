"""Schedule checkers against oracle labels (pure Python, no GPU)."""

from triton_equiv.msync.checkers import amk_rules, run_checker
from triton_equiv.msync.ir import barrier2, mlp_rms, moe_dispatch, reference_programs, split_reduce
from triton_equiv.msync.machine import Num, Sym, num_arena, num_inputs, run
from triton_equiv.msync.mutate import mutants
from triton_equiv.msync.oracle import label


def _mut(prog, mid):
    return next(m.program for m in mutants(prog) if m.id == mid)


def test_references_pass_every_checker():
    for ref in reference_programs():
        if ref.name == "moe_dispatch":
            continue
        assert amk_rules(ref) == [], ref.name
        for c in ("warm_rr_allclose", "nan_rr_allclose", "random20_nan_exact"):
            assert run_checker(c, ref, ref) is True, (ref.name, c)


def test_moe_dispatch_outcome_depends_on_order():
    ref = moe_dispatch()
    assert label(ref, ref, Sym())["label"] == "racy"
    assert amk_rules(ref) is None  # ticket dispatch is outside AMK's IR
    assert run_checker("random20_nan_exact", ref, ref) is False


def test_partial_wait_is_a_race_amk_rejects():
    ref = split_reduce(8)
    m = _mut(ref, "split_reduce_aligned:partial_wait:0")
    lab = label(m, ref, Sym())
    assert lab["label"] == "racy" and lab["witness_confirmed"]
    assert amk_rules(m)  # the all-join rule
    assert run_checker("random20_nan_exact", m, ref) is False


def test_unclamped_tail_overwrites_next_buffer_and_is_accepted():
    """The odd-size tail runs one row past ``y`` into ``z``; the norm overwrites ``z``
    afterwards, so the output is right and every run-and-compare checker accepts."""
    ref = mlp_rms(3, 2)
    m = _mut(ref, "mlp_rms_odd:tile_unclamped:0")
    lab = label(m, ref, Sym())
    assert lab["label"] == "oob" and lab["oob_regions"]
    for c in ("warm_rr_allclose", "nan_rr_allclose", "random20_nan_exact", "amk_rules"):
        assert run_checker(c, m, ref) is True, c


def test_barrier_reset_race_hangs_only_under_some_schedules():
    ref = barrier2()
    m = _mut(ref, "barrier2:barrier_reuse_reset:0")
    assert label(m, ref, Sym())["label"] == "deadlock"
    assert run(m, num_arena(m, num_inputs(m, 0)), Num())[0] == "ok"  # round robin finishes
    assert run_checker("random20_nan_exact", m, ref) is False


def test_transitive_same_sm_inversion_passes_direct_edge_rule():
    """g1 and h0 share SM 1 with h0 first, and h0 depends on g1 only through r on SM 0:
    a deadlock with no direct same-SM edge, which a direct-edge queue-order rule misses."""
    ref = mlp_rms(4, 2)
    m = next(x for x in mutants(ref) if x.operator == "swap_same_sm" and x.site == "g1<->h0").program
    assert label(m, ref, Sym())["label"] == "deadlock"
    assert amk_rules(m) == []


def test_counter_left_from_previous_launch_is_racy_and_passes_static_rules():
    ref = mlp_rms(4, 2)
    m = next(x for x in mutants(ref) if x.operator == "counter_not_zeroed" and x.site == "c_y").program
    assert label(m, ref, Sym())["label"] == "racy"
    assert amk_rules(m) == []
