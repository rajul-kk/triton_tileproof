"""Floating-point tier: real equivalence, structural bitwise proofs, witnesses and error bounds."""

import numpy as np
import pytest

pytest.importorskip("z3")

from triton_equiv import fporacle  # noqa: E402
from triton_equiv.kernels import KERNELS  # noqa: E402
from triton_equiv.mutate import mutants  # noqa: E402


def _mutant(kernel, op, site=0):
    return next(m for m in mutants(kernel, KERNELS[kernel].source, [op]) if m.site == site)


def _label(kernel, op, site=0):
    m = _mutant(kernel, op, site)
    return fporacle.label(KERNELS[kernel], m.source, m.id)


@pytest.mark.parametrize("kernel", list(KERNELS))
def test_reference_is_proved_bitwise_identical_to_itself(kernel):
    spec = KERNELS[kernel]
    lab = fporacle.label(spec, spec.source)
    assert lab.label == "benign" and lab.fp_proof == "uf"


def test_reorderings_are_real_equivalent_but_not_structurally_identical():
    spec = KERNELS["row_sum_loop"]
    for op in ("reverse_reduction_loop", "split_sum", "acc_dtype_downgrade"):
        m = _mutant("row_sum_loop", op)
        for size in spec.oracle_sizes:
            assert fporacle.real_counterexample(spec, m.source, size) is None, op
        assert not fporacle.uf_identical(spec, m.source, spec.fp_sizes[0]), op


def test_real_tier_finds_a_real_counterexample():
    # wrong_other on row_sum adds 1 per masked lane: different even over the reals.
    spec = KERNELS["row_sum"]
    m = _mutant("row_sum", "wrong_other")
    cx = fporacle.real_counterexample(spec, m.source, {"n_rows": 2, "n_cols": 5})
    assert cx is not None and cx["buffer"] == "out"


def test_two_iteration_loop_reversal_is_exact():
    # (0 + a) + b == (0 + b) + a in IEEE arithmetic, so with two K-tiles the reversed loop is
    # structurally different but still never differs; a witness needs >= 3 iterations.
    spec = KERNELS["row_sum_loop"]
    m = _mutant("row_sum_loop", "reverse_reduction_loop")
    w, _, n = fporacle.search_witness(spec, m.source, {"n_rows": 2, "n_cols": 8}, trials=5)
    assert w is None and n == 15


@pytest.mark.parametrize("kernel,op,loss", [
    ("row_sum", "split_sum", False),
    ("matmul", "reverse_reduction_loop", False),
    ("row_sum_loop", "acc_dtype_downgrade", True),
    ("matmul_fp16", "change_cast", True),
])
def test_numerical_label_with_reproducible_witness_and_sound_bounds(kernel, op, loss):
    spec = KERNELS[kernel]
    m = _mutant(kernel, op)
    lab = fporacle.label(spec, m.source, m.id)
    assert lab.label == "numerical" and lab.precision_loss is loss
    w = lab.witness
    launch = spec.make_launch(**w["size"])
    inputs = {b: np.array(v).astype(fporacle._np_dtype(launch.dtype(b))) for b, v in w["inputs"].items()}
    ref, _ = fporacle.emulate(spec.source, spec.fn_name, launch, inputs)
    mut, _ = fporacle.emulate(m.source, spec.fn_name, launch, inputs)
    i, buf = w["index"], w["buffer"]
    assert float(ref[buf][i]) == w["ref_value"] != w["mut_value"] == float(mut[buf][i])
    # Running error bounds hold against the fp64 value (tiny slack for fp64 itself).
    slack = 1e-12 * max(1.0, abs(w["fp64_value"]))
    assert abs(w["ref_value"] - w["fp64_value"]) <= w["ref_err_bound"] + slack
    assert abs(w["mut_value"] - w["fp64_value"]) <= w["mut_err_bound"] + slack


def test_structural_proofs_for_drop_cast_and_promoted_downgrade():
    # tl.store already rounds to the fp16 buffer, so the explicit .to(fp16) is redundant.
    assert _label("matmul_fp16", "drop_cast").fp_proof == "uf"
    # `acc += tl.dot(a, b)` promotes an fp16 accumulator back to fp32 on the first iteration
    # (interpreter semantics; compiled Triton would reject the loop-carried dtype change).
    lab = _label("matmul", "acc_dtype_downgrade")
    assert lab.label == "benign" and lab.fp_proof == "uf"


@pytest.mark.parametrize("kernel", ["matmul", "matmul_fp16"])
@pytest.mark.parametrize("site", [0, 1])
def test_other_change_on_matmul_is_proved_under_the_lane_product_model(kernel, site):
    spec, m = KERNELS[kernel], _mutant(kernel, "wrong_other", site)
    # The opaque-dot tier cannot see that a padded lane meets a zero on the other operand ...
    sizes = spec.oracle_sizes + spec.fp_sizes
    assert not all(fporacle.uf_identical(spec, m.source, s) for s in sizes)
    # ... the lane-product tier folds 1*0 and 0*0 to the same +0 product.
    assert all(fporacle.uf_identical(spec, m.source, s, lanes=True) for s in sizes)
    lab = _label(kernel, "wrong_other", site)
    assert lab.label == "benign" and lab.fp_proof == "uf-lane" and lab.witness is None


def test_lane_fold_only_folds_narrow_finite_constants_and_keeps_signed_zero():
    import z3
    dom = fporacle._UFDom(lanes=True)
    prod = lambda a, b: dom.lane_product(a, b, "float32")  # noqa: E731
    x = z3.Real("x")
    fold = fporacle._fold_lane
    assert z3.eq(fold(prod(z3.RealVal(1), z3.RealVal(0))), z3.RealVal(0))
    neg_zero_fill = prod(z3.RealVal(1), dom.const(-0.0))  # a -0.0 literal is not the numeral 0
    assert not z3.eq(dom.const(-0.0), dom.const(0.0))
    assert z3.eq(fold(neg_zero_fill), neg_zero_fill)
    assert z3.eq(fold(prod(z3.RealVal(0), z3.RealVal(0))), z3.RealVal(0))
    assert z3.eq(fold(prod(z3.RealVal(3), z3.RealVal(2))), z3.RealVal(6))
    neg_zero = prod(z3.RealVal(-1), z3.RealVal(0))  # IEEE gives -0, which a Z3 numeral cannot hold
    assert z3.eq(fold(neg_zero), neg_zero)
    with_var = prod(x, z3.RealVal(0))  # x may be inf or NaN
    assert z3.eq(fold(with_var), with_var)
    not_narrow = prod(z3.RealVal(0.1), z3.RealVal(1))  # 0.1 is not exact at fp16/bf16/tf32 inputs
    assert z3.eq(fold(not_narrow), not_narrow)


def test_value_padding_on_attention_is_proved_through_the_neg_inf_rules():
    spec, m = KERNELS["attention"], _mutant("attention", "wrong_other", 2)
    sizes = spec.oracle_sizes + spec.fp_sizes
    # v's padded rows become 1.0; they meet p = exp(-inf - max) = +0, as the 0.0 rows did.
    assert not all(fporacle.uf_identical(spec, m.source, s) for s in sizes)
    assert all(fporacle.uf_identical(spec, m.source, s, lanes=True) for s in sizes)
    lab = _label("attention", "wrong_other", 2)
    assert lab.label == "benign" and lab.fp_proof == "uf-lane" and lab.witness is None


def test_neg_inf_fold_needs_a_finite_subtrahend():
    import z3
    dom = fporacle._UFDom(lanes=True)
    neg_inf, pos_inf = dom.const(-float("inf")), dom.const(float("inf"))
    load = z3.Select(z3.Array("m", z3.BitVecSort(8), z3.RealSort()), z3.BitVecVal(0, 8))
    sub = lambda a, b: dom.sub(a, b, "float32")  # noqa: E731
    mx = lambda *xs: dom.max(list(xs), "float32")  # noqa: E731
    fold = fporacle._fold_lane
    assert z3.eq(fold(dom.exp(neg_inf, "float32")), z3.RealVal(0))
    assert z3.eq(fold(sub(neg_inf, load)), neg_inf)
    assert z3.eq(fold(sub(neg_inf, mx(neg_inf, load))), neg_inf)  # a row max with one finite entry
    assert z3.eq(fold(dom.exp(sub(neg_inf, mx(neg_inf, load)), "float32")), z3.RealVal(0))
    assert not z3.eq(fold(dom.exp(neg_inf, "float32")), dom.const(-0.0))  # exp(-inf) is +0, not -0
    for kept in (sub(neg_inf, mx(neg_inf, neg_inf)),  # -inf - (-inf) is NaN
                 sub(neg_inf, z3.Real("x")),  # a free real may be infinite
                 sub(neg_inf, mx(pos_inf, load)),  # +inf in the max
                 sub(pos_inf, load),
                 dom.exp(load, "float32"),
                 sub(neg_inf, dom.sqrt(load, "float32")),  # sqrt of a negative load is NaN
                 sub(neg_inf, dom.div(load, load, "float32")),  # 0/0 is NaN
                 sub(neg_inf, dom.exp(load, "float32")),  # exp overflows from finite inputs
                 sub(neg_inf, dom.add(load, load, "float16")),  # fp16 overflows at 65504
                 sub(neg_inf, z3.RealVal(10 ** 39)),  # beyond fp32 range
                 sub(neg_inf, z3.Select(z3.Store(load.arg(0), z3.BitVecVal(1, 8), neg_inf),
                                        z3.BitVecVal(2, 8)))):  # a read through a store may be -inf
        assert z3.eq(fold(kept), kept)


@pytest.mark.parametrize("op", ["mask_off_by_one", "wrong_pid_axis", "drop_where_mask"])
def test_neg_inf_rules_do_not_prove_real_differences_on_attention(op):
    spec, checked = KERNELS["attention"], 0
    for m in mutants("attention", spec.source, [op]):
        size = next((s for s in spec.oracle_sizes if fporacle.real_counterexample(spec, m.source, s)), None)
        if size is not None:
            checked += 1
            assert not fporacle.uf_identical(spec, m.source, size, lanes=True), m.id
    assert checked, f"no {op} mutant of attention has a real difference to check against"


@pytest.mark.parametrize("op", ["mask_off_by_one", "wrong_pid_axis", "missing_store_mask"])
def test_lane_product_tier_does_not_prove_real_differences_on_matmul(op):
    spec = KERNELS["matmul"]
    for m in mutants("matmul", spec.source, [op]):
        size = next((s for s in spec.oracle_sizes if fporacle.real_counterexample(spec, m.source, s)), None)
        if size is not None:
            assert not fporacle.uf_identical(spec, m.source, size, lanes=True), m.id


def test_error_bounds_hold_for_every_reference_on_random_inputs():
    for name, spec in KERNELS.items():
        for size in spec.oracle_sizes + spec.fp_sizes:
            launch = spec.make_launch(**size)
            for kind in fporacle.GENERATORS:
                inputs = fporacle.make_inputs(launch, kind, seed=1)
                val, err = fporacle.emulate(spec.source, spec.fn_name, launch, inputs)
                ex, _ = fporacle.emulate(spec.source, spec.fn_name, launch, inputs, exact=True)
                for buf in launch.outputs:
                    d = np.abs(val[buf].astype(np.float64) - ex[buf])
                    assert np.all(d <= err[buf] + 1e-12 * np.maximum(1, np.abs(ex[buf]))), (name, size, kind)


def test_promotion_and_bf16_truncation():
    assert fporacle.promote("float16", "float32") == "float32"
    assert fporacle.promote("float16", "bfloat16") == "float32"
    # bf16 keeps 7 mantissa bits. Round-to-nearest would give 1 + 2^-6 here; truncation gives 1 + 2^-7.
    x = np.array([1.0 + 2**-7 + 2**-8 + 2**-10], dtype=np.float32)
    assert fporacle._to_bf16(x)[0] == 1.0 + 2**-7


def test_real_domain_abstract_flag_is_per_trace():
    from triton_equiv.fporacle import _RealDom, _real_memory, trace_fp
    sm, va = KERNELS["softmax"], KERNELS["vector_add"]
    dom = _RealDom()  # reused across traces, as real_counterexample does
    ls, lv = sm.make_launch(**sm.oracle_sizes[0]), va.make_launch(**va.oracle_sizes[0])
    assert trace_fp(sm.source, sm.fn_name, ls, _real_memory(ls), "a", dom).abstract
    assert not trace_fp(va.source, va.fn_name, lv, _real_memory(lv), "b", dom).abstract


def test_layernorm_split_sum_gets_a_witness_from_the_row_widened_search():
    # An fp32 reordering of the mean has to survive the final fp16 rounding: too rare for the
    # per-size search, found by running 256 rows per input and moving the differing row into row 0.
    # Site 0 needs about 360 inputs (2.5 min); the labels CI job covers it.
    spec, m = KERNELS["layernorm"], _mutant("layernorm", "split_sum", 1)
    size = {"n_rows": 2, "n_cols": 5}
    w, n = fporacle.search_witness_wide(spec, m.source, size, seeds=40)
    assert w is not None and w["size"] == size and w["widened_from_rows"] == 256
    launch = spec.make_launch(**size)
    inputs = {b: np.array(v, dtype=fporacle._np_dtype(launch.dtype(b))) for b, v in w["inputs"].items()}
    ref, _ = fporacle.emulate(spec.source, spec.fn_name, launch, inputs)
    mut, _ = fporacle.emulate(m.source, spec.fn_name, launch, inputs)
    assert fporacle._bitwise_diff(ref["out"], mut["out"]).size


def test_row_widened_search_skips_kernels_without_rows():
    spec = KERNELS["matmul"]
    assert fporacle.search_witness_wide(spec, spec.source, spec.fp_sizes[0]) == (None, 0)
