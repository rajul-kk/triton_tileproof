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


def test_other_change_on_matmul_is_real_benign_but_not_structurally_proved():
    lab = _label("matmul", "wrong_other")
    assert lab.label == "benign" and lab.fp_proof is None and lab.witness is None


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
