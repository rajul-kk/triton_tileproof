"""Mutants whose label is known by hand analysis; the oracle must reproduce it."""

import pytest

pytest.importorskip("z3")

from triton_equiv import oracle  # noqa: E402
from triton_equiv.kernels import KERNELS  # noqa: E402
from triton_equiv.mutate import OPERATORS, count_sites, mutants  # noqa: E402


def _mutant(kernel, op, site):
    return next(m for m in mutants(kernel, KERNELS[kernel].source, [op]) if m.site == site)


@pytest.mark.parametrize("kernel", list(KERNELS))
def test_reference_is_benign_against_itself(kernel):
    spec = KERNELS[kernel]
    assert oracle.label(spec, spec.source).label == "benign"


def test_site_counts():
    assert [count_sites(KERNELS["matmul"].source, op) for op in OPERATORS] == [6, 1, 2, 2]
    assert [count_sites(KERNELS["row_sum"].source, op) for op in OPERATORS] == [1, 0, 1, 1]


def test_missing_store_mask_oob_only_on_non_multiple_size():
    spec, m = KERNELS["vector_add"], _mutant("vector_add", "missing_store_mask", 0)
    assert oracle.label_at(spec, m.source, {"n": 8}).label == "benign"  # tile multiple
    v = oracle.label_at(spec, m.source, {"n": 6})
    assert v.label == "oob"
    assert v.witness["access"] == "store" and v.witness["address"] == 6 and v.witness["pid"] == (1,)


def test_mask_off_by_one_is_oob_with_tail():
    spec, m = KERNELS["vector_add"], _mutant("vector_add", "mask_off_by_one", 0)
    assert oracle.label(spec, m.source).label == "oob"


def test_other_change_on_sum_is_inequivalent_with_counterexample():
    spec, m = KERNELS["row_sum"], _mutant("row_sum", "wrong_other", 0)
    # n_cols == BLOCK: no masked lanes, so the fill value is never read.
    assert oracle.label_at(spec, m.source, {"n_rows": 2, "n_cols": 8}).label == "benign"
    v = oracle.label_at(spec, m.source, {"n_rows": 2, "n_cols": 5})
    assert v.label == "inequivalent"
    assert v.witness["buffer"] == "out"
    # 3 masked lanes (BLOCK 8 - 5 cols) each add 1 instead of 0.
    assert (v.witness["mut_value"] - v.witness["ref_value"]) % 2**16 == 3


def test_other_change_where_masked_lanes_never_stored_is_benign():
    spec = KERNELS["vector_add"]
    for m in mutants("vector_add", spec.source, ["wrong_other"]):
        assert oracle.label(spec, m.source).label == "benign"


def test_other_change_on_matmul_is_benign_because_masked_lanes_meet_zeros():
    spec = KERNELS["matmul"]
    for m in mutants("matmul", spec.source, ["wrong_other"]):
        assert oracle.label(spec, m.source).label == "benign"


def test_pid_axis_swap_benign_on_square_grid_inequivalent_otherwise():
    spec, m = KERNELS["matmul"], _mutant("matmul", "wrong_pid_axis", 0)
    assert oracle.label_at(spec, m.source, {"M": 4, "N": 4, "K": 4}).label == "benign"  # 1x1 grid
    v = oracle.label_at(spec, m.source, {"M": 9, "N": 5, "K": 7})
    assert v.label == "inequivalent" and v.witness["reason"] == "output not written"


def test_unsupported_construct_raises():
    src = "@triton.jit\ndef k(x_ptr, BLOCK: tl.constexpr):\n    v = tl.load(x_ptr + tl.arange(0, BLOCK))\n    if v:\n        pass\n"
    spec = KERNELS["vector_add"]
    launch = spec.make_launch(n=4)
    with pytest.raises(oracle.Unsupported):
        oracle.trace(src, "k", type(launch)((1,), {"x_ptr": "x", "BLOCK": 4}, {"x": 4}),
                     oracle._memory(launch), "t")


def test_fp_mutants_are_invisible_to_the_bit_vector_oracle():
    # Reordering, accumulator dtype and casts are all exact in 16-bit modular arithmetic,
    # so the bit-vector tier calls every fp mutant benign; fporacle has to separate them.
    from triton_equiv.mutate import FP_OPERATORS
    for name, spec in KERNELS.items():
        for m in mutants(name, spec.source, FP_OPERATORS):
            assert oracle.label(spec, m.source).label == "benign", m.id
