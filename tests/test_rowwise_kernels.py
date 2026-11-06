"""Softmax, layernorm and attention: labels known by hand analysis.

These kernels use exp, sqrt, max and division, which the symbolic tiers treat as uninterpreted
functions. Equal terms are a proof; a difference is provisional until an fp64 run confirms it.
"""

import pytest

pytest.importorskip("z3")

from triton_equiv import fporacle, oracle  # noqa: E402
from triton_equiv.kernels import KERNELS  # noqa: E402
from triton_equiv.mutate import mutants  # noqa: E402


def _mutant(kernel, op, site=0):
    return next(m for m in mutants(kernel, KERNELS[kernel].source, [op]) if m.site == site)


def _label(kernel, op, site=0):
    m = _mutant(kernel, op, site)
    return fporacle.label(KERNELS[kernel], m.source, m.id)


def test_mutant_counts():
    assert {k: len(list(mutants(k, KERNELS[k].source))) for k in ("softmax", "layernorm", "attention")} == \
        {"softmax": 5, "layernorm": 13, "attention": 11}


def test_softmax_fill_bug_hidden_at_tile_size_and_confirmed_in_fp64():
    # other=-inf -> 0.0: padding lanes add exp(0 - max) to the denominator, only with a tail.
    spec, m = KERNELS["softmax"], _mutant("softmax", "wrong_other")
    assert oracle.label_at(spec, m.source, {"n_rows": 2, "n_cols": 8}).label == "benign"
    v = oracle.label_at(spec, m.source, {"n_rows": 2, "n_cols": 5})
    assert v.label == "inequivalent" and v.witness["abstract"] is True
    lab = fporacle.label(spec, m.source, m.id)
    assert lab.label == "inequivalent" and lab.witness["fp64"]["size"] == {"n_rows": 2, "n_cols": 5}


def test_softmax_split_sum_is_a_reordering():
    lab = _label("softmax", "split_sum")
    assert lab.label == "numerical" and lab.precision_loss is False and lab.real_proof


@pytest.mark.parametrize("op,site", [("divisor_block_size", 0), ("divisor_block_size", 1),
                                     ("drop_where_mask", 0), ("wrong_other", 0)])
def test_layernorm_tail_bugs(op, site):
    spec, m = KERNELS["layernorm"], _mutant("layernorm", op, site)
    assert oracle.label_at(spec, m.source, {"n_rows": 2, "n_cols": 8}).label == "benign"
    assert fporacle.label(spec, m.source, m.id).label == "inequivalent"


def test_layernorm_casts():
    # Output cast is redundant (tl.store rounds to fp16): proved bitwise identical.
    lab = _label("layernorm", "drop_cast", 1)
    assert lab.label == "benign" and lab.fp_proof == "uf"
    # Dropping the fp32 upcast keeps x, x - mean and diff * rstd in fp16.
    for op, site in [("drop_cast", 0), ("change_cast", 0), ("change_cast", 1)]:
        lab = _label("layernorm", op, site)
        assert lab.label == "numerical" and lab.precision_loss is True, (op, site)


def test_attention_labels():
    spec = KERNELS["attention"]
    labels = {m.id: fporacle.label(spec, m.source, m.id) for m in mutants("attention", spec.source)
              if m.operator in ("mask_off_by_one", "wrong_other", "drop_where_mask")}
    # Four of the five comparisons guard loads/stores (oob); the fifth is the -inf key mask.
    assert [labels[f"attention:mask_off_by_one:{i}"].label for i in range(5)] == \
        ["oob", "oob", "oob", "inequivalent", "oob"]
    assert labels["attention:drop_where_mask:0"].label == "inequivalent"
    # q and k fills never reach a stored output: proved.
    for i in (0, 1):
        lab = labels[f"attention:wrong_other:{i}"]
        assert lab.label == "benign" and lab.fp_proof == "uf"
    # v fill meets p = exp(-inf) = 0. Under uninterpreted exp that is not provable, and fp64
    # does not confirm a difference: benign, unproved.
    lab = labels["attention:wrong_other:2"]
    assert lab.label == "benign" and lab.fp_proof is None and lab.real_proof is False


def test_interpreter_dtype_rule():
    r = fporacle.result_dtype
    assert r("float16", False, "float32", True, "sub") == "float16"  # tile - 0-d mean
    assert r("float32", True, "float16", False, "mul") == "float16"
    assert r("float16", True, "float32", True, "add") == "float32"   # both 0-d: promote
    assert r("float16", False, "float16", True, "div") == "float32"  # fp16 division -> fp32
