"""The batch-invariance check (``batchinv``): verdicts, witnesses and row independence."""

import importlib.util

import numpy as np
import pytest

from triton_equiv import batchinv
from triton_equiv.batchinv import FAMILIES


@pytest.fixture(scope="module")
def results():
    return {n: batchinv.check(f, "sym", trials=6) for n, f in FAMILIES.items()}


@pytest.mark.parametrize("name", list(FAMILIES))
def test_verdict_matches_what_the_policy_was_written_to_show(results, name):
    r = results[name]
    if FAMILIES[name].expect == "invariant":
        assert r["verdict"] == "proved", r
        assert r["witness"] is None
    else:
        assert r["verdict"] == "refuted", r
        assert r["witness"] is not None, "terms differ but no input showed it"


def test_only_the_mixing_kernel_reads_other_rows(results):
    assert [n for n, r in results.items() if r["cross_row"]] == ["rowsum_mixed_rows"]
    c = results["rowsum_mixed_rows"]["cross_row"][0]
    assert c["from_row"] != c["row"]


def test_terms_read_the_row_they_belong_to():
    fam = FAMILIES["rowsum_loop_fixed"]
    terms, _ = batchinv.output_terms(fam, 3)
    for r in range(3):
        reads = {i for name, i in batchinv._selects(terms[r]) if name == "mem_x"}
        assert reads == set(range(r * 16, (r + 1) * 16))


def test_a_proof_is_not_vacuous_when_a_row_changes():
    """Proved means equal terms, not missing ones: the two rows of one batch differ."""
    terms, _ = batchinv.output_terms(FAMILIES["matmul_bm_by_batch"], 2)
    assert len(terms) == 16
    assert len({t.get_id() for t in terms.values()}) == 16


def test_witness_replays_in_the_emulator(results):
    fam = FAMILIES["splitk_by_batch"]
    w = results["splitk_by_batch"]["witness"]
    B1, B2 = fam.batches["sym"]
    big = batchinv._batch_inputs(fam, B2, "sym", w["generator"], w["seed"])
    small = batchinv._batch_inputs(fam, B1, "sym", w["generator"], w["seed"], base=big)
    n = fam.out_row("sym")
    i = w["row"] * n + w["col"]
    o1 = batchinv.emulate_family(fam, B1, "sym", small)
    o2 = batchinv.emulate_family(fam, B2, "sym", big)
    assert o1[i] != o2[i]
    assert np.float32(o1[i]) == np.float32(w["value_B1"])


def test_small_batch_inputs_are_the_first_rows_of_the_large_one():
    fam = FAMILIES["matmul_bk_by_batch"]
    big = batchinv._batch_inputs(fam, 5, "sym", "randn", 0)
    small = batchinv._batch_inputs(fam, 2, "sym", "randn", 0, base=big)
    assert np.array_equal(small["a"], big["a"][:2 * 16])
    assert np.array_equal(small["b"], big["b"])


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton not installed")
def test_interpreter_agrees_with_the_emulator_on_a_refuted_family():
    """The Triton CPU interpreter, run on the witness rows, shows the same difference."""
    import torch

    from triton_equiv.kernels import to_module
    fam = FAMILIES["rowsum_loop_block_by_batch"]
    w = batchinv.search_witness(fam, 2, 5, "sym", trials=6)
    big = batchinv._batch_inputs(fam, 5, "sym", w["generator"], w["seed"])
    small = batchinv._batch_inputs(fam, 2, "sym", w["generator"], w["seed"], base=big)
    st = fam.stages[0]
    fn = to_module(st.source, st.fn_name)
    outs = []
    for B, inp in ((2, small), (5, big)):
        launch = st.launch(B, "sym")
        x = torch.tensor(inp["x"], dtype=torch.float16)
        out = torch.zeros(B, dtype=torch.float32)
        fn[launch.grid](x, out, launch.args["n_cols"], BLOCK=launch.args["BLOCK"])
        outs.append(out.numpy())
    assert outs[0][w["row"]] == np.float32(w["value_B1"])
    assert outs[1][w["row"]] == np.float32(w["value_B2"])
