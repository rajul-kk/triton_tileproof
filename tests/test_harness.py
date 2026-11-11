"""Checker-audit harness: confusion bookkeeping (anywhere) and a small interpreter run (Linux/WSL)."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

from triton_equiv import harness


def test_confusion_counts_tile_and_odd_and_canary():
    rows = [
        {"id": "a", "oracle": "oob", "guard_write": True,
         "verdict": {"tile": {"kernelbench": "accept", "torch_default": "accept"},
                     "odd": {"kernelbench": "accept", "torch_default": "reject"}}},
        {"id": "b", "oracle": "numerical:precision", "guard_write": False,
         "verdict": {"tile": {"kernelbench": "accept", "torch_default": "reject"},
                     "odd": {"kernelbench": "accept", "torch_default": "reject"}}},
    ]
    c = harness.confusion(rows)
    assert c["tile/kernelbench"]["oob"] == {"accept": 1, "reject": 0}
    assert c["tile+odd/torch_default"]["oob"] == {"accept": 0, "reject": 1}
    assert c["tile+odd/kernelbench+canary"]["oob"] == {"accept": 0, "reject": 1}
    assert c["tile/torch_default"]["numerical:precision"] == {"accept": 0, "reject": 1}


def test_tritongym_table_only_when_every_row_has_its_verdict():
    rows = [{"id": "a", "oracle": "oob", "guard_write": False, "tritongym": "accept",
             "verdict": {"tile": {"kernelbench": "accept", "torch_default": "accept"},
                         "odd": {"kernelbench": "accept", "torch_default": "accept"}}}]
    assert harness.confusion(rows)["tritongym"]["oob"] == {"accept": 1, "reject": 0}
    del rows[0]["tritongym"]
    assert "tritongym" not in harness.confusion(rows)


def test_tritongym_rule_ignores_nan_and_rejects_inf():
    torch = pytest.importorskip("torch")
    ref = torch.zeros(4)
    assert harness.tritongym_close(ref + 0.009, ref)
    assert not harness.tritongym_close(ref + 0.02, ref)
    assert harness.tritongym_close(torch.tensor([float("nan"), 0.0, 0.0, 0.0]), ref)
    assert not harness.tritongym_close(torch.tensor([float("inf"), 0.0, 0.0, 0.0]), ref)


def test_family_names():
    assert harness._family(None) == "reference"
    assert harness._family({"label": "numerical", "precision_loss": True}) == "numerical:precision"
    assert harness._family({"label": "benign", "fp_proof": None}) == "benign:unproved"


SCRIPT = textwrap.dedent('''
    import json
    from triton_equiv import harness
    from triton_equiv.kernels import KERNELS
    from triton_equiv.mutate import mutants
    out = {}
    for kname, op, kind in [("row_sum_loop", "acc_dtype_downgrade", "tile"),
                            ("vector_add", "missing_store_mask", "odd")]:
        spec = KERNELS[kname]
        m = next(mutants(kname, spec.source, [op]))
        launch = spec.make_launch(**spec.checker_sizes[kind][0])
        inputs = harness.make_inputs(launch, 0)
        o, guard, err = harness.run(spec, m.source, launch, inputs)
        ref = harness.torch_reference(kname, launch, inputs)
        b = launch.outputs[0]
        out[m.id] = {p: harness.close(o[b], ref[b], launch.dtype(b), p) for p in harness.TOLERANCES}
        out[m.id]["guard"], out[m.id]["err"] = guard, err
    print(json.dumps(out))
''')


@pytest.mark.skipif(importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None,
                    reason="triton/torch not installed (run under WSL/Linux)")
def test_allclose_checker_misses_precision_loss_and_silent_oob_store():
    env = dict(os.environ, TRITON_INTERPRET="1",
               PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    res = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    # fp16 accumulation over 256 elements: inside KernelBench's 1e-2, outside torch's defaults.
    acc = out["row_sum_loop:acc_dtype_downgrade:0"]
    assert acc["kernelbench"] is True and acc["torch_default"] is False and acc["err"] is None
    # Unmasked store at n=70: correct output, so both tolerances accept; only the guard sees it.
    st = out["vector_add:missing_store_mask:0"]
    assert st["kernelbench"] is True and st["torch_default"] is True and st["guard"] is True


def test_label_counts_use_audit_families():
    import json
    from collections import Counter

    from triton_equiv.families import FAMILIES, family
    doc = json.load(open("results/labels.json"))
    assert doc["counts"] == dict(Counter(family(lab) for lab in doc["labels"]))
    assert set(doc["counts"]) <= set(FAMILIES)


def test_kernelgym_table_sets_apart_mutants_that_do_not_compile():
    from triton_equiv import kgym_table
    t = kgym_table.table(json.load(open("results/kernelgym_audit_t4.json"))["rows"])
    assert t["does not compile"]["ids"] == ["matmul:acc_dtype_downgrade:0", "matmul_fp16:acc_dtype_downgrade:0"]
    assert t["numerical:precision"] == {"n": 5, "tile/explicit_default": 0, "tile/kernelbench": 5,
                                        "odd/explicit_default": 0, "odd/kernelbench": 5}
    assert t["benign:proved"]["n"] == 6 and t["oob"]["tile/kernelbench"] == 27
