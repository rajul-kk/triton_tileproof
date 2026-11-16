"""U8 sentinel experiment: mutant selection, sentinel bookkeeping (anywhere) and an interpreter run (Linux/WSL)."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from triton_equiv import oobwrite


def _labels():
    return json.load(open("results/labels.json"))["labels"]


def test_store_and_load_oob_ids_split_the_oob_label():
    labels = _labels()
    st, ld = oobwrite.store_oob_ids(labels), oobwrite.load_oob_ids(labels)
    assert len(st) == 11 and len(ld) == 16
    assert "vector_add:missing_store_mask:0" in st and not set(st) & set(ld)


torch = pytest.importorskip("torch")


def test_adjacent_layout_counts_writes_on_each_side():
    from triton_equiv import harness
    from triton_equiv.kernels import KERNELS

    spec = KERNELS["vector_add"]
    launch = spec.make_launch(**spec.checker_sizes["odd"][0])
    views, sents, snaps, gap = oobwrite.place(torch, launch, harness.make_inputs(launch, 0), "adjacent", "cpu", 0)
    assert gap is None and views["out"].shape == (70,)
    assert oobwrite.changed(torch, sents, snaps) == {"out": [0, 0]}
    base = views["out"]
    torch.as_strided(base, (3,), (1,), base.storage_offset() + 70).fill_(7.0)  # 3 elements past the end
    assert oobwrite.changed(torch, sents, snaps) == {"out": [0, 3]}


def test_pool_layout_records_the_gap_and_rejects_unknown_layouts():
    from triton_equiv import harness
    from triton_equiv.kernels import KERNELS

    spec = KERNELS["vector_add"]
    launch = spec.make_launch(**spec.checker_sizes["odd"][0])
    _, sents, snaps, gap = oobwrite.place(torch, launch, harness.make_inputs(launch, 0), "pool", "cpu", 0)
    assert isinstance(gap, int) and oobwrite.changed(torch, sents, snaps) == {"out": [0]}
    with pytest.raises(ValueError, match="not in"):
        oobwrite.place(torch, launch, {}, "elsewhere", "cpu", 0)


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton not installed (run under WSL/Linux)")
def test_unmasked_store_is_accepted_but_overwrites_the_neighbour(tmp_path):
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    rows = {}
    for mid in ("vector_add:ref", "vector_add:missing_store_mask:0"):
        out = tmp_path / "r.json"
        args = [mid, "--trials", "1", "--device", "cpu", "--layouts", "adjacent", "--out", str(out)]
        res = subprocess.run([sys.executable, "-m", "triton_equiv.oobwrite", *args], env=env, capture_output=True,
                             text=True, timeout=600, check=False)
        assert res.returncode == 0, res.stderr[-2000:]
        rows[mid] = {r["layout"]: r for r in json.loads(out.read_text())}
    ref, mut = rows["vector_add:ref"]["adjacent"], rows["vector_add:missing_store_mask:0"]["adjacent"]
    assert ref["error"] is None and all(ref["accept"].values()) and ref["sentinel_changed"] == {"out": [0, 0]}
    # n = 70 with 16-wide tiles: the last tile stores 10 lanes past the end, and the output still matches.
    assert all(mut["accept"].values()) and mut["sentinel_changed"] == {"out": [0, 10]}


def test_summary_aggregates_trials_per_id():
    rows = [
        {
            "id": "m",
            "layout": "adjacent",
            "error": None,
            "gap_bytes": None,
            "accept": {"kb": True, "td": False},
            "sentinel_changed": {"out": [0, 10]},
        },
        {
            "id": "m",
            "layout": "pool",
            "error": None,
            "gap_bytes": 232,
            "accept": {"kb": True, "td": False},
            "sentinel_changed": {"out": [0]},
        },
        {
            "id": "m",
            "layout": "pool",
            "error": "RuntimeError: x",
            "gap_bytes": 232,
            "accept": {"kb": False, "td": False},
            "sentinel_changed": None,
        },
    ]
    s = oobwrite.summary(rows)["m"]
    assert s["accept"] == {"kb": 2 / 3, "td": 0.0} and s["adjacent_written"] == 10
    assert s["pool_hit"] == 0.0 and s["pool_gap_bytes"] == [232] and s["errors"] == 1
