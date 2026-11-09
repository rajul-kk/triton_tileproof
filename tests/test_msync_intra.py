"""Labels and mutants of the single-kernel synchronisation suite (``msync.intra``); the GPU
runs are in ``notebooks/msync_sanitizers.ipynb``."""

import pytest

from triton_equiv.msync import intra


def test_references_are_race_free():
    for k in intra.KERNELS:
        assert intra.hb_label(k)["label"] == "benign"


EXPECTED = {
    "reduce:pre_load": ("benign", None), "reduce:after_load": ("racy", "cross_warp"),
    "reduce:after_128": ("racy", "cross_warp"), "reduce:after_64": ("racy", "cross_warp"),
    "reduce:warp32_war": ("benign", None), "reduce:warp32_raw": ("racy", "intra_warp"),
    "reduce:warp16_war": ("racy", "intra_warp"), "reduce:warp16_raw": ("racy", "intra_warp"),
    "reduce:warp1_war": ("racy", "intra_warp"), "reduce:warp1_raw": ("benign", None),
    "reduce:warp_sync_free": ("racy", "intra_warp"), "matmul:after_load": ("racy", "mixed"),
    "matmul:before_next_load": ("racy", "mixed"), "matmul:after_loop": ("benign", None),
    "stencil:after_load": ("racy", "mixed"), "stencil:before_store": ("benign", None),
    "scan:after_load": ("racy", "mixed"), "scan:step1_war": ("racy", "mixed"),
    "scan:step1_raw": ("racy", "mixed"), "scan:step64_war": ("racy", "cross_warp"),
    "scan:step64_raw": ("racy", "cross_warp"), "scan:step128_raw": ("benign", None),
}


@pytest.mark.parametrize("m", intra.MUTANTS, ids=lambda m: m.id)
def test_happens_before_label(m):
    lab = intra.hb_label(m.kernel, m.mask)
    assert (lab["label"], lab["scope"]) == EXPECTED[m.id]


def test_mutant_ids_are_unique_and_drop_only_their_bits():
    assert len({m.id for m in intra.MUTANTS}) == len(intra.MUTANTS) == len(EXPECTED)
    m = next(m for m in intra.MUTANTS if m.id == "scan:after_load")
    assert m.mask == intra.ALL & ~(1 << 16)


def test_dropping_every_barrier_of_a_kernel_is_racy():
    assert intra.hb_label("matmul", 0)["label"] == "racy"
