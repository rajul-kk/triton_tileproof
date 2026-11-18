"""The KernelAudit-style rule: Eq. 1 states, slot combination, and contract-adapted inputs."""

import pytest

torch = pytest.importorskip("torch")

from triton_equiv import kaudit


def _ref(n=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    r64 = torch.randn(n, generator=g, dtype=torch.float64)
    return r64.float(), r64


def test_reference_itself_passes():
    r, r64 = _ref()
    assert kaudit.leaf_check(r.clone(), r, r64) == "pass"


def test_large_error_fails():
    r, r64 = _ref()
    k = r.clone()
    k[3] += 1.0
    assert kaudit.leaf_check(k, r, r64) == "fail"


def test_error_inside_calibrated_band_passes():
    r, r64 = _ref()
    k = r.clone()
    k[3] += 5e-6 * abs(float(r[3])) + 1e-6  # representable in fp32, inside the 1e-5 relative floor
    assert not torch.equal(k, r)
    assert kaudit.leaf_check(k, r, r64) == "pass"


def test_nonfinite_candidate_fails_even_when_calibration_is_capped():
    r64 = torch.randn(1000, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    r = (r64 * (1 + 1e-3)).float()
    k = r.clone()
    k[0] = float("nan")
    assert kaudit.leaf_check(k, r, r64) == "fail"


def test_nonfinite_candidate_fails_against_finite_reference():
    r, r64 = _ref()
    k = r.clone()
    k[0] = float("nan")
    assert kaudit.leaf_check(k, r, r64) == "fail"


def test_nonfinite_reference_is_unresolved():
    r, r64 = _ref()
    r[0] = float("inf")
    assert kaudit.leaf_check(r.clone(), r, r64) == "unresolved:ref_nonfinite"


def test_loose_calibration_is_unresolved_not_accepted():
    r64 = torch.randn(1000, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    r = (r64 * (1 + 1e-3)).float()  # native reference 1e-3 off fp64 everywhere: rho_bar = 4e-3 > 1e-3
    assert kaudit.leaf_check(r.clone(), r, r64) == "unresolved:rho_cap"


@pytest.mark.parametrize("states, want", [
    (["pass", "pass"], "accept"),
    (["accept", "unresolved"], "accept"),
    (["accept", "reject"], "reject"),
    (["pass", "fail"], "reject"),
    (["unresolved:rho_cap", "fail"], "reject"),
    (["unresolved:rho_cap", "pass"], "accept"),
    (["unresolved:rho_cap", "unresolved:input_range"], "unresolved"),
    ([], "unresolved"),
])
def test_combine(states, want):
    assert kaudit._combine(states) == want


def test_overflowing_scale_is_unresolved_input():
    from triton_equiv.kernels import KERNELS
    spec = KERNELS["matmul_fp16"]
    launch = spec.make_launch(**spec.checker_sizes["tile"][0])
    assert kaudit.slot_inputs(launch, "randn", 1e5, 0) is None
    assert kaudit.slot_inputs(launch, "randn", 1.0, 0) is not None


def test_sixteen_slots_with_one_repeated_run():
    assert len(kaudit.SLOTS) == 16
    assert [s[4] for s in kaudit.SLOTS].count(3) == 1
