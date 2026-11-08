"""The C++ reference kernels (CPU build) against PyTorch, on every kernel and size.

Skipped when the extension cannot be built (no compiler or ninja). The CUDA build runs the
same header and is checked against the CPU build in notebooks/cuda_ref.ipynb."""

import pytest

torch = pytest.importorskip("torch")

from triton_equiv import cuda_ref, harness  # noqa: E402
from triton_equiv.kernels import KERNELS  # noqa: E402

_ERR = cuda_ref.build_error(cuda_ref.Backend.CPU)
pytestmark = pytest.mark.skipif(_ERR is not None, reason=f"C++ extension: {_ERR}")


def _cases():
    for name, spec in KERNELS.items():
        sizes = list(spec.oracle_sizes) + [s for ss in spec.checker_sizes.values() for s in ss]
        for size in sizes:
            yield name, size


@pytest.mark.parametrize("name,size", list(_cases()), ids=lambda v: v if isinstance(v, str) else
                         "-".join(f"{k}{x}" for k, x in v.items()))
def test_matches_torch(name, size):
    spec = KERNELS[name]
    launch = spec.make_launch(**size)
    inputs = harness.make_inputs(launch, seed=1)
    (out,) = launch.outputs
    exact = harness.fp64_reference(name, launch, inputs)[out]
    got64 = cuda_ref.cpp_reference(name, launch, inputs, cuda_ref.Acc.F64)[out]
    assert got64.dtype == torch.float64
    assert torch.allclose(got64, exact, rtol=1e-12, atol=1e-12), (name, size)
    got32 = cuda_ref.cpp_reference(name, launch, inputs, cuda_ref.Acc.F32)[out]
    ref32 = harness.torch_reference(name, launch, inputs)[out]
    assert got32.dtype == ref32.dtype
    for p in harness.TOLERANCES:  # the two references agree under every tolerance policy
        assert harness.close(got32, ref32, launch.dtype(out), p), (name, size, p)


def test_specs_cover_every_kernel():
    assert set(cuda_ref.SPECS) == set(KERNELS)
    for name, (op, bufs, names) in cuda_ref.SPECS.items():
        launch = KERNELS[name].make_launch(**KERNELS[name].oracle_sizes[0])
        assert set(bufs) == set(launch.buffers) - set(launch.outputs), name
        assert len(cuda_ref.sizes(name, launch)) == len(names), name
