"""Reference outputs from plain C++/CUDA kernels, independent of both PyTorch and Triton.

``harness.torch_reference`` is what a benchmark's PyTorch model returns. This is a second
reference written from the kernels' specifications in ``kernels.py``: sequential loops in
``csrc/ref_kernels.h``, built either for the CPU (``csrc/ref_cpu.cpp``, testable without a
GPU) or for CUDA (``csrc/ref_cuda.cu``, one thread per output unit, no tiling, no shared
memory, no tensor cores). Both builds run the same ``__host__ __device__`` functions.

Arithmetic: fp16 inputs are widened to float32 exactly, the kernel computes in ``Acc``
(``F32`` or ``F64``) in index order, and an ``F32`` result is rounded to the output buffer's
dtype as the Triton kernel's store would; an ``F64`` result stays float64 (like
``harness.fp64_reference``).

Builds use ``torch.utils.cpp_extension.load_inline`` and need ninja and a C++ compiler
(plus nvcc for ``CUDA``). One build per backend per process, cached by backend.
"""

from __future__ import annotations

import enum
import functools
import os
import sys
from pathlib import Path

CSRC = Path(__file__).with_name("csrc")


class Backend(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"


class Acc(enum.Enum):
    F32 = "float32"
    F64 = "float64"


# kernel name -> (op in ref_*.cpp/cu, input buffer names in order, size names)
SPECS = {
    "vector_add": ("add", ("x", "y"), ("n",)),
    "row_sum": ("rowsum", ("x",), ("n_rows", "n_cols")),
    "row_sum_loop": ("rowsum", ("x",), ("n_rows", "n_cols")),
    "matmul": ("matmul", ("a", "b"), ("M", "N", "K")),
    "matmul_fp16": ("matmul", ("a", "b"), ("M", "N", "K")),
    "softmax": ("softmax", ("x",), ("n_rows", "n_cols")),
    "layernorm": ("layernorm", ("x",), ("n_rows", "n_cols")),
    "attention": ("attention", ("q", "k", "v"), ("M", "N", "D")),
}


def sizes(kernel: str, launch) -> list:
    """The size names of ``SPECS[kernel]`` read off a launch. Row-wise launches have no
    ``n_rows`` argument: it is the length of ``x`` over ``n_cols``."""
    a = dict(launch.args)
    if "n_cols" in a:
        a["n_rows"] = launch.buffers["x"] // a["n_cols"]
    return [int(a[k]) for k in SPECS[kernel][2]]


_DECL = ("at::Tensor run(const std::string& op, const std::vector<at::Tensor>& t, "
         "const std::vector<int64_t>& n, double eps, bool use_double);")


def ensure_ninja_on_path():
    """pip installs ninja into the interpreter's bin directory, which is often not on PATH
    (a venv used without activating it); put it there so ``load_inline`` finds it."""
    bindir = str(Path(sys.executable).parent)
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")


@functools.lru_cache(maxsize=None)
def module(backend: Backend):
    """The compiled extension for ``backend`` (built on first use)."""
    from torch.utils.cpp_extension import load_inline
    ensure_ninja_on_path()
    common = dict(functions=["run"], extra_include_paths=[str(CSRC)], extra_cflags=["-O2"], verbose=False)
    if backend is Backend.CPU:
        return load_inline("tileproof_ref_cpu", cpp_sources=[(CSRC / "ref_cpu.cpp").read_text()], **common)
    return load_inline("tileproof_ref_cuda", cpp_sources=[_DECL], cuda_sources=[(CSRC / "ref_cuda.cu").read_text()],
                       extra_cuda_cflags=["-O2"], **common)


def build_error(backend: Backend) -> str | None:
    """None if ``backend`` builds, else why not (for skip messages and notebooks)."""
    try:
        module(backend)
        return None
    except Exception as e:  # missing compiler, ninja or nvcc, or a compile error
        return f"{type(e).__name__}: {str(e)[:300]}"


def cpp_reference(kernel: str, launch, inputs: dict, acc: Acc = Acc.F32, backend: Backend = Backend.CPU) -> dict:
    """``{output buffer: tensor}`` on the CPU, like ``harness.torch_reference``."""
    import torch
    op, bufs, _ = SPECS[kernel]
    dev = "cuda" if backend is Backend.CUDA else "cpu"
    ts = [inputs[b].to(device=dev, dtype=torch.float32).contiguous() for b in bufs]
    n = sizes(kernel, launch)
    eps = float(launch.args.get("eps", 0.0))
    y = module(backend).run(op, ts, n, eps, acc is Acc.F64).cpu()
    (out,) = launch.outputs
    if acc is Acc.F32:
        y = y.to({"float16": torch.float16, "float32": torch.float32}[launch.dtype(out)])
    return {out: y}
