"""Reference Triton kernels for the stage-1 mutation audit.

Each kernel is stored as *source text* (no imports, ``@triton.jit`` decorator kept)
rather than as an imported function. One string then feeds three consumers:

* ``mutate``: parses it with ``ast`` and emits mutant source,
* ``oracle``: ``exec``s it against a symbolic ``tl`` shim (no Triton needed),
* real Triton (``TRITON_INTERPRET=1`` or GPU): ``to_module`` writes it to a file
  and imports it, because ``triton.jit`` needs ``inspect.getsource``.

The index-logic kernels (all buffers fp32):

``vector_add``  1-D masked add with a tail (N need not be a multiple of BLOCK).
``row_sum``     one program per row, masked row load with ``other=0.0``, ``tl.sum``.
``matmul``      2-D grid, K-loop, masked tile loads with ``other=0.0``, ``tl.dot``.

The mixed-precision kernels, added for the floating-point mutants (``fporacle``):

``row_sum_loop``  fp16 rows reduced in a K-loop into an fp32 accumulator tile
                  (``acc += x`` promotes fp16 to fp32), then ``tl.sum``. Output fp32.
``matmul_fp16``   fp16 inputs, ``acc = tl.dot(a, b, acc)`` with an fp32 accumulator,
                  explicit ``acc.to(tl.float16)`` before the store to an fp16 output.

Both are written so that each fp bug is one source edit. With ``acc += x`` (implicit
promotion) an fp16 ``tl.zeros`` really accumulates in fp16; with ``acc = tl.dot(a, b, acc)``
the accumulator dtype sets ``out_dtype``. (``acc += tl.dot(a, b)`` with an fp16 ``acc``
would instead be promoted back to fp32 by the interpreter.)

Tile sizes in ``oracle_sizes`` are deliberately tiny (BLOCK=4). The symbolic oracle
does not care, but real Triton requires ``tl.dot`` tiles >= 16, so the matmuls only run
in real Triton with ``BM=BN=BK>=16`` (see ``checker_sizes``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Launch:
    """One concrete launch: grid, kernel arguments and buffer lengths (in elements)."""

    grid: tuple[int, ...]
    args: dict  # kernel parameter name -> int (scalars) or buffer name (pointers)
    buffers: dict[str, int]  # buffer name -> number of elements
    dtypes: dict[str, str] = field(default_factory=dict)  # buffer -> "float16" etc.; default float32
    outputs: tuple[str, ...] = ()  # buffers the kernel writes (the rest are inputs)

    def dtype(self, buf: str) -> str:
        return self.dtypes.get(buf, "float32")


@dataclass(frozen=True)
class KernelSpec:
    name: str
    fn_name: str
    source: str
    make_launch: Callable[..., Launch]
    oracle_sizes: list[dict] = field(default_factory=list)
    # Sizes for real Triton runs (harness). "tile": every dimension a tile multiple;
    # "odd": partial last tiles. Matmul tiles are 16 because tl.dot needs >= 16.
    checker_sizes: dict[str, list[dict]] = field(default_factory=dict)
    # Name of the size parameter that is the reduction length (for the blind-band sweep).
    reduction_dim: str | None = None


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# --------------------------------------------------------------------------- vector add
VECTOR_ADD_SRC = '''
@triton.jit
def vector_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
'''


def _vector_add_launch(n: int, BLOCK: int = 4) -> Launch:
    return Launch(
        grid=(cdiv(n, BLOCK),),
        args={"x_ptr": "x", "y_ptr": "y", "out_ptr": "out", "n": n, "BLOCK": BLOCK},
        buffers={"x": n, "y": n, "out": n},
        outputs=("out",),
    )


# --------------------------------------------------------------------------- row sum
ROW_SUM_SRC = '''
@triton.jit
def row_sum(x_ptr, out_ptr, n_cols, row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * row_stride + cols, mask=cols < n_cols, other=0.0)
    tl.store(out_ptr + row, tl.sum(x, axis=0))
'''


def _row_sum_launch(n_rows: int, n_cols: int, BLOCK: int = 8) -> Launch:
    assert n_cols <= BLOCK, "row_sum loads a whole row in one tile"
    return Launch(
        grid=(n_rows,),
        args={"x_ptr": "x", "out_ptr": "out", "n_cols": n_cols, "row_stride": n_cols, "BLOCK": BLOCK},
        buffers={"x": n_rows * n_cols, "out": n_rows},
        outputs=("out",),
    )


# --------------------------------------------------------------------------- matmul
MATMUL_SRC = '''
@triton.jit
def matmul(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        rk = k * BK + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
'''


def _matmul_launch(M: int, N: int, K: int, BM: int = 4, BN: int = 4, BK: int = 4) -> Launch:
    return Launch(
        grid=(cdiv(M, BM), cdiv(N, BN)),
        args={"a_ptr": "a", "b_ptr": "b", "c_ptr": "c", "M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
        buffers={"a": M * K, "b": K * N, "c": M * N},
        outputs=("c",),
    )


# --------------------------------------------------------------------------- row sum, K-loop
ROW_SUM_LOOP_SRC = '''
@triton.jit
def row_sum_loop(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(0, tl.cdiv(n_cols, BLOCK)):
        cols = k * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=0.0)
        acc += x
    tl.store(out_ptr + row, tl.sum(acc, axis=0))
'''


def _row_sum_loop_launch(n_rows: int, n_cols: int, BLOCK: int = 4) -> Launch:
    return Launch(
        grid=(n_rows,),
        args={"x_ptr": "x", "out_ptr": "out", "n_cols": n_cols, "BLOCK": BLOCK},
        buffers={"x": n_rows * n_cols, "out": n_rows},
        dtypes={"x": "float16", "out": "float32"},
        outputs=("out",),
    )


# --------------------------------------------------------------------------- fp16 matmul
MATMUL_FP16_SRC = '''
@triton.jit
def matmul_fp16(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        rk = k * BK + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    c = acc.to(tl.float16)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c, mask=(rm[:, None] < M) & (rn[None, :] < N))
'''


def _matmul_fp16_launch(M: int, N: int, K: int, BM: int = 4, BN: int = 4, BK: int = 4) -> Launch:
    L = _matmul_launch(M, N, K, BM, BN, BK)
    return Launch(grid=L.grid, args=L.args, buffers=L.buffers,
                  dtypes={"a": "float16", "b": "float16", "c": "float16"}, outputs=("c",))


_T16 = {"BM": 16, "BN": 16, "BK": 16}
_MATMUL_CHECKER_SIZES = {"tile": [{"M": 32, "N": 32, "K": 64, **_T16}],
                         "odd": [{"M": 36, "N": 20, "K": 70, **_T16}]}

KERNELS: dict[str, KernelSpec] = {
    "vector_add": KernelSpec(
        "vector_add", "vector_add", VECTOR_ADD_SRC, _vector_add_launch,
        # 8 is a tile multiple; 6 and 9 leave a partial last tile.
        oracle_sizes=[{"n": 8}, {"n": 6}, {"n": 9}],
        checker_sizes={"tile": [{"n": 64, "BLOCK": 16}], "odd": [{"n": 70, "BLOCK": 16}]},
    ),
    "row_sum": KernelSpec(
        "row_sum", "row_sum", ROW_SUM_SRC, _row_sum_launch,
        # n_cols == BLOCK has no masked lanes; n_cols=5 < BLOCK=8 does.
        oracle_sizes=[{"n_rows": 2, "n_cols": 8}, {"n_rows": 2, "n_cols": 5}],
        checker_sizes={"tile": [{"n_rows": 4, "n_cols": 64, "BLOCK": 64}],
                       "odd": [{"n_rows": 4, "n_cols": 50, "BLOCK": 64}]},
        reduction_dim="n_cols",
    ),
    "matmul": KernelSpec(
        "matmul", "matmul", MATMUL_SRC, _matmul_launch,
        # (4,4,4): one tile; (6,5,7): square 2x2 grid with tails; (9,5,7): non-square 3x2 grid.
        oracle_sizes=[{"M": 4, "N": 4, "K": 4}, {"M": 6, "N": 5, "K": 7}, {"M": 9, "N": 5, "K": 7}],
        checker_sizes=_MATMUL_CHECKER_SIZES,
        reduction_dim="K",
    ),
    "row_sum_loop": KernelSpec(
        "row_sum_loop", "row_sum_loop", ROW_SUM_LOOP_SRC, _row_sum_loop_launch,
        # (2, 8): two full tiles per row; (2, 6) and (2, 11): partial last tile.
        oracle_sizes=[{"n_rows": 2, "n_cols": 8}, {"n_rows": 2, "n_cols": 6}, {"n_rows": 2, "n_cols": 11}],
        checker_sizes={"tile": [{"n_rows": 4, "n_cols": 256, "BLOCK": 16}],
                       "odd": [{"n_rows": 4, "n_cols": 250, "BLOCK": 16}]},
        reduction_dim="n_cols",
    ),
    "matmul_fp16": KernelSpec(
        "matmul_fp16", "matmul_fp16", MATMUL_FP16_SRC, _matmul_fp16_launch,
        oracle_sizes=[{"M": 4, "N": 4, "K": 8}, {"M": 6, "N": 5, "K": 7}, {"M": 9, "N": 5, "K": 7}],
        checker_sizes=_MATMUL_CHECKER_SIZES,
        reduction_dim="K",
    ),
}


def to_module(source: str, fn_name: str, directory: str | None = None):
    """Import ``source`` as a real Triton module and return the jitted function.

    Requires ``triton`` (Linux/WSL). Set ``TRITON_INTERPRET=1`` *before* importing
    triton to run on CPU.
    """
    d = Path(directory or tempfile.mkdtemp(prefix="triton_equiv_"))
    digest = hashlib.sha1(source.encode()).hexdigest()[:10]
    path = d / f"_k_{fn_name}_{digest}.py"
    path.write_text("import triton\nimport triton.language as tl\n" + source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)
