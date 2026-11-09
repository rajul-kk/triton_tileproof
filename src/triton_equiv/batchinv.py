"""Batch invariance: is a row's output bitwise the same whatever the batch size?

A kernel computes each row of a batch from that row alone, so mathematically row ``r`` of
the output does not depend on how many other rows are in the batch. In floating point it
can: libraries pick tile sizes, split-K factors and rows per program from the batch size,
and a different reduction order rounds differently. That is the batch-size
nondeterminism of LLM inference (Thinking Machines, 2025). Batch-invariant kernels already
exist (TBIK, arXiv 2511.17826; vLLM's ``VLLM_BATCH_INVARIANT``); this module *checks* a
kernel and its launch policy statically, using the uninterpreted-function tier of
``fporacle`` (check 2 there).

A **family** is Triton source (one or more stages, run in order, sharing buffers) plus a
**launch policy**: a function of the batch size ``B`` that picks the grid and the
constexprs, as a library's heuristics would. Batch rows are stored row-major at the same
offsets for every ``B``, so input cell ``(r, c)`` is the same Z3 term at every batch size.

Verdicts, per pair of batch sizes ``B1 < B2`` (rows ``r < B1`` compared):

``proved``    every output term of every shared row is syntactically identical with
              rounded operations uninterpreted (``fadd_float32``, ``fsum_float32_8``,
              ``fdot_float32_4``, ...). The rows are then bitwise equal under any semantics
              in which each primitive is a deterministic function of its operands and of
              its length. That is fporacle's assumption; on a GPU it can fail when the same
              reduction length is laid out differently (for example a 1-row and a 4-row
              tile), which ``gpu_check`` tests on hardware.
``refuted``   some term differs. The numpy emulator (the Triton interpreter's arithmetic)
              then looks for a **witness**: rows that, put in a batch of ``B1`` and in a
              batch of ``B2`` (the same first rows, the same shared inputs), give
              different bits. ``witness=None`` means the terms differ but no input tried
              showed it.

Separately, **row independence**: every output term of row ``r`` may read only row ``r`` of
each batch input (plus shared inputs such as weights). A term that reads another row is
reported under ``cross_row``; such a kernel is not batch invariant even in exact
arithmetic.

What this proves and does not: only at the listed sizes and batch pairs; relative to the
deterministic-primitive assumption above; nothing about atomics (not modelled by the
symbolic ``tl``) or about kernels whose policy changes with something other than ``B``.

Run ``python -m triton_equiv.batchinv --out results/batchinv.json`` (CPU, Z3), and with
``--gpu`` also the hardware check (needs a CUDA GPU and triton).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import z3

from . import oracle
from .fporacle import _UFDom, _bitwise_diff, _real_memory, emulate, make_inputs, trace_fp
from .kernels import KERNELS, MATMUL_SRC, ROW_SUM_LOOP_SRC, Launch, cdiv
from .oracle import IDX_BITS

# --------------------------------------------------------------------------- sources
ROWSUM_ROWS_SRC = '''
@triton.jit
def rowsum_rows(x_ptr, out_ptr, n_rows, n_cols, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + rows[:, None] * n_cols + cols[None, :],
                mask=(rows[:, None] < n_rows) & (cols[None, :] < n_cols), other=0.0)
    tl.store(out_ptr + rows, tl.sum(x, axis=1), mask=rows < n_rows)
'''

SPLITK_SRC = '''
@triton.jit
def matmul_splitk(a_ptr, b_ptr, p_ptr, M, N, K, KS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    s = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(KS, BK)):
        rk = s * KS + k * BK + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(p_ptr + s * M * N + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
'''

SPLITK_REDUCE_SRC = '''
@triton.jit
def splitk_reduce(p_ptr, c_ptr, MN, S: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(p_ptr + offs, mask=mask, other=0.0)
    for s in range(1, S):
        acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, acc, mask=mask)
'''

# A deliberately wrong kernel for the row-independence check: it sums a (ROWS, BLOCK)
# tile over both axes, so each row's output mixes in the other rows of its program.
ROWSUM_MIXED_SRC = '''
@triton.jit
def rowsum_mixed(x_ptr, out_ptr, n_rows, n_cols, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + rows[:, None] * n_cols + cols[None, :],
                mask=(rows[:, None] < n_rows) & (cols[None, :] < n_cols), other=0.0)
    t = tl.sum(tl.sum(x, axis=1), axis=0)
    tl.store(out_ptr + rows, tl.zeros((ROWS,), dtype=tl.float32) + t, mask=rows < n_rows)
'''


# --------------------------------------------------------------------------- families
@dataclass(frozen=True)
class Stage:
    source: str
    fn_name: str
    launch: Callable[[int, str], Launch]  # (batch size, "sym" | "gpu") -> Launch


@dataclass(frozen=True)
class Family:
    name: str
    stages: tuple
    batch_bufs: dict  # batch input buffer -> elements per row
    out: str  # output buffer, row-major over batch rows
    out_row: Callable[[str], int]  # size -> output elements per row
    policy: str  # the launch policy in words
    expect: str  # "invariant" or "variant": what the policy was written to show
    batches: dict = field(default_factory=lambda: {"sym": (2, 5), "gpu": (4, 32)})


def _rowsum_loop(block_of):
    def launch(B, size):
        n_cols = 16 if size == "sym" else 4096
        return Launch(grid=(B,), args={"x_ptr": "x", "out_ptr": "out", "n_cols": n_cols, "BLOCK": block_of(B, size)},
                      buffers={"x": B * n_cols, "out": B}, dtypes={"x": "float16", "out": "float32"},
                      outputs=("out",))
    return launch


def _rowsum_rows(rows_of, src_args=("x_ptr", "out_ptr")):
    def launch(B, size):
        n_cols = 8 if size == "sym" else 1024
        R = rows_of(B, size)
        return Launch(grid=(cdiv(B, R),),
                      args={src_args[0]: "x", src_args[1]: "out", "n_rows": B, "n_cols": n_cols, "ROWS": R,
                            "BLOCK": n_cols},
                      buffers={"x": B * n_cols, "out": B}, outputs=("out",))
    return launch


def _mm_dims(size):
    return (8, 16) if size == "sym" else (32, 64)  # N, K


def _matmul(tiles_of):
    def launch(B, size):
        N, K = _mm_dims(size)
        BM, BN, BK = tiles_of(B, size)
        return Launch(grid=(cdiv(B, BM), cdiv(N, BN)),
                      args={"a_ptr": "a", "b_ptr": "b", "c_ptr": "c", "M": B, "N": N, "K": K,
                            "BM": BM, "BN": BN, "BK": BK},
                      buffers={"a": B * K, "b": K * N, "c": B * N}, outputs=("c",))
    return launch


def _splitk(splits_of):
    def tiles(size):
        return (4, 4, 4) if size == "sym" else (16, 16, 16)

    def stage1(B, size):
        N, K = _mm_dims(size)
        BM, BN, BK = tiles(size)
        S = splits_of(B, size)
        return Launch(grid=(cdiv(B, BM), cdiv(N, BN), S),
                      args={"a_ptr": "a", "b_ptr": "b", "p_ptr": "p", "M": B, "N": N, "K": K, "KS": K // S,
                            "BM": BM, "BN": BN, "BK": BK},
                      buffers={"a": B * K, "b": K * N, "p": S * B * N}, outputs=("p",))

    def stage2(B, size):
        N, _ = _mm_dims(size)
        S = splits_of(B, size)
        BLOCK = 8 if size == "sym" else 128
        return Launch(grid=(cdiv(B * N, BLOCK),),
                      args={"p_ptr": "p", "c_ptr": "c", "MN": B * N, "S": S, "BLOCK": BLOCK},
                      buffers={"p": S * B * N, "c": B * N}, outputs=("c",))
    return (Stage(SPLITK_SRC, "matmul_splitk", stage1), Stage(SPLITK_REDUCE_SRC, "splitk_reduce", stage2))


def _rowwise_kernel(name):
    spec = KERNELS[name]

    def launch(B, size):
        n_cols, BLOCK = (5, 8) if size == "sym" else (1000, 1024)
        return spec.make_launch(n_rows=B, n_cols=n_cols, BLOCK=BLOCK)
    return Stage(spec.source, spec.fn_name, launch)


def _small(B, size):
    """The threshold the policies use: a small batch at either size."""
    return B <= (2 if size == "sym" else 4)


FAMILIES: dict[str, Family] = {f.name: f for f in [
    Family("softmax", (_rowwise_kernel("softmax"),), {"x": None}, "out", lambda s: 5 if s == "sym" else 1000,
           "one program per row, BLOCK from n_cols only", "invariant"),
    Family("layernorm", (_rowwise_kernel("layernorm"),), {"x": None}, "out", lambda s: 5 if s == "sym" else 1000,
           "one program per row, BLOCK from n_cols only", "invariant"),
    Family("rowsum_loop_fixed",
           (Stage(ROW_SUM_LOOP_SRC, "row_sum_loop", _rowsum_loop(lambda B, s: 4 if s == "sym" else 128)),),
           {"x": None}, "out", lambda s: 1, "K-loop tile fixed", "invariant"),
    Family("rowsum_loop_block_by_batch",
           (Stage(ROW_SUM_LOOP_SRC, "row_sum_loop",
                  _rowsum_loop(lambda B, s: (8 if _small(B, s) else 4) if s == "sym" else (256 if _small(B, s) else 128))),),
           {"x": None}, "out", lambda s: 1, "wider K-loop tile for a small batch", "variant"),
    Family("rowsum_rows_by_batch",
           (Stage(ROWSUM_ROWS_SRC, "rowsum_rows", _rowsum_rows(lambda B, s: 1 if _small(B, s) else 4)),),
           {"x": None}, "out", lambda s: 1,
           "one row per program for a small batch, four otherwise; each row reduced in one tl.sum of the same length",
           "invariant"),
    Family("matmul_fixed", (Stage(MATMUL_SRC, "matmul", _matmul(lambda B, s: (4, 4, 4) if s == "sym" else (16, 16, 16))),),
           {"a": None}, "c", lambda s: _mm_dims(s)[0], "BM, BN, BK fixed", "invariant"),
    Family("matmul_bm_by_batch",
           (Stage(MATMUL_SRC, "matmul", _matmul(
               lambda B, s: ((4 if _small(B, s) else 8), 4, 4) if s == "sym" else ((16 if _small(B, s) else 32), 16, 16))),),
           {"a": None}, "c", lambda s: _mm_dims(s)[0], "BM from the batch size, BK fixed", "invariant"),
    Family("matmul_bk_by_batch",
           (Stage(MATMUL_SRC, "matmul", _matmul(
               lambda B, s: (4, 4, (8 if _small(B, s) else 4)) if s == "sym" else (16, 16, (32 if _small(B, s) else 16)))),),
           {"a": None}, "c", lambda s: _mm_dims(s)[0], "BK from the batch size", "variant"),
    Family("splitk_fixed", _splitk(lambda B, s: 2), {"a": None}, "c", lambda s: _mm_dims(s)[0],
           "split-K into 2 partial products, then a second kernel adds them; 2 at every batch size", "invariant"),
    Family("splitk_by_batch", _splitk(lambda B, s: 2 if _small(B, s) else 1), {"a": None}, "c",
           lambda s: _mm_dims(s)[0], "split-K for a small batch only (as GEMM heuristics do)", "variant"),
    Family("rowsum_mixed_rows",
           (Stage(ROWSUM_MIXED_SRC, "rowsum_mixed", _rowsum_rows(lambda B, s: 1 if _small(B, s) else 4)),),
           {"x": None}, "out", lambda s: 1,
           "a wrong kernel: sums its whole (ROWS, BLOCK) tile, so rows mix when ROWS > 1", "variant"),
]}


def _row_len(fam: Family, buf: str, launch: Launch, B: int) -> int:
    n = fam.batch_bufs.get(buf)
    return launch.buffers[buf] // B if n is None else n


# --------------------------------------------------------------------------- symbolic
def output_terms(fam: Family, B: int, size: str = "sym", dom=None, mem=None):
    """``{output index: term}`` after all stages at batch ``B``, rounded ops uninterpreted.

    ``mem`` (buffer -> Z3 array) is shared between batch sizes so that input cells match;
    pass the same dict to both calls."""
    dom = dom or _UFDom()
    mem = {} if mem is None else mem
    cur = {}
    written: dict = {}
    last = None
    for st in fam.stages:
        launch = st.launch(B, size)
        for b, arr in _real_memory(launch).items():
            mem.setdefault(b, arr)
        cur = dict(mem)
        for (b, i), v in written.items():
            cur[b] = z3.Store(cur[b], z3.BitVecVal(i, IDX_BITS), v)
        tr = trace_fp(st.source, st.fn_name, launch, cur, f"B{B}", dom)
        vals, _ = oracle._final_memory(tr, launch)
        written.update(vals)
        last = launch
    out = {i: z3.simplify(v) for (b, i), v in written.items() if b == fam.out}
    return out, last


def _selects(term, acc=None):
    """Concrete ``(array name, index)`` reads in ``term``."""
    acc = set() if acc is None else acc
    seen = set()
    stack = [term]
    while stack:
        e = stack.pop()
        k = e.get_id()
        if k in seen:
            continue
        seen.add(k)
        if z3.is_select(e):
            arr, idx = e.arg(0), e.arg(1)
            if z3.is_const(arr) and z3.is_bv_value(idx):
                acc.add((arr.decl().name(), idx.as_signed_long()))
        stack.extend(e.children())
    return acc


def cross_row_reads(fam: Family, B: int, terms: dict, size: str, launch_in: dict) -> list:
    """Output cells whose term reads a batch input outside its own row."""
    n_out = fam.out_row(size)
    bad = []
    for i, t in sorted(terms.items()):
        r = i // n_out
        for name, idx in _selects(t):
            buf = name[len("mem_"):]
            if buf not in fam.batch_bufs:
                continue
            rl = launch_in[buf]
            if idx // rl != r:
                bad.append({"out_index": i, "row": r, "reads": buf, "index": idx, "from_row": idx // rl})
                break
    return bad


def symbolic(fam: Family, B1: int, B2: int, size: str = "sym") -> dict:
    dom, mem = _UFDom(), {}
    t1, l1 = output_terms(fam, B1, size, dom, mem)
    t2, l2 = output_terms(fam, B2, size, dom, mem)
    first_in = fam.stages[0].launch
    rl1 = {b: _row_len(fam, b, first_in(B1, size), B1) for b in fam.batch_bufs}
    rl2 = {b: _row_len(fam, b, first_in(B2, size), B2) for b in fam.batch_bufs}
    n_out = fam.out_row(size)
    differ = []
    for i in range(B1 * n_out):
        a, b = t1.get(i), t2.get(i)
        if a is None or b is None or not z3.eq(a, b):
            differ.append(i)
    cross = cross_row_reads(fam, B1, t1, size, rl1) + cross_row_reads(fam, B2, t2, size, rl2)
    return {"B1": B1, "B2": B2, "cells": B1 * n_out, "differing_cells": len(differ),
            "first_differing": differ[0] if differ else None,
            "verdict": "proved" if not differ else "refuted", "cross_row": cross[:5]}


# --------------------------------------------------------------------------- witness
def _batch_inputs(fam: Family, B: int, size: str, kind: str, seed: int, base: dict | None = None) -> dict:
    """Inputs for the first stage at batch ``B``. With ``base`` (inputs at a larger batch),
    the batch rows are its first ``B`` rows and the shared inputs are its own."""
    launch = fam.stages[0].launch(B, size)
    if base is None:
        return make_inputs(launch, kind, seed)
    out = {}
    for b, x in base.items():
        if b in fam.batch_bufs:
            out[b] = x[:launch.buffers[b]].copy()
        else:
            out[b] = x.copy()
    return out


def emulate_family(fam: Family, B: int, size: str, inputs: dict) -> np.ndarray:
    mem = dict(inputs)
    for st in fam.stages:
        launch = st.launch(B, size)
        res, _ = emulate(st.source, st.fn_name, launch, {b: mem[b] for b in launch.buffers if b in mem})
        for b in launch.outputs:
            mem[b] = res[b]
    return mem[fam.out]


def search_witness(fam: Family, B1: int, B2: int, size: str = "sym", trials: int = 10) -> dict | None:
    n_out = fam.out_row(size)
    for kind in ("randn", "mixed"):
        for seed in range(trials):
            big = _batch_inputs(fam, B2, size, kind, seed)
            small = _batch_inputs(fam, B1, size, kind, seed, base=big)
            o2 = emulate_family(fam, B2, size, big)[:B1 * n_out]
            o1 = emulate_family(fam, B1, size, small)
            d = _bitwise_diff(o1, o2)
            if d.size:
                i = int(d[0])
                return {"generator": kind, "seed": seed, "row": i // n_out, "col": i % n_out,
                        "value_B1": float(o1[i]), "value_B2": float(o2[i]), "cells_differing": int(d.size)}
    return None


def check(fam: Family, size: str = "sym", trials: int = 10) -> dict:
    t0 = time.time()
    B1, B2 = fam.batches[size]
    res = symbolic(fam, B1, B2, size)
    res["witness"] = search_witness(fam, B1, B2, size, trials) if res["verdict"] == "refuted" else None
    res.update(family=fam.name, policy=fam.policy, expect=fam.expect, size=size,
               seconds=round(time.time() - t0, 2))
    return res


# --------------------------------------------------------------------------- hardware
def gpu_check(fam: Family, trials: int = 5, device: str = "cuda") -> dict:
    """Run the real Triton kernels at both GPU batch sizes, the small batch being the first
    rows of the large one, and compare the shared rows bitwise."""
    import torch
    from .kernels import to_module
    fns = [to_module(st.source, st.fn_name) for st in fam.stages]
    B1, B2 = fam.batches["gpu"]
    n_out = fam.out_row("gpu")
    tdt = {"float16": torch.float16, "float32": torch.float32}

    def run(B, inputs):
        bufs = {}
        for st, fn in zip(fam.stages, fns):
            launch = st.launch(B, "gpu")
            for b, n in launch.buffers.items():
                if b not in bufs:
                    if b in inputs:
                        bufs[b] = torch.as_tensor(inputs[b]).to(device=device, dtype=tdt[launch.dtype(b)])
                    else:
                        bufs[b] = torch.zeros(n, device=device, dtype=tdt[launch.dtype(b)])
            fn[launch.grid](**{k: (bufs[v] if isinstance(v, str) else v) for k, v in launch.args.items()})
        torch.cuda.synchronize()
        return bufs[fam.out].float().cpu().numpy()

    tried, first, rows_differing = 0, None, 0
    for kind in ("randn", "mixed"):
        for seed in range(trials):
            big = _batch_inputs(fam, B2, "gpu", kind, seed)
            small = _batch_inputs(fam, B1, "gpu", kind, seed, base=big)
            o2 = run(B2, big)[:B1 * n_out]
            o1 = run(B1, small)
            again = run(B1, small)
            tried += 1
            if _bitwise_diff(o1, again).size:
                return {"B1": B1, "B2": B2, "error": "the same launch twice gave different bits", "tried": tried}
            d = _bitwise_diff(o1, o2)
            if d.size:
                rows_differing += len({int(i) // n_out for i in d})
                if first is None:
                    i = int(d[0])
                    first = {"generator": kind, "seed": seed, "row": i // n_out, "col": i % n_out,
                             "value_B1": float(o1[i]), "value_B2": float(o2[i]), "cells_differing": int(d.size)}
    return {"B1": B1, "B2": B2, "tried": tried, "bitwise_equal": first is None, "first_difference": first,
            "rows_differing_total": rows_differing}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/batchinv.json")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--gpu", action="store_true", help="also run the hardware check")
    ap.add_argument("--families", nargs="*", default=None)
    args = ap.parse_args(argv)
    names = args.families or list(FAMILIES)
    t0 = time.time()
    rows = []
    for n in names:
        fam = FAMILIES[n]
        r = check(fam, "sym", args.trials)
        if args.gpu:
            r["gpu"] = gpu_check(fam)
        rows.append(r)
        w = r["witness"]
        print(f"{n:<28} expect={fam.expect:<9} {r['verdict']:<8} cells={r['differing_cells']}/{r['cells']} "
              f"cross_row={len(r['cross_row'])} witness={'yes' if w else '-'}"
              + (f" gpu_equal={r['gpu'].get('bitwise_equal')}" if args.gpu else ""), flush=True)
    res = {"command": "python -m triton_equiv.batchinv " + " ".join(argv if argv is not None else sys.argv[1:]),
           "python": platform.python_version(), "z3": z3.get_version_string(), "seconds": round(time.time() - t0, 1),
           "rows": rows}
    if args.gpu:
        import torch
        import triton
        res.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__)
    from pathlib import Path
    Path(args.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
