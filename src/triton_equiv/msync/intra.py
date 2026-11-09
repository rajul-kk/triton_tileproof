"""Single-kernel synchronisation mutants: the RESOLVE-style comparison for the delay harness.

    python -m triton_equiv.msync.intra [--reps 200] [--dmax 20000] [--out results/msync_intra.json]

RESOLVE (arXiv 2610.05683, section 6.2) planted synchronisation bugs in single KernelBench
kernels and found that 90 of 92 caught mutants showed without any injected delay. On the
cross-SM counter protocols of ``races`` the T4 showed only about a quarter of the racy
mutants without delays. Are block-level shared-memory races simply easier to expose, or
does our harness expose less? This module runs block-level mutants through the same kind
of harness on the same GPU.

Kernels (``csrc/intra_sync.cu``), 256 threads per block, every barrier switchable by a
mask bit: ``reduce`` (a shared tree to 64, then a warp tail with ``__syncwarp``), ``matmul``
(16x16 shared tiles), ``stencil`` (3 points with a halo) and ``scan`` (Hillis-Steele, in
place). A mutant drops one barrier (or, ``reduce:warp_sync_free``, every ``__syncwarp`` of
the tail, the old warp-synchronous idiom).

Labels come from a happens-before model of each kernel's shared-memory accesses
(:func:`hb_label`), not from reading the code: two accesses by different threads to one
address, at least one a write, race unless an executed ``__syncthreads`` separates them, or
an executed ``__syncwarp`` does and both threads are in one warp. ``racy`` mutants have a
race; ``benign`` ones dropped a redundant barrier (controls for false alarms). Each racy
mutant's **scope** is ``intra_warp`` (every racing pair in one warp), ``cross_warp`` (none)
or ``mixed``.

Conditions, ``reps`` launches each, compared bitwise with the all-barrier kernel:

``none``  no delays.
``warp``  each (warp, delay point) gets, with probability 1/2, ``uniform{1..dmax}`` ns;
          every lane of the warp the same.
``lane``  the same draw per thread, so the lanes of one warp diverge.

A launch that differs is the bug made visible. A benign mutant or reference that ever
differs is a false alarm.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

NT, NP, WARP = 256, 8, 32
ALL = 0xFFFFFFFF
CONDITIONS = ("none", "warp", "lane")

# kernel -> (id in intra_run, blocks)
KERNELS = {"reduce": (0, 80), "matmul": (1, 64), "stencil": (2, 80), "scan": (3, 80)}
MM_N, MM_K = 128, 64  # matmul: 64 blocks of 16 x 16 = a 128 x 128 C, K = 64
WARP_TAIL_BITS = tuple(range(4, 16))


@dataclass(frozen=True)
class IntraMutant:
    id: str
    kernel: str
    bits: tuple  # barriers dropped
    what: str

    @property
    def mask(self) -> int:
        m = ALL
        for b in self.bits:
            m &= ~(1 << b)
        return m


def _m(kernel, name, bits, what):
    return IntraMutant(f"{kernel}:{name}", kernel, tuple(bits), what)


MUTANTS = [
    _m("reduce", "pre_load", [0], "barrier before the load"),
    _m("reduce", "after_load", [1], "barrier after the load"),
    _m("reduce", "after_128", [2], "barrier after the 128 level"),
    _m("reduce", "after_64", [3], "barrier after the 64 level"),
    _m("reduce", "warp32_war", [4], "__syncwarp between read and write, step 32"),
    _m("reduce", "warp32_raw", [5], "__syncwarp after the write, step 32"),
    _m("reduce", "warp16_war", [6], "__syncwarp between read and write, step 16"),
    _m("reduce", "warp16_raw", [7], "__syncwarp after the write, step 16"),
    _m("reduce", "warp1_war", [14], "__syncwarp between read and write, step 1"),
    _m("reduce", "warp1_raw", [15], "__syncwarp after the write, step 1"),
    _m("reduce", "warp_sync_free", WARP_TAIL_BITS, "every __syncwarp of the tail"),
    _m("matmul", "after_load", [0], "barrier after the tile loads"),
    _m("matmul", "before_next_load", [1], "barrier before the next tile load"),
    _m("matmul", "after_loop", [2], "barrier after the loop"),
    _m("stencil", "after_load", [0], "barrier after the loads"),
    _m("stencil", "before_store", [1], "barrier before the store"),
    _m("scan", "after_load", [16], "barrier after the load"),
    _m("scan", "step1_war", [0], "barrier between read and write, step 1"),
    _m("scan", "step1_raw", [1], "barrier after the write, step 1"),
    _m("scan", "step64_war", [12], "barrier between read and write, step 64"),
    _m("scan", "step64_raw", [13], "barrier after the write, step 64"),
    _m("scan", "step128_raw", [15], "barrier after the write, step 128"),
]


# ---------------------------------------------------------------- happens-before labels
def _program(kernel: str, t: int):
    """Thread ``t``'s shared-memory ops in order: ("r"|"w", addr), ("bar", k), ("wbar", k).
    Mirrors csrc/intra_sync.cu; addresses are (array, index)."""
    ops = []
    if kernel == "reduce":
        ops += [("bar", 0), ("w", ("s", t)), ("bar", 1)]
        if t < 128:
            ops += [("r", ("s", t)), ("r", ("s", t + 128)), ("w", ("s", t))]
        ops.append(("bar", 2))
        if t < 64:
            ops += [("r", ("s", t)), ("r", ("s", t + 64)), ("w", ("s", t))]
        ops.append(("bar", 3))
        if t < 32:
            bit, off = 4, 32
            while off >= 1:
                ops += [("r", ("s", t)), ("r", ("s", t + off)), ("wbar", bit), ("w", ("s", t)), ("wbar", bit + 1)]
                off //= 2
                bit += 2
            if t == 0:
                ops.append(("r", ("s", 0)))
    elif kernel == "matmul":
        tx, ty = t % 16, t // 16
        for _ in range(MM_K // 16):
            ops += [("w", ("A", ty * 16 + tx)), ("w", ("B", ty * 16 + tx)), ("bar", 0)]
            for k in range(16):
                ops += [("r", ("A", ty * 16 + k)), ("r", ("B", k * 16 + tx))]
            ops.append(("bar", 1))
        ops.append(("bar", 2))
    elif kernel == "stencil":
        ops.append(("w", ("s", t + 1)))
        if t == 0:
            ops.append(("w", ("s", 0)))
        if t == NT - 1:
            ops.append(("w", ("s", NT + 1)))
        ops += [("bar", 0), ("r", ("s", t)), ("r", ("s", t + 1)), ("r", ("s", t + 2)), ("bar", 1)]
    elif kernel == "scan":
        ops += [("w", ("s", t)), ("bar", 16)]
        off, i = 1, 0
        while off < NT:
            ops.append(("r", ("s", t)))
            if t >= off:
                ops.append(("r", ("s", t - off)))
            ops += [("bar", 2 * i), ("w", ("s", t)), ("bar", 2 * i + 1)]
            off *= 2
            i += 1
        ops.append(("r", ("s", t)))
    else:
        raise ValueError(kernel)
    return ops


@functools.lru_cache(maxsize=None)
def hb_label(kernel: str, mask: int = ALL) -> dict:
    """``{"label": "racy"|"benign", "scope": ..., "pairs": n}`` under ``mask``.

    Every thread meets the same sequence of block barriers (none sits under a condition)
    and the warp-tail ``__syncwarp`` is executed by all of warp 0, so an access's position
    is (executed block barriers before it, executed warp barriers before it), and access a
    happens before access b of another thread iff b's block count is larger, or both are
    in one warp and b's warp count is larger."""
    by_addr: dict = {}
    for t in range(NT):
        nb = nw = 0
        for op, arg in _program(kernel, t):
            if op == "bar":
                nb += (mask >> arg) & 1
            elif op == "wbar":
                nw += (mask >> arg) & 1
            else:
                by_addr.setdefault(arg, []).append((t, op, nb, nw))
    pairs, intra, cross = 0, 0, 0
    for accs in by_addr.values():
        writes = [a for a in accs if a[1] == "w"]
        for w in writes:
            for o in accs:
                if o[0] == w[0] or (o[1] == "w" and o >= w):
                    continue  # same thread; each write-write pair once
                same_warp = o[0] // WARP == w[0] // WARP
                ordered = o[2] != w[2] or (same_warp and o[3] != w[3])
                if not ordered:
                    pairs += 1
                    intra += same_warp
                    cross += not same_warp
    scope = None if not pairs else "intra_warp" if not cross else "cross_warp" if not intra else "mixed"
    return {"label": "racy" if pairs else "benign", "scope": scope, "pairs": pairs}


# ---------------------------------------------------------------- GPU
@functools.lru_cache(maxsize=None)
def module():
    from torch.utils.cpp_extension import load_inline
    from .. import cuda_ref
    cuda_ref.ensure_ninja_on_path()
    decl = ("void intra_run(int64_t kernel, at::Tensor x, at::Tensor y, at::Tensor out, int64_t mask, "
            "at::Tensor delays, int64_t nblocks, int64_t n, int64_t k);")
    return load_inline("tileproof_intra_sync", cpp_sources=[decl],
                       cuda_sources=[(cuda_ref.CSRC / "intra_sync.cu").read_text()], functions=["intra_run"],
                       extra_cuda_cflags=["-O2", "-arch=sm_75"], verbose=False)


class IntraRunner:
    """Inputs, buffers and the all-barrier reference output for one kernel."""

    def __init__(self, kernel: str, seed: int = 0, device: str = "cuda"):
        import torch
        self.torch, self.kernel, self.dev = torch, kernel, device
        self.kid, self.nblocks = KERNELS[kernel]
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = self.nblocks * NT
        if kernel == "matmul":
            M = self.nblocks * 16 * 16 // MM_N
            self.x = torch.randn(M * MM_K, generator=g).to(device)
            self.y = torch.randn(MM_K * MM_N, generator=g).to(device)
            self.out = torch.empty(M * MM_N, device=device)
        else:
            self.x = torch.randn(n, generator=g).to(device)
            self.y = self.x
            self.out = torch.empty(self.nblocks if kernel == "reduce" else n, device=device)
        self.n = n if kernel != "matmul" else MM_N
        self.k = MM_K
        self.zero = torch.zeros(self.nblocks * NT * NP, dtype=torch.int64, device=device)
        self.gen = torch.Generator(device=device)
        self.ref = self.run(ALL)

    def delays(self, cond: str, dmax: int):
        torch = self.torch
        if cond == "none":
            return self.zero
        shape = (self.nblocks, NT // WARP if cond == "warp" else NT, NP)
        on = torch.rand(shape, generator=self.gen, device=self.dev) < 0.5
        d = torch.randint(1, dmax + 1, shape, generator=self.gen, device=self.dev) * on
        if cond == "warp":
            d = d.repeat_interleave(WARP, dim=1)
        return d.reshape(-1).contiguous()

    def run(self, mask: int, delays=None):
        self.out.fill_(float("nan"))
        module().intra_run(self.kid, self.x, self.y, self.out, mask, self.zero if delays is None else delays,
                           self.nblocks, self.n, self.k)
        self.torch.cuda.synchronize()
        return self.out.clone()

    def same(self, out) -> bool:
        return bool(self.torch.equal(out.view(self.torch.int32), self.ref.view(self.torch.int32)))


def trial(r: IntraRunner, mask: int, reps: int, dmax: int, seed: int) -> dict:
    r.gen.manual_seed(seed)
    res = {}
    for cond in CONDITIONS:
        res[cond] = sum(not r.same(r.run(mask, r.delays(cond, dmax))) for _ in range(reps))
    return res


def summarise(rows: list, reps: int) -> dict:
    out = {}
    for r in rows:
        if r["kind"] != "mutant":
            continue
        key = r["label"] if r["label"] == "benign" else f"racy/{r['scope']}"
        for k in (key, r["label"]):
            s = out.setdefault(k, {"mutants": 0, **{c: {"seen": 0, "mean_rate": 0.0} for c in CONDITIONS}})
            s["mutants"] += 1
            for c in CONDITIONS:
                s[c]["seen"] += r["differs"][c] > 0
                s[c]["mean_rate"] += r["differs"][c] / reps
    for s in out.values():
        for c in CONDITIONS:
            s[c]["mean_rate"] = round(s[c]["mean_rate"] / s["mutants"], 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--dmax", type=int, default=20_000, help="ns")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/msync_intra.json")
    args = ap.parse_args(argv)
    import torch
    t0 = time.time()
    runners = {k: IntraRunner(k, args.seed) for k in KERNELS}
    rows = []
    for k, r in runners.items():
        again = [r.same(r.run(ALL)) for _ in range(20)]
        rows.append({"id": f"{k}:reference", "kind": "reference", "kernel": k, "label": "benign", "scope": None,
                     "deterministic": all(again),
                     "differs": trial(r, ALL, args.reps, args.dmax, args.seed)})
    for i, m in enumerate(MUTANTS):
        lab = hb_label(m.kernel, m.mask)
        d = trial(runners[m.kernel], m.mask, args.reps, args.dmax, args.seed + 1 + i)
        rows.append({"id": m.id, "kind": "mutant", "kernel": m.kernel, "bits": list(m.bits), "what": m.what,
                     "label": lab["label"], "scope": lab["scope"], "hb_pairs": lab["pairs"], "differs": d})
        print(f"{m.id:<26} {lab['label']:<7} {str(lab['scope']):<11} "
              + "  ".join(f"{c}={d[c] / args.reps:.2f}" for c in CONDITIONS), flush=True)
    false_alarms = [r["id"] for r in rows if r["label"] == "benign" and any(r["differs"].values())]
    res = {"command": "python -m triton_equiv.msync.intra " + " ".join(argv if argv is not None else sys.argv[1:]),
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "reps": args.reps, "dmax": args.dmax,
           "seed": args.seed, "seconds": round(time.time() - t0, 1), "false_alarms": false_alarms,
           "summary": summarise(rows, args.reps), "rows": rows}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))
    print("false alarms:", false_alarms or "none")
    return res


if __name__ == "__main__":
    main()
