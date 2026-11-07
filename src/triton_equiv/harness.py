"""Standard-test checker audit: allclose over random inputs, run on the Triton CPU interpreter.

The checker is the usual benchmark correctness test: allocate inputs with ``torch.randn``, run
the candidate kernel, compare its outputs to a PyTorch reference with an rtol/atol test, and
accept if every trial passes. Two tolerance policies are scored from the same runs:

``kernelbench``   ``rtol = atol = 1e-2`` for every dtype (KernelBench's default check).
``torch_default`` dtype-aware ``torch.testing.assert_close`` defaults: fp32 ``rtol 1.3e-6,
                  atol 1e-5``; fp16 ``rtol 1e-3, atol 1e-5``.

Sizes come from ``KernelSpec.checker_sizes``. ``tile`` sizes make every dimension a tile
multiple, which is what most fixed-shape benchmarks use. ``odd`` sizes leave partial tiles.
A checker "with odd sizes" runs both.

Each buffer lives inside one larger tensor with a guard band on each side, filled with seeded
random values. Out-of-bounds loads therefore read plausible data, and out-of-bounds stores land
in the guard instead of corrupting the process. The standard checker never looks at the guard;
``guard_write`` is recorded separately to show what a canary check would add. Output buffers
start at zero.

Subcommands (run under WSL with ``TRITON_INTERPRET=1``; see README):

``audit``       every mutant in ``results/labels.json`` -> confusion table of oracle label
                against checker verdict -> ``results/checker_audit.json``.
``blind-band``  ``numerical`` mutants swept over the reduction length (``n_cols`` or ``K``), with
                the miss rate of the kernelbench check per length -> ``results/blind_band.json``.

Both take ``--kernels`` to restrict the run (the softmax/layernorm sweep is stored separately in
``results/blind_band_rowwise.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from .families import FAMILIES
from .families import family as _family

GUARD = 4096  # elements on each side of every buffer
TOLERANCES = {
    "kernelbench": {"float32": (1e-2, 1e-2), "float16": (1e-2, 1e-2)},
    "torch_default": {"float32": (1.3e-6, 1e-5), "float16": (1e-3, 1e-5)},
}


def _torch():
    os.environ.setdefault("TRITON_INTERPRET", "1")
    import torch
    return torch


# ------------------------------------------------------------------ PyTorch references
def torch_reference(kernel: str, launch, inputs: dict):
    """What a benchmark's PyTorch model would return, in the output buffer's dtype."""
    torch = _torch()
    a = launch.args
    if kernel == "vector_add":
        return {"out": inputs["x"] + inputs["y"]}
    if kernel in ("row_sum", "row_sum_loop"):
        x = inputs["x"].view(-1, a["n_cols"])
        return {"out": x.to(torch.float64 if x.dtype == torch.float64 else torch.float32).sum(1)}
    if kernel == "matmul":
        return {"c": (inputs["a"].view(a["M"], a["K"]) @ inputs["b"].view(a["K"], a["N"])).reshape(-1)}
    if kernel == "matmul_fp16":
        A, B = inputs["a"].view(a["M"], a["K"]), inputs["b"].view(a["K"], a["N"])
        if A.dtype == torch.float64:
            return {"c": (A @ B).reshape(-1)}
        return {"c": (A.float() @ B.float()).to(torch.float16).reshape(-1)}
    if kernel == "softmax":
        return {"out": torch.softmax(inputs["x"].view(-1, a["n_cols"]), dim=1).reshape(-1)}
    if kernel == "layernorm":
        x = inputs["x"].view(-1, a["n_cols"])
        wide = x if x.dtype == torch.float64 else x.float()
        y = torch.nn.functional.layer_norm(wide, (a["n_cols"],), eps=a["eps"])
        return {"out": (y if x.dtype == torch.float64 else y.to(x.dtype)).reshape(-1)}
    if kernel == "attention":
        D = a["D"]
        Q, K, V = (inputs[b].view(-1, D) for b in ("q", "k", "v"))
        return {"o": (torch.softmax(Q @ K.T, dim=1) @ V).reshape(-1)}
    raise KeyError(kernel)


def fp64_reference(kernel: str, launch, inputs: dict):
    return torch_reference(kernel, launch, {b: t.double() for b, t in inputs.items()})


# ------------------------------------------------------------------ running one kernel
_COMPILED: dict = {}


def _compiled(source: str, fn_name: str):
    from .kernels import to_module
    key = (source, fn_name)
    if key not in _COMPILED:
        _COMPILED[key] = to_module(source, fn_name)
    return _COMPILED[key]


def make_inputs(launch, seed: int, dist: str = "randn"):
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    dt = {"float16": torch.float16, "float32": torch.float32}
    out = {}
    for b, n in launch.buffers.items():
        if b in launch.outputs:
            continue
        x = torch.randn(n, generator=g, dtype=torch.float64) if dist == "randn" else \
            torch.rand(n, generator=g, dtype=torch.float64)
        out[b] = x.to(dt[launch.dtype(b)])
    return out


def run(spec, source: str, launch, inputs: dict, seed: int = 0):
    """Run ``source`` once. Returns (outputs, guard_write, error string or None)."""
    torch = _torch()
    dt = {"float16": torch.float16, "float32": torch.float32}
    g = torch.Generator().manual_seed(10_000 + seed)
    big, views = {}, {}
    for b, n in launch.buffers.items():
        t = torch.randn(n + 2 * GUARD, generator=g).to(dt[launch.dtype(b)])
        t[GUARD:GUARD + n] = inputs[b] if b in inputs else 0
        big[b], views[b] = t, t[GUARD:GUARD + n]
    before = {b: (t[:GUARD].clone(), t[GUARD + launch.buffers[b]:].clone()) for b, t in big.items()}
    try:
        fn = _compiled(source, spec.fn_name)
        fn[launch.grid](**{k: (views[v] if isinstance(v, str) else v) for k, v in launch.args.items()})
        err = None
    except Exception as e:  # a crash is a rejection by any checker
        err = f"{type(e).__name__}: {str(e)[:200]}"
    guard_write = any(
        not torch.equal(big[b][:GUARD], lo) or not torch.equal(big[b][GUARD + launch.buffers[b]:], hi)
        for b, (lo, hi) in before.items())
    return {b: views[b].clone() for b in launch.outputs}, guard_write, err


def close(out, ref, dtype: str, policy: str) -> bool:
    torch = _torch()
    rtol, atol = TOLERANCES[policy][dtype]
    return bool(torch.allclose(out.double(), ref.double(), rtol=rtol, atol=atol))


# ------------------------------------------------------------------ audit


def audit(labels_path: str, trials: int, kernels=None):
    from .kernels import KERNELS
    from .mutate import mutants
    with open(labels_path) as f:
        labels = {lab["id"]: lab for lab in json.load(f)["labels"]}
    rows = []
    for name, spec in KERNELS.items():
        if kernels and name not in kernels:
            continue
        cands = [(f"{name}:reference", spec.source, None)] + \
                [(m.id, m.source, labels[m.id]) for m in mutants(name, spec.source)]
        for mid, src, lab in cands:
            row = {"id": mid, "oracle": _family(lab), "verdict": {}, "guard_write": False, "errors": []}
            for kind, sizes in spec.checker_sizes.items():
                ok = {p: True for p in TOLERANCES}
                for size in sizes:
                    launch = spec.make_launch(**size)
                    for seed in range(trials):
                        inputs = make_inputs(launch, seed)
                        out, gw, err = run(spec, src, launch, inputs, seed)
                        row["guard_write"] |= gw
                        if err and err not in row["errors"]:
                            row["errors"].append(err)
                        ref = torch_reference(name, launch, inputs)
                        for p in TOLERANCES:
                            ok[p] &= err is None and all(
                                close(out[b], ref[b], launch.dtype(b), p) for b in launch.outputs)
                row["verdict"][kind] = {p: "accept" if v else "reject" for p, v in ok.items()}
            rows.append(row)
            print(mid, row["oracle"], row["verdict"], "guard" if row["guard_write"] else "", flush=True)
    return rows


def confusion(rows):
    """{checker: {family: {"accept": n, "reject": n}}}; checker = sizes + tolerance policy."""
    out = {}
    for p in TOLERANCES:
        for checker, kinds in ((f"tile/{p}", ("tile",)), (f"tile+odd/{p}", ("tile", "odd"))):
            tab = {f: {"accept": 0, "reject": 0} for f in FAMILIES}
            for r in rows:
                acc = all(r["verdict"][k][p] == "accept" for k in kinds)
                tab[r["oracle"]]["accept" if acc else "reject"] += 1
            out[checker] = tab
    tab = {f: {"accept": 0, "reject": 0} for f in FAMILIES}
    for r in rows:  # tile+odd kernelbench check plus a guard-band canary
        acc = all(r["verdict"][k]["kernelbench"] == "accept" for k in ("tile", "odd")) and not r["guard_write"]
        tab[r["oracle"]]["accept" if acc else "reject"] += 1
    out["tile+odd/kernelbench+canary"] = tab
    return out


# ------------------------------------------------------------------ blind band
_NS = (16, 64, 256, 1024, 4096)
SWEEP = {
    "row_sum": [{"n_rows": 4, "n_cols": n, "BLOCK": n} for n in _NS],
    "row_sum_loop": [{"n_rows": 4, "n_cols": n, "BLOCK": 16} for n in _NS],
    "matmul": [{"M": 16, "N": 16, "K": k, "BM": 16, "BN": 16, "BK": 16} for k in _NS],
    "matmul_fp16": [{"M": 16, "N": 16, "K": k, "BM": 16, "BN": 16, "BK": 16} for k in _NS],
    # Row-wise kernels hold a whole row in one tile, so n stops at 1024.
    "softmax": [{"n_rows": 4, "n_cols": n, "BLOCK": n} for n in _NS[:4]],
    "layernorm": [{"n_rows": 4, "n_cols": n, "BLOCK": n} for n in _NS[:4]],
}


def blind_band(labels_path: str, trials: int, dists=("randn", "rand"), kernels=None):
    torch = _torch()
    from .kernels import KERNELS
    from .mutate import mutants
    with open(labels_path) as f:
        labels = {lab["id"]: lab for lab in json.load(f)["labels"]}
    rows = []
    for name, sizes in SWEEP.items():
        if kernels and name not in kernels:
            continue
        spec = KERNELS[name]
        muts = [(m.id, m.source) for m in mutants(name, spec.source) if labels[m.id]["label"] == "numerical"]
        cands = [(f"{name}:reference", spec.source)] + muts
        for dist in dists:
            for size in sizes:
                launch = spec.make_launch(**size)
                b = launch.outputs[0]
                n = size[spec.reduction_dim]
                stats = {mid: {"accept": 0, "differs": 0, "err": [], "excess": []} for mid, _ in cands}
                for seed in range(trials):
                    inputs = make_inputs(launch, seed, dist)
                    ref = torch_reference(name, launch, inputs)[b]
                    exact = fp64_reference(name, launch, inputs)[b]
                    outs = {mid: run(spec, src, launch, inputs, seed)[0][b] for mid, src in cands}
                    base = outs[f"{name}:reference"]
                    e0 = (base.double() - exact).abs().max().item()
                    for mid, _ in cands:
                        o, s = outs[mid], stats[mid]
                        s["accept"] += close(o, ref, launch.dtype(b), "kernelbench")
                        s["differs"] += not torch.equal(o, base)
                        e = (o.double() - exact).abs().max().item()
                        s["err"].append(e)
                        s["excess"].append(e / e0 if e0 > 0 else (1.0 if e == 0 else float("inf")))
                for mid, _ in cands:
                    s = stats[mid]
                    fam = "reference" if mid.endswith(":reference") else _family(labels[mid])
                    rows.append({
                        "id": mid, "family": fam, "dist": dist, "n": n, "size": size, "trials": trials,
                        "accept_rate": s["accept"] / trials,
                        "differs_from_reference_kernel_rate": s["differs"] / trials,
                        "median_max_abs_err_vs_fp64": sorted(s["err"])[trials // 2],
                        "median_err_ratio_vs_reference_kernel": sorted(s["excess"])[trials // 2],
                    })
                    r = rows[-1]
                    print(f"{dist:5s} n={n:5d} {mid:40s} accept={r['accept_rate']:.2f} "
                          f"err={r['median_max_abs_err_vs_fp64']:.3g} "
                          f"ratio={r['median_err_ratio_vs_reference_kernel']:.3g}", flush=True)
    return rows


# ------------------------------------------------------------------ CLI
def _env():
    torch = _torch()
    import triton
    return {"python": platform.python_version(), "torch": torch.__version__, "triton": triton.__version__,
            "TRITON_INTERPRET": os.environ.get("TRITON_INTERPRET"), "platform": platform.platform()}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="allclose checker audit on the Triton interpreter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--labels", default="results/labels.json")
    a.add_argument("--trials", type=int, default=3)
    a.add_argument("--out", default="results/checker_audit.json")
    a.add_argument("--kernels", nargs="*", help="only these kernels (default: all)")
    b = sub.add_parser("blind-band")
    b.add_argument("--labels", default="results/labels.json")
    b.add_argument("--trials", type=int, default=20)
    b.add_argument("--out", default="results/blind_band.json")
    b.add_argument("--kernels", nargs="*", help="only these kernels (default: all in SWEEP)")
    args = ap.parse_args(argv)
    t0 = time.time()
    doc = {"command": "TRITON_INTERPRET=1 PYTHONPATH=src python -m triton_equiv.harness " + " ".join(argv),
           "env": _env(), "tolerances": TOLERANCES}
    if args.cmd == "audit":
        rows = audit(args.labels, args.trials, args.kernels)
        doc.update(trials_per_size=args.trials, confusion=confusion(rows), rows=rows)
    else:
        doc.update(trials=args.trials, rows=blind_band(args.labels, args.trials, kernels=args.kernels))
    doc["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    if args.cmd == "audit":
        print(json.dumps(doc["confusion"], indent=1))


if __name__ == "__main__":
    main()
