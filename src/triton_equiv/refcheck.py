"""Does the standard checker's verdict depend on which reference it compares against?

    python -m triton_equiv.refcheck [--trials 3] [--kernels ...] [--device cpu|cuda]
                                   [--out results/refcheck.json]

The checker in ``harness`` compares a candidate kernel with PyTorch. This reruns the same
checker on every reference kernel and every mutant in ``results/labels.json`` against three
references (:data:`REFERENCES`):

``torch``   ``harness.torch_reference`` (what a benchmark's PyTorch model returns)
``cpp``     ``cuda_ref.cpp_reference`` with ``Acc.F32``: naive C++/CUDA loops, rounded to
            the output dtype
``cpp64``   the same kernels in float64 (unrounded)

and records ``accept``/``reject`` per (reference, size kind, tolerance policy). The result
lists every candidate whose verdict changes with the reference, and, for each kernel, the
largest error of the unmutated Triton kernel against each reference.

On the CPU (default) the candidates run in the Triton interpreter (``TRITON_INTERPRET=1``)
and the C++ references use the CPU build. With ``--device cuda`` the candidates run in real
Triton and ``cpp``/``cpp64`` use the CUDA build. Both builds run the same source.
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys
import time

from . import cuda_ref, harness
from .families import family


class Verdict(enum.Enum):
    ACCEPT = "accept"
    REJECT = "reject"   # includes a crash: a crash is a rejection by any checker


REFERENCES = ("torch", "cpp", "cpp64")


def references(name, launch, inputs, backend: cuda_ref.Backend) -> dict:
    return {"torch": harness.torch_reference(name, launch, inputs),
            "cpp": cuda_ref.cpp_reference(name, launch, inputs, cuda_ref.Acc.F32, backend),
            "cpp64": cuda_ref.cpp_reference(name, launch, inputs, cuda_ref.Acc.F64, backend)}


def check_candidate(name, spec, src, trials: int, device: str, backend) -> dict:
    """``{"verdict": {ref: {kind: {policy: Verdict value}}}, "max_err": {ref: float},
    "errors": [...]}``. ``max_err`` is the largest absolute difference from each reference
    over all runs that did not crash (None if every run crashed)."""
    ok = {r: {k: {p: True for p in harness.TOLERANCES} for k in spec.checker_sizes} for r in REFERENCES}
    max_err = {r: None for r in REFERENCES}
    errors = []
    for kind, sizes in spec.checker_sizes.items():
        for size in sizes:
            launch = spec.make_launch(**size)
            for seed in range(trials):
                inputs = harness.make_inputs(launch, seed)
                out, _, err = harness.run(spec, src, launch, inputs, seed, device=device)
                if err and err not in errors:
                    errors.append(err)
                refs = references(name, launch, inputs, backend)
                for r, ref in refs.items():
                    for p in harness.TOLERANCES:
                        ok[r][kind][p] &= err is None and all(
                            harness.close(out[b], ref[b], launch.dtype(b), p) for b in launch.outputs)
                    if err is None:
                        e = max(float((out[b].double() - ref[b].double()).abs().nan_to_num(float("inf")).max())
                                for b in launch.outputs)
                        max_err[r] = e if max_err[r] is None else max(max_err[r], e)
    verdict = {r: {k: {p: (Verdict.ACCEPT if v else Verdict.REJECT).value for p, v in pk.items()}
                   for k, pk in kp.items()} for r, kp in ok.items()}
    return {"verdict": verdict, "max_err": max_err, "errors": errors}


def disagreements(rows) -> list:
    """Candidates whose verdict under some (kind, policy) differs between references."""
    out = []
    for row in rows:
        v = row["verdict"]
        diffs = sorted({(k, p) for k in v["torch"] for p in v["torch"][k]
                        if len({v[r][k][p] for r in REFERENCES}) > 1})
        if diffs:
            out.append({"id": row["id"], "oracle": row["oracle"],
                        "differs_at": [f"{k}:{p}" for k, p in diffs],
                        "verdict": {r: {f"{k}:{p}": v[r][k][p] for k, p in diffs} for r in REFERENCES}})
    return out


def audit(labels_path: str, trials: int, kernels=None, device: str = "cpu") -> list:
    from .kernels import KERNELS
    from .mutate import mutants
    backend = cuda_ref.Backend.CUDA if device == "cuda" else cuda_ref.Backend.CPU
    with open(labels_path) as f:
        labels = {lab["id"]: lab for lab in json.load(f)["labels"]}
    rows = []
    for name, spec in KERNELS.items():
        if kernels and name not in kernels:
            continue
        cands = [(f"{name}:reference", spec.source, None)] + \
                [(m.id, m.source, labels[m.id]) for m in mutants(name, spec.source)]
        for mid, src, lab in cands:
            row = {"id": mid, "kernel": name, "oracle": family(lab),
                   **check_candidate(name, spec, src, trials, device, backend)}
            rows.append(row)
            print(mid, row["oracle"], {r: row["verdict"][r] for r in REFERENCES} if
                  len({json.dumps(row["verdict"][r], sort_keys=True) for r in REFERENCES}) > 1 else "agree",
                  flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="results/labels.json")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--kernels", nargs="*")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--out", default="results/refcheck.json")
    args = ap.parse_args(argv)
    if args.device == "cpu":
        os.environ.setdefault("TRITON_INTERPRET", "1")
    t0 = time.time()
    rows = audit(args.labels, args.trials, args.kernels, args.device)
    ref_err = {r["kernel"]: r["max_err"] for r in rows if r["id"].endswith(":reference")}
    res = {"command": "python -m triton_equiv.refcheck " + " ".join(argv or sys.argv[1:]),
           "device": args.device, "trials": args.trials, "seconds": round(time.time() - t0, 1),
           **harness._env(), "reference_kernel_max_err": ref_err,
           "disagreements": disagreements(rows), "rows": rows}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{len(res['disagreements'])} of {len(rows)} candidates change verdict with the reference")
    return res


if __name__ == "__main__":
    main()
