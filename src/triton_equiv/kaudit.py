"""A KernelAudit-style semantic oracle, re-implemented from its paper, scored on our labelled mutants.

KernelAudit (Li, Song, Qi, Yang and Xie, arXiv 2611.15506, 11 Nov 2026) audits kernel validators with a semantic
oracle: reference-calibrated stress tests at the task's shape, judged per output leaf by its Eq. 1. No code is
copied; the rule is re-implemented from the paper's text (read 18 Nov 2026), so this is "KernelAudit-style", not
KernelAudit.

Eq. 1, per floating output leaf, with ``r`` the native-precision reference and ``r64`` the fp64 one::

    d_i      = |r_i - r64_i|
    eps_abs  = p99(d),  eps_rel = p99(d / max(|r64|, 1e-12))
    a        = max(1e-8, 4 * eps_abs)
    rho_bar  = 4 * eps_rel;  rho = max(1e-5, rho_bar)
    accept   iff |k_i - r_i| <= a + rho * |r_i| for every element i

The comparison is *unresolved* (not passed) when ``rho_bar > 1e-3`` or the reference is non-finite. A non-finite
candidate against a finite reference fails.

The paper gives the schedule only in outline: 16 slots, "contract-adapted values spanning 10^-4 to 10^5", native-seed
slots, and one repeated-run slot of three calls, all at the task's shape (it "does not establish alternative-shape
coverage"). ``SLOTS`` below is our instance of that outline. Inputs whose scaled values overflow the buffer's dtype
leave the slot unresolved ("contract-adapted": the value is outside the dtype's range).

Candidate verdict: ``reject`` if any slot fails; else ``accept`` if at least one slot resolved; else ``unresolved``.
The ``tile`` sizes are the task's shape; ``odd`` sizes are scored too, as a sensitivity column KernelAudit does not run.
Out-of-bounds stores are recorded (``guard_write``) but, as in KernelAudit, never decide the verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .families import FAMILIES
from .families import family as _family
from .harness import (
    candidates,
    env_info,
    fp64_reference,
    get_torch,
    load_labels,
    make_inputs,
    run,
    torch_reference,
)

# Eq. 1 constants, one place (A9).
A_MIN, RHO_MIN, RHO_MAX, C, P = 1e-8, 1e-5, 1e-3, 4.0, 0.99
REL_FLOOR = 1e-12

# (name, distribution, scale, seed, calls). 16 slots: value slots spanning 1e-4..1e5, uniform slots, native seeds,
# and one repeated-run slot of three calls on the same input.
SLOTS = (
    [("value_normal", "randn", 1.0, 0, 1)]
    + [(f"value_scale_{s:g}", "randn", s, 10 + i, 1)
       for i, s in enumerate((1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e5))]
    + [("value_uniform", "rand", 1.0, 2, 1), ("value_uniform_1e3", "rand", 1e3, 3, 1)]
    + [(f"native_seed_{k}", "randn", 1.0, 100 + k, 1) for k in range(4)]
    + [("repeated_run", "randn", 1.0, 200, 3)]
)
FP_MAX = {"float16": 65504.0, "float32": 3.4028234663852886e38}


def _p99(t) -> float:
    torch = get_torch()
    return float(torch.quantile(t.flatten().double(), P).item()) if t.numel() else 0.0


def leaf_check(cand, ref, ref64) -> str:
    """Eq. 1 on one output leaf: ``"pass"``, ``"fail"``, ``"unresolved:ref_nonfinite"`` or
    ``"unresolved:rho_cap"``."""
    torch = get_torch()
    r, r64, k = ref.double(), ref64.double(), cand.double()
    if not bool(torch.isfinite(r).all()) or not bool(torch.isfinite(r64).all()):
        return "unresolved:ref_nonfinite"
    if r.numel() == 0:
        return "pass"
    if not bool(torch.isfinite(k).all()):
        return "fail"  # a non-finite candidate against a finite reference fails, whatever the calibration
    d = (r - r64).abs()
    a = max(A_MIN, C * _p99(d))
    rho_bar = C * _p99(d / r64.abs().clamp_min(REL_FLOOR))
    if rho_bar > RHO_MAX:
        return "unresolved:rho_cap"
    rho = max(RHO_MIN, rho_bar)
    return "pass" if bool(((k - r).abs() <= a + rho * r.abs()).all()) else "fail"


def slot_inputs(launch, dist: str, scale: float, seed: int):
    """Scaled inputs in each buffer's dtype, or None when the scaled values overflow that dtype."""
    base = make_inputs(launch, seed, dist)
    out = {}
    for b, x in base.items():
        y = x.double() * scale
        if float(y.abs().max().item()) > FP_MAX[launch.dtype(b)]:
            return None
        out[b] = y.to(x.dtype)
    return out


def _combine(parts) -> str:
    """Verdict from slot states (``pass``/``fail``/``unresolved:*``) or from verdicts (``accept``/``reject``/
    ``unresolved``): any failure rejects; else any pass accepts; else unresolved."""
    parts = list(parts)
    if any(s in ("fail", "reject") for s in parts):
        return "reject"
    if any(s in ("pass", "accept") for s in parts):
        return "accept"
    return "unresolved"


def run_slot(spec, name: str, src: str, launch, slot, device: str = "cpu"):
    """One slot on one launch -> (state, guard_write, error). State is ``pass``/``fail``/``unresolved:<why>``.
    References are computed on the CPU whatever ``device`` the candidate runs on."""
    _, dist, scale, seed, calls = slot
    inputs = slot_inputs(launch, dist, scale, seed)
    if inputs is None:
        return "unresolved:input_range", False, None
    ref = torch_reference(name, launch, inputs)
    ref64 = fp64_reference(name, launch, inputs)
    states, gw = [], False
    for call in range(calls):  # ``seed + call`` only changes the guard-band fill between repeated calls
        out, g, e = run(spec, src, launch, inputs, seed + call, device=device)
        gw |= g
        if e:
            return "fail", gw, e  # a crash is a rejection by any checker
        states += [leaf_check(out[b], ref[b], ref64[b]) for b in launch.outputs]
    if not states:
        raise ValueError(f"{name}: launch has no outputs")  # L5: an empty slot must never read as a pass
    if "fail" in states:
        return "fail", gw, None
    bad = [s for s in states if s != "pass"]
    return (bad[0] if bad else "pass"), gw, None


def audit(labels_path: str, kernels=None, kinds=("tile", "odd"), device: str = "cpu", ids=None):
    """Rows for every candidate of ``kernels`` (all by default), or only the candidate ids in ``ids``."""
    from .kernels import KERNELS
    labels = load_labels(labels_path)
    rows = []
    for name, spec in KERNELS.items():
        if kernels and name not in kernels:
            continue
        for mid, src, lab in candidates(name, spec, labels):
            if ids is not None and mid not in ids:
                continue
            row = {"id": mid, "oracle": _family(lab), "slots": {}, "verdict": {}, "guard_write": False,
                   "errors": []}
            for kind in kinds:
                states = []
                for si, size in enumerate(spec.checker_sizes[kind]):
                    launch = spec.make_launch(**size)
                    for slot in SLOTS:
                        st, gw, err = run_slot(spec, name, src, launch, slot, device)
                        row["slots"][f"{kind}{si}/{slot[0]}"] = st
                        row["guard_write"] |= gw
                        if err and err not in row["errors"]:
                            row["errors"].append(err)
                        states.append(st)
                row["verdict"][kind] = _combine(states)
            rows.append(row)
            print(mid, row["oracle"], row["verdict"], "guard" if row["guard_write"] else "", flush=True)
    return rows


def confusion(rows):
    """{checker: {family: {"accept": n, "reject": n, "unresolved": n}}} for ``tile`` and ``tile+odd``."""
    out = {}
    for checker, kinds in (("kaudit/tile", ("tile",)), ("kaudit/tile+odd", ("tile", "odd"))):
        tab = {f: {"accept": 0, "reject": 0, "unresolved": 0} for f in FAMILIES}
        for r in rows:
            if all(k in r["verdict"] for k in kinds):
                tab[r["oracle"]][_combine(r["verdict"][k] for k in kinds)] += 1
        out[checker] = tab
    return out


def slot_state_counts(rows):
    """Count of slot states by reason, over every row and slot."""
    out: dict = {}
    for r in rows:
        for st in r["slots"].values():
            out[st] = out.get(st, 0) + 1
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="KernelAudit-style oracle on the labelled mutants")
    ap.add_argument("--labels", default="results/labels.json")
    ap.add_argument("--out", default="results/kaudit_audit.json")
    ap.add_argument("--kernels", nargs="*", help="only these kernels (default: all)")
    ap.add_argument("--ids", nargs="*", help="only these candidate ids, e.g. one per process on a GPU")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="cuda runs real Triton (set TRITON_INTERPRET=0)")
    args = ap.parse_args(argv)
    t0 = time.time()
    rows = audit(args.labels, args.kernels, device=args.device, ids=set(args.ids) if args.ids else None)
    if args.ids and len(rows) != len(set(args.ids)):  # L5: an unknown id must not pass silently as "no rows"
        raise SystemExit(f"unknown candidate ids: {sorted(set(args.ids) - {r['id'] for r in rows})}")
    doc = {"command": "PYTHONPATH=src python -m triton_equiv.kaudit " + " ".join(argv),
           "env": env_info(), "source": "arXiv 2611.15506 Eq. 1, re-implemented (read 2026-11-18)",
           "constants": {"a_min": A_MIN, "rho_min": RHO_MIN, "rho_max": RHO_MAX, "c": C, "percentile": P},
           "slots": [list(s) for s in SLOTS], "confusion": confusion(rows),
           "slot_states": slot_state_counts(rows), "rows": rows}
    doc["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(doc["confusion"], indent=1))


if __name__ == "__main__":
    main()
