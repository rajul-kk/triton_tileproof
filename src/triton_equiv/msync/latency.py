"""Launch latency of each reference program as one megakernel, as a dynamic megakernel and
as one kernel per task, with tinytail's EVT/pWCET analysis of the tails.

    pip install git+https://github.com/rajul-kk/tinytail
    python -m triton_equiv.msync.latency [--n 100000] [--out results/msync_gpu_latency.json]

Needs a CUDA GPU, torch, triton and tinytail's ``pwcet`` package. For every reference
program whose label is ``benign`` it times ``n`` launches (after 50 warm-up launches) in
each :data:`SCHEDULES` (``gpu.Runner.time``: CUDA events around the launch, arena copied in
before the start event, no delays):

``static``    ``parallel`` mode: each program walks a fixed queue (an AMK-style megakernel).
``dynamic``   ``dynamic`` mode: programs take tasks from a shared atomic head.
``per_task``  one launch per task in topological order (the eager baseline).

The first third of each series is the fit set and the rest the hold-out set
(``pwcet.analysis.analyse``, origin ``hardware``). The question is the one in the docs:
do the schedules give timings that pass tinytail's i.i.d. gates (independence: runs and
Ljung-Box; stationarity: KPSS and a KS test of the two halves), and where they do, what
pWCET does the GPD fit give and does the hold-out set agree with it? A failed gate means no
bound, by design: that verdict is the result.

Raw series go to a gzipped JSON next to ``--out`` so the analysis can be rerun offline.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

from . import gpu
from .ir import reference_programs
from .machine import Sym, num_arena, num_inputs
from .oracle import label

SCHEDULES = {"static": "parallel", "dynamic": "dynamic", "per_task": "per_task"}


def series(prog, n: int, device="cuda") -> dict:
    """``{schedule: [microseconds]}`` for one program."""
    arena = num_arena(prog, num_inputs(prog, 0), "nan")
    enc = gpu.encode(prog)
    return {s: gpu.Runner(enc, arena, mode, n_programs=prog.n_sms, device=device).time(n)
            for s, mode in SCHEDULES.items()}


def tail_report(x, cfg) -> dict:
    """tinytail's report, cut down to what the summary needs (the full one is long)."""
    from pwcet.analysis import analyse
    k = int(len(x) * cfg.fit_frac)
    r = analyse(x[:k], x[k:], cfg, origin="hardware", meta={"unit": "us"})
    keep = ("gates", "verdict", "primary_outcome", "fit_set", "holdout_set", "baseline_max_plus_margin",
            "estimates", "validation", "extrapolation_refuted")
    return {key: r[key] for key in keep if key in r}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)  # tinytail PROTOCOL.md: >= 100k rows per session
    ap.add_argument("--gof-reps", type=int, default=1000)
    ap.add_argument("--out", default="results/msync_gpu_latency.json")
    args = ap.parse_args(argv)
    import pwcet
    import torch
    import triton
    from pwcet.analysis import Config

    cfg = Config(gof_reps=args.gof_reps)
    t0, sym = time.time(), Sym()
    raw, rows = {}, []
    for prog in reference_programs():
        if label(prog, prog, sym)["label"] != "benign":
            continue  # moe_dispatch's ticket order makes its output schedule-dependent
        s = series(prog, args.n)
        raw[prog.name] = s
        for sched, x in s.items():
            rep = tail_report(x, cfg)
            rows.append({"program": prog.name, "tasks": len(prog.tasks), "sms": prog.n_sms, "schedule": sched,
                         "median_us": sorted(x)[len(x) // 2], "gate_verdict": rep["gates"]["verdict"],
                         "primary_outcome": rep["primary_outcome"], "report": rep})
            print(f"{prog.name:<22} {sched:<9} median={rows[-1]['median_us']:8.2f}us "
                  f"gates={rep['gates']['verdict']:<22} {rep['primary_outcome']}", flush=True)
    out = Path(args.out)
    res = {"command": "python -m triton_equiv.msync.latency " + " ".join(argv or sys.argv[1:]),
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
           "pwcet": pwcet.__version__, "n": args.n, "seconds": round(time.time() - t0, 1), "rows": rows}
    out.write_text(json.dumps(res, indent=1, default=float))
    with gzip.open(out.with_suffix(".samples.json.gz"), "wt") as f:
        json.dump(raw, f)
    return res


if __name__ == "__main__":
    main()
