"""Launch latency of each reference program as one megakernel, as a dynamic megakernel and
as one kernel per task, with tinytail's EVT/pWCET analysis of the tails.

    pip install git+https://github.com/rajul-kk/tinytail
    python -m triton_equiv.msync.latency [--backend triton|cuda] [--n 100000]
                                         [--design back_to_back|paced] [--gap-us 200]
                                         [--out results/msync_gpu_latency.json]

Needs a CUDA GPU, torch and tinytail's ``pwcet`` package, plus triton for the default
backend. For every reference program whose label is ``benign`` it times ``n`` launches
(after 50 warm-up launches) in each :data:`SCHEDULES` (``Runner.time``: CUDA events around
the launch, arena copied in before the start event, no delays):

``static``    ``parallel`` mode: each program walks a fixed queue (an AMK-style megakernel).
``dynamic``   ``dynamic`` mode: programs take tasks from a shared atomic head.
``per_task``  one launch per task in topological order (the eager baseline).

With ``--backend cuda`` (``cuda_vm.CudaRunner``) there is one more series:

``static_in_kernel``  ``parallel`` mode, time from the first task's start to the last
                      task's end as read from ``%globaltimer`` inside the kernel: no launch
                      overhead, no host jitter. (Collected with one synchronising run per
                      sample, so its samples are not back to back.)

The first third of each series is the fit set and the rest the hold-out set
(``pwcet.analysis.analyse``, origin ``hardware``). The question is the one in the docs:
do the schedules give timings that pass tinytail's i.i.d. gates (independence: runs and
Ljung-Box; stationarity: KPSS and a KS test of the two halves), and where they do, what
pWCET does the GPD fit give and does the hold-out set agree with it? A failed gate means no
bound, by design: that verdict is the result.

Two designs (``--design``):

``back_to_back``  each schedule's ``n`` launches one after another, one schedule after the
                  other (the first T4 run). Every series failed the i.i.d. gates: state
                  carries from launch to launch, and slow drift lands in one schedule's
                  series only.
``paced``         ``n`` rounds. Each round runs every schedule once, in a fresh random
                  order, and each launch follows an idle gap drawn uniformly from
                  ``[0, gap_us]`` microseconds (host busy-wait after the previous launch
                  has finished, so the GPU is idle). Drift is shared by all schedules, and
                  no launch directly follows one of its own kind at a fixed spacing.
                  Rounds pair the schedules' samples, so each row also gets the paired
                  difference to ``static`` (:func:`paired`), which needs no i.i.d.
                  assumption across rounds.

Raw series (and, for ``paced``, the gaps) go to a gzipped JSON next to ``--out`` so the
analysis can be rerun offline.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

from . import gpu
from .ir import reference_programs
from .machine import Sym, num_arena, num_inputs
from .oracle import label

SCHEDULES = {"static": "parallel", "dynamic": "dynamic", "per_task": "per_task"}
BACKENDS = ("triton", "cuda")
DESIGNS = ("back_to_back", "paced")
WARMUP = 50


def in_kernel_us(run) -> float:
    """First task start to last task end of one ``cuda_vm.Run``, in microseconds."""
    spans = [t for t in run.task_ns if t is not None]
    if run.launch.outcome is not gpu.RunOutcome.COMPLETED or len(spans) != len(run.task_ns):
        raise RuntimeError(f"in-kernel timing of an unfinished launch: {run.launch}")
    return (max(e for _, e in spans) - min(s for s, _ in spans)) / 1000.0


def timers(prog, backend: str = "triton", device="cuda") -> dict:
    """``{schedule: f}`` where each ``f()`` times one launch of ``prog`` in microseconds."""
    if backend not in BACKENDS:
        raise ValueError(backend)
    arena = num_arena(prog, num_inputs(prog, 0), "nan")
    enc = gpu.encode(prog)
    if backend == "triton":
        return {s: gpu.Runner(enc, arena, mode, n_programs=prog.n_sms, device=device).time_one
                for s, mode in SCHEDULES.items()}
    from ..cuda_ref import Backend
    from .cuda_vm import CudaRunner
    out = {s: CudaRunner(enc, arena, mode, n_programs=prog.n_sms, backend=Backend.CUDA).time_one
           for s, mode in SCHEDULES.items()}
    r = CudaRunner(enc, arena, "parallel", backend=Backend.CUDA)
    out["static_in_kernel"] = lambda: in_kernel_us(r.run())
    return out


def series(prog, n: int, backend: str = "triton", device="cuda") -> dict:
    """``{schedule: [microseconds]}`` for one program, back to back: each schedule's
    ``n`` launches in a row, after ``WARMUP`` untimed ones."""
    out = {}
    for s, f in timers(prog, backend, device).items():
        x = [f() for _ in range(WARMUP + n)]
        out[s] = x[WARMUP:]
    return out


def idle(us: float):
    """Busy-wait on the host for ``us`` microseconds (finer than ``time.sleep``)."""
    t_end = time.perf_counter() + us * 1e-6
    while time.perf_counter() < t_end:
        pass


def paced_series(prog, n: int, backend: str = "triton", device="cuda", gap_us: float = 200.0,
                 seed: int = 0, timer=None, wait=idle):
    """``({schedule: [microseconds]}, {schedule: [gap before each sample]})``: ``n`` rounds,
    each running every schedule once in a random order, each launch after a random idle
    gap. Sample ``i`` of every schedule comes from round ``i``. ``timer`` (default
    :func:`timers`) and ``wait`` are parameters so the design can be tested without a GPU."""
    fs = (timer or timers)(prog, backend, device)
    rng, names = random.Random(seed), list(fs)
    for _ in range(WARMUP):
        for f in fs.values():
            f()
    out, gaps = {s: [] for s in names}, {s: [] for s in names}
    for _ in range(n):
        rng.shuffle(names)
        for s in names:
            g = rng.uniform(0.0, gap_us)
            wait(g)
            gaps[s].append(g)
            out[s].append(fs[s]())
    return out, gaps


def paired(x: dict, base: str = "static") -> dict:
    """For each other schedule, the per-round difference to ``base`` (needs a paced run):
    median, the share of rounds where it is positive, and the median of the base."""
    b = x[base]
    res = {}
    for s, y in x.items():
        if s == base or len(y) != len(b):
            continue
        d = sorted(yi - bi for yi, bi in zip(y, b))
        res[f"{s}-{base}"] = {"median_us": d[len(d) // 2], "share_positive": sum(v > 0 for v in d) / len(d),
                              "rounds": len(d)}
    return res


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
    ap.add_argument("--backend", default="triton", choices=BACKENDS)
    ap.add_argument("--n", type=int, default=100_000)  # tinytail PROTOCOL.md: >= 100k rows per session
    ap.add_argument("--gof-reps", type=int, default=1000)
    ap.add_argument("--design", default="back_to_back", choices=DESIGNS)
    ap.add_argument("--gap-us", type=float, default=200.0, help="paced: largest idle gap before a launch")
    ap.add_argument("--seed", type=int, default=0, help="paced: round orders and gaps")
    ap.add_argument("--out", default="results/msync_gpu_latency.json")
    args = ap.parse_args(argv)
    import pwcet
    import torch
    from pwcet.analysis import Config
    triton_version = None
    if args.backend == "triton":
        import triton
        triton_version = triton.__version__

    cfg = Config(gof_reps=args.gof_reps)
    t0, sym = time.time(), Sym()
    raw, rows, pairs = {}, [], {}
    for prog in reference_programs():
        if label(prog, prog, sym)["label"] != "benign":
            continue  # moe_dispatch's ticket order makes its output schedule-dependent
        if args.design == "paced":
            s, gaps = paced_series(prog, args.n, args.backend, gap_us=args.gap_us, seed=args.seed)
            raw[prog.name] = {"samples": s, "gaps_us": gaps}
            pairs[prog.name] = paired(s)
            print(prog.name, json.dumps(pairs[prog.name]), flush=True)
        else:
            s = series(prog, args.n, args.backend)
            raw[prog.name] = s
        for sched, x in s.items():
            rep = tail_report(x, cfg)
            rows.append({"program": prog.name, "tasks": len(prog.tasks), "sms": prog.n_sms, "schedule": sched,
                         "median_us": sorted(x)[len(x) // 2], "gate_verdict": rep["gates"]["verdict"],
                         "primary_outcome": rep["primary_outcome"], "report": rep})
            print(f"{prog.name:<22} {sched:<16} median={rows[-1]['median_us']:8.2f}us "
                  f"gates={rep['gates']['verdict']:<22} {rep['primary_outcome']}", flush=True)
    out = Path(args.out)
    res = {"command": "python -m triton_equiv.msync.latency " + " ".join(argv or sys.argv[1:]),
           "backend": args.backend, "design": args.design,
           "gap_us": args.gap_us if args.design == "paced" else None,
           "seed": args.seed if args.design == "paced" else None,
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "triton": triton_version,
           "pwcet": pwcet.__version__, "n": args.n, "seconds": round(time.time() - t0, 1),
           "paired": pairs or None, "rows": rows}
    out.write_text(json.dumps(res, indent=1, default=float))
    with gzip.open(out.with_suffix(".samples.json.gz"), "wt") as f:
        json.dump(raw, f)
    return res


if __name__ == "__main__":
    main()
