"""How often does each schedule bug actually show on a GPU, with and without injected delays?

    python -m triton_equiv.msync.races [--reps 100] [--dmax 1000] [--out results/msync_gpu_races.json]

Needs a CUDA GPU, torch and triton. For every reference program and every part-1 mutant it
runs the persistent kernel (``gpu``, ``parallel`` mode) ``reps`` times in each condition:

``none``    no delays: whatever interleaving the GPU produces on its own.
``random``  each of the 3 delay points of each task gets, with probability 1/2, a spin of
            ``uniform{1..dmax}`` volatile loads (fresh draw per launch, seeded).

Each launch is one :class:`Obs`, compared with the step machine's float64 output for the
*reference* program on the same inputs (uninitialised memory is NaN, so a stale read
poisons the output instead of hiding):

``match``      completed, output equals the reference (float32 tolerance)
``differs``    completed, output differs
``timed_out``  some program gave up waiting (``max_spin`` polls)
``faulted``    some program's region ran outside the arena

A launch other than ``match`` is the bug made visible. The oracle's label for the same
program (``oracle.label``) says what is possible; this says what a test that runs the
kernel would actually see. A reference or ``benign`` program with any launch other than
``match`` is a false alarm, i.e. a fault of the kernel or of ``max_spin`` (recorded in
``false_alarms``; there should be none).

``max_spin`` is set per program to ``3 * n_tasks * dmax + spin_floor`` polls (floor 20 000): more than the
longest possible chain of injected delays, so a correct program never times out
(checked by the false alarms above).
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import random
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

from . import gpu
from .ir import reference_programs
from .machine import Sym, num_arena, num_inputs, run
from .mutate import mutants
from .oracle import explore, label


class Obs(enum.Enum):
    MATCH = "match"
    DIFFERS = "differs"
    TIMED_OUT = "timed_out"
    FAULTED = "faulted"


CONDITIONS = ("none", "random")


def observe(launch: gpu.Launch, expected) -> Obs:
    if launch.outcome is gpu.RunOutcome.FAULTED:
        return Obs.FAULTED
    if launch.outcome is gpu.RunOutcome.TIMED_OUT:
        return Obs.TIMED_OUT
    same = len(expected) == len(launch.output) and all(
        not math.isnan(g) and math.isclose(g, e, rel_tol=1e-3, abs_tol=1e-4)
        for g, e in zip(launch.output, expected))
    return Obs.MATCH if same else Obs.DIFFERS


def random_delays(rng: random.Random, n_tasks: int, dmax: int) -> list:
    return [[rng.randint(1, dmax) if rng.random() < 0.5 else 0 for _ in range(3)] for _ in range(n_tasks)]


SPIN_FLOOR = 20_000


def max_spin_for(prog, dmax: int, floor: int = SPIN_FLOOR) -> int:
    return 3 * len(prog.tasks) * dmax + floor


def trial(prog, expected, arena, reps: int, dmax: int, seed: int, mode="parallel", device="cuda",
          spin_floor: int = SPIN_FLOOR) -> dict:
    """``{condition: {obs value: count}}`` for one program."""
    runner = gpu.Runner(gpu.encode(prog), arena, mode, device=device)
    spin = max_spin_for(prog, dmax, spin_floor)
    rng = random.Random(seed)
    res = {}
    for cond in CONDITIONS:
        c = Counter()
        for _ in range(reps):
            d = None if cond == "none" else random_delays(rng, len(prog.tasks), dmax)
            c[observe(runner.run(d, spin), expected).value] += 1
        res[cond] = {o.value: c.get(o.value, 0) for o in Obs}
    return res


def _rate(counts: dict) -> float:
    n = sum(counts.values())
    return (n - counts[Obs.MATCH.value]) / n if n else 0.0


def summarise(rows: list) -> dict:
    """Per oracle label: mutants, and for each condition how many showed the bug at least
    once and the mean fraction of launches that showed it."""
    by = {}
    for r in rows:
        if r["kind"] != "mutant":
            continue
        s = by.setdefault(r["label"], {"mutants": 0, **{c: {"seen": 0, "mean_rate": 0.0} for c in CONDITIONS}})
        s["mutants"] += 1
        for c in CONDITIONS:
            rate = _rate(r["obs"][c])
            s[c]["seen"] += rate > 0
            s[c]["mean_rate"] += rate
    for s in by.values():
        for c in CONDITIONS:
            s[c]["mean_rate"] = round(s[c]["mean_rate"] / s["mutants"], 4)
    return by


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--dmax", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/msync_gpu_races.json")
    ap.add_argument("--mode", default="parallel", choices=("parallel", "serial"),
                    help="serial: one program, list order (a CPU smoke test under TRITON_INTERPRET=1)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spin-floor", type=int, default=SPIN_FLOOR)
    args = ap.parse_args(argv)
    import torch
    import triton

    t0 = time.time()
    sym, rows = Sym(), []
    for ref in reference_programs():
        inputs = num_inputs(ref, args.seed)
        arena = num_arena(ref, inputs, "nan")
        status, m = run(ref, arena)
        if status != "ok":
            raise RuntimeError(f"reference {ref.name}: {status}")
        expected = list(m.output())
        ref_out = explore(ref, sym)[0]
        ref_label = label(ref, ref, sym, ref_out)["label"]
        progs = [("reference", ref.name, None, ref_label, ref)]
        if ref_label == "benign":  # as in audit: mutants need a reference with one outcome
            progs += [("mutant", mu.id, mu.operator, label(mu.program, ref, sym, ref_out)["label"], mu.program)
                      for mu in mutants(ref)]
        for kind, pid, op, lab, prog in progs:
            obs = trial(prog, expected, num_arena(prog, inputs, "nan"), args.reps, args.dmax,
                        seed=zlib.crc32(f"{args.seed}:{pid}".encode()), mode=args.mode, device=args.device,
                        spin_floor=args.spin_floor)
            rows.append({"id": pid, "kind": kind, "program": ref.name, "operator": op, "label": lab,
                         "max_spin": max_spin_for(prog, args.dmax, args.spin_floor), "obs": obs})
            print(f"{pid:<48} {lab:<13} " + "  ".join(
                f"{c}={_rate(obs[c]):.2f}" for c in CONDITIONS), flush=True)
    false_alarms = [r["id"] for r in rows if r["label"] == "benign" and
                    any(_rate(r["obs"][c]) > 0 for c in CONDITIONS)]
    res = {"command": "python -m triton_equiv.msync.races " + " ".join(argv or sys.argv[1:]),
           "gpu": torch.cuda.get_device_name() if args.device == "cuda" else args.device, "mode": args.mode, "torch": torch.__version__, "triton": triton.__version__,
           "reps": args.reps, "dmax": args.dmax, "spin_floor": args.spin_floor, "seed": args.seed, "seconds": round(time.time() - t0, 1),
           "false_alarms": false_alarms, "by_label": summarise(rows), "rows": rows}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["by_label"], indent=1))
    print("false alarms:", false_alarms or "none")
    return res


if __name__ == "__main__":
    main()
