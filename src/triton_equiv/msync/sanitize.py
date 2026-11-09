"""Audit GPU race checkers on labelled synchronisation mutants.

    python -m triton_equiv.msync.sanitize [--tools racecheck synccheck memcheck gsan]
                                          [--gsan-python PY] [--gsan-path DIR]
                                          [--out results/msync_sanitizers.json]

Two suites of mutants with ground-truth labels:

``vm``     the schedule programs of part 3 (cross-SM counter protocols, ``mutate``) plus
           the warp-level ``no_barrier`` mutants, labelled by the step-machine oracle.
``intra``  the single-kernel shared-memory mutants of ``intra``, labelled by its
           happens-before model.

Checkers:

``racecheck``, ``synccheck``, ``memcheck``  NVIDIA compute-sanitizer tools, run on the CUDA
           VM (``vm``) and on the intra kernels. One sanitized process per suite and tool;
           before each program the child prints a marker line and after it synchronises, so
           each report line is attributed to the program whose marker precedes it (the
           references come first and must stay clean, which checks the attribution).
``gsan``   Triton's experimental global-memory race sanitizer
           (``triton.experimental.gsan``, vector clocks over ld/st and atomics with their
           memory semantics; within-CTA races are out of its scope by design). Run on the
           Triton VM (``gpu.Runner``, so ``vm`` without the warp mutants), one child
           process per program, because a report is a device-side assert that ends the
           CUDA context. ``--gsan-python``/``--gsan-path`` select an interpreter or an extra
           ``sys.path`` entry with a Triton that has GSan (3.9 has it).
``consan`` Triton's ``consan`` instrumentation mode (shared and tensor memory, mbarriers),
           run the same way as gsan, if requested.

Every program runs once, without injected delays: these checkers look for unordered
accesses, not for wrong outputs, so the question is whether they flag a race that the
output comparison of ``races`` mostly misses without delays.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .ir import reference_programs
from .machine import num_arena, num_inputs, run
from .mutate import mutants, warp_mutants

CS_TOOLS = ("racecheck", "synccheck", "memcheck")
MARK = "@@ "
GSAN_RE = re.compile(r"(Read after write|Write after read|Write after write) race detected")


# ---------------------------------------------------------------- program catalogue
def find(pid: str):
    """``(program, reference)`` for a reference name or a mutant id ``<ref>:<op>:<k>``."""
    refs = {r.name: r for r in reference_programs()}
    if pid in refs:
        return refs[pid], refs[pid]
    name = pid.split(":")[0]
    ref = refs[name]
    for mu in mutants(ref) + warp_mutants(ref):
        if mu.id == pid:
            return mu.program, ref
    raise KeyError(pid)


def catalogue(warp: bool = True) -> list:
    """Rows ``{id, kind, program, operator, label}`` in the order of ``races``: each
    reference, then (benign references only) its mutants, then its warp mutants."""
    from .machine import Sym
    from .oracle import explore, label
    sym, rows = Sym(), []
    for ref in reference_programs():
        ref_out = explore(ref, sym)[0]
        ref_label = label(ref, ref, sym, ref_out)["label"]
        rows.append({"id": ref.name, "kind": "reference", "program": ref.name, "operator": None, "label": ref_label})
        if ref_label != "benign":
            continue
        for mu in mutants(ref) + (warp_mutants(ref) if warp else []):
            rows.append({"id": mu.id, "kind": "mutant", "program": ref.name, "operator": mu.operator,
                         "label": label(mu.program, ref, sym, ref_out)["label"]})
    return rows


def _expected(ref, inputs):
    status, m = run(ref, num_arena(ref, inputs, "nan"))
    assert status == "ok", (ref.name, status)
    return list(m.output())


# ---------------------------------------------------------------- children
def child_vm(ids: list, timeout_ns: int):
    import torch
    from ..cuda_ref import Backend
    from . import gpu
    from .cuda_vm import CudaRunner
    from .races import observe
    for pid in ids:
        prog, ref = find(pid)
        inputs = num_inputs(ref, 0)
        r = CudaRunner(gpu.encode(prog), num_arena(prog, inputs, "nan"), "parallel", backend=Backend.CUDA)
        print(f"{MARK}{pid}", flush=True)
        launch = r.run(None, timeout_ns).launch
        torch.cuda.synchronize()
        print(f"{MARK}obs {pid} {observe(launch, _expected(ref, inputs)).value}", flush=True)
        time.sleep(0.3)


def child_intra(_ids, _timeout):
    import torch
    from . import intra
    for k in intra.KERNELS:
        r = intra.IntraRunner(k)
        for pid, mask in [(f"{k}:reference", intra.ALL)] + [(m.id, m.mask) for m in intra.MUTANTS if m.kernel == k]:
            print(f"{MARK}{pid}", flush=True)
            out = r.run(mask)
            torch.cuda.synchronize()
            print(f"{MARK}obs {pid} {'match' if r.same(out) else 'differs'}", flush=True)
            time.sleep(0.3)


def child_triton(pid: str, mode: str):
    """One program on the Triton VM with ``mode`` instrumentation. Prints ``RESULT {...}``."""
    import torch
    import triton
    from . import gpu
    from .races import max_spin_for, observe
    triton.knobs.compilation.instrumentation_mode = mode
    ctx = contextlib.nullcontext()
    if mode == "gsan":
        from triton.experimental.gsan import create_mem_pool
        ctx = torch.cuda.use_mem_pool(create_mem_pool())
    prog, ref = find(pid)
    inputs = num_inputs(ref, 0)
    expected = _expected(ref, inputs)
    with ctx:
        r = gpu.Runner(gpu.encode(prog), num_arena(prog, inputs, "nan"), "parallel", device="cuda")
        launch = r.run(None, max_spin_for(prog, 1000))
        torch.cuda.synchronize()
    print("RESULT " + json.dumps({"obs": observe(launch, expected).value, "triton": triton.__version__}), flush=True)


# ---------------------------------------------------------------- parents
def sanitizer() -> str | None:
    return shutil.which("compute-sanitizer") or next(
        (p for p in ("/usr/local/cuda/bin/compute-sanitizer", "/usr/local/cuda/compute-sanitizer/compute-sanitizer")
         if os.path.exists(p)), None)


def _classify(line: str) -> str:
    low = line.lower()
    if "race reported" in low or "hazard" in low:
        return "race"
    if "barrier" in low:
        return "barrier"
    if "invalid" in low or "out of bounds" in low or "misaligned" in low:
        return "memory"
    return "other"


def parse_cs(text: str) -> dict:
    """``{id: {"lines": Counter by class, "obs": ..., "sample": [...]}}`` from a sanitized run."""
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith(MARK):
            rest = line[len(MARK):].strip()
            if rest.startswith("obs "):
                _, pid, obs = rest.split(" ", 2)
                out.setdefault(pid, {"lines": Counter(), "sample": []})["obs"] = obs
            else:
                cur = rest
                out.setdefault(cur, {"lines": Counter(), "sample": []})
            continue
        if cur is None or not line.startswith("========="):
            continue
        body = line.strip("= ").strip()
        if not body or body.startswith(("ERROR SUMMARY", "RACECHECK SUMMARY", "COMPUTE-SANITIZER", "Saved host backtrace",
                                         "Host Frame", "Program hit", "at ", "by thread", "in ", "Address")):
            continue
        if not re.match(r"(Error|Warning|WARN|ERROR|Invalid|Race|Barrier)", body):
            continue
        rec = out[cur]
        rec["lines"][_classify(body)] += 1
        if len(rec["sample"]) < 3:
            rec["sample"].append(body[:240])
    return {k: {"hits": sum(v["lines"].values()), "by_class": dict(v["lines"]), "obs": v.get("obs"),
                "sample": v["sample"]} for k, v in out.items()}


def run_cs(tool: str, suite: str, ids: list, timeout_ns: int, log_dir: Path) -> dict:
    cs = sanitizer()
    if cs is None:
        return {"error": "compute-sanitizer not found"}
    cmd = [cs, "--tool", tool, "--print-limit", "100000"]
    if tool == "racecheck":
        cmd += ["--racecheck-report", "analysis"]  # one line per racing pair of source locations
    cmd += [sys.executable, "-m", "triton_equiv.msync.sanitize", "--child", suite, "--timeout-ns", str(timeout_ns),
            "--ids", *ids]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([p for p in (str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH")) if p])
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
    text = p.stdout + "\n" + p.stderr
    (log_dir / f"cs_{tool}_{suite}.log").write_text(text)
    summary = re.findall(r"ERROR SUMMARY: (\d+) error", text)
    return {"returncode": p.returncode, "seconds": round(time.time() - t0, 1),
            "error_summary": int(summary[-1]) if summary else None, "rows": parse_cs(p.stdout)}


def run_triton(mode: str, ids: list, python: str, extra_path: str | None, log_dir: Path) -> dict:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join([p for p in (extra_path, src, env.get("PYTHONPATH")) if p])
    env["TRITON_INTERPRET"] = "0"
    rows, t0, log = {}, time.time(), []
    for pid in ids:
        try:
            p = subprocess.run([python, "-m", "triton_equiv.msync.sanitize", "--triton-child", pid, "--mode", mode],
                               capture_output=True, text=True, env=env, timeout=900)
            text, rc = p.stdout + "\n" + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            text, rc = f"timeout: {e}", None
        res = next((json.loads(l[7:]) for l in text.splitlines() if l.startswith("RESULT ")), None)
        kinds = Counter(GSAN_RE.findall(text))
        rows[pid] = {"returncode": rc, "result": res, "races": dict(kinds), "reported": bool(kinds),
                     "error": None if (res or kinds) else text.strip()[-600:]}
        log.append(f"### {pid} rc={rc}\n{text[-4000:]}")
        print(f"{mode} {pid:<44} rc={rc} reported={bool(kinds)} obs={(res or {}).get('obs')}", flush=True)
        if mode != "gsan" and rows[pid]["error"] and len(rows) == 1:
            break  # the mode does not work here; record the first failure only
    (log_dir / f"triton_{mode}.log").write_text("\n".join(log))
    return {"seconds": round(time.time() - t0, 1), "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="*", default=["racecheck", "synccheck", "memcheck", "gsan"])
    ap.add_argument("--out", default="results/msync_sanitizers.json")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--timeout-ns", type=int, default=2_000_000_000)
    ap.add_argument("--gsan-python", default=sys.executable)
    ap.add_argument("--gsan-path", default=None)
    ap.add_argument("--child", choices=("vm", "intra"))
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--triton-child")
    ap.add_argument("--mode", default="gsan")
    args = ap.parse_args(argv)
    if args.child:
        return (child_vm if args.child == "vm" else child_intra)(args.ids, args.timeout_ns)
    if args.triton_child:
        return child_triton(args.triton_child, args.mode)

    from . import intra
    out = Path(args.out)
    log_dir = Path(args.log_dir or out.with_suffix(""))
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vm = catalogue(warp=True)
    vm_ids = [r["id"] for r in vm]
    vm_triton_ids = [r["id"] for r in vm if r["operator"] != "no_barrier"]
    intra_rows = [{"id": f"{k}:reference", "kind": "reference", "kernel": k, "label": "benign", "scope": None}
                  for k in intra.KERNELS]
    for m in intra.MUTANTS:
        lab = intra.hb_label(m.kernel, m.mask)
        intra_rows.append({"id": m.id, "kind": "mutant", "kernel": m.kernel, "label": lab["label"],
                           "scope": lab["scope"]})
    res = {"command": "python -m triton_equiv.msync.sanitize " + " ".join(argv if argv is not None else sys.argv[1:]),
           "sanitizer": sanitizer(), "tools": {}, "vm": vm, "intra": intra_rows}
    if res["sanitizer"]:
        v = subprocess.run([res["sanitizer"], "--version"], capture_output=True, text=True)
        res["sanitizer_version"] = v.stdout.strip().splitlines()[-1] if v.stdout.strip() else v.stderr[-200:]
    for tool in args.tools:
        if tool in CS_TOOLS:
            res["tools"][tool] = {"vm": run_cs(tool, "vm", vm_ids, args.timeout_ns, log_dir),
                                  "intra": run_cs(tool, "intra", [], args.timeout_ns, log_dir)}
            print(tool, "done", flush=True)
        elif tool in ("gsan", "consan"):
            res["tools"][tool] = {"vm_triton": run_triton(tool, vm_triton_ids, args.gsan_python, args.gsan_path,
                                                          log_dir)}
        else:
            raise ValueError(tool)
        out.write_text(json.dumps(res, indent=1))
    res["seconds"] = round(time.time() - t0, 1)
    res["summary"] = summarise(res)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))
    return res


def summarise(res: dict) -> dict:
    """Per tool and suite: flagged / total for racy mutants, benign mutants and references."""
    def groups(rows, suite):
        for r in rows:
            g = "reference" if r["kind"] == "reference" else r["label"]
            if suite == "intra" and r["label"] == "racy":
                g = f"racy/{r['scope']}"
            yield r["id"], g, r

    out = {}
    for tool, by_suite in res["tools"].items():
        for suite, data in by_suite.items():
            got = data.get("rows") or {}
            base = res["intra"] if suite == "intra" else res["vm"]
            s = {}
            for pid, g, r in groups(base, suite):
                if suite == "vm_triton" and r.get("operator") == "no_barrier":
                    continue
                rec = got.get(pid)
                flagged = None if rec is None else (rec.get("reported") if "reported" in rec else rec["hits"] > 0)
                e = s.setdefault(g, {"programs": 0, "flagged": 0, "missing": 0})
                e["programs"] += 1
                if flagged is None:
                    e["missing"] += 1
                else:
                    e["flagged"] += bool(flagged)
            out[f"{tool}/{suite}"] = s
    return out


if __name__ == "__main__":
    main()
