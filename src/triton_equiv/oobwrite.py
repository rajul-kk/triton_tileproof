"""U8: what out-of-bounds stores do to neighbouring memory, and whether an output-comparing checker notices.

Each out-of-bounds-store mutant runs at its checker's odd size (partial last tiles, where the store mask matters)
with its output placed two ways:

* ``adjacent``: the output is the middle of one allocation, with a sentinel of ``SENTINEL`` elements directly before
  and after it. This is the worst case, a neighbour starting at the output's last element, and it shows every
  element the kernel writes outside the output.
* ``pool``: the output and then a sentinel are separate ``torch.empty`` calls after the inputs, as a benchmark's
  ``forward`` allocates them. The caching allocator decides what lies past the output; ``gap_bytes`` records how far
  the sentinel starts from the output's end.

The checker verdict is ``torch.allclose`` against the PyTorch reference under each tolerance policy of
``harness.TOLERANCES``. One mutant per process (``main``), because an illegal-address fault poisons the CUDA context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SENTINEL = 4096  # elements on each side (adjacent) or in the pool sentinel
LAYOUTS = ("adjacent", "pool")


def store_oob_ids(labels: list[dict]) -> list[str]:
    """Mutants the oracle labels ``oob`` with a store as its witness access."""
    return [x["id"] for x in labels if x["label"] == "oob" and x["witness"]["access"] == "store"]


def load_oob_ids(labels: list[dict]) -> list[str]:
    return [x["id"] for x in labels if x["label"] == "oob" and x["witness"]["access"] != "store"]


def _pattern(torch, n, dtype, device, seed):
    g = torch.Generator().manual_seed(20_000 + seed)
    return torch.randn(n, generator=g).to(dtype).to(device)


def place(torch, launch, inputs, layout, device, seed):
    """Allocate inputs, outputs and sentinels. Returns (views, sentinels, snapshots, gap_bytes)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r} not in {LAYOUTS}")
    dt = {"float16": torch.float16, "float32": torch.float32}
    views = {b: inputs[b].to(device) for b in launch.buffers if b not in launch.outputs}
    sentinels, gap = {}, None
    for b in launch.outputs:
        n, d = launch.buffers[b], dt[launch.dtype(b)]
        if layout == "adjacent":
            blk = _pattern(torch, n + 2 * SENTINEL, d, device, seed)
            blk[SENTINEL : SENTINEL + n] = 0
            views[b] = blk[SENTINEL : SENTINEL + n]
            sentinels[b] = (blk[:SENTINEL], blk[SENTINEL + n :])
        else:
            views[b] = torch.zeros(n, dtype=d, device=device)
            s = _pattern(torch, SENTINEL, d, device, seed)
            sentinels[b] = (s,)
            gap = s.data_ptr() - (views[b].data_ptr() + n * views[b].element_size())
    snaps = {b: tuple(t.clone() for t in ts) for b, ts in sentinels.items()}
    return views, sentinels, snaps, gap


def changed(torch, sentinels, snaps) -> dict:
    """Per output: elements changed below and above it (adjacent) or in the pool sentinel (pool)."""
    out = {}
    for b, ts in sentinels.items():
        out[b] = [
            int(
                (
                    t.view(torch.int16 if t.element_size() == 2 else torch.int32)
                    != s.view(torch.int16 if s.element_size() == 2 else torch.int32)
                ).sum()
            )
            for t, s in zip(ts, snaps[b])
        ]
    return out


def run_mutant(spec, source, launch, inputs, layout, device, seed=0) -> dict:
    from . import harness

    torch = harness.get_torch()
    views, sentinels, snaps, gap = place(torch, launch, inputs, layout, device, seed)
    try:
        fn = harness.compiled(source, spec.fn_name)
        fn[launch.grid](**{k: (views[v] if isinstance(v, str) else v) for k, v in launch.args.items()})
        if device == "cuda":
            torch.cuda.synchronize()
        err = None
    except Exception as e:  # a crash is a rejection by any checker
        err = f"{type(e).__name__}: {str(e)[:200]}"
    ref = harness.torch_reference(spec.name, launch, inputs) if err is None else {}
    accept = {
        p: (err is None and all(harness.close(views[b].cpu(), ref[b], launch.dtype(b), p) for b in launch.outputs))
        for p in harness.TOLERANCES
    }
    return {
        "layout": layout,
        "seed": seed,
        "gap_bytes": gap,
        "error": err,
        "accept": accept,
        "sentinel_changed": changed(torch, sentinels, snaps) if err is None else None,
    }


def run_id(mutant_id: str, trials: int, device: str, layouts=LAYOUTS) -> list[dict]:
    from . import harness
    from .kernels import KERNELS
    from .mutate import mutants

    name = mutant_id.split(":")[0]
    spec = KERNELS[name]
    source = (
        spec.source
        if mutant_id == f"{name}:ref"
        else next(m.source for m in mutants(name, spec.source) if m.id == mutant_id)
    )
    size = spec.checker_sizes["odd"][0]
    launch = spec.make_launch(**size)
    rows = []
    for seed in range(trials):
        inputs = harness.make_inputs(launch, seed)
        for layout in layouts:
            rows.append(
                {"id": mutant_id, "size": size, **run_mutant(spec, source, launch, inputs, layout, device, seed)}
            )
    return rows


def summary(rows: list[dict]) -> dict:
    """Per id: share of trials each policy accepts, elements written outside (adjacent, max over trials), share of
    pool trials whose sentinel changed, the pool gaps seen, and the error count."""
    out = {}
    for mid in dict.fromkeys(r["id"] for r in rows):
        rs = [r for r in rows if r["id"] == mid]
        adj = [r for r in rs if r["layout"] == "adjacent" and r["error"] is None]
        pool = [r for r in rs if r["layout"] == "pool" and r["error"] is None]
        out[mid] = {
            "trials": len(rs),
            "accept": {p: sum(r["accept"][p] for r in rs) / len(rs) for p in rs[0]["accept"]},
            "adjacent_written": max((sum(sum(v) for v in r["sentinel_changed"].values()) for r in adj), default=None),
            "pool_hit": (sum(any(sum(v) for v in r["sentinel_changed"].values()) for r in pool) / len(pool))
            if pool
            else None,
            "pool_gap_bytes": sorted({r["gap_bytes"] for r in pool}),
            "errors": sum(r["error"] is not None for r in rs),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mutant_id", help="a mutant id, or '<kernel>:ref' for the reference kernel")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--layouts",
        nargs="+",
        choices=LAYOUTS,
        default=list(LAYOUTS),
        help="pool on the CPU interpreter writes into the process heap and can crash it",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.device == "cuda":
        os.environ["TRITON_INTERPRET"] = "0"
    rows = run_id(a.mutant_id, a.trials, a.device, a.layouts)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f)
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
