"""The oracle's labels for the warp-level ``no_barrier`` mutants against a GPU race run.

    python -m triton_equiv.msync.warp_check [--gpu results/msync_gpu_races_cuda.json]
                                            [--out results/msync_warp_labels.json]

Before warps were in the step machine, ``races --backend cuda`` recorded these mutants
with label ``unmodelled``. This labels each one with the oracle (``mutate.warp_mutants``,
the same ids in the same site order), and puts it next to what the GPU showed: a mutant
was *seen* if any launch with random delays did not match. The prediction holds where
``racy`` coincides with seen and ``benign`` with never seen. Needs no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ir import reference_programs
from .machine import Sym
from .mutate import warp_mutants
from .oracle import explore, label


def predictions() -> dict:
    """``{mutant id: (site task, oracle label)}`` for the warp mutants of every benign reference."""
    sym, out = Sym(), {}
    for ref in reference_programs():
        ref_out = explore(ref, sym)[0]
        if label(ref, ref, sym, ref_out)["label"] != "benign":
            continue
        for mu in warp_mutants(ref):
            out[mu.id] = (mu.site, label(mu.program, ref, sym, ref_out)["label"])
    return out


def compare(gpu_rows: list) -> list:
    """One row per warp mutant: the oracle's label, the GPU's mismatch rates, agreement."""
    obs = {r["id"]: r["obs"] for r in gpu_rows}
    rows = []
    for mid, (site, lab) in predictions().items():
        if mid not in obs:
            raise KeyError(f"{mid} is not in the GPU run")
        rate = {c: 1 - o["match"] / sum(o.values()) for c, o in obs[mid].items()}
        seen = rate["random"] > 0
        rows.append({"id": mid, "site": site, "label": lab, "rate_none": round(rate["none"], 4),
                     "rate_random": round(rate["random"], 4), "seen": seen,
                     "agrees": seen == (lab == "racy")})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="results/msync_gpu_races_cuda.json")
    ap.add_argument("--out", default="results/msync_warp_labels.json")
    args = ap.parse_args(argv)
    run = json.loads(Path(args.gpu).read_text())
    rows = compare(run["rows"])
    for r in rows:
        print(f"{r['id']:<30} {r['site']:<4} {r['label']:<7} random={r['rate_random']:.2f} "
              f"{'agrees' if r['agrees'] else 'DISAGREES'}")
    res = {"command": "python -m triton_equiv.msync.warp_check --gpu " + args.gpu, "gpu_run": run["command"],
           "gpu": run["gpu"], "agree": sum(r["agrees"] for r in rows), "mutants": len(rows), "rows": rows}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"{res['agree']} of {res['mutants']} agree")
    return res


if __name__ == "__main__":
    main()
