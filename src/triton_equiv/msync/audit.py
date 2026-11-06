"""Label every schedule mutant and audit every checker against the labels.

    python -m triton_equiv.msync.audit            # writes results/msync_audit.json

The run takes seconds on a laptop CPU; no GPU or Triton is needed.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

from .checkers import CHECKERS, run_checker
from .ir import reference_programs
from .machine import Sym
from .mutate import mutants
from .oracle import explore, label

LABELS = ("oob", "deadlock", "racy", "inequivalent", "benign")


def audit():
    sym = Sym()
    refs, rows = [], []
    for ref in reference_programs():
        ref_outcomes, dead, fault, n = explore(ref, sym)
        rl = label(ref, ref, sym, ref_outcomes)
        refs.append({"program": ref.name, "label": rl["label"], "outcomes": len(ref_outcomes), "states": n,
                     "tasks": len(ref.tasks),
                     "checkers": {c: run_checker(c, ref, ref) for c in CHECKERS}})
        if rl["label"] != "benign":
            continue  # mutants are only meaningful against a reference with one outcome
        for m in mutants(ref):
            lab = label(m.program, ref, sym, ref_outcomes)
            rows.append({"id": m.id, "program": ref.name, "operator": m.operator, "site": m.site, **lab,
                         "checkers": {c: run_checker(c, m.program, ref) for c in CHECKERS}})
    return refs, rows


def table(rows):
    """Accepted / applicable per (label, checker)."""
    out = {}
    for lab in LABELS:
        rs = [r for r in rows if r["label"] == lab]
        out[lab] = {"n": len(rs)}
        for c in CHECKERS:
            app = [r for r in rs if r["checkers"][c] is not None]
            out[lab][c] = f"{sum(r['checkers'][c] for r in app)}/{len(app)}"
    return out


def main(argv=None):
    t0 = time.time()
    refs, rows = audit()
    res = {"command": "python -m triton_equiv.msync.audit",
           "seconds": round(time.time() - t0, 1),
           "references": refs,
           "label_counts": dict(Counter(r["label"] for r in rows)),
           "operator_counts": dict(Counter(r["operator"] for r in rows)),
           "accepted_by_label": table(rows),
           "mutants": rows}
    path = Path(argv[0] if argv else "results/msync_audit.json")
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("seconds", "label_counts", "accepted_by_label")}, indent=1))
    for r in refs:
        print(r)
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
