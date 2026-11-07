"""Accepted-by-label table from a KernelGYM audit run, with non-compiling mutants set apart.

    python -m triton_equiv.kgym_table results/kernelgym_audit_t4.json

A Triton kernel compiles on its first launch, so a compile error reaches KernelGYM as an
evaluation error while ``compiled`` stays True. A run counts as "does not compile" when its
error is Triton's compiler diagnostic (``... failed: at <line>:<col>:`` followed by the
source). Mutants that do not compile in every run are reported as their own row and left out
of the accepted/rejected counts for their oracle label: the label describes a program the
compiler would never accept.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

from .families import FAMILIES

_COMPILE_ERR = re.compile(r"failed: at \d+:\d+:\n")
COLUMNS = (("tile", "explicit_default"), ("tile", "kernelbench"), ("odd", "explicit_default"), ("odd", "kernelbench"))


def does_not_compile(row) -> bool:
    return bool(row.get("error")) and bool(_COMPILE_ERR.search(row["error"]))


def table(rows):
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    nocompile = sorted(i for i, rs in by_id.items() if all(does_not_compile(r) for r in rs))
    out = {}
    for fam in FAMILIES[1:]:
        ids = sorted(i for i, rs in by_id.items() if rs[0]["oracle"] == fam and i not in nocompile)
        out[fam] = {"n": len(ids)}
        for kind, pol in COLUMNS:
            out[fam][f"{kind}/{pol}"] = sum(any(r["size_kind"] == kind and r["policy"] == pol
                                                and r["correctness"] is True for r in by_id[i]) for i in ids)
    out["does not compile"] = {"n": len(nocompile), "ids": nocompile,
                               "oracle": [by_id[i][0]["oracle"] for i in nocompile]}
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    res = table(json.load(open(argv[0] if argv else "results/kernelgym_audit_t4.json"))["rows"])
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
