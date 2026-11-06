"""Source-level (AST) mutation operators for Triton kernels.

Stage-1 operators (all Triton-specific; see docs/scoop_check.md, differentiator 1):

``mask_off_by_one``    flip one strict/non-strict comparison: ``<`` <-> ``<=``, ``>`` <-> ``>=``.
``missing_store_mask`` drop the ``mask=`` keyword from one ``tl.store``.
``wrong_pid_axis``     ``tl.program_id(0)`` <-> ``tl.program_id(1)`` at one call site.
``wrong_other``        on one masked ``tl.load``: ``other=c`` -> ``other=c+1``; if no
                       ``other=`` is given, add ``other=1.0``.

Each operator yields **one mutant per application site** (first-order mutants), so a
kernel with three comparisons gets three ``mask_off_by_one`` mutants.

Why AST rewriting (and the alternatives rejected)
-------------------------------------------------
* **Regex/text substitution**: simple, but it cannot tell ``<`` in a mask from ``<`` in a
  string or a comment, and it cannot reliably find "the ``mask=`` keyword of a
  ``tl.store``". Rejected because sites would be ambiguous.
* **Mutating Triton IR (TTIR/TTGIR via MLIR)**: closest to what runs, but it needs a working
  Triton compiler on every machine (no official Windows wheel), and IR mutants do not map
  back to source a user can read. The mutations we model are source-level authoring bugs,
  so source is the right level. Rejected for stage 1.
* **Hand-written mutant files**: no systematic site enumeration, so coverage cannot be
  counted. Rejected.
* **Python ``ast``** (chosen): exact sites, the stdlib only, and ``ast.unparse`` gives
  valid source that real Triton, the interpreter and our symbolic shim can all run. The
  cost is that comments and formatting are lost in mutants, which does not matter here.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Callable, Iterator

OPERATORS = ("mask_off_by_one", "missing_store_mask", "wrong_pid_axis", "wrong_other")


@dataclass(frozen=True)
class Mutant:
    kernel: str
    operator: str
    site: int  # index of the application site within the kernel (0-based, source order)
    lineno: int
    description: str
    source: str

    @property
    def id(self) -> str:
        return f"{self.kernel}:{self.operator}:{self.site}"


def _is_tl_call(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
    )


def _kw(call: ast.Call, name: str) -> ast.keyword | None:
    return next((k for k in call.keywords if k.arg == name), None)


# Each site finder returns the nodes the operator applies to, in source order.
# Each applier mutates one node in place and returns a short description.

_FLIP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}


def _sites_mask_off_by_one(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _FLIP]


def _apply_mask_off_by_one(node: ast.Compare) -> str:
    old = type(node.ops[0])
    node.ops[0] = _FLIP[old]()
    sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
    return f"{sym[old]} -> {sym[_FLIP[old]]}"


def _sites_missing_store_mask(tree):
    return [n for n in ast.walk(tree) if _is_tl_call(n, "store") and _kw(n, "mask") is not None]


def _apply_missing_store_mask(node: ast.Call) -> str:
    node.keywords = [k for k in node.keywords if k.arg != "mask"]
    return "removed mask= from tl.store"


def _sites_wrong_pid_axis(tree):
    return [n for n in ast.walk(tree)
            if _is_tl_call(n, "program_id") and n.args
            and isinstance(n.args[0], ast.Constant) and n.args[0].value in (0, 1)]


def _apply_wrong_pid_axis(node: ast.Call) -> str:
    old = node.args[0].value
    node.args[0] = ast.Constant(1 - old)
    return f"program_id({old}) -> program_id({1 - old})"


def _sites_wrong_other(tree):
    out = []
    for n in ast.walk(tree):
        if _is_tl_call(n, "load") and _kw(n, "mask") is not None:
            other = _kw(n, "other")
            if other is None or (isinstance(other.value, ast.Constant)
                                 and isinstance(other.value.value, (int, float))):
                out.append(n)
    return out


def _apply_wrong_other(node: ast.Call) -> str:
    other = _kw(node, "other")
    if other is None:
        node.keywords.append(ast.keyword(arg="other", value=ast.Constant(1.0)))
        return "added other=1.0 to masked tl.load"
    old = other.value.value
    other.value = ast.Constant(old + 1)
    return f"other={old!r} -> other={old + 1!r}"


_OPS: dict[str, tuple[Callable, Callable]] = {
    "mask_off_by_one": (_sites_mask_off_by_one, _apply_mask_off_by_one),
    "missing_store_mask": (_sites_missing_store_mask, _apply_missing_store_mask),
    "wrong_pid_axis": (_sites_wrong_pid_axis, _apply_wrong_pid_axis),
    "wrong_other": (_sites_wrong_other, _apply_wrong_other),
}


def _ordered(nodes):
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def count_sites(source: str, operator: str) -> int:
    return len(_OPS[operator][0](ast.parse(source)))


def mutants(kernel: str, source: str, operators=OPERATORS) -> Iterator[Mutant]:
    """Yield every first-order mutant of ``source`` for the given operators."""
    base = ast.parse(source)
    for op in operators:
        find, apply = _OPS[op]
        for i in range(len(find(base))):
            tree = copy.deepcopy(base)
            node = _ordered(find(tree))[i]
            desc = apply(node)
            ast.fix_missing_locations(tree)
            yield Mutant(kernel, op, i, node.lineno, desc, ast.unparse(tree) + "\n")
