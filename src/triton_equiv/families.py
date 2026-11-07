"""The seven mutant families the audits report, from one oracle label.

``fporacle`` writes labels and counts them, and ``harness`` groups checker verdicts by them.
Both derive the family here so the two tables cannot name the same mutants differently.
"""

from __future__ import annotations

FAMILIES = ("reference", "oob", "inequivalent", "numerical:precision", "numerical:reorder",
            "benign:proved", "benign:unproved")


def family(lab) -> str:
    """Family of a label (an ``FpLabel`` or its ``asdict``); ``None`` is the reference kernel."""
    if lab is None:
        return "reference"
    get = lab.get if isinstance(lab, dict) else lambda k: getattr(lab, k)
    if get("label") == "numerical":
        return "numerical:precision" if get("precision_loss") else "numerical:reorder"
    if get("label") == "benign":
        return "benign:proved" if get("fp_proof") else "benign:unproved"
    return get("label")
