"""Bookkeeping of the race-checker audit (``msync.sanitize``): program lookup, report
attribution and the summary. The sanitizers themselves run on Kaggle."""

from collections import Counter

from triton_equiv.msync import sanitize


def test_catalogue_ids_resolve_to_programs():
    rows = sanitize.catalogue(warp=True)
    assert rows[0]["kind"] == "reference"
    for r in rows:
        prog, ref = sanitize.find(r["id"])
        assert ref.name == r["program"]
    ops = Counter(r["operator"] for r in rows)
    assert ops["no_barrier"] > 0 and ops[None] == 6


SAMPLE = """\
@@ reduce:reference
@@ obs reduce:reference match
@@ reduce:after_load
========= Error: Race reported between Write access at k_reduce(const float *, float *, unsigned int, const long long *)+0x2a0 in intra_sync.cu:45
=========     and Read access at k_reduce(const float *, float *, unsigned int, const long long *)+0x330 in intra_sync.cu:48 [128 hazards]
=========
@@ obs reduce:after_load differs
@@ scan:after_load
========= Warning: Race reported between Write access at k_scan+0x10 in intra_sync.cu:120
=========     and Read access at k_scan+0x20 in intra_sync.cu:125 [12 hazards]
@@ obs scan:after_load match
========= RACECHECK SUMMARY: 2 hazards displayed (1 error, 1 warning)
"""


def test_parse_attributes_reports_to_the_preceding_marker():
    got = sanitize.parse_cs(SAMPLE)
    assert got["reduce:reference"]["hits"] == 0 and got["reduce:reference"]["obs"] == "match"
    assert got["reduce:after_load"]["hits"] == 1 and got["reduce:after_load"]["by_class"] == {"race": 1}
    assert got["scan:after_load"]["hits"] == 1 and got["scan:after_load"]["obs"] == "match"


def test_summary_counts_flagged_programs_per_label():
    res = {"vm": [], "intra": [{"id": "a", "kind": "reference", "label": "benign", "scope": None},
                               {"id": "b", "kind": "mutant", "label": "racy", "scope": "mixed"},
                               {"id": "c", "kind": "mutant", "label": "benign", "scope": None}],
           "tools": {"racecheck": {"intra": {"rows": {"a": {"hits": 0}, "b": {"hits": 3}}}}}}
    s = sanitize.summarise(res)["racecheck/intra"]
    assert s["reference"] == {"programs": 1, "flagged": 0, "missing": 0}
    assert s["racy/mixed"] == {"programs": 1, "flagged": 1, "missing": 0}
    assert s["benign"] == {"programs": 1, "flagged": 0, "missing": 1}
