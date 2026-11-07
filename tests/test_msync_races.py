"""The race driver's bookkeeping (no GPU, no triton needed)."""

import math
import random

from triton_equiv.msync import gpu, races


def _launch(outcome, out=()):
    return gpu.Launch(outcome, [], [], list(out))


def test_observe():
    exp = [1.0, 2.0]
    assert races.observe(_launch(gpu.RunOutcome.COMPLETED, [1.0, 2.00001]), exp) is races.Obs.MATCH
    assert races.observe(_launch(gpu.RunOutcome.COMPLETED, [1.0, math.nan]), exp) is races.Obs.DIFFERS
    assert races.observe(_launch(gpu.RunOutcome.COMPLETED, [1.0]), exp) is races.Obs.DIFFERS
    assert races.observe(_launch(gpu.RunOutcome.TIMED_OUT), exp) is races.Obs.TIMED_OUT
    assert races.observe(_launch(gpu.RunOutcome.FAULTED), exp) is races.Obs.FAULTED


def test_random_delays_shape_and_range():
    d = races.random_delays(random.Random(0), 5, 7)
    assert len(d) == 5 and all(len(r) == 3 and all(0 <= v <= 7 for v in r) for r in d)


def test_summarise():
    z = {o.value: 0 for o in races.Obs}
    rows = [{"kind": "mutant", "label": "racy", "obs": {"none": z | {"match": 4}, "random": z | {"match": 2, "differs": 2}}},
            {"kind": "mutant", "label": "racy", "obs": {"none": z | {"match": 4}, "random": z | {"match": 4}}},
            {"kind": "reference", "label": "benign", "obs": {"none": z | {"match": 4}, "random": z | {"match": 4}}}]
    s = races.summarise(rows)
    assert s == {"racy": {"mutants": 2, "none": {"seen": 0, "mean_rate": 0.0},
                          "random": {"seen": 1, "mean_rate": 0.25}}}
