"""The paced latency design, with a fake timer: no GPU needed."""

from triton_equiv.msync import latency
from triton_equiv.msync.ir import reference_programs


def fake(log):
    def timers(prog, backend, device):
        return {s: (lambda s=s: log.append(s) or float(len(log)))
                for s in ("static", "dynamic", "per_task")}
    return timers


def test_rounds_run_every_schedule_once_in_shuffled_orders():
    log, waits = [], []
    x, gaps = latency.paced_series(reference_programs()[0], 200, gap_us=50.0, seed=3,
                                   timer=fake(log), wait=waits.append)
    timed = log[3 * latency.WARMUP:]
    rounds = [timed[i:i + 3] for i in range(0, len(timed), 3)]
    assert len(rounds) == 200 and all(sorted(r) == ["dynamic", "per_task", "static"] for r in rounds)
    assert len({tuple(r) for r in rounds}) == 6          # every order shows up
    assert {s: len(v) for s, v in x.items()} == {"static": 200, "dynamic": 200, "per_task": 200}
    assert len(waits) == 600 and all(0.0 <= g <= 50.0 for g in waits)
    assert sorted(sum(gaps.values(), [])) == sorted(waits)


def test_same_seed_same_design():
    a = latency.paced_series(reference_programs()[0], 20, seed=1, timer=fake([]), wait=lambda g: None)[1]
    b = latency.paced_series(reference_programs()[0], 20, seed=1, timer=fake([]), wait=lambda g: None)[1]
    assert a == b


def test_paired_differences():
    x = {"static": [1.0, 2.0, 3.0], "dynamic": [2.0, 2.5, 2.0], "short": [1.0]}
    p = latency.paired(x)
    assert set(p) == {"dynamic-static"}                  # unequal lengths are not paired
    assert p["dynamic-static"] == {"median_us": 0.5, "share_positive": 2 / 3, "rounds": 3}
