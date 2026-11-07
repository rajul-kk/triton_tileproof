"""The persistent kernel against the step machine, in Triton's CPU interpreter.

The interpreter runs launched programs one after another, so spin-waits between programs
cannot be tested here; ``serial`` mode runs every task on one program in list order and is
compared with ``machine`` on the same program moved onto one SM. Skipped without triton."""

import dataclasses
import math
import os

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
pytest.importorskip("torch")
pytest.importorskip("triton")

from triton_equiv.msync import gpu, machine  # noqa: E402
from triton_equiv.msync.ir import reference_programs  # noqa: E402
from triton_equiv.msync.mutate import mutants  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1",
                                reason="these tests need the CPU interpreter")


def one_sm(prog):
    return prog.copy(n_sms=1, tasks=[dataclasses.replace(t, sm=0) for t in prog.tasks])


def _programs():
    for ref in reference_programs():
        yield ref.name, ref
        for m in mutants(ref):
            yield m.id, m.program


EXPECT = {"ok": gpu.RunOutcome.COMPLETED, "deadlock": gpu.RunOutcome.TIMED_OUT,
          "fault": gpu.RunOutcome.FAULTED}


def _same(a, b):
    return all((math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-5)
               for x, y in zip(a, b)) and len(a) == len(b)


@pytest.mark.parametrize("name,prog", list(_programs()), ids=lambda v: v if isinstance(v, str) else "")
def test_serial_kernel_matches_machine(name, prog):
    arena = machine.num_arena(prog, machine.num_inputs(prog, 0), "nan")
    status, m = machine.run(one_sm(prog), arena)
    got = gpu.launch(gpu.encode(prog), arena, mode="serial", max_spin=4, device="cpu")
    assert got.outcome == EXPECT[status], (name, status, got)
    if status == "ok":
        assert _same(list(m.output()), got.output), (name, m.output(), got.output)


def test_states_are_final():
    with pytest.raises(RuntimeError):
        gpu.outcome_of([gpu.ProgState.DONE, gpu.ProgState.RUNNING])
    assert gpu.outcome_of([1, 2, 3]) is gpu.RunOutcome.FAULTED
    assert gpu.outcome_of([1, 2]) is gpu.RunOutcome.TIMED_OUT


@pytest.mark.parametrize("mode", ["per_task", "dynamic"])
def test_topological_modes_match_reference(mode):
    # the interpreter runs programs in turn, so ``dynamic``'s first program takes every task
    for ref in reference_programs():
        arena = machine.num_arena(ref, machine.num_inputs(ref, 3), "nan")
        status, m = machine.run(ref, arena)
        got = gpu.Runner(gpu.encode(ref), arena, mode, n_programs=3, device="cpu").run(max_spin=4)
        assert status == "ok" and got.outcome is gpu.RunOutcome.COMPLETED, (ref.name, got)
        assert _same(list(m.output()), got.output), ref.name


def test_delays_do_not_change_a_serial_run():
    ref = reference_programs()[0]
    arena = machine.num_arena(ref, machine.num_inputs(ref, 0), "nan")
    r = gpu.Runner(gpu.encode(ref), arena, "serial", device="cpu")
    a = r.run()
    b = r.run(delays=[[3, 2, 1]] * len(ref.tasks))
    assert a.outcome is b.outcome is gpu.RunOutcome.COMPLETED and _same(a.output, b.output)
    with pytest.raises(ValueError):
        r.run(delays=[1, 2])


def test_kernel_literals_match_the_host_side():
    lit = gpu.KERNEL_LITERALS
    assert lit["states"] == {s.name: s.value for s in gpu.ProgState}
    assert lit["steps"] == {s.name: s.value for s in gpu.StopStep}
    assert lit["fields"] == {k: getattr(gpu, k) for k in lit["fields"]}
    assert lit["ops"] == gpu.OPS
    assert lit["delays"] == {k: getattr(gpu, k) for k in lit["delays"]}
