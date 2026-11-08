"""Shared by the two VM backends' tests (Triton kernel and C++ VM): neither needs a GPU."""

import dataclasses
import math

from triton_equiv.msync import gpu
from triton_equiv.msync.ir import reference_programs
from triton_equiv.msync.mutate import mutants


def one_sm(prog):
    return prog.copy(n_sms=1, tasks=[dataclasses.replace(t, sm=0) for t in prog.tasks])


def programs():
    """``(name, program)`` for every reference and every mutant of it."""
    for ref in reference_programs():
        yield ref.name, ref
        for m in mutants(ref):
            yield m.id, m.program


# machine.run status -> the launch outcome a serial run must give
EXPECT = {"ok": gpu.RunOutcome.COMPLETED, "deadlock": gpu.RunOutcome.TIMED_OUT,
          "fault": gpu.RunOutcome.FAULTED}


def same(a, b):
    return len(a) == len(b) and all(
        (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-5) for x, y in zip(a, b))
