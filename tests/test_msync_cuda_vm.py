"""The C++ schedule VM (CPU build of csrc/msync_vm.h) against the step machine.

As with the Triton interpreter, programs run one after another on the CPU, so only serial
and per-task runs are checked here; the CUDA build is exercised in notebooks/msync_gpu.ipynb.
Skipped when the extension cannot be built."""

import pytest

pytest.importorskip("torch")

from triton_equiv.cuda_ref import Backend  # noqa: E402
from triton_equiv.msync import cuda_vm, gpu, machine  # noqa: E402
from triton_equiv.msync.ir import reference_programs  # noqa: E402

from msync_helpers import EXPECT, one_sm, programs  # noqa: E402
from msync_helpers import same as _same  # noqa: E402

_ERR = cuda_vm.build_error(Backend.CPU)
pytestmark = pytest.mark.skipif(_ERR is not None, reason=f"C++ extension: {_ERR}")


@pytest.mark.parametrize("name,prog", list(programs()), ids=lambda v: v if isinstance(v, str) else "")
def test_serial_vm_matches_machine(name, prog):
    arena = machine.num_arena(prog, machine.num_inputs(prog, 0), "nan")
    status, m = machine.run(one_sm(prog), arena)
    r = cuda_vm.CudaRunner(gpu.encode(prog), arena, "serial", backend=Backend.CPU).run(timeout_ns=100_000)
    assert r.launch.outcome == EXPECT[status], (name, status, r.launch)
    if status == "ok":
        assert _same(list(m.output()), r.launch.output), name
        assert all(t is not None and t[1] >= t[0] for t in r.task_ns), name


@pytest.mark.parametrize("mode", ["per_task", "dynamic"])
def test_topological_modes(mode):
    for ref in reference_programs():
        arena = machine.num_arena(ref, machine.num_inputs(ref, 3), "nan")
        _, m = machine.run(ref, arena)
        r = cuda_vm.CudaRunner(gpu.encode(ref), arena, mode, n_programs=3, backend=Backend.CPU).run(timeout_ns=100_000)
        assert r.launch.outcome is gpu.RunOutcome.COMPLETED and _same(list(m.output()), r.launch.output), ref.name


def test_layout_matches_the_python_side():
    lay, lit = cuda_vm.layout(), gpu.KERNEL_LITERALS
    for k, v in {**lit["states"], **lit["steps"], **lit["ops"], **lit["delays"]}.items():
        assert lay[k] == v, k
    for k in ("F_OP", "F_OC", "F_SIGFIRST", "F_NW", "F_W0C", "F_W0T", "F_R0B", "F_R0L", "F_R1B", "F_R1L",
              "F_WB", "F_WL", "F_K", "F_CAP", "F_AUX"):
        assert lay[k] == lit["fields"][k], k
    assert lay["NF"] == gpu.NF


def test_no_barrier_sites_and_single_thread_run():
    ref = reference_programs()[0]
    sites = cuda_vm.no_barrier_sites(ref)
    assert sites
    arena = machine.num_arena(ref, machine.num_inputs(ref, 0), "nan")
    a = cuda_vm.CudaRunner(gpu.encode(ref), arena, "serial", backend=Backend.CPU).run()
    b = cuda_vm.CudaRunner(gpu.encode(ref), arena, "serial", backend=Backend.CPU, no_barrier=sites).run(
        delays_ns=[[0, 0, 5]] * len(ref.tasks))
    # one thread has no other warp to lag behind, so the mutant cannot show on the CPU
    assert a.launch.outcome is b.launch.outcome is gpu.RunOutcome.COMPLETED and _same(a.launch.output, b.launch.output)


def test_in_kernel_time_of_a_cpu_run():
    from triton_equiv.msync import latency
    ref = reference_programs()[0]
    arena = machine.num_arena(ref, machine.num_inputs(ref, 0), "nan")
    run = cuda_vm.CudaRunner(gpu.encode(ref), arena, "serial", backend=Backend.CPU).run()
    assert latency.in_kernel_us(run) >= 0.0
    dead = cuda_vm.CudaRunner(gpu.encode(ref), arena, "parallel", backend=Backend.CPU).run(timeout_ns=1000)
    with pytest.raises(RuntimeError):  # cross-program waits time out on one CPU thread
        latency.in_kernel_us(dead)
