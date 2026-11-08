"""Warps in the step machine: the no_barrier mutant, its oracle labels and the T4 run."""

import dataclasses
import json
from pathlib import Path

import pytest

from triton_equiv.msync import ir, warp_check
from triton_equiv.msync.ir import Buffer, Region, Task, Wait, reference_programs
from triton_equiv.msync.machine import Machine, Num, Sym, num_arena, num_inputs, replay, run, sym_arena
from triton_equiv.msync.mutate import mutants, no_barrier_sites, warp_mutants
from triton_equiv.msync.oracle import explore, label

ROOT = Path(__file__).resolve().parents[1]

# The oracle's labels, stated here so a change to the model shows up as a test failure.
LABELS = {
    "mlp_rms_aligned:no_barrier:0": "benign", "mlp_rms_aligned:no_barrier:1": "racy",
    "mlp_rms_aligned:no_barrier:2": "racy", "mlp_rms_odd:no_barrier:0": "benign",
    "mlp_rms_odd:no_barrier:1": "racy", "barrier2:no_barrier:0": "racy", "barrier2:no_barrier:1": "racy",
    "barrier2:no_barrier:2": "benign", "barrier2:no_barrier:3": "racy",
}


def owner(e, nthreads):  # csrc/msync_vm.h
    if nthreads < 64:
        return e % nthreads
    warps = nthreads // 32
    return (e % warps) * 32 + (e // warps) % 32


def test_warp_of_is_the_cuda_vm_owner():
    assert ir.WARPS == 128 // 32
    for e in range(64):
        assert ir.warp_of(e) == owner(e, 128) // 32


def two_sm(no_barrier=True):
    """SM 0 writes y[0:4] and signals; SM 1 waits and copies y to out (scale)."""
    bufs = [Buffer("x", 4, "input"), Buffer("y", 4, "act"), Buffer("out", 4, "output")]
    tasks = [Task("p", 0, "scale", reads=(Region("x", 0, 4),), writes=(Region("y", 0, 4),), out_counter="c",
                  no_barrier=no_barrier),
             Task("q", 1, "scale", reads=(Region("y", 0, 4),), writes=(Region("out", 0, 4),), out_counter="d",
                  waits=(Wait("c", 1),))]
    return ir.Program("two_sm", 2, bufs, {"c": 0, "d": 0}, tasks)


def test_warps_write_in_their_own_steps_and_the_task_waits_for_them():
    prog = two_sm()
    m = Machine(prog, Num(), num_arena(prog, num_inputs(prog, 0), "nan"))
    assert m.actions() == [(0, 0)]
    m.step(0)                                   # acquire
    assert m.actions() == [(0, 0), (0, 1), (0, 2), (0, 3)]
    m.step(0)                                   # warp 0 writes y[0]
    m.step(0)                                   # signal
    assert m.phase[0] == 3 and m.actions() == [(0, 1), (0, 2), (0, 3), (1, 0)]
    m.step(1)                                   # SM 1 reads y while y[1:4] are unwritten
    for k in (1, 2, 3):
        m.step(0, k)
    assert m.current(0) is None                 # the task ended with its last warp
    m.step(1), m.step(1)
    out = m.output()
    assert not any(v != v for v in out[:1]) and all(v != v for v in out[1:])  # NaN where stale


def test_round_robin_lets_the_warps_finish_first():
    prog, ref = two_sm(), two_sm(no_barrier=False)
    arena = num_arena(prog, num_inputs(prog, 0), "nan")
    assert run(prog, arena)[1].output() == run(ref, arena)[1].output()


def test_two_sm_labels():
    sym = Sym()
    ref_out = explore(two_sm(False), sym)[0]
    lab = label(two_sm(), two_sm(False), sym, ref_out)
    assert lab["label"] == "racy" and lab["witness_confirmed"]
    trace = json.loads(json.dumps(lab["witness_trace"]))  # (sm, lane) pairs survive as lists
    prog = two_sm()
    m = replay(prog, num_arena(prog, num_inputs(prog, 1), "nan"), trace)
    assert m.done()


def test_same_sm_consumer_is_protected_by_the_closing_barrier():
    prog = two_sm()
    prog = prog.copy(tasks=[prog.tasks[0], dataclasses.replace(prog.tasks[1], sm=0)])
    sym = Sym()
    ref = two_sm(False)
    ref = ref.copy(tasks=[ref.tasks[0], dataclasses.replace(ref.tasks[1], sm=0)])
    assert label(prog, ref, sym)["label"] == "benign"


def test_flag_off_changes_nothing():
    sym = Sym()
    for ref in reference_programs():
        m = Machine(ref, sym, sym_arena(ref, sym))
        assert all(k == 0 for _, k in m.actions())
        assert not any(t.no_barrier for mu in mutants(ref) for t in mu.program.tasks)


def test_warp_mutant_sites_and_ids():
    for ref in reference_programs():
        ms = warp_mutants(ref)
        assert [m.id for m in ms] == [f"{ref.name}:no_barrier:{k}" for k in range(len(no_barrier_sites(ref)))]
        for m, i in zip(ms, no_barrier_sites(ref)):
            assert [j for j, t in enumerate(m.program.tasks) if t.no_barrier] == [i]


def test_oracle_labels_of_the_warp_mutants():
    assert {k: v[1] for k, v in warp_check.predictions().items()} == LABELS


def test_labels_match_the_t4_run():
    """results/msync_gpu_races_cuda.json: seen on the T4 with random delays iff racy."""
    path = ROOT / "results" / "msync_gpu_races_cuda.json"
    if not path.exists():
        pytest.skip("no T4 run in results/")
    rows = warp_check.compare(json.loads(path.read_text())["rows"])
    assert len(rows) == len(LABELS) and all(r["agrees"] for r in rows), rows


def test_triton_refuses_and_cuda_vm_takes_the_flag():
    torch = pytest.importorskip("torch")
    from triton_equiv.msync import gpu
    prog = warp_mutants(reference_programs()[0])[1].program
    arena = num_arena(prog, num_inputs(prog, 0), "nan")
    with pytest.raises(ValueError, match="CUDA VM"):
        gpu.Runner(gpu.encode(prog), arena, "serial", device="cpu")
    cuda_vm = pytest.importorskip("triton_equiv.msync.cuda_vm")
    from triton_equiv.cuda_ref import Backend
    if cuda_vm.build_error(Backend.CPU):
        pytest.skip(cuda_vm.build_error(Backend.CPU))
    r = cuda_vm.CudaRunner(gpu.encode(prog), arena, "serial", backend=Backend.CPU)
    assert torch.nonzero(r.no_barrier).flatten().tolist() == cuda_vm.no_barrier_tasks(prog)
