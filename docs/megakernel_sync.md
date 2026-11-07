# Checking megakernel synchronisation

A megakernel runs a whole model forward pass as one persistent kernel. SMs pass work to
each other through counters in global memory instead of kernel boundaries (Hazy Research's
Llama megakernels, MPK arXiv 2512.22219, AutoMegaKernel arXiv 2606.09682). That moves
correctness from "each kernel is right" to "each kernel is right **and** the cross-SM
protocol orders every read after its writes, never hangs and leaves no stale state".

What is known (literature read 2026-11-07):

* Hazy Research and MPK publish no correctness tests. AutoMegaKernel checks its schedule
  graph with a static validator ("not a mechanized proof"), then compares logits within a
  tolerance and greedy tokens against eager PyTorch.
* RESOLVE (arXiv 2610.05683) inserted random delays with NVBit and compared outputs
  bitwise. It found a grid-barrier reset race and an unoffset scratch buffer in one
  megakernel, and order-dependent token dropping in two MoE kernels. Most of them showed up
  only under injected delays. MPK hung under the tool.
* The only formal result is a Lean proof of a 2-CTA, 15-instruction ticket kernel under
  the PTX memory model (arXiv 2611.04447). I found no planted-bug benchmark for megakernel
  synchronisation and no audit of the checkers megakernel projects use.

This is the gap this work targets. It reuses the repo's method (plant labelled bugs, audit
checkers against the labels) on the synchronisation layer instead of on Triton masks.

## Plan

**Part 1: model, oracle, mutants, audit (CPU; done, below).** A schedule IR cut down from
AutoMegaKernel's, an oracle that explores every interleaving, 12 mutation operators and an
audit of six checkers. Deliverable: labelled mutants and an accepted-by-label table.

**Part 2: the real AutoMegaKernel.** Pin AMK at a commit, translate each mutant into its
`MegakernelProgram` JSON and run its own `validate()`, `simulate_counters()` and
`simulate_adversarial()`, to confirm or refute the two static-rule gaps found in part 1
with its code rather than our reading of its docs. Then, on a Kaggle T4 (sm_75 is in AMK's
range), plant tile, tail and counter bugs in megakernels AMK generates for its small Llama
test models and run its logit-tolerance and greedy-token gate. Report upstream if the
gaps hold. Needs: cloning and running AMK (not done in part 1), one Kaggle GPU session.

**Part 3: hardware.** Move the delay-injection checker onto the GPU: a Triton persistent
kernel with counter waits, `__nanosleep`-style delays and the part-1 mutants, to measure
how often each race actually shows on a T4 with and without delays. Then measure launch
latency of the same model as a megakernel and as one kernel per op, and fit the tails with
tinytail's EVT/pWCET code, checking whether static schedules give i.i.d. timings and
dynamic ones do not. No published tail-latency study of megakernels was found.

## Part 1: what was built

`src/triton_equiv/msync/`, pure Python, `python -m triton_equiv.msync.audit` writes
`results/msync_audit.json` in under a second. Tests: `tests/test_msync_*.py`.

**IR** (`ir.py`). Tasks run on SMs in queue order; a task waits until counters reach
static thresholds, reads, writes and increments one counter, as in AMK's IR. Buffers sit
back to back in one arena, so an out-of-bounds tile touches the next buffer, as separate
allocations do on a GPU; only accesses past the arena (plus 8 spare cells, for allocation
rounding) fault. Two operations outside AMK's IR model RESOLVE's bugs: `reset` (a reusable
barrier zeroes its counter) and `ticket` (MoE slot from an atomic counter). A task flag
moves the counter increment before the writes (a missing release fence).

**Reference programs** (`ir.py`), each small enough to explore exhaustively:

| Program | Shape |
|---|---|
| `mlp_rms_aligned` / `_odd` | GEMV tiles on 2 SMs, all-join on a counter, RMSNorm, second GEMV. N=4 / N=3 with tile 2 (a short tail tile) |
| `split_reduce_aligned` / `_odd` | 2 workers write partial sums to scratch slots, one task adds them. n=8 / n=7 |
| `barrier2` | two layers with a grid barrier each (one counter per barrier, AMK style), then a norm |
| `moe_dispatch` | 3 tokens take slots from an atomic ticket, capacity 2. Nondeterministic by design (not mutated) |

**Oracle** (`oracle.py`, `machine.py`). Each task is three atomic steps (acquire and
read, write, signal) and the oracle explores every interleaving of the SMs' steps by
memoised DFS (at most 88 states here). Values are hash-consed terms with every operation
uninterpreted, so "every interleaving gives the reference's output terms" holds for all
input values. Labels, in priority order: `oob` (a region outside its buffer, or a fault),
`deadlock` (a reachable stuck state), `racy` (more than one output), `inequivalent` (one
output, not the reference's), `benign`. Every `racy` and `inequivalent` label has a
witness interleaving that was replayed in float64 and confirmed to differ (all 47 did).
Limits: proofs hold only at these sizes and under sequential consistency at step
granularity; weaker PTX orderings inside a step are not modelled.

**Mutants** (`mutate.py`): 99 over the five deterministic references. Labels: 39 racy,
27 deadlock, 14 benign, 11 oob, 8 inequivalent.

| Operator | n | Labels |
|---|---|---|
| `partial_wait` | 7 | 7 racy |
| `threshold_plus_one` | 11 | 11 deadlock |
| `drop_wait` | 11 | 9 racy, 2 benign |
| `wrong_wait_counter` | 9 | 4 deadlock, 3 racy, 2 benign |
| `signal_before_write` | 14 | 9 racy, 5 benign |
| `swap_same_sm` | 11 | 11 deadlock |
| `tile_shift` | 16 | 8 oob, 8 inequivalent |
| `tile_unclamped` | 3 | 3 oob (odd sizes only) |
| `scratch_no_offset` | 2 | 2 racy |
| `counter_not_zeroed` | 13 | 8 racy, 5 benign |
| `barrier_reuse_reset` | 1 | deadlock (RESOLVE's barrier bug) |
| `barrier_reuse_no_reset` | 1 | racy |

The benign ones are real: a dropped wait whose producer already runs earlier on the same
SM, an early signal on a counter nobody waits on in time, a stale counter nobody reads.

## Part 1 results: accepted by checker

Accepted / applicable. A checker is "not applicable" when the mutant cannot be written in
AMK's IR (reset, ticket, early signal) or, for argmax, when the output is one value.

| Oracle label | n | warm, round robin, allclose | zeroed, rr, allclose | NaN, rr, allclose | warm, rr, argmax | 20 random schedules, NaN, exact | AMK rules (our reading) |
|---|---|---|---|---|---|---|---|
| oob | 11 | 7/11 | 3/11 | 2/11 | 8/8 | 2/11 | **11/11** |
| deadlock | 27 | 1/27 | 1/27 | 1/27 | 1/23 | 0/27 | 2/26 |
| racy | 39 | **37/39** | 29/39 | 29/39 | 29/29 | 0/39 | 8/30 |
| inequivalent | 8 | 6/8 | 0/8 | 0/8 | 6/6 | 0/8 | 2/8 |
| benign | 14 | 14/14 | 14/14 | 14/14 | 10/10 | 14/14 | 5/9 |

Reading:

* **One launch on warm memory catches almost nothing.** Repeated benchmark launches leave
  the previous launch's correct values in every buffer, so a consumer that runs early reads
  the right numbers anyway: 37 of 39 races and 6 of 8 wrong-tile mutants pass. Poisoning
  memory with NaN catches the wrong tiles but still passes 29 of 39 races, because the
  round-robin schedule happens to order them correctly. Greedy-token (argmax) agreement
  accepts every race and every wrong-tile mutant it applies to.
* **Schedule perturbation catches every race** in this model: 20 random interleavings on
  poisoned memory reject all 39 races and 27 deadlocks, including the barrier-reset race
  that the round-robin schedule completes. This is RESOLVE's finding reproduced in a model;
  on hardware schedules are far more correlated, which part 3 measures.
* **Static rules catch most synchronisation bugs but none of the bounds bugs.** Our
  reading of AMK's validator rejects 22 of 30 applicable races and 24 of 26 deadlocks, and
  accepts all 11 out-of-bounds tiles (it does not check that tiles stay inside buffers) and
  both tile-coverage bugs. The two unclamped odd-size tails in `mlp_rms_odd` also pass
  every run-and-compare checker: one writes a row into the next buffer, which the norm
  overwrites before anyone reads it, the other into the allocation's rounding padding.
  This is the same tile/tail blind spot as in KernelGYM's check, one level up.
* **Two possible rule gaps, checked against AMK's code in part 2:**
  1. *Transitive same-SM inversion* (`mlp_rms_*:swap_same_sm:2`). `h0` comes before `g1`
     in SM 1's task list, and `h0` depends on `g1` only through the norm on SM 0. With SM
     queues in task-list order that deadlocks on every schedule, and a queue-order rule
     that checks only direct edges passes it. **Not a gap in AMK** (part 2): its CUDA
     loader builds each SM's queue from the topological order, not the task list, so this
     schedule cannot hang there.
  2. *Counters not zeroed* (8 racy mutants). The rules never look at a counter's initial
     value; AMK leaves zeroing to the host. Confirmed in AMK's code (part 2).
* **False rejections:** 4 of 9 benign mutants fail the happens-before rule because it uses
  counter edges only and ignores ordering that the SM queue already guarantees.

Caveat: `amk_rules` is our re-implementation from AMK's `docs/IR_SPEC.md` and a summary of
`schedule/ir.py`; it is not AMK's code. Part 2 exists to replace it with the real validator.

## Part 2: the real AutoMegaKernel (code written, results pending)

Pinned commit: `884534c4d2dbed4381576f81a79ae97feca53c34`. AMK is not a dependency;
both scripts read `AMK_ROOT`.

**What reading AMK's code confirmed.** `schedule/ir.py` `validate()` checks
SM queue order, in task-list order, only on direct producer-to-consumer edges. That is not
a hole: `lower()` leaves `Task.sm` unset, and the CUDA loader (`vm/loader.py`
`_assign_sms`) builds every SM's queue as a subsequence of `topological_order()`, which
cannot deadlock on an acyclic graph. The transitive inversion from part 1 therefore does
not exist on AMK's GPU path. `validate()` never reads `Counter.init`.
It never compares a GEMV tile's `n_off + N_tile` with the output buffer's shape.
`simulate_counters()` and `simulate_adversarial()` fire any ready task in any order and
ignore the per-SM queues. Correctness is decided by the CPU `ReferenceVM`, which also
ignores SMs and sweeps the task list in order, with `logit_equivalence` (allclose and
top-1 ≥ 0.99). A GPU mismatch in `harness._evaluate_program` adds a note but does not
change `correct`.

**`msync/amk_bridge.py`** translates part-1 programs into AMK's `MegakernelProgram`
(see the module docstring) and runs AMK's `validate()` and both of its simulators on them.
It adds `queue_sim`, a simulator that lets each SM start only the head of its own queue.
Programs that use `reset`, `ticket` or an early signal are skipped.

**`msync/amk_e2e.py`** lowers AMK's own toy Llama (`make_toy(n_layers=1)`, fp32, one
decode step) with AMK's lowerer. It then plants four kinds of bug in that real schedule:
a stale counter init, the last GEMV tile overrunning its buffer, the last tile one column
short (a control), and the transitive queue inversion from part 1. `lower()` leaves SMs
unassigned, so the script assigns 4 SMs round robin over the topological order, as the
loader's `sm_round_robin` mode does.
It records the four schedule checks above and AMK's authoritative gate for each mutant,
and writes `results/amk_toy_mutants.json`.

**Tests**: `tests/test_msync_amk.py` is skipped unless `AMK_ROOT` is set.

Expected, from the code but **not yet run**:

* the queue inversion passes `validate()`, both AMK simulators and the ReferenceVM gate;
  `queue_sim` in task-list order reports it stuck, and in the loader's order it does not
  (`loader_queue_stuck` empty), so it does not hang AMK;
* the stale counter init passes `validate()`;
* the overrunning tile passes `validate()`, and the ReferenceVM truncates it through
  torch slicing.

The GPU half of part 2 (running AMK's CUDA `MegakernelVM` on a T4 for the mutants the
gate accepts) is still to do.

Run:

    AMK_ROOT=/path/to/AutoMegaKernel PYTHONPATH=src python -m triton_equiv.msync.amk_e2e
    AMK_ROOT=/path/to/AutoMegaKernel PYTHONPATH=src python -m pytest tests/test_msync_amk.py

## Part 3: the same programs on a GPU (code written, results pending)

`msync/gpu.py` runs a schedule program as one persistent Triton kernel. Each launched
program plays one SM and walks its queue in task-list order (the order `machine` and
part 1 use). Programs synchronise through the program's counters in global memory:

* a wait is an `atomic_add(c, 0, sem="acquire")` poll, then a block barrier;
* a signal is a block barrier, then an `atomic_add(c, 1, sem="release")`;
* `signal_first` signals before the writes, which is the missing fence the mutant models;
* `ticket` takes its slot with an atomic fetch-and-add, and `reset` is an `atomic_xchg`.

All buffers live in one float32 arena laid out as in `ir.Program`. An off-by-one region
therefore reads or writes the neighbouring buffer, as on real hardware. A region outside
the arena is a fault.

Every program's state is explicit, in `ProgState`:

    RUNNING -> DONE | TIMED_OUT | FAULTED     (all three final)

A wait gives up after `max_spin` polls, so a deadlock ends the launch as `TIMED_OUT`
instead of hanging the GPU. `RunOutcome` folds the programs' states for one launch, and
`races.Obs` compares a launch with the reference output.

Delays can be injected at three points of every task: before its waits, between its
reads and its write (a slow producer), and between write and signal. Each delay is a
spin of volatile loads.

Modes:

* `parallel`: the megakernel.
* `dynamic`: programs pull tasks from an atomic head over a topological order.
* `per_task`: one launch per task, the eager baseline.
* `serial`: one program in list order, for the CPU interpreter.

**Tests**: `tests/test_msync_gpu.py` runs under `TRITON_INTERPRET=1` (Triton 3.9 in
WSL, no GPU needed). It checks that serial mode gives `machine`'s outcome and output on
all 6 reference programs and all 106 of their mutants (the 99 audited ones, plus
the 7 of `moe_dispatch`, which the audit skips because that reference is racy).
It also checks `per_task` and `dynamic` on the references, and that delays do not change
a serial run. The interpreter runs
launched programs one after another, so it cannot test spin-waits between programs:
only the GPU run can.

Two experiments, in `notebooks/msync_gpu.ipynb` (Kaggle T4, **not yet run**):

1. `python -m triton_equiv.msync.races` runs every reference and mutant 100 times in
   `parallel` mode, once with no delays and once with random delays (each delay point,
   with probability 1/2, gets up to 1000 spins). Each launch is recorded as `match`,
   `differs`, `timed_out` or `faulted`. Grouped by oracle label, this gives how many
   mutants a GPU test would ever catch, and how often. Any non-`match` launch of a
   benign program counts as a false alarm and is listed. There should be none.
2. `python -m triton_equiv.msync.latency` times 100k launches of each benign reference
   as `static` (`parallel`), `dynamic` and `per_task`, using CUDA events around the
   launches only. Each series goes through tinytail's `analyse()` (fit on the first
   third, hold-out on the rest, origin `hardware`).

What to expect, stated before the run:

* Deadlock mutants should time out on every launch, because waits that can never be
  met do not depend on timing. The exception is deadlocks that depend on the
  interleaving, such as the barrier-reuse ones.
* Without delays, races should rarely show. The programs are tiny and the GPU runs its
  SMs nearly in lockstep, which is the point RESOLVE makes about sanitizers.
* Random delays should expose many more races. Some should still never show: the oracle
  can prove an interleaving exists that hardware timing almost never produces.
* `per_task` should be slowest, by roughly one launch overhead per task.
* `dynamic` timings should be less i.i.d. than `static` ones, because the order in which
  programs win the atomic head changes from launch to launch. Whether the gates actually
  separate the two schedules is the open question.
