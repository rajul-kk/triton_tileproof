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

## Part 2: the real AutoMegaKernel

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

Expected, from the code, before the run:

* the queue inversion passes `validate()`, both AMK simulators and the ReferenceVM gate;
  `queue_sim` in task-list order reports it stuck, and in the loader's order it does not
  (`loader_queue_stuck` empty), so it does not hang AMK;
* the stale counter init passes `validate()`;
* the overrunning tile passes `validate()`, and the ReferenceVM truncates it through
  torch slicing.

### Part 2 results

`results/amk_toy_mutants.json` (AMK `884534c`, torch 2.10 CPU, 38 tasks, 12 mutants).
Every prediction held.

| mutant | sites | `validate()` | `simulate_counters` | `queue_sim` stuck | loader stuck | gate |
|---|---|---|---|---|---|---|
| counter_init_stale | 3 | pass | pass | 0 | 0 | correct (err 0) |
| tile_oob | 3 | pass | pass | 0 | 0 | correct (err 0) |
| tile_gap (control) | 3 | pass | pass | 0 | 0 | 1 correct, 2 wrong (err 5e-3, 1e-2) |
| queue_inversion | 3 | pass | pass | 8 tasks | 0 | correct (err 0) |

* **`validate()` passes all 12 mutants.** That includes three stale counter inits, which
  would let a consumer start early on a GPU, and three tiles that write 4 elements past
  their buffer. AMK's gate calls all six `correct` with zero error. The ReferenceVM
  ignores the counters, and torch slicing silently clips the overrun.
* **The queue inversion is real in task-list order but unreachable in AMK.** `queue_sim`
  in task-list order leaves 8 tasks stuck. In the loader's topological order it leaves
  none. Part 1's transitive inversion is a hazard of the IR, not of AMK's GPU path.
* **The control behaves as a control should, with one exception.** `tile_gap` drops the
  last column of a tile, and AMK catches it in `o_proj` and `gate_proj` (allclose fails;
  top-1 still agrees). It does not catch it in `q_proj`, and that is correct: at `pos=0`
  attention has one key, so the softmax weight is 1 whatever `q` is, and `q` cannot
  change the output. A one-step decode cannot test the query path.
* **AMK's own race hunt flags its own reference.** `simulate_adversarial(seeds=96)` reports
  two read-before-write races on the unmutated schedule. `L0.k_append` and `L0.v_append`
  read `kcache`/`vcache`, which the same pass also writes, so AMK's
  `kv_written_this_pass` rule counts them as unwritten. The caches exist before the step,
  so these are false alarms. The `amk_sim_adv` column is therefore `False` on every row
  and does not separate mutants from the reference.

Not done yet: running AMK's CUDA `MegakernelVM` on a GPU (AMK lists a `t4` target) for the
ten mutants the gate accepts. It is the run that would show whether the stale counters
and overrunning tiles do damage on real hardware.

Run:

    AMK_ROOT=/path/to/AutoMegaKernel PYTHONPATH=src python -m triton_equiv.msync.amk_e2e
    AMK_ROOT=/path/to/AutoMegaKernel PYTHONPATH=src python -m pytest tests/test_msync_amk.py

## Part 3: the same programs on a GPU

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

Two experiments, in `notebooks/msync_gpu.ipynb` (Kaggle T4; results below):

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

### A second backend in CUDA C++

`csrc/msync_vm.h` is the same VM as the Triton kernel, written once over a hardware
policy. It reads the same task tables and queues, and gives the same `ProgState`,
`StopStep` and `RunOutcome`. `msync/cuda_vm.py` runs it (`CudaRunner`, with the same
interface as `gpu.Runner`), and `races`/`latency` take `--backend cuda`. Against the
Triton kernel it adds:

* waits as `ld.acquire.gpu` polls and signals as `atom.add.release.gpu`, written in PTX;
* a timeout in nanoseconds of `%globaltimer`, and delays of `__nanosleep` in nanoseconds;
* a time stamp at every task's start (after its waits) and end (after its signal).
  `latency --backend cuda` adds a `static_in_kernel` series, first task start to last
  task end, with no launch overhead or host jitter in it;
* the warp-level `no_barrier` mutant, which Triton cannot express. A task's output
  elements are dealt over the block's 4 warps. With `no_barrier`, thread 0 signals once
  warp 0 has written, without the block barrier, while the other warps' writers first
  wait out the D_POST delay. The step machine has no warps, so these rows are labelled
  `unmodelled`. `races --backend cuda` plants one at every `cuda_vm.no_barrier_sites`
  site: tasks that signal a waited-on counter and write more than one element.

`csrc/msync_vm_cpu.cpp` builds the header for one host thread.
`tests/test_msync_cuda_vm.py` checks that build in WSL:

* serial mode gives `machine`'s outcome and output on all 6 references and 106 mutants;
* `per_task` and `dynamic` match on the references;
* the C++ enum values equal `gpu.KERNEL_LITERALS`;
* `no_barrier` cannot show on a single thread.

The CUDA build is compiled on Kaggle by `notebooks/msync_gpu.ipynb`, section 3 (there is
no nvcc here).

## Part 3 results: Tesla T4

Kaggle T4, torch 2.11.0+cu130, Triton 3.6.0, tinytail 0.0.1. Both backends compiled and ran;
the control (every reference matches with and without delays) passed on both. Results are
in `results/msync_gpu_races.json`, `results/msync_gpu_races_cuda.json`,
`results/msync_gpu_latency.json` and `results/msync_gpu_latency_cuda.json`, with the raw
samples in `.samples.json.gz`.

### Races

100 launches per program, without delays (`none`) and with random delays (`random`).
Triton: delays up to 1000 spins. CUDA: delays up to 20 µs of `__nanosleep`. "Seen" counts
mutants with at least one non-`match` launch. The rate is the mean fraction of
non-`match` launches.

| label | mutants | Triton seen, none | Triton seen, random (rate) | CUDA seen, none | CUDA seen, random (rate) |
|---|---|---|---|---|---|
| racy | 39 | 9 | 39 (0.40) | 11 | 39 (0.43) |
| deadlock | 27 | 27 | 27 (0.99) | 26 | 27 (0.98) |
| inequivalent | 8 | 8 | 8 (1.00) | 8 | 8 (1.00) |
| oob | 11 | 9 | 9 (0.82) | 9 | 9 (0.82) |
| benign | 14 | 0 | 0 | 0 | 0 |
| unmodelled (`no_barrier`) | 9 | - | - | 0 | 6 (0.18) |

* **No false alarms.** No launch of a benign reference or benign mutant ever timed out,
  faulted or differed, on either backend. The bounded spins, release/acquire counters and
  outcome folding hold up on real hardware.
* **Without delays, most races do not show.** That matches the prediction. Only 9 (Triton)
  or 11 (CUDA) of 39 racy mutants show at all. They are the ones whose race does not need
  a slow producer: `scratch_no_offset` (always), `wrong_wait_counter`, and some `drop_wait`
  and `counter_not_zeroed` sites. No `partial_wait` or `barrier_reuse_no_reset` mutant
  shows on either backend. `signal_before_write` shows only on CUDA (2 of 9).
* **With random delays, every racy mutant shows.** This was against the prediction: no
  oracle-racy mutant stayed hidden. Rates range from 0.22 (`partial_wait`) to 1.0
  (`scratch_no_offset`). Under delay injection, a test catches everything the oracle calls
  racy, but a run without it misses about three quarters.
* **Deadlocks time out on every launch** except `barrier2:barrier_reuse_reset:0`. That
  one depends on the interleaving, as predicted. On CUDA it never deadlocked without
  delays and timed out on 46% of delayed launches. On Triton it timed out on every
  undelayed launch.
* **Out-of-bounds mutants never fault.** The arena is contiguous, so an off-by-one region
  lands in the neighbouring buffer and shows as `differs`, as designed. 2 of 11
  (`mlp_rms_odd:tile_unclamped:0/1`) never show at all. Their stray elements fall where
  they do not change the compared output. Only the oracle sees them.
* **`moe_dispatch`**, the racy reference, gave a different output from the list-order
  interleaving on 100% of undelayed launches and 77% of delayed ones.
* **`no_barrier`**, the warp-level mutant the step machine could not express at the time,
  never showed without delays. With delays it showed at 6 of 9 sites, at a mean rate of
  0.18. This was a real bug class, invisible to the part-1 oracle, that delay injection
  finds. Part 4 adds warps to the model, and the oracle then predicts exactly these 6.

### Latency

100k launches per series. tinytail fits on the first third and checks the rest. Medians
(µs) of the held-out two thirds:

| program | tasks | Triton static | dynamic | per_task | CUDA static | dynamic | per_task | in-kernel |
|---|---|---|---|---|---|---|---|---|
| mlp_rms_aligned | 5 | 50.9 | 55.6 | 155.1 | 28.7 | 32.2 | 45.1 | 16.0 |
| mlp_rms_odd | 5 | 49.5 | 55.3 | 153.6 | 29.1 | 30.8 | 45.1 | 14.6 |
| split_reduce_aligned | 3 | 49.7 | 50.8 | 102.4 | 22.5 | 23.8 | 34.0 | 7.7 |
| split_reduce_odd | 3 | 49.2 | 49.6 | 102.4 | 22.3 | 23.9 | 34.2 | 7.7 |
| barrier2 | 5 | 49.3 | 55.3 | 154.4 | 30.2 | 31.6 | 45.2 | 15.6 |

* **`per_task` is slowest**, as predicted. On Triton that is about 26 µs per extra launch,
  which is the Python launch path. On CUDA it is 4–6 µs per extra launch.
* **`dynamic` costs 0–6 µs over `static`**: the atomic head, and losing the static
  placement.
* **The Triton static megakernel is around 49 µs regardless of program**, so it measures
  launch overhead, not the schedule. The CUDA build takes 22–30 µs. Its in-kernel time
  (first task start to last task end, by `%globaltimer`) is 8–16 µs, with a spread under
  0.6 µs.
* **Every series fails tinytail's i.i.d. gates** (`EVT-invalid-dependent`). Ljung–Box
  rejects on all 35 and the runs test on 34, and most also fail KPSS or the KS-halves test. So
  tinytail computes no EVT or MBPTA bound for any of them. It does report only the
  hold-out maximum plus a margin, marked non-probabilistic. Back-to-back launches carry
  state between iterations (clocks, caches, the launch queue), and the gates catch it.
* **The gates do not separate `dynamic` from `static`.** Both fail, which answers the
  open question above: at this sample size and back-to-back pacing, the dependence from
  repeated launching swamps any extra non-i.i.d.-ness from the dynamic queue. Telling
  them apart would need randomised pacing, or a design with launches interleaved across
  schedules.
* **The in-kernel series are quantised.** `%globaltimer` on the T4 takes only 18–59
  distinct values over 66k samples, so the timer's resolution, not the schedule, sets
  their tail. They are useful as medians, not for EVT.

## Part 4: warps in the step machine

The T4 run found a bug the oracle could not see: `no_barrier`, where a task's signal goes
out before all of its warps have written. Part 4 puts warps into the model, as far as
that bug needs them.

* `ir.Task.no_barrier` marks the bug. `ir.WARPS = 4` and `ir.warp_of(e) = e % 4` give
  element `e`'s writer, the same dealing as `owner()` in `csrc/msync_vm.h`
  (`tests/test_msync_warps.py` checks the two agree for 128 threads).
* In `machine`, a step is now an action `(sm, lane)`. Lane 0 is the SM's main line. For
  a `no_barrier` task it is: acquire, write warp 0's elements, signal. Lanes 1–3 are the
  other warps' writes. Each is enabled from the acquire on, so it can land before or after
  the signal. The SM moves to its next task only when the signal and every warp's write
  are done: the `hw.sync()` that closes every task in the CUDA VM. The SM waits in a
  fourth phase, "signalled, waiting for its slow warps".
* Without the flag nothing changes. Every SM has only lane 0, the explored state spaces
  are the same, and the part-1 audit (`mutate.mutants`) is untouched. The new mutants come
  from `mutate.warp_mutants`, at the same sites as the T4 run and with the same ids.
* Round robin (`machine.run`) lets the warps finish before the main line moves on. So the
  default schedule never shows the bug, and the oracle has to find it.
* The Triton runner refuses `no_barrier` programs, since Triton has no warp-level control.
  The CUDA runner takes the flag from the program.

### The oracle against the T4

`python -m triton_equiv.msync.warp_check` labels the 9 warp mutants and compares them with
`results/msync_gpu_races_cuda.json`, which recorded them as `unmodelled`. Result:
`results/msync_warp_labels.json`.

| mutant | site | oracle | T4 mismatch rate, random delays |
|---|---|---|---|
| mlp_rms_aligned:no_barrier:0 | g0 | benign | 0 |
| mlp_rms_aligned:no_barrier:1 | g1 | racy | 0.25 |
| mlp_rms_aligned:no_barrier:2 | r | racy | 0.36 |
| mlp_rms_odd:no_barrier:0 | g0 | benign | 0 |
| mlp_rms_odd:no_barrier:1 | r | racy | 0.41 |
| barrier2:no_barrier:0 | a0 | racy | 0.22 |
| barrier2:no_barrier:1 | a1 | racy | 0.19 |
| barrier2:no_barrier:2 | b0 | benign | 0 |
| barrier2:no_barrier:3 | b1 | racy | 0.20 |

**9 of 9 agree.** The GPU showed exactly the mutants the oracle calls `racy`, and never
showed the 3 it calls `benign`. Every racy label comes with a witness interleaving that
the float64 replay confirms.

The three benign sites are protected by the barrier that closes each task. `g0` (in both
`mlp_rms` programs) and `b0` (in `barrier2`) run on SM 0, and the only task that waits for
their counter (`r`, `fin`) is also on SM 0, later in its queue. SM 0 cannot start that
consumer until all of its own warps have written, so a late signal cannot be seen early.
At every racy site, a consumer on the other SM waits for the counter.

**The prediction held a second time, made before the run.**
`notebooks/msync_paced.ipynb` reran `races --backend cuda` on a T4 with the oracle
labelling the warp mutants first (`results/msync_gpu_races_cuda_warps.json`). All 6
`racy` ones showed with random delays, at rates of 0.17 to 0.41, and none showed without
delays. The 3 `benign` ones never showed. No benign program, warp mutant or otherwise, had
a false alarm.

So a model at the granularity of SMs and steps is enough for these programs once warps
are modelled as separate writers. Two things are still not modelled: weaker PTX
orderings inside a step, and reads split over warps.

`tests/test_msync_warps.py` covers:

* the warp steps and the closing barrier;
* round robin hiding the bug;
* the same-SM case;
* that the flag off changes nothing;
* the labels above;
* their agreement with the T4 file, when it is present.

`tests/conftest.py` now sets `TRITON_INTERPRET=1` before collection. `triton.jit` reads
it when `msync.gpu` is first imported. With the variable set only in
`test_msync_gpu.py`, a full run that collected `test_msync_cuda_vm.py` first built
compiled kernels, and the 115 interpreter tests failed with "0 active drivers". They
passed only when the file ran on its own.

## Part 5: paced latency

Every back-to-back series in part 3 failed tinytail's i.i.d. gates, so those gates could
not separate `static` from `dynamic`. `latency --design paced` changes the measurement and
not the analysis:

* there are `n` rounds;
* each round runs every schedule once, in a fresh random order;
* each launch follows an idle gap drawn uniformly from 0–200 µs, a host busy-wait after
  the previous launch has finished.

Drift is then shared by all schedules, and no launch follows one of its own kind at a
fixed spacing. Because the rounds pair the samples, each program also gets the per-round
difference of each schedule from `static` (`latency.paired`). That comparison needs no
i.i.d. assumption across rounds. `tests/test_msync_latency.py` checks the design with a
fake timer.

`notebooks/msync_paced.ipynb` ran it on a Kaggle T4, from a private dataset of the source
because the code was not on GitHub yet. Triton backend, 100k rounds:
`results/msync_gpu_latency_paced.json`.

**The gates still fail.** All 15 Triton series are `EVT-invalid-dependent`. The runs and
Ljung–Box tests reject on every series (p below 1e-100), and so does the KS test of the two
halves. KPSS rejects on 5. Randomising the order and the spacing did not remove the
dependence, so it is not an artefact of launching back to back. The likely source is
state of the GPU and driver that drifts over the session, such as clocks and power state.
Pacing on the host cannot break that. This run does not measure it directly. The gates are doing their job: there is no defensible pWCET for these
series, and tinytail refuses to give one.

**The paired differences separate the schedules anyway:**

| program | tasks | dynamic − static, median | rounds dynamic slower | per_task − static, median | rounds per_task slower |
|---|---|---|---|---|---|
| mlp_rms_aligned | 5 | +4.4 µs | 80% | +106 µs | 99.98% |
| mlp_rms_odd | 5 | +4.2 µs | 77% | +110 µs | 99.98% |
| split_reduce_aligned | 3 | +0.2 µs | 52% | +61 µs | 99.5% |
| split_reduce_odd | 3 | +0.4 µs | 54% | +56 µs | 99.7% |
| barrier2 | 5 | +4.7 µs | 80% | +107 µs | 99.98% |

* **The dynamic queue costs about 4.5 µs on the 5-task programs** and is slower in about
  four rounds out of five. On the 3-task programs the cost is within noise: half the rounds
  each way. There, each SM takes at most two tasks from the atomic head, so there is
  little contention to pay for.
* **`per_task` costs 55–110 µs**, 26–30 µs per extra launch, in nearly every round.
* **Medians are 4–10 µs higher than back to back** (static is 53–59 µs, against 49–51 µs).
  A GPU left idle is slower to start a launch than one that is already busy. The back-to-back
  numbers are the warm case.

A tail bound on this hardware would need a cleaner timing source: in-kernel cycle
counters per SM, or fixed clocks, which Kaggle does not allow. A pWCET is not what these
programs need anyway. The paired design answers the schedule question.

### The CUDA backend, paced

The first CUDA paced run crashed after measuring its first program. The series it was
analysing was `static_in_kernel`, which is quantised. Its top 5% were all one value, so
nothing lay strictly above the POT threshold, and tinytail's GPD fit failed on an empty
array. tinytail now refuses a fit with fewer than 10 exceedances, and `analyse` records
that quantile as an error and moves on (tinytail `0fb113a`). `latency.tail_report`
also records an analysis error for a series instead of losing the run. The rerun
(`notebooks/msync_paced_cuda.ipynb`, T4, 100k rounds, about 16 minutes) used that tinytail:
`results/msync_gpu_latency_paced_cuda.json`.

| program | tasks | dynamic − static, median | rounds dynamic slower | per_task − static, median | rounds per_task slower |
|---|---|---|---|---|---|
| mlp_rms_aligned | 5 | +2.1 µs | 92% | +25.6 µs | 99.8% |
| mlp_rms_odd | 5 | +1.9 µs | 90% | +24.7 µs | 99.7% |
| split_reduce_aligned | 3 | +1.1 µs | 76% | +11.8 µs | 96.7% |
| split_reduce_odd | 3 | +0.9 µs | 72% | +12.1 µs | 96.7% |
| barrier2 | 5 | +2.0 µs | 92% | +26.1 µs | 99.8% |

* **The CUDA VM's dynamic queue costs about 2 µs on the 5-task programs and about 1 µs on
  the 3-task ones.** Unlike Triton, it separates the 3-task programs too, in three rounds
  out of four. Its launches are shorter (static medians 22.5–30.4 µs, against 53–59 µs for
  Triton), so less noise hides the cost.
* **`per_task` costs 12–26 µs,** about 6 µs per extra launch, against 26–30 µs for Triton.
* **Idle gaps barely move the CUDA medians.** Static is within 0.5 µs of back to back
  except `mlp_rms_aligned` (+3 µs). `per_task` rises by 8–10 µs on the 5-task programs and
  not at all on the 3-task ones. So most of the idle penalty seen with Triton comes from
  its launch path rather than the GPU.
* **`static_in_kernel` sits 14–15 µs below the event-timed `static`** in every round, so
  half to two thirds of each event-timed launch is outside the kernel.

**The gates:** all 15 event-timed series still fail (runs and Ljung–Box on every one).
The in-kernel series differ:

* 3 of 5 pass every gate: both `split_reduce` programs and `barrier2`. These are the
  first series in this work to pass.
* `mlp_rms_odd` is refused as degenerate (18 distinct values).
* `mlp_rms_aligned` fails.

None of the three passing series gets a GPD bound. The fit is rejected at every quantile
(p = 0.001), and on both `split_reduce` programs the 0.975 and 0.99 quantiles have no
exceedances at all: the case that crashed the first run. With 38–65 distinct values, the
in-kernel clock is too coarse for a tail fit. So the conclusion above stands for both
backends: the paired differences answer the schedule question; a pWCET needs a finer
timer.
