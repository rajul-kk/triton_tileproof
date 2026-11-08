# triton-equiv

Research repo: formal labels for Triton kernel mutants, and an audit of the correctness checks
that kernel benchmarks and RL reward environments use.

**Status.** Eight reference kernels (vector add, row sum, tiled matmul, two mixed-precision
variants, softmax, layernorm, single-head attention), eleven single-site AST mutation operators
(72 mutants), a Z3 bit-vector oracle for index/mask logic, a floating-point tier that separates
"exact over the reals but rounds differently" from truly benign mutants, and an audit of a
standard allclose checker against those labels on the Triton CPU interpreter. The KernelGYM
audit has also run once on a GPU (Kaggle Tesla T4, Triton 3.6.0); see
[`docs/kernelgym_audit.md`](docs/kernelgym_audit.md).

## Contribution, and what is not new

Two observations this audit reproduces are **already published** for CUDA kernels by
*Measuring the Checker* (arXiv 2609.22220, read in full): that tile-aligned problem sizes hide
dropped guards and tails (its Boundary family, "remainder blocks never execute", 22.9% escape),
and that the tolerance blind band grows with reduction size (largest undetectable relative error
0.39 at d=64, 2.5e3 at d=393,216; an all-zeros softmax passes). Our tables below are
Triton-interpreter replications of those effects, not new findings. It is also not the first
mutation analysis of kernel checks.

What this repo adds:

1. **Triton-specific mutants.** Mask comparisons, a dropped store mask, `other=` fill values
   (including softmax's `other=-inf` -> `0.0`), the `program_id` axis, `tl.where` padding masks,
   dividing by `BLOCK` instead of the valid length, accumulator dtype, casts, and reduction-order
   rewrites, on kernels with partial tiles. 2609.22220's 124 rules are CUDA only.
2. **Formal labels.** Each mutant is labelled by a Z3 oracle at fixed small sizes: `oob` and
   `inequivalent` with witnesses, and `benign` **with a proof**. A floating-point tier splits
   real-equivalent mutants into *proved bitwise identical*, *reorderings* and a
   **precision-loss** tier, with fp64 witnesses and Higham error bounds. 2609.22220 labels by kill
   witness or quarantine; Volta (arXiv 2511.12638) treats values as reals, so reorderings and
   precision loss count as equal there.
3. **An audit with formally labelled, Triton-specific mutants, which isolates the failure
   class**: OOB at tails, `undef` lanes, precision loss, reorderings. Auditing Triton checkers is
   itself **not new**. *Red-Teaming the Reward* (NeurIPS 2026) already reports that three
   published verifiers accept nearly all of the invalid agent submissions (25% of those it
   audited). Its audit rests on agent runs and a verifier that is not released. Ours is a
   complement, not a first. It covers the allclose harness below and KernelGYM's `kernel_simple`
   check, run on a Kaggle T4. TritonRL has no released code, and DRTriton and TritonGym have not
   been read in code.

## Related work (summary; details and read levels in `docs/scoop_check.md`)

| Work | Level | Values | Sizes / tails | Loops, `tt.dot` | Labels or output |
|---|---|---|---|---|---|
| Measuring the Checker, 2609.22220 | CUDA source mutants | concrete runs | reports aligned shapes hide tails | yes (CUDA) | kill witness or quarantine, no proofs |
| Volta, 2511.12638 | PTX | reals | fixed; tails not discussed | yes | equivalence proof; 4 LLM kernels, no hidden bugs found |
| kernel-tv / triton-tv | TTIR, compiler translation validation | opaque UF tags (IEEE mode designed) | fixed tiles; `program_id` free, not grid-bounded | `scf.for` and `tt.dot` **not implemented** (source read 2026-11-06) | equivalent / not / unknown; no OOB model |
| Tatva | TTIR + Z3 | | | loop-free only | equivalence proof |
| Red-Teaming the Reward (KernelQuest) | Triton agent submissions | concrete runs, 5 seeded inputs, up to 5 distributions | | | 4-gate verifier (structure, correctness, precision/CUDA-graph policy, performance), hidden; audits three published verifiers |
| TritonRL, 2510.17891 | Triton RL reward | 5 random inputs, exact equality as written | | | linter + LLM judge; no code released |
| triton-verify | TTIR | bit-vectors | symbolic pid in grid | | pointer safety only |
| **this repo** | Triton source (Python AST) | BV + reals + UF + fp64 emulator | fixed, tails included | unrolled loops, `tt.dot` | oob / inequivalent / numerical (precision, reorder) / benign (proved or not) |

kernel-tv is the closest Triton-level SMT tool. Its abstract floating-point mode (every op an
uninterpreted function) is the same idea as our structural bitwise check. Read on GitHub only
(nothing cloned or run; no license), it models masked loads with `other`, but has no memory
bounds, no loops and no `tt.dot` yet, and its purpose is checking compiler passes, not checking
kernels or reward checkers. Red-Teaming's policy gate (no lower-precision casts on fp32 tasks)
targets the same failures as our precision-loss tier, but by a rule rather than an error bound.

## Contents

| Path | What it is |
|---|---|
| `docs/scoop_check.md` | Prior-work check: what each related paper or tool covers, whether code is released, and how each source was read (two papers in full; kernel-tv source read on GitHub) |
| `docs/learning_path.md` | Two-week plan with exercises: Triton tutorials, PMPP ch. 1-5, Z3 basics, Alive2-style refinement |
| `src/triton_equiv/kernels.py` | Reference kernels as source text. fp32: masked `vector_add`, `row_sum` (`tl.sum`), tiled `matmul` (`tl.dot`, K-loop), `softmax` (`other=-inf`, max subtraction, `tl.exp`), single-head `attention` (`tl.dot(q, tl.trans(k))`, `-inf` key mask with `tl.where`, softmax, `tl.dot(p, v)`; all keys in one tile, no scaling). Mixed precision: `row_sum_loop` (fp16 in, fp32 accumulator, K-loop), `matmul_fp16` (fp32 accumulator via `tl.dot(a, b, acc)`, cast to fp16 on store), `layernorm` (fp16 in and out, fp32 compute, `tl.where` padding, `1.0 / tl.sqrt(var + eps)`, no affine) |
| `src/triton_equiv/mutate.py` | AST mutation operators. Index and mask logic: `mask_off_by_one`, `missing_store_mask`, `wrong_pid_axis`, `wrong_other`, `drop_where_mask`, `divisor_block_size`. Floating point: `reverse_reduction_loop`, `split_sum`, `acc_dtype_downgrade`, `drop_cast`, `change_cast` |
| `src/triton_equiv/oracle.py` | Bit-vector oracle: runs kernel source against a symbolic `tl` shim and labels a mutant `oob`, `inequivalent` (with counterexample) or `benign` (proof at that size). exp, sqrt, max and `/` are uninterpreted functions |
| `src/triton_equiv/fporacle.py` | Floating-point tier for every bit-vector-benign mutant: Z3 real-arithmetic equivalence, a structural bitwise-identity proof with rounded ops as uninterpreted functions, and an fp64-referenced witness search with running error bounds. It also confirms, in fp64, any difference that was found only under uninterpreted functions. The docstring says what it proves and what it does not |
| `src/triton_equiv/harness.py` | Allclose checker audit on the Triton CPU interpreter: confusion table of oracle label against checker verdict, and the tolerance blind-band sweep |
| `results/labels.json`, `results/checker_audit.json`, `results/blind_band.json`, `results/blind_band_rowwise.json` | Outputs, each with the command that produced it |
| `notebooks/kernelgym_audit.ipynb` | Kaggle notebook that plants all 72 mutants into KernelGYM's `kernel_simple` correctness check. Run on a Kaggle T4; see `docs/kernelgym_audit.md` |
| `notebooks/kernelgym_readme_check.py` | Runs the `kernel_simple` example from KernelGYM's README three ways (verbatim, cases split out, explicit tolerance) |
| `results/kernelgym_audit_t4.json`, `results/kernelgym_audit_t4_unpatched.json`, `results/kernelgym_readme_check_t4.json` | KernelGYM GPU results |
| `src/triton_equiv/msync/` | Megakernel synchronisation: a schedule IR cut down from AutoMegaKernel's, an oracle that explores every SM interleaving, 12 schedule mutation operators and a checker audit. See `docs/megakernel_sync.md` |
| `src/triton_equiv/msync/amk_bridge.py`, `amk_e2e.py` | Run AutoMegaKernel's own validator, simulators and correctness gate on our mutants and on bugs planted in its toy-Llama lowering (needs `AMK_ROOT`) |
| `src/triton_equiv/msync/gpu.py`, `races.py`, `latency.py` | The same schedule programs as one persistent Triton kernel on a GPU: how often each mutant shows with and without injected delays, and launch-latency tails checked with tinytail (`notebooks/msync_gpu.ipynb`, run on a T4: no false alarms, every racy mutant shown under delays, all latency series fail the i.i.d. gates) |
| `src/triton_equiv/msync/cuda_vm.py`, `csrc/msync_vm.h` | The same VM in CUDA C++: PTX acquire/release, `%globaltimer` timeouts and per-task timestamps, and a warp-level `no_barrier` mutant (`--backend cuda`) |
| `src/triton_equiv/cuda_ref.py`, `csrc/ref_kernels.h`, `refcheck.py` | C++/CUDA reference kernels independent of PyTorch and Triton, and a rerun of the checker against them. See `docs/cuda_reference.md` |
| `src/triton_equiv/kgym_table.py` | KernelGYM accepted-by-label table from `results/kernelgym_audit_t4.json`, with mutants that do not compile on GPU counted separately |
| `results/msync_audit.json` | 99 labelled schedule mutants and six checkers' verdicts (`python -m triton_equiv.msync.audit`) |
| `results/refcheck_cpu.json` | Checker verdicts against PyTorch and against the C++ reference kernels: no verdict changes (`python -m triton_equiv.refcheck`). `results/refcheck_t4.json`: the same in real Triton on a T4, none of 80 changes |
| `src/triton_equiv/symexec.py` | Placeholder for later work |
| `tests/` | Known-label oracle tests, fp-tier tests, mutation tests, interpreter tests |

## Labels (all 72 mutants)

Produced by `PYTHONPATH=src python -m triton_equiv.fporacle --out results/labels.json`
(127 s on a laptop CPU, Windows, Python 3.10, z3 5.3.3, numpy 2.2.6).

| Kernel | Mutants | oob | inequivalent | numerical: precision loss | numerical: reorder | benign: proved bitwise | benign: no witness, not proved |
|---|---|---|---|---|---|---|---|
| vector_add | 5 | 2 | 1 | 0 | 0 | 2 | 0 |
| row_sum | 4 | 1 | 2 | 0 | 1 (`split_sum`) | 0 | 0 |
| matmul | 13 | 7 | 2 | 0 | 1 (loop reversed) | 1 (fp16 `tl.zeros`, promoted back to fp32) | 2 (`other` 0 -> 1) |
| row_sum_loop | 6 | 1 | 2 | 1 (fp16 accumulator) | 2 (loop reversed, `split_sum`) | 0 | 0 |
| matmul_fp16 | 15 | 7 | 2 | 2 (fp16 accumulator, cast to bf16) | 1 (loop reversed) | 1 (redundant `.to(fp16)` before the store) | 2 (`other` 0 -> 1) |
| softmax | 5 | 2 | 2 (pid axis, `other=-inf` -> `0.0`) | 0 | 1 (`split_sum` of the denominator) | 0 | 0 |
| layernorm | 13 | 2 | 5 (pid axis, `other`, `tl.where` dropped, 2x `/ BLOCK`) | 3 (fp32 upcast dropped or changed to fp16; output cast to bf16) | 0 | 1 (redundant output `.to(fp16)`) | 2 (`split_sum` of mean and variance) |
| attention | 11 | 5 | 3 (pid axis, key-mask `<=`, `tl.where` dropped) | 0 | 0 | 2 (`other` on q and k) | 1 (`other` on v) |
| **total** | **72** | **27** | **19** | **6** | **6** | **7** | **7** |

The four label families:

* `oob` / `inequivalent`: from the bit-vector tier, with a witness (program id, lane, address;
  or buffer, index and model).
* `numerical`: Z3 proves the mutant computes the same real-valued function at the oracle sizes,
  and a concrete input makes the rounded outputs differ. Every one of the 12 carries a witness
  input, and all 12 witnesses reproduce bit for bit when replayed in the Triton interpreter
  (WSL; seven of them are in `tests/test_interpreter.py`). The mutant is labelled
  `precision_loss` when the mutant's running error bound is more than 4x the reference's
  (max over the inputs tried).
  The measured ratios are 8204x (fp16 accumulator in `row_sum_loop`), 15x and 17x
  (`matmul_fp16` downgrade and bf16 cast), 55x on `randn` and 108x on `rand` (layernorm computed
  in fp16; unbounded on the wide-exponent inputs, where fp16 overflows) and 17x (layernorm output
  through bf16). Reorderings measure 0.6x-1.5x.
* `benign`: real-equivalent and no witness. 7 are **proved** bitwise identical by the
  uninterpreted-function check. The other 7 found no witness in 660-720 inputs, which is
  evidence, not proof: `other=1.0` in both matmuls and on attention's v (lanes that always meet a
  zero), and layernorm's two `split_sum`s, where an fp32 reordering has to survive the final
  fp16 rounding.

### exp, sqrt, max and division (softmax, layernorm, attention)

The bit-vector tier has no exact meaning for these, so it treats them as **uninterpreted
functions**. Non-integral constants (`eps`, `-inf`) become named uninterpreted constants. The
real tier keeps `/`, `max` and `tl.where` exact and leaves only `exp` and `sqrt` uninterpreted.
Terms that are equal under uninterpreted functions are equal under every interpretation, so a
`benign` proof still holds. A *difference*, though, only holds for some interpretation. So each
of the 10 `inequivalent` labels on these kernels was confirmed by an fp64 run of the emulator
(relative difference > 1e-6), and the label records that run. One mutant shows why the check is
needed: `attention:wrong_other:2` (`other=1.0` on v) differs under uninterpreted `exp`, but
the padded lanes have `p = exp(-inf) = 0` exactly. fp64 finds no difference, so it stays
`benign` with `real_proof: false`.

The emulator follows the interpreter's dtype rules, which were checked with a probe. A 0-d
operand (a scalar argument or a reduced tile) does not promote a tile, so an fp16 tile minus an
fp32 mean stays fp16. `/` on fp16 returns fp32, and `tl.exp`/`tl.sqrt` reject fp16.

Labels depend on size. Every OOB mutant is benign at the tile-multiple size, and so is every
softmax, layernorm and attention tail bug (`other=`, `tl.where`, `/ BLOCK`). The matmul pid-axis
mutant is benign on a 1x1 grid. A reduction loop needs at least 3 iterations before reversing it
can change any rounding. The fp tier therefore adds `K = 16` (4 K-tiles) and `n_cols = 16` to the
bit-vector sizes. One witness is rare: `matmul_fp16:reverse_reduction_loop` needed 690 inputs,
because the fp32 difference has to survive the final fp16 rounding.

## Checker audit on CPU (Triton interpreter, WSL)

`src/triton_equiv/harness.py` is a standard allclose checker. It draws `torch.randn` inputs (3
seeds per size), runs the candidate under `TRITON_INTERPRET=1`, compares with a PyTorch
reference (`torch.softmax`, `F.layer_norm`, `softmax(Q K^T) V` for the new kernels), and accepts if
every trial passes. Two tolerance policies are scored from the same runs: `kernelbench`
(`rtol = atol = 1e-2`) and `torch_default` (dtype-aware `torch.testing` defaults: fp32
1.3e-6/1e-5, fp16 1e-3/1e-5). "tile" sizes are tile multiples; "odd" sizes leave partial tiles
(softmax and layernorm: 64 columns vs 50 in a 64-wide tile; attention: M=32, N=16 vs M=20,
N=13 with 16-wide tiles). Buffers sit inside random-filled guard bands, so out-of-bounds accesses
are harmless and stores into the guard can be detected.

Command: `TRITON_INTERPRET=1 PYTHONPATH=src python -m triton_equiv.harness audit --trials 3`
(135 s, 365 MB peak) -> `results/checker_audit.json`. All 8 reference kernels pass every
checker below. The 43 original mutants give exactly the same verdicts as the earlier run.

Accepted / total, by oracle label (accepting an `oob`, `inequivalent` or
`numerical: precision` mutant is a miss). Each cell is *all 72* (*new kernels only*: softmax,
layernorm, attention):

| Oracle label | n | tile, 1e-2 | tile+odd, 1e-2 | tile, torch defaults | tile+odd, torch defaults | tile+odd, 1e-2 + guard canary |
|---|---|---|---|---|---|---|
| oob | 27 (9) | 27 (9) | 19 (7) | 27 (9) | 19 (7) | 11 (3) |
| inequivalent | 19 (10) | 9 (7) | 0 (0) | 9 (7) | 0 (0) | 0 (0) |
| numerical: precision loss | 6 (3) | 6 (3) | 6 (3) | 0 (0) | 0 (0) | 6 (3) |
| numerical: reorder | 6 (1) | 6 (1) | 6 (1) | 6 (1) | 6 (1) | 6 (1) |
| benign: proved | 7 (3) | 7 (3) | 7 (3) | 7 (3) | 7 (3) | 7 (3) |
| benign: unproved | 7 (3) | 7 (3) | 7 (3) | 7 (3) | 7 (3) | 7 (3) |

* At tile-multiple sizes the 1e-2 check accepts **every** OOB mutant, all 6 precision-loss
  mutants, and 9 of the 19 inequivalent ones. On the new kernels it accepts 7 of 10: the
  softmax `other=-inf` -> `0.0` fill, layernorm's `other`, dropped `tl.where` and both
  `/ BLOCK` divisors, and attention's key-mask `<=` and dropped `tl.where`. Every one of them
  only matters when a tile has padding lanes. This is the aligned-shape effect that 2609.22220
  reports for CUDA (Boundary family). Here it is reproduced for Triton masks, and each case
  carries an oracle label. Adding one odd size rejects all 19.
* The 19 OOB mutants still accepted at odd sizes either write past the end of the output with
  correct in-bounds values (8: the dropped store mask in vector_add, softmax, layernorm and
  attention, and a `<=` in the store mask of vector_add, both matmuls and attention), or read
  out of bounds with values that never reach a stored output (11: 8 `tl.dot` operand masks in the
  matmuls, 3 in attention). The new reads are
  attention's q, k and v loads with `<=`: the extra q row is never stored, the extra key is
  masked to `-inf` by `tl.where`, and the extra v row is multiplied by `p = 0`. A guard-band canary
  catches the stores. No output comparison can see the reads.
* Tighter `torch.testing` tolerances reject all 6 precision-loss mutants and still accept the
  6 reorderings. They add nothing for OOB or tail bugs.
* The oracle and the checker agree on every `benign` and `numerical: reorder` mutant: no false rejects.

### Tolerance blind band

Commands: `TRITON_INTERPRET=1 PYTHONPATH=src python -m triton_equiv.harness blind-band --trials 20`
(330 s) -> `results/blind_band.json`, and the same with `--kernels softmax layernorm --out
results/blind_band_rowwise.json` (84 s). Each `numerical` mutant is swept over the reduction
length n (`n_cols` or `K`) = 16 ... 4096 (row-wise kernels: 16 ... 1024, one tile per row), with
20 seeds per point. "Miss" = the 1e-2 check accepts. That the blind band depends on reduction
size is already reported by 2609.22220; this is a Triton-interpreter replication.

| Mutant | Inputs | n=16 | 64 | 256 | 1024 | 4096 | error vs reference kernel (median) |
|---|---|---|---|---|---|---|---|
| `row_sum_loop` fp16 accumulator | randn | 1.00 | 1.00 | 1.00 | 0.85 | 0.70 | 2.4e4-5.5e4x (n >= 256) |
| `row_sum_loop` fp16 accumulator | rand | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8e3-3e4x (n >= 64) |
| `matmul_fp16` fp16 accumulator | randn | 1.00 | 1.00 | 0.60 | 0.00 | 0.00 | 1-12x |
| `matmul_fp16` fp16 accumulator | rand | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1-11x |
| `matmul_fp16` cast to bf16 | both | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16x |
| `layernorm` computed in fp16 (2 mutants) | randn | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.0x |
| `layernorm` computed in fp16 (2 mutants) | rand | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.9-2.4x |
| `layernorm` output cast to bf16 | both | 1.00 | 1.00 | 1.00 | 1.00 | - | 16x |
| reorderings (6 mutants, incl. softmax `split_sum`) | both | 1.00 everywhere | | | | | 0.75-1.09x |

The miss rate of a precision-loss mutant falls only as the reduction grows, and only for
zero-mean inputs. With positive inputs (`rand`), the sum grows with n and `rtol` grows with
it, so the fp16 accumulator passes at every size tried, with an absolute error of 1.4 at n=4096
(`row_sum_loop`) and 5.2 (`matmul_fp16`). The bf16 casts are never caught: their error is a fixed
16x the reference's, always inside 1e-2 relative.

**A caveat on the precision-loss tier**, from the layernorm rows. The oracle flags
`layernorm` computed in fp16 because its running error *bound* is 55x (randn) to 108x (rand)
the reference's. The *measured* output error is only 1.0-2.4x, because the final fp16
rounding of the output dominates both kernels. The bound is sound but loose here. So
`precision_loss` means "the guaranteed accuracy is much worse", not "the observed error is much
worse". Whether a given loss is acceptable is a policy decision; the oracle only shows that it
exists. These numbers come from the CPU interpreter only.

## KernelGYM audit (run on a Kaggle T4)

`notebooks/kernelgym_audit.ipynb` clones KernelGYM at commit `3a84417` and this repo, then wraps
each reference kernel and all 72 mutants as a KernelGYM `ModelNew` module. Each one goes through
`KernelSimpleToolkit.evaluate` (`kernelgym/toolkit/kernel_simple/toolkit.py`, called in-process
with the `kernelbench` backend adapter that `kernelgym/workflow/kernel_simple.py` selects). It ran
on Kaggle (Tesla T4, torch 2.11.0, Triton 3.6.0) on 2026-11-07. Full write-up:
[`docs/kernelgym_audit.md`](docs/kernelgym_audit.md); raw results in `results/kernelgym_*_t4*.json`.

Two problems in the `kernel_simple` path, both reproduced on KernelGYM's own README example
(`notebooks/kernelgym_readme_check.py`):

1. The toolkit `exec`s Triton source from a string when loading cases or init inputs, and
   Triton 3.6 refuses `@jit` functions not defined in a file. The README example fails at
   "Failed to load cases" before the kernel runs.
2. With no `rtol`/`atol` in a case, `_normalize_case` stores `None`, `case.get("rtol", 1e-4)`
   returns `None`, `torch.allclose` raises, and the comparison counts as a failure. A correct
   `x + y` kernel is marked incorrect; with explicit tolerances it passes.

To reach the comparison, the notebook bypasses the init-inputs loader (documented in the
notebook). With that patch, all 16 reference controls pass with explicit tolerances. At
tile-multiple sizes KernelGYM accepts all 27 out-of-bounds mutants and 9 of 19 output-changing
mutants; odd sizes catch the 9 but still accept 17 of the 27 OOB mutants. Two
`acc_dtype_downgrade` mutants do not compile on GPU Triton, so their interpreter labels do not
apply there; the KernelGYM table counts them separately (`python -m triton_equiv.kgym_table`).

TritonRL, DRTriton and Red-Teaming the Reward are the other audit targets. Their checkers have
not been read in code or run.

## Megakernel synchronisation (CPU model)

Megakernels run a whole forward pass as one persistent kernel, with SMs handing work to
each other through counters in global memory. `src/triton_equiv/msync/` applies this
repo's method to that layer: a small schedule IR modelled on AutoMegaKernel's, an oracle
that explores every interleaving of the SMs and labels a schedule `oob`, `deadlock`,
`racy`, `inequivalent` or `benign` (benign holds for all inputs at that size), and 99
mutants of five reference schedules (missing or partial waits, early signals, swapped
queues, shifted and unclamped tiles, unoffset scratch, counters not zeroed, and RESOLVE's
barrier-reset race).

Accepted by six checkers (details and the three-part plan in `docs/megakernel_sync.md`):

| Oracle label | n | one launch, warm memory, allclose | one launch, NaN memory | argmax only | 20 random schedules, exact | AMK-style static rules |
|---|---|---|---|---|---|---|
| racy | 39 | 37/39 | 29/39 | 29/29 | 0/39 | 8/30 |
| deadlock | 27 | 1/27 | 1/27 | 1/23 | 0/27 | 2/26 |
| oob | 11 | 7/11 | 2/11 | 8/8 | 2/11 | 11/11 |
| inequivalent | 8 | 6/8 | 0/8 | 6/6 | 0/8 | 2/8 |
| benign | 14 | 14/14 | 14/14 | 10/10 | 14/14 | 5/9 |

A single launch on memory left by a previous launch misses almost every race; schedule
perturbation catches all of them in this model; static rules catch most synchronisation
bugs but no out-of-bounds tile. The "AMK-style rules" are our reading of AutoMegaKernel's
validator from its docs, not its code. Reading its code since: counters not zeroed is a
real gap in its validator; the transitive same-SM inversion is not, because AMK's GPU loader
orders each SM's queue topologically rather than by task list. This is a model with sequential consistency at step granularity, not a GPU.

## Scope limits

* **Floating-point tier: fp64 reference plus bounded error analysis, not Z3 floating point.**
  The reasons are in the `fporacle` docstring. A toy-format proof does not transfer to fp32 at
  realistic reduction sizes, and its witnesses could not be replayed in Triton. It proves
  real equivalence and structural bitwise identity at the oracle sizes, plus a rigorous
  per-input error bound. It does **not** prove that an unproved `benign` mutant never differs. It
  does not model GPU reduction trees, FMA contraction or round-to-nearest bf16 (the
  interpreter truncates), and it does not model overflow, NaN/inf (except exact `-inf` padding)
  or subnormals.
* **exp, sqrt, max, division are uninterpreted** in the bit-vector tier (and exp and sqrt in the
  real tier). Every `inequivalent` label on softmax, layernorm and attention therefore rests on an
  fp64 emulator run, not on a Z3 model. A real-equivalence claim with `real_proof: false` is only
  "not refuted". `-inf` is a named constant in the symbolic tiers, so proofs that involve it hold
  for every value of that constant, but the infinite arithmetic itself is not modelled
  symbolically.
* The emulator copies the interpreter's numpy calls and dtype rules (probed on triton 3.9.0). All
  12 numerical witnesses replay bit for bit in the interpreter. The error bounds for `exp` assume
  numpy's `exp` is within one ulp.
* Numeric semantics are those of the Triton **CPU interpreter**: `tl.sum` is `np.sum`,
  `tl.dot` is `np.matmul(a, b, dtype=acc) + acc`. On a GPU an fp32 `tl.dot` defaults to TF32
  inputs, which the interpreter run does not show. The
  error bounds hold for any summation order; the bitwise witnesses are only guaranteed in the
  interpreter.
* `matmul:acc_dtype_downgrade` is benign in the interpreter because `acc += tl.dot(...)`
  promotes the fp16 accumulator to fp32. Compiled Triton rejects the loop-carried dtype
  change (confirmed on Triton 3.6.0, T4), as it does for `matmul_fp16:acc_dtype_downgrade`.
* The new kernels are deliberately small. Softmax and layernorm hold a whole row in one tile (no
  column loop); layernorm has no weight or bias; attention has no `1/sqrt(d)` scaling, a single
  key tile and no online-softmax loop.
* The allclose checker audit ran only on the CPU interpreter. The KernelGYM audit ran on one
  GPU (T4) with one Triton version; no OOB mutant faulted there, but corruption of
  neighbouring allocations was not measured.
* Labels are proofs only at the listed sizes. Problem sizes are not symbolic.
* No symbolic executor against a PyTorch spec yet.

## Running

```bash
pip install -e ".[dev]"     # oracle + tests: Windows, Linux or WSL
pytest                      # Windows: 66 passed, 2 skipped (no triton)
```

### Triton CPU interpreter (WSL)

`TRITON_INTERPRET=1` **works under WSL Ubuntu** (Python 3.14, torch 2.15.0+cpu, triton 3.9.0).
`tests/test_interpreter.py` runs the reference vector add and the `missing_store_mask` mutant
at n=6, BLOCK=4, with a canary guard band after the output. The mutant's in-bounds output is
still correct (so `allclose` would accept it), but it writes the canary at address 6, which
matches the oracle's witness. It also replays seven fp-tier witnesses (including softmax and
layernorm) in the interpreter and checks that they match the emulator bit for bit. Triton has no
official Windows wheel, so these tests are skipped on Windows.

```bash
# WSL, from the repo root
python3 -m venv ~/venv-triton-equiv
~/venv-triton-equiv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/venv-triton-equiv/bin/pip install triton z3-solver numpy pytest
~/venv-triton-equiv/bin/python -m pytest -q      # 69 passed
```

## License

MIT. See `LICENSE`.
