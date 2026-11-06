# triton-equiv

Research repo: formal labels for Triton kernel mutants, and an audit of the correctness checks
that kernel benchmarks and RL reward environments use.

**Status.** Eight reference kernels (vector add, row sum, tiled matmul, two mixed-precision
variants, softmax, layernorm, single-head attention), eleven single-site AST mutation operators
(72 mutants), a Z3 bit-vector oracle for index/mask logic, a floating-point tier that separates
"exact over the reals but rounds differently" from truly benign mutants, and an audit of a
standard allclose checker against those labels on the Triton CPU interpreter. Nothing has run
on a GPU, and the KernelGYM notebook has not been run.

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
   complement, not a first. So far it covers only the allclose harness below. The KernelGYM
   notebook is written but **not run** (no GPU). TritonRL has no released code, and DRTriton and
   TritonGym have not been read in code.

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
| `notebooks/kernelgym_audit.ipynb` | Kaggle notebook that plants the original 43 mutants into KernelGYM's `kernel_simple` correctness check. **Written, not run** |
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

## KernelGYM audit notebook (written, **not run**)

`notebooks/kernelgym_audit.ipynb` is for a Kaggle GPU session. It clones KernelGYM at commit
`3a84417` and this repo, then wraps each reference kernel and every mutant (it iterates over
`KERNELS`, so now all 72) as a KernelGYM `ModelNew` module. Each one goes through
`KernelSimpleToolkit.evaluate` (`kernelgym/toolkit/kernel_simple/toolkit.py`, called in-process
with the `kernelbench` backend adapter that `kernelgym/workflow/kernel_simple.py` selects). The
notebook records which faulty mutants are accepted. It has never been executed: there is no GPU
here, so it is untested.

One thing it checks comes from reading the toolkit code, not from running it.
`_normalize_case` stores `rtol=None` and `atol=None` when a case gives no tolerance. The later
`case.get("rtol", 1e-4)` then returns `None` rather than the default. `torch.allclose`
raises `TypeError` on `rtol=None` (checked on torch 2.15 CPU), and the toolkit catches the
exception as a failed comparison. If that reading is right, cases
without explicit tolerances reject every kernel, correct or not. The notebook's reference-kernel
control tests this.

TritonRL, DRTriton and Red-Teaming the Reward are the other audit targets. Their checkers have
not been read in code or run.

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
  promotes the fp16 accumulator to fp32. Compiled Triton would reject the loop-carried
  dtype change. That was not run here (no GPU).
* The new kernels are deliberately small. Softmax and layernorm hold a whole row in one tile (no
  column loop); layernorm has no weight or bias; attention has no `1/sqrt(d)` scaling, a single
  key tile and no online-softmax loop.
* The checker audit ran only on the CPU interpreter. Out-of-bounds behaviour on a GPU (faults,
  corrupting a neighbouring allocation) and GPU reduction orders were not observed. The
  KernelGYM notebook is untested.
* Labels are proofs only at the listed sizes. Problem sizes are not symbolic.
* No symbolic executor against a PyTorch spec yet.

## Running

```bash
pip install -e ".[dev]"     # oracle + tests: Windows, Linux or WSL
pytest                      # Windows: 55 passed, 2 skipped (no triton)
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
~/venv-triton-equiv/bin/python -m pytest -q      # 58 passed
```

## License

MIT. See `LICENSE`.
