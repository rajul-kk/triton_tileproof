# triton-equiv

Research repo: tile-loop-aware equivalence checking for Triton kernels, starting with a
mutation audit of the correctness checks that kernel benchmarks and RL environments use.

**Status.** Five reference kernels, nine single-site AST mutation operators (43 mutants), a
Z3 bit-vector oracle for index/mask logic, a floating-point tier that separates
"exact over the reals but rounds differently" from truly benign mutants, and an audit of a
standard allclose checker against those labels on the Triton CPU interpreter. There is no
symbolic executor against a PyTorch spec yet, and nothing has run on a GPU.

This is not the first mutation analysis of kernel correctness checks (see
`docs/scoop_check.md`, especially arXiv 2609.22220). What differs here is that the mutants
are Triton-specific and the labels come from a formal oracle that can also prove a mutant benign.

## Contents

| Path | What it is |
|---|---|
| `docs/scoop_check.md` | Prior-work check: what each related paper covers, whether code is released, and how each source was read |
| `docs/learning_path.md` | Two-week plan with exercises: Triton tutorials, PMPP ch. 1-5, Z3 basics, Alive2-style refinement |
| `src/triton_equiv/kernels.py` | Reference kernels as source text. fp32: masked `vector_add`, `row_sum` (`tl.sum`), tiled `matmul` (`tl.dot`, K-loop). Mixed precision: `row_sum_loop` (fp16 in, fp32 accumulator, K-loop) and `matmul_fp16` (fp32 accumulator via `tl.dot(a, b, acc)`, cast to fp16 on store) |
| `src/triton_equiv/mutate.py` | AST mutation operators. Index logic: `mask_off_by_one`, `missing_store_mask`, `wrong_pid_axis`, `wrong_other`. Floating point: `reverse_reduction_loop`, `split_sum`, `acc_dtype_downgrade`, `drop_cast`, `change_cast` |
| `src/triton_equiv/oracle.py` | Bit-vector oracle: runs kernel source against a symbolic `tl` shim and labels a mutant `oob`, `inequivalent` (with counterexample) or `benign` (proof at that size) |
| `src/triton_equiv/fporacle.py` | Floating-point tier for every bit-vector-benign mutant: Z3 real-arithmetic equivalence, a structural bitwise-identity proof with rounded ops as uninterpreted functions, and an fp64-referenced witness search with running error bounds. The docstring says what it proves and what it does not |
| `src/triton_equiv/harness.py` | Allclose checker audit on the Triton CPU interpreter: confusion table of oracle label against checker verdict, and the tolerance blind-band sweep |
| `results/labels.json`, `results/checker_audit.json`, `results/blind_band.json` | Outputs, each with the command that produced it |
| `notebooks/kernelgym_audit.ipynb` | Kaggle notebook that plants the mutants into KernelGYM's `kernel_simple` correctness check. **Written, not run** |
| `src/triton_equiv/symexec.py` | Placeholder for later work |
| `tests/` | Known-label oracle tests, fp-tier tests, mutation tests, interpreter tests |

## Labels (all 43 mutants)

Produced by `PYTHONPATH=src python -m triton_equiv.fporacle --out results/labels.json`
(35 s on a laptop CPU, Windows, Python 3.10, z3 5.3.3, numpy 2.2.6).

| Kernel | Mutants | oob | inequivalent | numerical: precision loss | numerical: reorder | benign: proved bitwise | benign: no witness, not proved |
|---|---|---|---|---|---|---|---|
| vector_add | 5 | 2 | 1 | 0 | 0 | 2 | 0 |
| row_sum | 4 | 1 | 2 | 0 | 1 (`split_sum`) | 0 | 0 |
| matmul | 13 | 7 | 2 | 0 | 1 (loop reversed) | 1 (fp16 `tl.zeros`, promoted back to fp32) | 2 (`other` 0 -> 1) |
| row_sum_loop | 6 | 1 | 2 | 1 (fp16 accumulator) | 2 (loop reversed, `split_sum`) | 0 | 0 |
| matmul_fp16 | 15 | 7 | 2 | 2 (fp16 accumulator, cast to bf16) | 1 (loop reversed) | 1 (redundant `.to(fp16)` before the store) | 2 (`other` 0 -> 1) |
| **total** | **43** | **18** | **9** | **3** | **5** | **4** | **4** |

The four label families:

* `oob` / `inequivalent`: from the bit-vector tier, as before, with a witness (program id,
  lane, address; or buffer, index and model).
* `numerical`: Z3 proves the mutant computes the same real-valued function at the oracle sizes,
  and a concrete input makes the rounded outputs differ. Every one of the 8 carries a witness
  input, and all 8 witnesses reproduce bit for bit in the Triton interpreter (WSL). It is
  `precision_loss` when the mutant's running error bound is more than 4x the reference's
  (max over the inputs tried).
  The measured ratios are 8204x (fp16 accumulator in `row_sum_loop`), 15x and 17x
  (`matmul_fp16` downgrade and bf16 cast). Reorderings measure 0.6x-1.5x.
* `benign`: real-equivalent and no witness. 4 are **proved** bitwise identical by the
  uninterpreted-function check. The other 4 (`other=1.0` in both matmuls, on a lane that always
  meets a zero) found no witness in 720 inputs, which is evidence, not proof.

Labels depend on size. Every OOB mutant is benign at the tile-multiple size, and the matmul
pid-axis mutant is benign on a 1x1 grid. A reduction loop needs at least 3 iterations
before reversing it can change any rounding. The fp tier therefore adds `K = 16`
(4 K-tiles) and `n_cols = 16` to the bit-vector sizes. One witness is rare:
`matmul_fp16:reverse_reduction_loop` needed 690 inputs, because the fp32 difference has
to survive the final fp16 rounding.

## Checker audit on CPU (Triton interpreter, WSL)

`src/triton_equiv/harness.py` is a standard allclose checker. It draws `torch.randn` inputs (3
seeds per size), runs the candidate under `TRITON_INTERPRET=1`, compares with a PyTorch
reference, and accepts if every trial passes. Two tolerance policies are scored from the
same runs: `kernelbench` (`rtol = atol = 1e-2`) and `torch_default` (dtype-aware
`torch.testing` defaults: fp32 1.3e-6/1e-5, fp16 1e-3/1e-5). "tile" sizes are tile
multiples; "odd" sizes leave partial tiles. Buffers sit inside random-filled guard bands, so
out-of-bounds accesses are harmless and stores into the guard can be detected.

Command: `TRITON_INTERPRET=1 PYTHONPATH=src python -m triton_equiv.harness audit --trials 3`
(11 s, 340 MB peak) -> `results/checker_audit.json`. All 5 reference kernels pass every
checker below.

Accepted / total, by oracle label (accepting an `oob`, `inequivalent` or
`numerical: precision` mutant is a miss):

| Oracle label | n | tile, 1e-2 | tile+odd, 1e-2 | tile, torch defaults | tile+odd, torch defaults | tile+odd, 1e-2 + guard canary |
|---|---|---|---|---|---|---|
| oob | 18 | 18 | 12 | 18 | 12 | 8 |
| inequivalent | 9 | 2 | 0 | 2 | 0 | 0 |
| numerical: precision loss | 3 | 3 | 3 | 0 | 0 | 3 |
| numerical: reorder | 5 | 5 | 5 | 5 | 5 | 5 |
| benign: proved | 4 | 4 | 4 | 4 | 4 | 4 |
| benign: unproved | 4 | 4 | 4 | 4 | 4 | 4 |

* At tile-multiple sizes the 1e-2 check accepts **every** OOB mutant, the 3 precision-loss
  mutants, and 2 of the 9 inequivalent ones (the `other=` changes, which only matter when a
  tile has masked lanes). Adding one odd size rejects all 9 inequivalent and 6 of the 18 OOB.
* The 12 OOB mutants still accepted at odd sizes either write past the end of the output
  with correct in-bounds values (4), or read out of bounds with values that never reach a
  stored output (8, all `tl.dot` operand masks). A guard-band canary catches the 4 stores. No
  output comparison can see the 8 reads.
* Tighter `torch.testing` tolerances reject the 3 precision-loss mutants and still accept the
  5 reorderings, which is the intended split. They add nothing for OOB.
* The oracle and the checker agree on every `benign` and `numerical: reorder` mutant: no false rejects.

### Tolerance blind band

Command: `TRITON_INTERPRET=1 PYTHONPATH=src python -m triton_equiv.harness blind-band --trials 20`
(330 s) -> `results/blind_band.json`. Each `numerical` mutant is swept over the reduction
length n (`n_cols` or `K`) = 16 ... 4096, with 20 seeds per point. "Miss" = the 1e-2 check accepts.

| Mutant | Inputs | n=16 | 64 | 256 | 1024 | 4096 | error vs reference kernel (median) |
|---|---|---|---|---|---|---|---|
| `row_sum_loop` fp16 accumulator | randn | 1.00 | 1.00 | 1.00 | 0.85 | 0.70 | 2.4e4-5.5e4x (n >= 256) |
| `row_sum_loop` fp16 accumulator | rand | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8e3-3e4x (n >= 64) |
| `matmul_fp16` fp16 accumulator | randn | 1.00 | 1.00 | 0.60 | 0.00 | 0.00 | 1-12x |
| `matmul_fp16` fp16 accumulator | rand | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1-11x |
| `matmul_fp16` cast to bf16 | both | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 16x |
| reorderings (5 mutants) | both | 1.00 everywhere | | | | | 0.75-1.09x |

The miss rate of a precision-loss mutant falls only as the reduction grows, and only for
zero-mean inputs. With positive inputs (`rand`), the sum grows with n and `rtol` grows with
it, so the fp16 accumulator passes at every size tried, with an absolute error of 1.4 at n=4096
(`row_sum_loop`) and 5.2 (`matmul_fp16`). The bf16 cast is never caught: its error is a fixed
16x the reference's, always inside 1e-2. Note that the precision-loss mutants that pass
here are flagged by the oracle without running any GPU or choosing a tolerance. Whether a
given loss is acceptable is a policy decision; the oracle only shows that it exists.
These numbers come from the CPU interpreter only.

## KernelGYM audit notebook (written, **not run**)

`notebooks/kernelgym_audit.ipynb` is for a Kaggle GPU session. It clones KernelGYM at commit
`3a84417` and this repo, then wraps each reference kernel and all 43 mutants as a KernelGYM
`ModelNew` module. Each one goes through `KernelSimpleToolkit.evaluate`
(`kernelgym/toolkit/kernel_simple/toolkit.py`, called in-process with the `kernelbench` backend
adapter that `kernelgym/workflow/kernel_simple.py` selects). The notebook records which faulty
mutants are accepted. It has never been executed: there is no GPU here, so it is untested.

One thing it checks comes from reading the toolkit code, not from running it.
`_normalize_case` stores `rtol=None` and `atol=None` when a case gives no tolerance. The later
`case.get("rtol", 1e-4)` then returns `None` rather than the default. `torch.allclose`
raises `TypeError` on `rtol=None` (checked on torch 2.15 CPU), and the toolkit catches the
exception as a failed comparison. If that reading is right, cases
without explicit tolerances reject every kernel, correct or not. The notebook's reference-kernel
control tests this.

## Scope limits

* **Floating-point tier: fp64 reference plus bounded error analysis, not Z3 floating point.**
  The reasons are in the `fporacle` docstring. A toy-format proof does not transfer to fp32 at
  realistic reduction sizes, and its witnesses could not be replayed in Triton. It proves
  real equivalence and structural bitwise identity at the oracle sizes, plus a rigorous
  per-input error bound. It does **not** prove that an unproved `benign` mutant never differs. It
  does not model GPU reduction trees, FMA contraction or round-to-nearest bf16 (the
  interpreter truncates), and it does not model overflow, NaN/inf or subnormals.
* Numeric semantics are those of the Triton **CPU interpreter**: `tl.sum` is `np.sum`,
  `tl.dot` is `np.matmul(a, b, dtype=acc) + acc`. The error bounds hold for any summation
  order; the bitwise witnesses are only guaranteed in the interpreter.
* `matmul:acc_dtype_downgrade` is benign in the interpreter because `acc += tl.dot(...)`
  promotes the fp16 accumulator to fp32. Compiled Triton would reject the loop-carried
  dtype change. That was not run here (no GPU).
* The checker audit ran only on the CPU interpreter. Out-of-bounds behaviour on a GPU (faults,
  corrupting a neighbouring allocation) and GPU reduction orders were not observed. The
  KernelGYM notebook is untested.
* Labels are proofs only at the listed sizes. Problem sizes are not symbolic.
* No symbolic executor against a PyTorch spec yet.

## Running

```bash
pip install -e ".[dev]"     # oracle + tests: Windows, Linux or WSL
pytest                      # Windows: 38 passed, 2 skipped (no triton)
```

### Triton CPU interpreter (WSL)

`TRITON_INTERPRET=1` **works under WSL Ubuntu** (Python 3.14, torch 2.15.0+cpu, triton 3.9.0).
`tests/test_interpreter.py` runs the reference vector add and the `missing_store_mask` mutant
at n=6, BLOCK=4, with a canary guard band after the output. The mutant's in-bounds output is
still correct (so `allclose` would accept it), but it writes the canary at address 6, which
matches the oracle's witness. It also replays four fp-tier witnesses in the interpreter and
checks that they match the emulator bit for bit. Triton has no official Windows wheel, so
these tests are skipped on Windows.

```bash
# WSL, from the repo root
python3 -m venv ~/venv-triton-equiv
~/venv-triton-equiv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/venv-triton-equiv/bin/pip install triton z3-solver numpy pytest
~/venv-triton-equiv/bin/python -m pytest -q      # 41 passed
```

## License

MIT. See `LICENSE`.
