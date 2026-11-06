# triton-equiv

Research repo: tile-loop-aware equivalence checking for Triton kernels, starting with a
mutation audit of the correctness checks that kernel benchmarks and RL environments use.

**Status.** Five reference kernels, nine single-site AST mutation operators (43 mutants), a
Z3 bit-vector oracle for index/mask logic, and a floating-point tier that separates
"exact over the reals but rounds differently" from truly benign mutants. There is no
symbolic executor against a PyTorch spec yet.

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
| `results/labels.json` | All 43 labels with witnesses, plus the command that produced them |
| `src/triton_equiv/harness.py`, `symexec.py` | Placeholders for later work |
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
  `precision_loss` when the mutant's a-priori error bound is more than 4x the reference's.
  The measured ratios are 8204x (fp16 accumulator in `row_sum_loop`), 15x and 17x
  (`matmul_fp16` downgrade and bf16 cast). Reorderings measure 0.6x-1.5x.
* `benign`: real-equivalent and no witness. 4 are **proved** bitwise identical by the
  uninterpreted-function check. The other 4 (matmul `other=1.0` on a lane that always meets a
  zero) found no witness in 720 inputs, which is evidence, not proof.

Labels depend on size. Every OOB mutant is benign at the tile-multiple size, and the matmul
pid-axis mutant is benign on a 1x1 grid. A reduction loop needs at least 3 iterations
before reversing it can change any rounding. The fp tier therefore adds `K = 16`
(4 K-tiles) and `n_cols = 16` to the bit-vector sizes. One witness is rare:
`matmul_fp16:reverse_reduction_loop` needed 690 inputs, because the fp32 difference has
to survive the final fp16 rounding.

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
* Labels are proofs only at the listed sizes. Problem sizes are not symbolic.
* No symbolic executor against a PyTorch spec yet.

## Running

```bash
pip install -e ".[dev]"     # oracle + tests: Windows, Linux or WSL
pytest                      # Windows: 36 passed, 1 skipped (no triton)
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
~/venv-triton-equiv/bin/python -m pytest -q      # 38 passed
```

## License

MIT. See `LICENSE`.
