# triton-equiv

Research repo: tile-loop-aware equivalence checking for Triton kernels, starting with a
mutation audit of the correctness checks that kernel benchmarks and RL environments use.

**Status: stage 1.** The repo has three reference kernels, four Triton-specific AST mutation
operators, and a Z3 bit-vector oracle that labels each mutant at small fixed sizes. There is
no evaluation protocol, test-harness comparison, Kaggle notebook or symbolic executor yet.

## Contents

| Path | What it is |
|---|---|
| `docs/scoop_check.md` | Prior-work check: what each related paper covers, whether code is released, and how each source was read |
| `docs/learning_path.md` | Two-week plan with exercises: Triton tutorials, PMPP ch. 1-5, Z3 basics, Alive2-style refinement |
| `src/triton_equiv/kernels.py` | Reference kernels as source text: masked `vector_add`, `row_sum` (`tl.sum`), tiled `matmul` (`tl.dot`, K-loop) |
| `src/triton_equiv/mutate.py` | AST mutation operators: `mask_off_by_one`, `missing_store_mask`, `wrong_pid_axis`, `wrong_other`. The module docstring explains why AST rewriting was chosen over regex, IR mutation and hand-written mutants. |
| `src/triton_equiv/oracle.py` | Z3 oracle. It runs kernel source against a symbolic `tl` shim and labels a mutant `oob`, `inequivalent` (with counterexample) or `benign` (proof at that size). The docstring lists what is out of scope. |
| `src/triton_equiv/harness.py`, `symexec.py` | Placeholders for later stages |
| `tests/` | Known-label oracle tests, mutation tests, and one Triton-interpreter test |

## Stage-1 labels (all 19 mutants, from running `oracle.label`)

| Kernel | Mutants | oob | inequivalent | benign |
|---|---|---|---|---|
| vector_add (n = 8, 6, 9; BLOCK 4) | 5 | 2 (mask `<=`, missing store mask) | 1 (pid axis) | 2 (`other=` added: masked lanes are never stored) |
| row_sum (2 rows x 8 or 5 cols; BLOCK 8) | 3 | 1 (mask `<=`) | 2 (pid axis; `other` 0 -> 1 when cols < BLOCK) | 0 |
| matmul (M,N,K = 4,4,4 / 6,5,7 / 9,5,7; tiles 4) | 11 | 7 (six mask `<=`, missing store mask) | 2 (pid axis) | 2 (`other` 0 -> 1: a masked A lane always meets a masked B zero, or lands in an unstored row) |

Labels depend on size. Every OOB mutant is benign at the tile-multiple size, and the matmul
pid-axis mutant is benign on a 1x1 grid. This is why the oracle sizes include non-multiples.
Each mutant takes under 1 s to label on a laptop CPU.

## Scope limits (stage 1)

* Floating point is **not** modelled. Values are 16-bit bit-vectors, and dtype casts, rounding,
  NaN/inf and `tl.dot` numerics are ignored. Accumulator-dtype and reduction-order mutants
  therefore need a later fp tier.
* Labels are proofs only at the listed sizes. Problem sizes are not symbolic. Program ids are
  symbolic in the OOB query and enumerated for the output comparison.
* No symbolic executor against a PyTorch spec yet. That is stage 2.

## Running

```bash
pip install -e ".[dev]"     # oracle + tests: Windows, Linux or WSL
pytest                      # Windows: 15 passed, 1 skipped (no triton)
```

### Triton CPU interpreter (WSL)

`TRITON_INTERPRET=1` **works under WSL Ubuntu** (Python 3.14, torch 2.15.0+cpu, triton 3.9.0).
Vector add, `tl.sum` and a `tl.dot` matmul with 16x16 tiles all match torch on CPU.
`tests/test_interpreter.py` runs the reference vector add and the `missing_store_mask` mutant
at n=6, BLOCK=4, with a canary guard band after the output. The mutant's in-bounds output is
still correct (so `allclose` would accept it), but it writes the canary at address 6, which
matches the oracle's witness. Triton has no official Windows wheel, so the test is skipped on Windows.

```bash
# WSL, from the repo root
python3 -m venv ~/venv-triton-equiv
~/venv-triton-equiv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/venv-triton-equiv/bin/pip install triton z3-solver numpy pytest
~/venv-triton-equiv/bin/python -m pytest -q      # 16 passed
```

## License

MIT. See `LICENSE`.
