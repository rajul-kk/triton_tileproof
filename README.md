# triton-equiv

Research repo: tile-loop-aware equivalence checking for Triton kernels, starting with a
mutation audit of the correctness checks that kernel benchmarks and RL environments use.

**Status: skeleton only.** This repo holds the literature/scoop check, a two-week learning
path, and an empty package layout. No mutation suite, oracle, evaluation protocol or
symbolic executor has been written yet.

## Contents

| Path | What it is |
|---|---|
| `docs/scoop_check.md` | Prior-work check: what each related paper covers, whether code is released, and how each source was read |
| `docs/learning_path.md` | Two-week plan with exercises: Triton tutorials, PMPP ch. 1-5, Z3 basics, Alive2-style refinement |
| `src/triton_equiv/` | Placeholder modules for the planned components (`kernels`, `mutate`, `oracle`, `harness`, `symexec`) |
| `tests/` | Smoke test that checks the package imports |

## Planned work (not started)

1. **Mutation suite.** Reference kernels: vector add, softmax, layernorm, matmul with `tl.dot`,
   row reduction with `tl.sum`, masked tails. Mutation operators: off-by-one masks, wrong tail
   handling, wrong accumulator dtype, reduction-order changes, wrong strides/offsets, missing
   store mask, wrong `program_id` axis. Each mutant gets a label (truly inequivalent or benign)
   from a stated oracle, and we measure which mutants random-input testing catches.
2. **Symbolic executor** for a Triton subset (masked load/store, `tl.dot`, `tl.sum`, static loops
   at small tile sizes), compared against a PyTorch-style spec in Z3, with bounded-size limits stated.

See `docs/scoop_check.md` before starting: *Measuring the Checker* (arXiv 2609.22220) already
runs a mutation analysis of KernelBench oracles on CUDA, so the audit half needs a
Triton-specific angle to be new.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Linux / WSL
pip install -e ".[dev]"
pytest
```

Optional extras: `.[smt]` (Z3) and `.[triton]` (torch + triton). Triton has no official Windows
wheel. Use WSL/Linux with `TRITON_INTERPRET=1` for CPU runs, and Kaggle for GPU runs.

## License

MIT. See `LICENSE`.
