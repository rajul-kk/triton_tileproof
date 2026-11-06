# Two-week learning path: Triton, GPU basics, Z3, refinement

Plan for about 2 to 3 hours per weekday and one longer weekend session. There is no local GPU.
Run Triton on CPU with `TRITON_INTERPRET=1` under WSL/Linux, and use Kaggle (free T4/P100) for
real GPU runs. Each day has a deliverable: put it in `scratch/` (git-ignored if you like) or in a notebook.

## Week 1: GPU model and Triton

| Day | Read / watch | Exercise (deliverable) |
|---|---|---|
| 1 | PMPP (Hwu, Kirk, El Hajj, 4th ed.) ch. 1-2: heterogeneous computing, data-parallel model, vector add | Explain in 5 sentences how a 1-D grid of thread blocks covers an array of length N that is not a multiple of the block size, and what goes wrong without a bounds check. |
| 2 | PMPP ch. 3: multidimensional grids, row-major indexing | Write (in plain NumPy) the index function `(block_x, block_y, tx, ty) -> (row, col)` for a 2-D grid and check by brute force that every element of an MxN matrix is covered exactly once for M, N in 1..20. |
| 3 | Triton tutorial 01 *Vector Addition* (triton-lang.org > Tutorials). Triton programming model: one *program* per block, `tl.program_id`, `tl.arange`, masks | Run vector add under `TRITON_INTERPRET=1` in WSL. Then break it on purpose: remove `mask=` from `tl.store`, and change `<` to `<=`. Record what each run prints for N=1000, BLOCK=256 and for N=1024. Which breakage does `torch.allclose` miss? |
| 4 | PMPP ch. 4: compute architecture, warps, SIMT, divergence, occupancy | Explain why Triton exposes *blocks* (tiles) rather than threads. List three things the Triton compiler decides for you (layout, vectorisation, shared memory, ...). |
| 5 | Triton tutorial 02 *Fused Softmax* | Re-implement it from memory. Change `other=-float('inf')` to `other=0.0`. Work out by hand which inputs (row lengths) make the result wrong, then confirm in the interpreter. |
| 6 | PMPP ch. 5: memory hierarchy, tiling, tiled matmul | On paper, trace a tiled matmul with M=N=K=4 and tile 2: which tiles of A and B each output tile reads, and in what order. |
| 7 (long) | Triton tutorial 03 *Matrix Multiplication* (`tl.dot`, K loop, strides, `program_id` grouping) and tutorial 05 *Layer Norm* (forward only) | Run matmul in the interpreter with M, N, K not multiples of the block sizes. Make the accumulator `tl.float16` and measure the max error against an fp64 `torch` reference for K = 64, 512, 4096. Plot error against K. |

Week 1 checkpoint: you can read any of the five reference kernels and say, for each `tl.load`
and `tl.store`, which addresses it touches for given `pid`, N and BLOCK.

## Week 2: SMT and refinement

| Day | Read / watch | Exercise (deliverable) |
|---|---|---|
| 8 | Z3 Python tutorial ("Programming Z3", Bjorner, de Moura et al., sections on basics, arithmetic, bit-vectors) | Use Z3 to find x, y with `x + y == 10, x - y == 4`. Then prove `(x & (x - 1)) == 0` iff x is a power of two or zero, for 8-bit bit-vectors. |
| 9 | Programming Z3: arrays, uninterpreted functions, quantifier-free reasoning, `prove` vs `check` | Model a 1-D masked load: `offs = pid*B + i`, `mask = offs < N`. With B=4 and symbolic `pid`, `N` in [1, 64], ask Z3 whether any lane with `mask` true has `offs >= N` (it should be UNSAT). Repeat with `<=` and read the counterexample. |
| 10 | Alive2 paper (Lopes et al., PLDI 2021) sections 1-3: refinement, undefined behaviour, poison | In your own words: why is "target has UB where source is defined" a refinement failure, while "source has UB" allows anything? Map an out-of-bounds store in a mutant onto this. |
| 11 | Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 4 (summation) | Prove the recursive-summation bound \|err\| <= (n-1)u sum\|x_i\| (to first order). Empirically compare sequential, pairwise and reversed fp32 sums of 1024 randn values against an fp64 reference. |
| 12 | Revisit Day 9 | Encode the whole 1-D vector add for **all** programs at N <= 16, B = 4 with symbolic float inputs as Z3 Reals. Prove that the mutant with a reversed add order (`y + x`) is equivalent and that a missing store mask is not (read the witness). |
| 13 | Skim the abstracts in `docs/scoop_check.md`, then the full PDFs of Tatva and 2511.12638 | Write one paragraph: what can each tool *not* handle that a K-loop matmul with masked tails needs? |
| 14 (long) | No new reading | Mini-project: write 3 mutants of softmax by hand and label each inequivalent or benign. Use exact Z3 over Reals at row length <= 8 and an fp64 + tolerance check. Note where the two disagree (for example, removing the max-subtraction is exact over the reals but overflows in fp32). |

Week 2 checkpoint: you can state precisely what "equivalent" means for a float kernel and
why exact real-number equivalence alone is not enough to label a mutant.

## Resources

- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
- PMPP, 4th ed. (Morgan Kaufmann, 2022), ch. 1-5
- Programming Z3: https://theory.stanford.edu/~nikolaj/programmingz3.html
- Z3 Python API: `pip install z3-solver`
- Alive2: Lopes, Lee, Hur, Liu, Regehr, "Alive2: Bounded Translation Validation for LLVM", PLDI 2021
- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., ch. 4
