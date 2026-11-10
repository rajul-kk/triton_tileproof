# Batch invariance

`src/triton_equiv/batchinv.py` asks whether one row of a kernel's output is bitwise the
same at two batch sizes. The kernel is given as Triton source together with the launch
policy that picks tiles, split-K or rows per program from the batch size. The module
docstring defines the verdicts and the assumption they rest on. This page reports the
run.

```
python -m triton_equiv.batchinv --gpu --out results/batchinv.json
```

The run took 8.9 s on a Kaggle T4 with torch 2.11.0+cu130 and Triton 3.6.0. The symbolic
check compares batch 2 with batch 5 at the small sizes. The hardware check compares
batch 4 with batch 32 at the GPU sizes, over 10 inputs (5 seeds of each of 2 generators).

## Results

| Family | Policy | Expected | Symbolic, B = 2 vs 5 | Emulator witness | T4, B = 4 vs 32 |
|---|---|---|---|---|---|
| softmax | one program per row | invariant | proved | — | equal |
| layernorm | one program per row | invariant | proved | — | equal |
| rowsum_loop_fixed | K-loop tile fixed | invariant | proved | — | equal |
| rowsum_loop_block_by_batch | wider K tile for a small batch | variant | refuted | yes | differs, 28 rows |
| rowsum_rows_by_batch | 1 row per program for a small batch, 4 otherwise | invariant | proved | — | equal |
| matmul_fixed | BM, BN, BK fixed | invariant | proved | — | equal |
| matmul_bm_by_batch | BM from the batch, BK fixed | invariant | proved | — | equal |
| matmul_bk_by_batch | BK from the batch | variant | refuted | yes | **equal** |
| splitk_fixed | split-K = 2 at every batch | invariant | proved | — | equal |
| splitk_by_batch | split-K for a small batch only | variant | refuted | yes | differs, 40 rows |
| rowsum_mixed_rows | wrong kernel: rows mix | variant | refuted, cross-row read | yes | differs, 40 rows |

- **All 7 proved families were bitwise equal on the T4.** That includes
  `rowsum_rows_by_batch`, where a 1-row tile and a 4-row tile reduce the same length.
  This is the layout case the deterministic-primitive assumption could miss. It held
  here, on 10 inputs and one GPU.
- **3 of the 4 refuted families also differ on the T4.** A different K tile, split-K
  factor or row mix changes the bits, as the witness predicted.
- **`rowsum_mixed_rows` is caught twice.** The row-independence check finds an output of
  row 0 that reads row 1, so it is not batch invariant even in exact arithmetic.

## matmul_bk_by_batch: refuted, but equal on the T4

The policy uses BK = 32 for a small batch and 16 otherwise, with K = 64. The symbolic
terms differ, and the numpy emulator finds inputs whose result differs by one ulp
(−1.7401631 against −1.7401633). On the T4 every one of the 10 inputs gave identical
bits.

The cause is a rewrite in the Triton compiler. The kernel writes `acc += tl.dot(a, b)`.
Triton's combiner (`CombineDotAddFPattern` in
`lib/Dialect/Triton/Transforms/Combine.cpp`) folds an add of a dot with a zero
accumulator into the dot itself, giving `tl.dot(a, b, acc)`. It applies only when the dot
has one use and `maxNumImpreciseAcc` is 0, and both hold here.

- **Emulator.** It computes `acc + (sum of one BK chunk)`. Its rounding therefore depends
  on BK.
- **T4.** The T4 (sm_75) has no TF32. An fp32 dot is lowered to FMAs that accumulate
  into `acc` one k at a time, in order. The chain over all 64 products is then the same
  at any BK.

So the refutation is correct for the source semantics, but not for the code Triton
generates for this pattern.

**Status of this explanation.** The rewrite was checked in Triton's source. That the T4
FMA chain runs in plain k order was not checked in the generated TTGIR or PTX. A GPU with
`mma` (Ampere or later, TF32 or fp16 inputs) may group k differently, and there BK could
change the bits. Two follow-ups:

- model the folded form (`dot(a, b, acc)` as one accumulating chain) in the
  uninterpreted-function tier;
- rerun the family on an sm_80+ GPU.

## What this does and does not show

- Proofs hold at the listed sizes and batch pairs only, under the deterministic-primitive
  assumption in the module docstring.
- The policies are the module's own 11 families, not policies taken from real libraries.
- Each hardware check used one GPU and one Triton version.
