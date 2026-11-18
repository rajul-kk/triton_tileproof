# A KernelAudit-style oracle on the labelled mutants

KernelAudit (Li, Song, Qi, Yang and Xie, arXiv 2611.15506, 11 Nov 2026) audits six GPU-kernel validators
(CUDA-Agent, KernelBench, KernelGYM, robust-kbench, DeepReinforce, MusaCoder) against its own semantic oracle. That
oracle is a set of reference-calibrated stress tests at the task's shape, judged per output leaf by its Eq. 1. Its
corpora include Triton (TritonBench-G, KernelBenchX). It has no mutation testing, no formal labels, and no
out-of-bounds stores (read 18 Nov 2026 in the arXiv HTML; `docs/scoop_check.md` row 12).

**Question:** does a KernelAudit-style oracle catch the mutants that `results/labels.json` labels wrong (`oob`,
`inequivalent`, `numerical:precision`)? If it catches nearly all of them, the formal labels add little beyond it.

## What was run

`src/triton_equiv/kaudit.py`, run on the CPU interpreter (`TRITON_INTERPRET=1`, WSL, `~/venv-triton-equiv`). The
output is `results/kaudit_audit.json`: 80 candidates (8 reference kernels and 72 labelled mutants), in 238 s (the
file's `seconds`). The rule and schedule are re-implemented from the paper's text; no code is copied.

- **Eq. 1, per output leaf:**
  - `|k - r| <= a + rho |r|`, where `a = max(1e-8, 4 p99|r - r64|)` and
    `rho = max(1e-5, 4 p99(|r - r64| / max(|r64|, 1e-12)))`.
  - The leaf is **unresolved** when `rho` before its floor exceeds 1e-3, or when the reference is non-finite.
  - A non-finite candidate fails.
- **Schedule:** the paper gives it only in outline. Our instance has 16 slots at the task's `tile` shape:
  - `value_normal`;
  - eight value scales from 1e-4 to 1e5, each with its own seed;
  - two uniform slots;
  - four native seeds;
  - one repeated-run slot of three calls.

  A scale that overflows a buffer's dtype leaves that slot unresolved.
- **Two columns:**
  - `kaudit/tile` is the shape KernelAudit runs (the paper says its plan "does not establish alternative-shape
    coverage").
  - `kaudit/tile+odd` adds sizes that leave partial tiles, which KernelAudit does not run.

## Result (`results/kaudit_audit.json`, `confusion`)

| Oracle label | `kaudit/tile` accept / reject / unresolved | `kaudit/tile+odd` accept / reject / unresolved |
|---|---|---|
| reference (8) | 6 / 0 / 2 | 6 / 0 / 2 |
| oob (27) | 18 / 0 / 9 | 13 / 5 / 9 |
| inequivalent (19) | 5 / 7 / 7 | 0 / 12 / 7 |
| numerical:precision (6) | 0 / 1 / 5 | 0 / 1 / 5 |
| numerical:reorder (8) | 5 / 0 / 3 | 5 / 0 / 3 |
| benign:proved (12) | 8 / 0 / 4 | 8 / 0 / 4 |

**How many wrong mutants it catches.** There are 52 wrong mutants (27 `oob`, 19 `inequivalent`, 6
`numerical:precision`).
- At the task shape, it rejects **8 of 52** (8 of the 31 it resolves).
- With odd sizes added, it rejects **18 of 52** (18 of 31).

So goals.md P1's kill threshold (90% caught) is not reached, and the paper's scope stays as planned.

**What it gets right.**
- It rejects no acceptable mutant (`numerical:reorder`, `benign:proved`) in either column.
- With odd sizes it rejects every resolved `inequivalent` mutant. The 5 it accepts at the task shape (`row_sum`,
  `row_sum_loop` and `softmax` `wrong_other:0`; `attention:mask_off_by_one:3`; `attention:drop_where_mask:0`) are
  all rejected once tails exist (`rows`).

**What it misses.** Out-of-bounds mutants pass:
- 18 of 27 are accepted at the task shape, and 13 of 27 even with odd sizes.
- Of those 13, 6 write outside their output buffer on the interpreter (`guard_write`):
  - `vector_add:mask_off_by_one:0` and `vector_add:missing_store_mask:0`;
  - `matmul:mask_off_by_one:4`;
  - `softmax:missing_store_mask:0`;
  - `attention:mask_off_by_one:4` and `attention:missing_store_mask:0`.
- Like every value-comparing oracle, it never inspects memory outside the outputs. On a T4, U8 measured what these
  stores overwrite (`docs/kernelgym_audit.md` section 3).

**Where it abstains.** The fp16-output kernels `matmul_fp16` and `layernorm` are unresolved in every slot
(`slot_states` and `rows`). There are 804 `rho_cap` slot states, all on these two kernels. The cause:
- fp16 rounding against fp64 is up to 2^-11 (about 4.9e-4) relative, so 4 x p99 is above the 1e-3 cap;
- the remaining slots overflow the input dtype (`input_range`) or the reference (`ref_nonfinite`).

This follows from the rule as we read it. It is not a measured property of KernelAudit's own code, which we have not
run. For these kernels, 5 of the 6 `numerical:precision` mutants and 7 of the 19 `inequivalent` mutants go unjudged.

## P2: the same audit on a T4 GPU (`results/p2_kaudit_gpu.json`)

All 80 candidates were run on a Kaggle Tesla T4 (torch 2.11.0+cu130, triton 3.6.0; `gpu`, `torch`, `triton`).
References were computed on the CPU, as in P1, and each candidate ran in its own process. The run took 430 s and had
no crashes (`seconds`, `crashes`). The source hashes are in `source`.

The confusion is the same as P1's except for 4 of 80 candidates (row-by-row diff of `rows` against
`results/kaudit_audit.json`):

| Candidate | Label | P1 interpreter (tile / odd) | P2 T4 (tile / odd) |
|---|---|---|---|
| `matmul:acc_dtype_downgrade:0` | benign | accept / accept | reject / reject |
| `matmul_fp16:acc_dtype_downgrade:0` | numerical | unresolved / unresolved | reject / reject |
| `matmul:missing_store_mask:0` | oob | accept / reject | accept / accept |
| `softmax:missing_store_mask:0` | oob | accept / accept | accept / reject |

- **The accumulator downgrades.** Both are rejected in every slot on the GPU. The fp32 one is labelled benign because
  its proof is over the reals (`labels.json`: `real_proof` true, `fp_proof` "uf"). On hardware, the narrower
  accumulator's rounding is above the band, while the interpreter hides it. So the "benign" label is a real-number
  statement, and a floating-point oracle on hardware disagrees with it. This is a labelling caveat for the claim table
  (P3), not a defect in the oracle.
- **The missing store masks.** Out-of-bounds stores depend on the hardware: here the odd-shape verdict flips in both
  directions. These mutants are OOB by label, and on a T4 neither run catches them reliably (U8).
- **Slot states** (`slot_states`, P1 → P2): `pass` 1195 → 1181, `fail` 391 → 435, `rho_cap` 804 → 780,
  `ref_nonfinite` 96 → 90, `input_range` 74 unchanged. The per-slot diff of `rows` accounts for every change:
  - `matmul_fp16:acc_dtype_downgrade:0` now fails 30 slots: 24 that were `rho_cap` and 6 that were `ref_nonfinite`.
    References still run on the CPU; the candidate's definite failure is checked first (L59), so those slots no
    longer reach the abstentions.
  - `matmul:acc_dtype_downgrade:0` goes from pass to fail on 32 slots.
  - The OOB store masks and `matmul:mask_off_by_one:5` move both ways: `softmax:missing_store_mask:0` +13 fail;
    `matmul:missing_store_mask:0` −16 and `matmul:mask_off_by_one:5` −15. The off-by-one's verdict does not change,
    because some slots still fail.

## Caveats

- **The rule is a re-implementation.** The paper's Appendix B schedule is given only in outline, so our 16 slots are
  an instance of it, not a copy. Different value choices could change individual verdicts.
- **One GPU, one run.** P2 is a single T4 run. The OOB flips show that some verdicts are not stable across runs or
  devices, and P2 did not measure that rate.
- **Small task shapes.** The `tile` shapes are this repo's checker sizes (`kernels.py`, `checker_sizes`), not
  KernelBench's.
