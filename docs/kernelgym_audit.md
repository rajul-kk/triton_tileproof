# KernelGYM correctness check: GPU run with planted Triton mutants

Run on Kaggle, Tesla T4, 2026-11-07. torch 2.11.0+cu130, Triton 3.6.0, KernelGYM commit
`3a84417f8c0efaadb215ef638b37d12e71ed20f3`. Each case calls `KernelSimpleToolkit.evaluate` in-process with the
`kernelbench` backend adapter. The Redis/scheduler layer is not used.

Files:

| File | What |
|---|---|
| `notebooks/kernelgym_audit.ipynb` | Mutant audit: 8 kernels, 72 mutants, tile and odd sizes, two tolerance settings |
| `notebooks/kernelgym_readme_check.py` | Three runs of the `kernel_simple` example from KernelGYM's own README |
| `results/kernelgym_audit_t4.json` | Mutant audit, with the init-inputs patch described below |
| `results/kernelgym_audit_t4_unpatched.json` | The same notebook without the patch (summary only) |
| `results/kernelgym_readme_check_t4.json` | README example results |

## 1. Two problems in the `kernel_simple` path

Both are reproduced with the `kernel_simple` example from KernelGYM's README, copied verbatim, with no patch.

| Run | Setup | compiled | correctness | Message |
|---|---|---|---|---|
| A | README `kernel_code` as written (Triton kernel and `get_cases()` in one string) | False | False | `Failed to load cases: @jit functions should be defined in a Python file` |
| B | Same kernel; `get_cases()` moved to `cases_code`; no `rtol`/`atol` | True | **False** | none (a correct `x + y` kernel is marked incorrect) |
| C | As B, with `rtol=1e-4, atol=1e-5` in the case | True | True | |

1. **Triton source is `exec`'d as a string.** `_load_cases_from_code` (when cases come from `kernel_code`) and
   `_load_init_inputs_from_code` (when cases are passed inline) both run `exec(kernel_code)` on a string. With
   Triton 3.6, `@triton.jit` refuses functions that are not defined in a file, so evaluation stops before the
   kernel runs. Only the `cases_code` path avoids it. The model itself is loaded from a temp file and is fine.
2. **Unset tolerance rejects everything.** `_normalize_case` stores `rtol=None`/`atol=None` when a case omits
   them, so `case.get("rtol", 1e-4)` returns `None`. `torch.allclose` then raises, and `_compare_tensors` turns the
   exception into `False`. The README example sets no tolerance, so it fails this check even once it loads.

Scope of the claim: one Triton version (3.6.0, the Kaggle default with torch 2.11), one commit, and the
`kernel_simple` toolkit only. The KernelBench workflow (`/evaluate`) loads code differently and was not tested.
KernelGYM does not pin a Triton version (`requirements.txt` asks only for `torch>=2.0.0`).

## 2. Mutant audit (with the patch)

To reach the comparison, the notebook replaces `_load_init_inputs_from_code` with `lambda code: []` (our kernels
take no init inputs). Without the patch, all 48 controls and 288 mutant runs stop at the load error above.

Controls: the 8 reference kernels at tile and odd sizes (16 configurations) **all pass** with explicit tolerances
(`1e-4/1e-5` and `1e-2/1e-2`) and **all fail** when the case omits them (problem 2).

Accepted by KernelGYM (`correctness == True`), by oracle label. Counts are mutants out of the number with that label.
The two mutants that do not compile on GPU Triton (below) are left out of their labels' rows
(`python -m triton_equiv.kgym_table`):

| Oracle label | n | tile sizes, 1e-4/1e-5 | tile sizes, 1e-2 | odd sizes, 1e-4/1e-5 | odd sizes, 1e-2 |
|---|---|---|---|---|---|
| oob | 27 | **27** | **27** | 17 | 17 |
| inequivalent | 19 | 9 | 9 | 0 | 0 |
| numerical: precision loss | 5 | 0 | 5 | 0 | 5 |
| numerical: reorder | 6 | 5 | 6 | 5 | 6 |
| benign (proved) | 6 | 6 | 6 | 6 | 6 |
| benign (unproved) | 7 | 7 | 7 | 7 | 7 |
| does not compile | 2 | – | – | – | – |

Reading:

- Every out-of-bounds mutant passes at tile-multiple sizes, and 17 of 27 still pass at odd sizes. KernelGYM
  compares outputs only and has no guard band, so out-of-bounds reads and writes that leave the output intact are
  invisible. None of them crashed on the GPU; the 10 rejected at odd sizes were rejected for output mismatch.
- 9 of the 19 output-changing mutants pass at tile sizes; all of them are tail bugs, and odd sizes catch every one.
  This is the aligned-shape effect reported by arXiv 2609.22220 for CUDA, reproduced here for Triton.
- KernelGYM's fallback tolerance (1e-4/1e-5) rejects every precision-loss mutant, while a KernelBench-style 1e-2
  accepts all 5 that compile.
- **Two mutants do not compile on GPU Triton**: `matmul:acc_dtype_downgrade:0` (labelled benign) and
  `matmul_fp16:acc_dtype_downgrade:0` (labelled precision loss). Triton 3.6 rejects a loop-carried accumulator whose
  dtype changes inside the loop. The CPU interpreter ran both. Their labels describe programs the compiler would
  never accept, so the table above lists them as "does not compile" instead of under their labels. Their
  rejection was the compile error, not a wrong output. KernelGYM still reports `compiled: True` for them,
  because Triton compiles on the first launch, inside the correctness run.

Not covered: other GPUs, other Triton versions, KernelGYM's performance and decoy checks, and the KernelBench
workflow.
