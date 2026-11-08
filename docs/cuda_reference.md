# C++/CUDA reference kernels

The standard checker (`harness`) compares a candidate Triton kernel with PyTorch. That
makes PyTorch part of the measurement: a verdict could depend on how PyTorch computes the
reference (its reduction order, TF32, fused kernels) rather than on the candidate. This
adds a second reference that shares no code with PyTorch or Triton, and asks whether any
verdict changes when the reference changes.

## What was built

* `src/triton_equiv/csrc/ref_kernels.h` holds the eight kernels of `kernels.py`, written
  from their specifications as sequential loops, one output unit per call. The arithmetic
  type is `Acc = float` or `double`. Every function is `__host__ __device__`.
* `csrc/ref_cpu.cpp` is the CPU build: one loop iteration per output unit.
* `csrc/ref_cuda.cu` is the CUDA build: one thread per output unit (an element, or a row
  for the row-wise kernels and attention). It uses no tiling, no shared memory and no
  tensor cores.
* `cuda_ref.cpp_reference(kernel, launch, inputs, acc, backend)` is a drop-in for
  `harness.torch_reference`.
  * fp16 inputs are widened to float32, which is exact.
  * An `F32` result is rounded to the output buffer's dtype. An `F64` result stays float64.
  * Builds use `torch.utils.cpp_extension.load_inline`, which needs ninja and a C++
    compiler, plus nvcc for CUDA.
* `refcheck.py` reruns the standard checker on every reference kernel and every labelled
  mutant against three references:
  * `torch`: PyTorch;
  * `cpp`: the C++ kernels in fp32;
  * `cpp64`: the C++ kernels in fp64.

  It records the verdict per (reference, size kind, tolerance policy), every candidate
  whose verdict differs between references, and the reference Triton kernel's largest
  error against each reference.

Both builds run the same header, so the CPU build stands for the CUDA one up to the
device's `exp`/`sqrt`. Sums are sequential, not the tree order of `tl.sum` or a tensor-core
MMA. This is an independent reference, not a bit-for-bit model of the Triton kernels.

**Tests**: `tests/test_cuda_ref.py` checks the CPU build on every kernel at every oracle
and checker size:

* in fp64 it agrees with `harness.fp64_reference` to 1e-12;
* in fp32 it agrees with `harness.torch_reference` under both tolerance policies,
  including the tight `torch_default` one (fp32 rtol 1.3e-6).

The tests are skipped, with the build error as the reason, where the extension cannot be
built.

## Result on the CPU (Triton interpreter)

`python -m triton_equiv.refcheck --trials 3` (WSL, Triton 3.9.0 interpreter, torch 2.15
CPU, 44 s) gives `results/refcheck_cpu.json`. It covers 80 candidates: the 8 reference
kernels and 72 mutants.

**No candidate changes verdict with the reference**, under either policy or size kind.
On the CPU, the checker's verdicts are a property of the candidate, not of PyTorch.

Largest error of each unmutated Triton kernel against each reference:

| kernel | torch | cpp (fp32) | cpp64 |
|---|---|---|---|
| vector_add | 0 | 0 | 2.4e-7 |
| row_sum | 1.9e-6 | 2.9e-6 | 1.6e-6 |
| matmul | 7.6e-6 | 9.5e-6 | 3.6e-6 |
| row_sum_loop | 9.5e-7 | 0 | 9.5e-7 |
| matmul_fp16 | 2.0e-3 | 2.0e-3 | 7.7e-3 |
| softmax | 1.5e-8 | 6.0e-8 | 2.0e-8 |
| layernorm | 0 | 0 | 9.7e-4 |
| attention | 1.0e-6 | 1.2e-6 | 1.2e-6 |

The `cpp64` column for the fp16-output kernels (`matmul_fp16`, `layernorm`) is the
rounding of the output to fp16. Everything else is fp32 reordering at the 1e-6 level.

## On the T4 (not yet run)

`notebooks/cuda_ref.ipynb`:

1. Builds both extensions and checks the CUDA build against the CPU build and against
   PyTorch on every kernel and size.
2. Runs `refcheck --device cuda`: real Triton on the GPU, `cpp`/`cpp64` from the CUDA
   build.

The GPU is where a difference is plausible. There, `tl.dot` uses tensor cores (fp16
inputs, and possibly TF32 for fp32), while the CUDA reference uses plain fp32 FMA loops.
