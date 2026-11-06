# Scoop check: prior work on Triton/GPU-kernel equivalence and checker audits

Checked 2026-11-06. **Read-level tags** say what was actually read for each source:

- `abstract`: the arXiv abstract page or the conference listing page (title, authors, abstract, links)
- `README`: a repository README read on GitHub
- `snippet`: a search-engine result snippet only
- `full PDF`: the full paper (**none of the sources below were read at this level**)

"No code" means no repository link appeared on the abstract or listing page and none turned
up in a web search. A repo released later, or linked only inside the PDF, would be missed.
Verify against the full PDFs before making any novelty claim.

## Summary table

| # | Work | Read level | Code released? | What it covers | Overlap with this repo |
|---|---|---|---|---|---|
| 1 | **Tatva**, "Formally Verifying AI-Generated GPU Kernels" (Taneja, St John, Serrino), NeurIPS 2026 ML for Systems workshop | abstract (neurips.cc listing) | No (listing links only an OpenReview forum) | Lowers a PyTorch baseline and an optimized candidate to a common DAG, scalarizes them and checks in Z3. Proves equivalence for **straight-line, loop-free** kernels on KernelBench. Masks are not mentioned. | Same goal at the PyTorch/Triton boundary. Tile **loops**, masks and reduction order are explicitly outside its stated scope, which leaves the gap this repo targets. |
| 2 | arXiv 2511.12638, "Equivalence Checking of ML GPU Kernels" (Driscoll, Dubey, Wei, Kayal, Sharma, Aiken), OOPSLA 2026 | abstract, plus search snippets for the system name | No repo on the abstract page | Described as the first equivalence checker for GPU kernels. Sound, and complete for a defined kernel class. Handles convolutions, matmul and attention. Snippets say it works on CUDA kernel assembly by symbolic evaluation, and also checks race and deadlock freedom. | The strongest prior work on soundness and completeness. It works at the CUDA/PTX level, not on Triton source. **System name "Volta" is confirmed** by the arXiv abstract page and the search snippets. |
| 3 | ProofWright, arXiv 2511.12294 (Chatterjee, Zagieboylo, Damani, Hari, Kozyrakis) | abstract | No repo on the abstract page | Agentic formal verification of **CUDA**. Proves safety properties for 74% of KernelBench L1 kernels. Proves semantic equivalence only for element-wise kernels. About 3 minutes per kernel. | CUDA only, and equivalence only for element-wise kernels. Reductions, `tl.dot` and loops are not covered. |
| 4 | Contract-Grade Verifier, arXiv 2608.12700 (Shah, Shrestha) | abstract | No repo found. A Zenodo archive is cited (doi 10.5281/zenodo.21563213), contents not checked. | Twelve adversarial "gates", several tolerance-free (NaN/inf, determinism, shape robustness, accumulation precision). Audits **2,638 accepted kernels: 39.5% broken** beyond any tolerance argument, 62.1% with at least one violation, 1,487 accepted by the standard test but rejected by the verifier. | A testing-based checker, not a formal one. Its numbers are the baseline our audit must be compared against. Check the Zenodo archive for kernels we could reuse. |
| 5 | Measuring the Checker, arXiv 2609.22220 (Du, Luu, Huang, Ng) | abstract | **Data released**: KernelBench-M on Hugging Face (`Elfsong/KernelBench-M`). No code repo linked. | **Mutation analysis of benchmark oracles.** 10,303 compilable faults in 188 KernelBench problems (CUDA). The official check misses 16.9% of witnessed faults, and 78.6% of precision faults. Shows a tolerance blind band that grows with reduction size, and audits KernelBench-Verified. | **Largely pre-empts the "mutation audit" half of Plan H.** It targets CUDA, labels mutants with kill witnesses rather than an SMT/fp64 oracle, and does not cover Triton masks/tails or `program_id` axes. Our audit is only new if it is Triton-specific and its labels come from a formal oracle (including proofs that a mutant is benign). |
| 6 | The Correctness Illusion, arXiv 2606.20128 (Sarkar) | abstract | No repo on the abstract page | 24, later 26, Triton/CPU stand-in kernels with 9 to 10 seeded transcription bugs. Uses seeded fuzzing, an fp64 CPU reference and per-(op, dtype) tolerances. Caught 10 of 10 bugs with 16 of 16 controls clean on five GPUs. | Includes Triton. Fuzzing is the baseline we should compare against. Its corpus is small and hand-seeded. |
| 7 | KernelBench-Verified, arXiv 2607.16241 (Zhang, Yu, Wang, Fan, Reed, Mirhoseini, Su) | abstract | No repo on the abstract page | Hidden test suite with four input distributions, TF32 baseline and memory metrics. Reports reward hacking via hardcoded bypasses. The best model's speedup falls from 1.43x to 0.88x. | The stronger testing baseline. 2609.22220 reports that its gain splits into +4.0 points from hidden inputs and +4.5 from tighter tolerance. |
| 8 | TritonRL, arXiv 2510.17891 (Woo, Zhu, Nie, Jia, Wang, Park) | abstract | No repo on the abstract page | 8B Triton-writing LLM trained with RL. A "multi-layered verification system" counters reward hacking. The abstract does not detail the checks. | Its checker would be an audit target. Read the PDF for the exact checks. |
| 9 | Dr. Kernel / KernelGYM, arXiv 2602.05885 | README (github.com/hkust-nlp/KernelGYM). Paper not read. | **Yes**: environment, training code (`drkernel/`, VERL integration), models and data (HF collection `hkust-nlp/drkernel`). Apache-2.0. | Distributed GPU environment. Correctness is checked by rtol/atol against a reference over configurable test cases. The README mentions "reward hacking detection" and "decoy kernel detection" without details. Requires CUDA 11.8+, Redis and Linux. | The most practical audit target because it is released. Its checks need a GPU, so on this machine they can only run on Kaggle. Checker code to read first: `kernelgym/toolkit/kernel_simple/toolkit.py` and `kernelgym/workflow/kernel_simple.py`. |
| 10 | triton-verify (github.com/alinamiretai/triton-verify) | README | **Yes** (27 commits, 0 stars, no license stated) | Proves **pointer safety** (no out-of-bounds access) on pre-optimization TTIR with Z3 bit-vectors. Program id is symbolic in [0, grid). Returns SAT with a counterexample pid, or UNSAT. Numerical equivalence is out of scope. README reports 20/20 tests on 10 kernels. | Overlaps with the memory-safety part of the oracle: missing store mask and off-by-one OOB mutants. Equivalence is not covered. Possible reuse or comparison for OOB labels, but there is no license. |

### Additional related work found during the search (not in the original list)

| Work | Read level | Code | Note |
|---|---|---|---|
| "Red-Teaming the Reward: A Verifier and Benchmark for Triton Kernel Optimization Agents" (Shiri, Hosseini), NeurIPS 2026 | abstract (neurips.cc listing) | None linked | Triton verifier plus KernelQuest (100 tasks). Rejected 813 of 1,800 agent runs and accepted 100 of 100 certified solutions. Targets shortcuts such as reduced precision and CUDA Graph replay. The listing mentions neither mutation analysis nor formal methods. **Another Triton checker we must compare against.** |
| DRTriton, arXiv 2603.21465 | snippet | Unknown | Triton verifier pipeline: linter, then compile, then output comparison. Monkey-patches kernels with no-ops to check they are actually used. |
| Verified Lifting, arXiv 2412.20992 | not re-read (cited from the earlier research doc) | Unknown | Earlier notes say masks are excluded and matmul fails. |
| gpucheck (github.com/Akasxh/gpucheck) | snippet | Yes | pytest-style correctness, benchmarking and shape fuzzing for CUDA and Triton. A possible fuzzing baseline. |
| ouroboros-kernelsmith (github.com/ymrohit/ouroboros-kernelsmith) | snippet (GitHub search result) | Yes | "Exploit-tested verification for LLM-generated Triton GPU kernels." Not inspected. |

## Verdict

- **Checker half (formal equivalence for Triton with tile loops): still open, but contested.**
  Tatva states it is loop-free. Volta works at the CUDA/assembly level. ProofWright is CUDA and
  element-wise only. triton-verify covers pointer safety only. Based on abstracts and READMEs,
  none of them claims Triton-source equivalence with **tile loops, masked load/store, `tl.dot`
  and reduction-order awareness**. Tatva's authors are the most likely group to extend to loops.
  Volta's completeness result is the bar any soundness claim will be measured against.
- **Audit half (mutation suite): largely pre-empted for CUDA by 2609.22220** (10k mutants,
  KernelBench-M released). It is also close to 2608.12700 (twelve gates, 39.5% broken) and
  2606.20128 (seeded Triton bugs). The audit is still worth doing as an **entry project and
  benchmark for the checker**, if it differs on three points:
  1. Triton-specific operators (mask/tail, `program_id` axis, `other=` values, strides),
  2. labels from a formal oracle that can also *prove* a mutant benign (for example a reduction
     reorder that is exact over the reals and within fp tolerance), not just kill witnesses,
  3. an audit of the released KernelGYM checks, run on Kaggle.
  Do not claim to be the first mutation analysis of kernel oracles.
- **Before any novelty claim:** read the full PDFs of Tatva (OpenReview `cEh56JC9TB`), 2511.12638
  (scope: loops and masks?), 2609.22220 (exact operator list, any Triton subset), and the Red-Teaming
  verifier.

## Sources

- https://neurips.cc/virtual/2026/178980 (Tatva)
- https://arxiv.org/abs/2511.12638 ; https://theory.stanford.edu/~aiken/publications/papers/oopsla26c.pdf (not opened)
- https://arxiv.org/abs/2511.12294
- https://arxiv.org/abs/2608.12700
- https://arxiv.org/abs/2609.22220 ; https://huggingface.co/datasets/Elfsong/KernelBench-M (not opened)
- https://arxiv.org/abs/2606.20128
- https://arxiv.org/abs/2607.16241
- https://arxiv.org/abs/2510.17891
- https://github.com/hkust-nlp/KernelGYM
- https://github.com/alinamiretai/triton-verify
- https://neurips.cc/virtual/2026/173031 (Red-Teaming the Reward)
- https://arxiv.org/pdf/2603.21465 (DRTriton, snippet only)
