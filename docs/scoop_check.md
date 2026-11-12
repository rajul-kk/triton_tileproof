# Scoop check: prior work on Triton/GPU-kernel equivalence and checker audits

First checked 2026-11-06 (abstracts and READMEs). Updated the same day after reading two papers
in full and reading the source of one new competitor (kernel-tv). **Read-level tags** say what
was actually read for each source:

- `abstract`: the arXiv abstract page or the conference listing page (title, authors, abstract, links)
- `README`: a repository README read on GitHub
- `source`: repository files read on GitHub (named in the row). Nothing was cloned, built or run
- `snippet`: a search-engine result snippet only
- `full PDF`: the full paper. **Only 2609.22220 and 2511.12638 (Volta) were read at this level**
- `full text (partial)`: the arXiv HTML, read in part (the share is given in the row)
- `abstract + blog`: the conference abstract plus a vendor blog post about the work

"No code" means no repository link appeared on the abstract or listing page and none turned
up in a web search. A repo released later, or linked only inside the PDF, would be missed.

## Summary table

| # | Work | Read level | Code released? | What it covers | Overlap with this repo |
|---|---|---|---|---|---|
| 1 | **Tatva**, "Formally Verifying AI-Generated GPU Kernels" (Taneja, St John, Serrino; Gimlet Labs), NeurIPS 2026 ML for Systems workshop | abstract (neurips.cc listing), plus the owner's notes that it goes through TTIR and Z3. Full paper **could not be opened** (2026-11-11): OpenReview `cEh56JC9TB` and its API both redirect to a bot check, and a title search finds no arXiv version | No (listing links only an OpenReview forum) | Lowers a PyTorch baseline and an optimized candidate to a common form (TTIR) and checks in Z3. Proves equivalence for **straight-line, loop-free** kernels on KernelBench. Masks are not mentioned. | Same goal at the PyTorch/Triton boundary. Update (2026-11-09, Gimlet blog read in full): it now handles reductions by bounded symbolic unrolling (a 256-term dot product unrolled to k=2), still over exact reals, and was presented at ARRAY 2026. On 26 KernelBench Level 1 Triton kernels: 16 proved equivalent, 2 pass numeric tests but proved inequivalent, 8 unknown. Masks and OOB are not discussed. |
| 2 | **Volta**, arXiv 2511.12638, "Equivalence Checking of ML GPU Kernels" (Driscoll, Dubey, Wei, Kayal, Sharma, Aiken), OOPSLA 2026 | **full PDF** | **Yes**: github.com/willtunnels/volta, MIT. README read 2026-11-11 (nothing cloned or run) | Equivalence checker for GPU kernels by symbolic evaluation of **PTX**. Values are **reals**, so a reordered reduction counts as equal. Sound, and complete for its kernel class. Problem **sizes are fixed**; partial tiles and tails are not discussed. Also checks race freedom. Evaluated on 4 LLM-generated kernels and found **no** kernel that the tests accepted but that was buggy. | Strongest prior work on soundness. Different level (PTX, not Triton source), real-valued only (no rounding, no precision-loss tier), no tails, no checker audit. Its "no hidden bugs in 4 LLM kernels" result is the opposite of what a mutation audit measures. Repo README: input is **PTX** from nvcc (Triton is not mentioned); kernels must have static sizes and statically resolvable addresses; floating point canonicalised to rational functions over the reals, with `sqrt`, `log`, comparisons, `select` and data-dependent reads as uninterpreted atoms (sound, not complete); an optional f64 oracle (`--verify-numeric`) rechecks verdicts at random inputs; masks and out-of-bounds are not discussed; 39 benchmark kernels (reduction, matmul, attention, conv, race). Nothing in it changes the overlap. |
| 3 | ProofWright, arXiv 2511.12294 (Chatterjee, Zagieboylo, Damani, Hari, Kozyrakis) | abstract | No repo on the abstract page | Agentic formal verification of **CUDA**. Proves safety properties for 74% of KernelBench L1 kernels. Proves semantic equivalence only for element-wise kernels. About 3 minutes per kernel. | CUDA only, and equivalence only for element-wise kernels. |
| 4 | Contract-Grade Verifier, arXiv 2608.12700 (Shah, Shrestha) | abstract | No repo found. A Zenodo archive is cited (doi 10.5281/zenodo.21563213), contents not checked. | Twelve adversarial "gates", several tolerance-free. Audits **2,638 accepted kernels: 39.5% broken** beyond any tolerance argument. | A testing-based checker. Baseline numbers for any audit. |
| 5 | **Measuring the Checker**, arXiv 2609.22220 (Du, Luu, Huang, Ng) | **full PDF** | **Data released**: KernelBench-M on Hugging Face (`Elfsong/KernelBench-M`). No code repo linked. | Mutation analysis of benchmark oracles: **124 mutation rules, CUDA only**, 10,303 compilable faults in 188 KernelBench problems. Labels come from **kill witnesses or quarantine; there are no formal proofs**. Its *Boundary* family already shows that aligned shapes hide dropped guards and tails ("remainder blocks never execute", **22.9% escape**). It already reports that the **tolerance blind band grows with reduction size**: an all-zeros softmax passes, and the largest undetectable relative error is 0.39 at d=64 and 2.5e3 at d=393,216. It does not mention KernelGYM. | **Pre-empts** "aligned sizes hide OOB/tail bugs" and "the blind band grows with n". Both must be cited to it, not claimed. Not covered by it: Triton-specific operators (`mask=`/`other=`, `program_id` axis, accumulator dtype), proof-backed labels (proved-benign, real-equivalent-but-rounding-different, precision loss), and the RL reward checkers. |
| 6 | The Correctness Illusion, arXiv 2606.20128 (Sarkar) | abstract | No repo on the abstract page | 24-26 Triton/CPU stand-in kernels with 9-10 seeded bugs; seeded fuzzing, fp64 CPU reference, per-(op, dtype) tolerances. | Includes Triton. Fuzzing baseline. Small hand-seeded corpus. |
| 7 | KernelBench-Verified, arXiv 2607.16241 | abstract | No repo on the abstract page | Hidden test suite with four input distributions, TF32 baseline, memory metrics. | Stronger testing baseline. |
| 8 | TritonRL, arXiv 2510.17891 | full text: arXiv HTML, first ~60% read by the coordinating session, the rest (appendices C, E, G, H) on 2026-11-11 | **No** ("Codes and datasets will be available soon") | 8B Triton-writing LLM trained with RL. Reward is `valid = syntax x func`. *Syntax*: a rule-based linter that checks for `@triton.jit`. *Func*: a linter that detects real Triton kernel calls and flags PyTorch module use, plus an LLM judge (Qwen3-235B-Instruct) for semantic correctness. Correctness runs on **5 random test inputs**; the paper writes the comparison as exact equality (`o_code(x) == g(x)`) and states no tolerance. | **Audit target.** No released code to run. As written, a fixed set of 5 random inputs at fixed shapes has the aligned-shape blind spot unless the shapes leave tails; the exact-equality reading would reject every reordering, which we label benign or reorder. Not checked against code. The appendices add no tolerance (appendix C: outputs on "five randomly generated inputs" must all match), list the LLM judge's five rules (all core computation in Triton, no identity or no-op, no omitted components, a real bottleneck, real parallelism; appendix H), and name formal guarantees as future work (section 3.2). No release statement beyond the one above. |
| 9 | Dr. Kernel / KernelGYM, arXiv 2602.05885 | README and `kernelgym/toolkit/kernel_simple/toolkit.py`, `kernelgym/workflow/kernel_simple.py` (source) | **Yes**, Apache-2.0 | Distributed GPU RL environment. Correctness by rtol/atol against a reference over test cases. | **Audit target.** Run once on a Kaggle T4 (Triton 3.6.0); see `docs/kernelgym_audit.md`. |
| 10 | triton-verify (github.com/alinamiretai/triton-verify) | README | Yes, no license stated | Proves **pointer safety** on pre-optimization TTIR with Z3 bit-vectors; symbolic pid in [0, grid). Equivalence out of scope. | Overlaps with our `oob` query only. |
| 11 | **kernel-tv / triton-tv** (github.com/Youngzt998/kernel-tv; fork jaeHbk/kernel-tv) | **source**, read on GitHub 2026-11-06 (see the next section) | **Yes**, **no license** (GitHub reports none; no LICENSE file). 288 commits; created 2026-09-04, last push 2026-09-16. Fork is 2 commits ahead (2026-10). | **Translation validation of the Triton compiler**: proves two TTIR functions (one kernel before and after a pass pipeline) equivalent with Z3. Includes a GPU equivalence fuzzer that compares compilations bitwise. | Different question (miscompiles, not LLM kernels or reward checkers). Today it models **no loops and no `tt.dot`**, no OOB, no tolerance. See below. |

### kernel-tv in detail (source read, nothing run)

Files read: `README.md`, `doc/ttir-modeling.md`, `doc/roadmap.md`, `doc/kernel-smt-goals.md`,
`doc/dot-encoding.md`, `paper/noticable.md`, `semantics/AbstractFp.h`, `semantics/Memory.cpp`,
`semantics/Context.cpp`, `builder/triton/TritonOps.cpp`, `builder/mlir/ArithOps.cpp`,
`builder/mlir/State.cpp`, `eq_fuzzing/README.md`, `eval/inequal/cases.tsv`; in the fork,
`doc/ttir-bug-mechanisms.md` and `eval/dot_accumulator_witness.py`. The repo tree lists
`paper/noticable.md` as the only file in `paper/`: an engineering log about memory encoding,
not a paper draft.

| Question | What the code does (main branch) |
|---|---|
| Purpose | Compiler translation validation: "the same kernel before vs after a compiler pass/pipeline is equivalent". LLM kernels, KernelBench, RL rewards, mutation and tolerances are not mentioned in the goals doc. |
| Masked `tt.load` with `other=` | `Memory::load` returns `ite(mask_i, load(ptr_i), other_i)`. If `other` is absent, a **zero** tile is synthesized (Triton leaves those lanes undefined; we model them as fresh `undef`). The heap is read at `ptr_i` regardless of the mask. **Unmasked** loads and stores hit an `assert` ("not yet supported"). |
| Masked `tt.store` | Byte-wise `ite(mask_i && addr == byteAddr, byte, old)` over a lambda array; masked-off lanes leave memory unchanged. |
| Out-of-bounds | **Not modelled.** Memory is one unbounded `Array(BV64, BV8)` per pointer argument; no allocation sizes, no bounds or UB checks. A dropped guard that only touches memory outside the buffer is invisible. |
| `scf.for` | Dispatched in `State::interpretOp`, but `interpretFor` (and `interpretIf`, `interpretWhile`) are **unimplemented**: each ends in `llvm_unreachable("... not yet implemented")` (confirmed independently by the coordinating session; `tt.dot` likewise has no branch in `interpretOp` and falls through to `llvm_unreachable`). The roadmap lists unrolled loops for milestone M1-MVP and summarized loops for M1-complete; loops "block 197 kernels" of their corpus. |
| Reductions | `tt.reduce`: single operand, **1-D only** (`axis` ignored), folded left in index order and statically unrolled. The combiner is interpreted from the region. |
| `tt.dot` | **Not dispatched.** `doc/dot-encoding.md` is a "proposed design note, NOT an adopted decision": an uninterpreted `fp_dot` with optional recurrence axioms and one UF family per `inputPrecision`. Fork notes say a saved TTIR pair "cannot yet be interpreted" because `tt.dot` and `scf.for` are unsupported. |
| Floating point | Default **Abstract**: each value is an opaque `BitVec(width)` tag, every op an uninterpreted function, with axioms that +0, -0, +inf, -inf and NaN are distinct, `add`/`mul`/`max` commute and `neg` is an involution; associativity withheld. Ops: add, sub, mul, div, max, neg, abs, `math.exp`; no sqrt, no fma. Real and FPA (IEEE) modes are designed; `Real` throws in `load`/`store`. The fork adds IEEE constants and rounded add/sub/mul/div/fma in the core, but the TTIR binary still uses Abstract mode. No error bounds, no precision-loss notion. |
| Sizes | **Fixed**: tile shapes are compile-time constants and the witness-index `exists` is unfolded statically. `program_id` is a free symbolic `BV32` shared by both sides and **not constrained to a grid**. Problem sizes (`n_elements`) are scalar arguments assumed equal on both sides. |
| Test corpus | `eval/inequal/`: 6 NEQ pairs (vector add: `sub`, `mul`, `pid*5`; softmax with 8-column tiles: `mul_not_div`, `add_not_sub`, `sum_not_max`). `eval/pairs/` and `eval/compile-options/`: vector add and softmax under pass permutations. Reported: 1024-element add self-equivalence in 0.30 s. |

**Overlap with our claims.** kernel-tv is the closest Triton-level SMT tool, and its Abstract
FP mode is the same idea as our uninterpreted-function bitwise check (`fporacle`, check 2).
With commutativity axioms it is slightly stronger than ours, which is purely structural. It does
**not** currently handle any of what our labels depend on: tails at sizes that are not tile
multiples, OOB, loops, `tt.dot`, an undefined value for an unmasked lane without `other=`, a
real-equivalent tier, error bounds, or tolerance checkers. Its softmax NEQ cases are operator
swaps, not mask or tail bugs. If its M1 milestones land (unrolled loops, `tt.dot`), it could
produce `inequivalent`/`benign` labels for our matmul mutants, but still not `oob` (no bounds)
and not precision loss. No license, so its code cannot be reused.

### Additional related work (not in the original list)

| Work | Read level | Code | Note |
|---|---|---|---|
| **"Red-Teaming the Reward: A Verifier and Benchmark for Triton Kernel Optimization Agents"** (Shiri, Hosseini; Turing), NeurIPS 2026 | abstract + blog (neurips.cc listing; turing.com/blog/kernelquest-benchmark-ai-agents-triton-kernels), read by the coordinating session. OpenReview `JK85pn7BYN` blocked by a bot check, again on 2026-11-11 (forum and API), and a title search finds no arXiv version | **Hidden**: verifier and certified solutions not released | Verifier with 4 gates: *structural* (a real Triton kernel does the work); *correctness* (5 seeded inputs across up to 5 distributions, plus dtype, storage-freshness, sensitivity and parameter-structure checks); *policy* (no TF32 or lower-precision casts on fp32 tasks, no CUDA-graph capture); *performance*. KernelQuest, 100 tasks. 25% of audited agent submissions were invalid, and "three published verifiers and a stopwatch baseline accepted nearly all" of them, while their verifier accepted almost none. Rejected 813 of 1,800 agent runs. The blog compares against KernelBench, KernelBench-Verified, robust-kbench, TritonBench, GEAK-eval, TritonGym, MultiKernelBench, AgentKernelArena, FastKernels and SOL-ExecBench; which three verifiers were audited is not stated in what was read. | **An audit of published Triton checkers already exists** here, built from agent-produced submissions and a hidden verifier. Ours is a complement: formally labelled, Triton-specific mutants that isolate the failure class (OOB at tails, `undef` lanes, precision loss, reorderings). Their policy gate (no lower-precision casts) targets the same thing as our precision-loss tier, by rule instead of by error bound. |
| TritonGym (Guan et al., ICML 2026, PMLR 306) | **full PDF** (proceedings PDF, text extracted) and README (github.com/yil384/TritonGym), 2026-11-11 | **Yes**, MIT | An agentic benchmark: tools for compile, verify and evaluate. The verifier is "probabilistic": for each task shape, N random inputs with fixed seeds, all of which must meet absolute and relative thresholds set "with reference to data format and operator type"; the README states a max absolute error of 0.01 against PyTorch. The paper calls random testing "the best practice for numerical kernels". No masks, tails, OOB or formal checks are discussed. The structure, freshness, precision and degeneracy gates in the earlier snippet are Turing's description of a tier, not in this paper. | **Audit target**, and released (`benchmark/eval/runner.py`, `tools.py`). A fixed absolute tolerance at task-specified shapes is the setting where our OOB-at-tail and precision-loss mutants are expected to pass. Checked 11 Nov: the code uses one draw (`shape_index` 0, fixed seed) where the paper says N, and only the absolute bound; 5 of its 139 tasks have a first shape with a dimension above 16 that is not a multiple of 16. Its rule is now a column of our audit (README): it accepts all 27 `oob`, 9 of 19 `inequivalent` and 2 of 6 precision-loss mutants. |
| DRTriton, arXiv 2603.21465 (Guo, Lin, Yang) | full text (partial): arXiv HTML, section 4.1 and appendix F, 2026-11-11 | No repo linked | Correct only if three checks pass: a linter (at least one `@triton.jit` kernel) and compilation; *faithfulness*, where each kernel is swapped for a no-op to confirm the Triton code runs; and 5 random test cases whose outputs must be "matched precisely" against PyTorch, with no tolerance given. No formal checks. | **Audit target** with no code. As written, exact matching on 5 random inputs would reject reorderings and accept OOB at tails whenever the shapes are tile multiples. |
| Verified Lifting, arXiv 2412.20992 | not re-read | Unknown | Earlier notes say masks are excluded and matmul fails. |
| gpucheck (github.com/Akasxh/gpucheck) | snippet | Yes | Shape fuzzing for CUDA and Triton. Possible baseline. |
| ouroboros-kernelsmith (github.com/ymrohit/ouroboros-kernelsmith) | snippet | Yes | "Exploit-tested verification for LLM-generated Triton GPU kernels." Not inspected. |

## Verdict (revised)

**Do not claim**, because 2609.22220 already shows them (cite it):
- that aligned (tile-multiple) sizes hide dropped guards, tails and OOB accesses (its Boundary
  family, 22.9% escape);
- that the tolerance blind band grows with reduction size (0.39 relative at d=64 to 2.5e3 at
  d=393,216; an all-zeros softmax passes);
- that this is the first mutation analysis of kernel correctness checks.

Our results on these points are **Triton-interpreter replications**, and should be presented that way.

**What remains ours**, in three parts:
1. **Triton-specific mutants.** Masks (`mask=` comparisons, a dropped store mask), `other=`
   fill values, `program_id` axis, accumulator dtype, cast changes and reduction-order rewrites,
   over Triton kernels with tails. 2609.22220 is CUDA only; kernel-tv's mutants are 6 hand-written
   operator swaps used to test its validator.
2. **Formal labels.** A Z3 bit-vector oracle gives `oob` and `inequivalent` with witnesses and
   **proves `benign`** at fixed sizes; a floating-point tier separates real-equivalent mutants into
   *proved bitwise identical*, *reorderings* and a **precision-loss** tier with fp64 witnesses and
   Higham error bounds. 2609.22220 labels by kill witness or quarantine only. Volta is real-valued,
   so reorderings and precision loss are all "equal" to it. kernel-tv has no bounds, loops, `tt.dot`
   or error analysis.
3. **An audit with formally labelled, Triton-specific mutants, which isolates the failure class**
   (OOB at tails, `undef` lanes, precision loss, reorderings). It complements Red-Teaming the
   Reward, whose audit is built on agent runs and a hidden verifier, and which already reports
   that three published verifiers accept nearly all invalid submissions. **Do not claim** that
   nobody has audited Triton RL checkers. Targets: KernelGYM (released; run once on a
   Kaggle T4), TritonRL (no code released), DRTriton, TritonGym. 2609.22220 does not mention
   KernelGYM.

**Competitive risk.** kernel-tv is active and Triton-level; once it supports unrolled `scf.for`
and `tt.dot` it covers the `inequivalent` part of our oracle for compiler-produced pairs. Volta
has released code and a completeness result. Tatva's group has since handled reductions by bounded unrolling and published a Triton
result (2 of 26 kernels pass tests but are proved inequivalent); masks and OOB remain
outside its stated scope. arXiv 2610.24548 (Lean and VerCors, CUDA, floating-point values)
and Kernel Contracts (precision and ordering classes) narrow the floating-point claim to
the per-mutant graded split with proofs. The parts least
likely to be scooped are the formal labels themselves: the proved-benign and precision-loss tiers,
`undef` lanes and OOB at tails. The checker audit as such is not new (Red-Teaming the Reward).

**Reading finished on 2026-11-11.** TritonRL, DRTriton, TritonGym and the Volta README are read (rows above).
Two could not be opened: the Tatva paper (OpenReview `cEh56JC9TB`) and the Red-Teaming paper (OpenReview
`JK85pn7BYN`, so which three verifiers it audited is still unknown). Both forums and the OpenReview API redirect
to a bot check, and neither paper has an arXiv version. Someone with an OpenReview login can read them.
Re-checked 2026-11-13: title searches still find no arXiv version of either paper. The NeurIPS listing of the
Red-Teaming paper adds that the verifier accepted all 100 certified solutions and rewarded none of 26 exploit-based
runs; it still does not name the three published verifiers it audited.

**What is still new after this reading.** None of the checkers read here (TritonRL, DRTriton, TritonGym,
KernelGYM, the Red-Teaming verifier as described) goes beyond testing (random or seeded inputs at fixed
shapes, a tolerance or exact match) plus rule, LLM or sensitivity gates; none of the verifiers (Volta, Tatva, kernel-tv, triton-verify) models a
masked lane without `other=` as undefined, checks bounds at tails together with equivalence, or grades
floating-point differences. So the formal labels remain the contribution: `oob` with witnesses at tails,
`undef` lanes, proved-benign mutants (all 12 now proved), and the split of real-equivalent mutants into
bitwise-identical, reordering and precision-loss tiers with fp64 witnesses and error bounds. The audit itself is
not new (Red-Teaming the Reward) and is presented as a complement.

## Sources

- https://neurips.cc/virtual/2026/178980 (Tatva)
- https://arxiv.org/abs/2511.12638 (Volta, full PDF read); https://github.com/willtunnels/volta (README read)
- https://arxiv.org/abs/2511.12294
- https://arxiv.org/abs/2608.12700
- https://arxiv.org/abs/2609.22220 (full PDF read); https://huggingface.co/datasets/Elfsong/KernelBench-M (not opened)
- https://arxiv.org/abs/2606.20128
- https://arxiv.org/abs/2607.16241
- https://github.com/hkust-nlp/KernelGYM
- https://github.com/alinamiretai/triton-verify
- https://github.com/Youngzt998/kernel-tv and https://github.com/jaeHbk/kernel-tv (files listed above, via github.com, raw.githubusercontent.com and api.github.com)
- https://neurips.cc/virtual/2026/173031 (Red-Teaming the Reward); https://www.turing.com/blog/kernelquest-benchmark-ai-agents-triton-kernels (read by the coordinating session)
- https://arxiv.org/abs/2510.17891 (TritonRL; HTML read in part by the coordinating session)
- https://arxiv.org/html/2603.21465 (DRTriton, section 4.1 and appendix F read)
- https://proceedings.mlr.press/v306/guan26d.html and its PDF (TritonGym, read); https://github.com/yil384/TritonGym (README)
