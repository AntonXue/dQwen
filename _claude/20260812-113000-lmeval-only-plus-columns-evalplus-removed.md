# lm-eval-only ruling: the "+" columns are grid cells; EvalPlus removed

> 2026-08-12, Anton: "I'm willing to rerun *everything* under lm-eval if it
> lets us simplify the codebase." Evidence gathered first, then the cut.

## The evidence

**Grader A/B** — 10 models × 164 saved HumanEval completions, identical
strings graded by EvalPlus (gt-calibrated timeouts, own sandbox) and by
lm-eval's `code_eval` path (fixed 3s timeout) against the same plus suite:

- **9 verdict flips in 1,640 gradings (0.55%).** All six dQwen-family cells
  agree EXACTLY; worst aggregate delta 1.22pp (Qwen3.5-2B AR).
- One systematic case: HumanEval/92 passes under EvalPlus and fails under
  code_eval on all four Qwen3.5 AR models — a per-problem
  timeout/tolerance semantics difference, not noise.

**Prompts** — HumanEval/HE+ prompts are byte-identical across frameworks
(both render the raw `prompt` field). MBPP+ prompts are NOT standardized:
EvalPlus codegen uses a 0-shot docstring format; lm-eval uses the
[BEGIN]/[DONE] scaffold, 3-shot — i.e. OUR house MBPP protocol.

**Stock-task trap** — lm-eval's `mbpp_plus` never executes the plus suite:
its `doc_to_target` renders `test_list[0..2]` (the three ORIGINAL asserts)
while the ~9.5KB plus suite sits unused in the dataset's `test` field.
Stock `mbpp_plus` is MBPP+ in name only. `humaneval_plus` is sound.

## The ruling

One framework end-to-end. `humaneval-plus` and `mbpp-plus` are ordinary
BENCH rows (grid cells: striping, sentinel, provenance, SLURM for free):

- `humaneval-plus` → stock `humaneval_plus` (same prompts as humaneval;
  regenerating 164 docs is cheaper than keeping a second framework).
- `mbpp-plus` → local `dqeval/tasks/mbpp_plus_full.yaml`: sanitized
  problems + REAL plus suite + our scaffold. Smoked end-to-end via run.py.

Deleted: `evalplus_driver.py` (born 2026-08-11, SLURM-hardened and killed
the same day — the store/resume work it grew is superseded by the grid,
which had all of it already), the `_runs/evalplus/` store, the `evalplus`
pin in requirements-eval.txt.

## Consequences for numbers

- `_runs/regrades/` is FROZEN: the 2026-08-12 EvalPlus-graded HE+ record.
  The manuscript's current HEval+ cells come from it and stand as printed;
  the grid restamp regrades under code_eval — expect 0 movement on family
  cells, ≤1.22pp on AR cells (measured above).
- MBPP+ column = house scaffold + plus suite: a re-measurement, NOT
  comparable to published EvalPlus MBPP+ (Dream-Coder 61.6, CoDA) —
  tripwire-style disclosure required wherever that column is narrated.
- HE+ under lm-eval keeps published comparability (same prompts; grader
  delta bounded by the A/B).

## UPDATE: flip decomposition (Anton asked "purely timeouts?") — NO

Reran all 9 flips with 60s limits on BOTH sides:

- **2 were timeouts**: HumanEval/139 (lm's flat 3s too tight; passes @60s)
  and HumanEval/100 (EvalPlus's gt-calibrated limit too tight; passes
  @60s/100x).
- **7 are test-SUITE skew, not grading mechanics**, concentrated on 4
  problems. The two distribution channels ship subtly different suites:
  case counts differ (HumanEval/9: package 963 vs HF 962; HumanEval/96:
  190 vs 180), and on HumanEval/92 the channels disagree about the same
  input: `any_int(1.5, 5, 3.5)` — our AR solutions return True, the HF
  suite expects False and fails them (correctly: the spec requires all
  ints), while the evalplus package PASSES them. On /92 the lm-eval
  channel is the stricter-and-right one.

So: not slow code, and not a grader bug on our side — upstream suite
version skew between `evalplus==0.3.1`'s internal data and the
`evalplus/humanevalplus` HF dataset. Protocol = the HF channel (what
lm-eval loads), pinnable by dataset revision. Bound unchanged: <=1.22pp,
family cells unaffected.

## UPDATE 2: the flip hunt found a REAL BUG — four manuscript cells corrected

Chasing Anton's timeout question to ground exposed that the A/B (and the
morning's Table-4 fill) read EvalPlus's per-task `plus_status` ALONE.
Official EvalPlus "+" = **base AND plus**. `plus_status` overcounts
solutions that fail a base input but pass the 1000 extra tests —
HumanEval/92 exactly: `any_int(1.5, 5, 3.5)` is a BASE input, our AR
solutions fail it under both graders, but plus_status said "pass".

Corrections (manuscript 9c655e8): Qwen3.5 AR HEval+ 21.95→21.34,
34.76→34.15, 52.44→51.22, 64.02→63.41. Family + Qwen3 cells verified
unaffected. `summarize.py` and the regrades README now compute base∧plus.

Consequently the A/B numbers above OVERSTATE disagreement: under official
semantics the graders disagree on **4/1640 (0.24%)** — HE/139 + HE/100
are pure timeouts (each side's limit too tight once), HE/62 + HE/96 are
genuine upstream suite skew (e.g. /96: 190 package cases vs 180 HF).
The /92 and /9 "flips" and their suite-skew attribution in UPDATE 1 were
artifacts of the semantics bug. The lm-eval-only ruling only strengthens:
correct-semantics agreement is 99.76%.

## UPDATE 3: /62 and /96 are NOT different test sets — the HF comparator is unsound on empty-expected

Case-by-case diff of both channels:

- **/62**: inputs AND expected outputs are IDENTICAL in both channels
  (953 cases). Our solution returns `[0]` where both expect `[]`.
- **/96**: package has 10 extra cases (190 vs 180), but the flipping case
  exists in BOTH channels with the same expected `[]`.

The real cause of both flips is a bug in the HF-rendered script's inline
comparator: `is_floats([])` is vacuously True (`all()` on empty), which
routes empty-expected comparisons into `np.allclose`, and numpy
broadcasts shape (1,) against (0,) to an empty result — so
`np.allclose([0], []) == True`. Any WRONG non-empty output vacuously
PASSES when the expected value is an empty list/tuple. The evalplus
package's internal comparator handles this correctly (fails both).

So the reconciled A/B floor is: 2 timeouts (grader config) + 2 cases of
the HF script over-crediting genuinely wrong solutions via the
empty-expected bug. UPDATE 2's "suite skew" attribution for /62–/96 was
wrong. Not yet ruled: whether to patch the comparator via process_docs
(a string fix on doc["test"] in a local task variant) or accept +
disclose the over-credit (it only fires on wrong-nonempty vs
expected-empty).
