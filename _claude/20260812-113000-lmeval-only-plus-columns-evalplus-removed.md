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
