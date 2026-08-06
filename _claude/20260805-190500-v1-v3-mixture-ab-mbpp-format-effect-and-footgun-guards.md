# v1-vs-v3 mixture A/B, an MBPP format effect, and the foot-guns that caused/hid both

> 2026-08-05, eval box. Numbers live in `docs/results/mixture_v1_vs_v3_08b.md`; this file is
> the progression — what was run, what was refuted, and what changed in the harness as a result.

## What was asked
Run the 0.8B v3 endpoint (`EER6b/dQwen3.5-0.8B-Base-v3`, 50k, just published) through the
battery and compare against released v1. v1 was **re-run rather than quoted**: the July jsons
record no decode config, and `gen_length`'s default moved 512→1024 in between, so quoting them
would have been an uncontrolled comparison.

## Result
v3 wins every benchmark; **code rose ~4.4pp while code share fell 71%→50%**. See the results
doc. The re-run v1 arm reproduced July on all four benchmarks (two bitwise), which is what makes
the deltas trustworthy.

## Three priors refuted, in order

**1. "The MMLU gap vs July is cross-process nondeterminism."** WRONG. The known cuBLAS bf16
GEMM roulette on the 248k head was the obvious suspect for 26.36→24.73, and the hypothesis was
even shaped correctly (MMLU letter-loglik on a near-chance model is 14k four-way near-ties;
greedy code decode is immune). Pinning `CUBLAS_WORKSPACE_CONFIG=:4096:8` changed **nothing** —
24.73 pinned and unpinned, 29.84 pinned and unpinned. **These evals are exactly reproducible
across processes.** Do not re-propose nondeterminism as an explanation for eval drift here.

**2. "The real cause was model drift on the Hub."** Also wrong — `main` and `step50000-swa`
sha-match (`3d76d47e58fc12b4`) and the last weight commit predates the July run.

**3. The actual cause: `--num-fewshot` defaulting to the TASK default (0), not 5.** Diffing
per-subject exposed it (`CONFIG DIFF num_fewshot: july=5 today=0`; 38 subjects down, 20 up —
a protocol signature, not noise). Cost a full round of MMLU runs.

**4. "v3's MBPP gain could be `code_sft` scaffold familiarity."** Refuted by measurement, not
argument: `[BEGIN]` is **0.000%** of both corpora. Also refuted independently by v3's own
numbers — if it had learned the harness, `[BEGIN]` would be its best format; it is its worst
by 6pp.

**5. "v1 is slow on the fence arm because it never closes the fence and burns the canvas."**
Wrong — generation lengths are near-identical (v1 mean 151 chars, v3 140; both 1.6% over 1000).
The 37-vs-7-minute gap was GPU contention. The fence gain is a real format effect, not a
stopping artifact.

## The MBPP format finding (the durable one)
`[BEGIN]`→```` ```python ```` is worth **+4.00pp (v1) / +6.00pp (v3)**, both significant.
v1-fenced ≈ v3-scaffolded: **format ≈ the whole mixture overhaul**. Mechanism is the
*task→fenced-solution pattern* (0%→17% of v3 tokens), not fences per se — v1 already had 42.9%
fence exposure from `code_cc` web prose. New task: `dqeval/tasks/mbpp_ticks.yaml`, one variable
changed vs stock, few-shot problems imported from lm-eval rather than copied so they cannot drift.

## Harness changes (this is the part future-you cares about)
1. **`--mode append` now requires `--allow-append`** (mirrors the existing `--allow-code`
   idiom) and fails in argparse *before the model loads*. Full-canvas beat append head-to-head;
   append is retained only because SDAR's architecture has no full-canvas variant. Second,
   softer guard in `harness.py` for callers that build `model_args` directly.
2. **`--num-fewshot` warning** naming the published shot count per task (mmlu 5, gsm8k_cot 8,
   mbpp/mbpp_ticks 3, humaneval 0) — the exact slip above.
3. **Provenance in every result json**: `decode` block (mode/gen_length/block/steps/order/
   temperature/mc_num) + the literal `model_args`. The ambiguity that forced the v1 re-run
   cannot recur.
4. **Config banner as the first line of every run's log**, so a wrong protocol is visible
   immediately instead of reconstructed afterwards.
5. **`.gitignore` bug fixed**: line 1 was `.claude/          # comment`. Git only honours `#`
   at the start of a line, so the pattern was literal and matched nothing — the agent worktree
   (a second full checkout plus `_runs/`, GBs of dumps) was untracked-but-visible and one
   `git add -A` from being committed.
6. Registry: added `dqwen3.5-{0.8b,4b,9b}-base-v3`. Flagged (did not silently re-point) three
   **dead** entries — `dqwen3.5-0.8b-base-v1/-v2`, `dqwen3.5-4b-base-v2` — whose repos 404
   since the v2* consolidation; kept so the `ov_*` runs in `_runs/` stay traceable.

## Open
- **GSM8K v3** still running at time of writing (v1 = 6.67 flex; 0.8B is near the floor here,
  so expect it to be the least informative number).
- **Comparator format matrix in flight**: LLaDA-8B-Base, Dream-v0-Base-7B, Dream-Coder-7B-Base
  × {stock, fence}. Needed before any fenced number can be quoted next to published values.
- **Run the battery on 4B-v3 / 9B-v3** — 4B is the size where code is not floored and MMLU
  already retains, so it is the legible test, and it would inform the 50k forks still training.
