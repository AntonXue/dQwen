# summarize.py silently blanked the entire HumanEval+ column — 22 cells, fixed

> 2026-08-14, workstation, found while harvesting the campaign into the
> manuscript's benchmark grid. Not a data problem: every HEval+ cell ran
> correctly and holds a valid score. The aggregation layer could not read
> them, so the column rendered as `?` — which looks like missing runs.

## The bug

`headline()` resolved the task name by EXACT lookup:

    r = results.get(HEAD_TASK.get(bench, bench))
    if not r: return "?", None

`HEAD_TASK["humaneval-plus"]` is `"humaneval_plus"`, but the cells carry
lm-eval's variant name **`humaneval_plus_sound`**. Miss → `"?"`.

**Why it is worth a doc.** The failure is silent and it fails toward
"missing", not toward "wrong". A harvester reading the grid sees an empty
column and reasonably concludes those cells were never run — the same
failure shape as the CoDA row earlier the same night, one layer up. Nothing
raised, nothing logged.

Scope: **22 cells** — every model's HumanEval+ at block32-static-s32 plus the
AR band. Confirmed against the vista store.

## The fix

Exact lookup first; on a miss, accept a **unique prefix match**, else keep
`"?"`:

    cand = [k for k in results if k.startswith(want)]
    if len(cand) == 1:
        r = results[cand[0]]

Unique is the load-bearing word: one candidate is unambiguous, and anything
else still reports `"?"` so a genuine problem is not papered over. This also
absorbs any future `<task>_<variant>` naming without another edit.

## Verification (before/after over the whole vista store)

    old cells reading '?' : 22
    new cells reading '?' : 0
    changed lines         : 22  (all of them '? ?' -> a value)
    previously-readable cells with a changed value: NONE

The recovered values were cross-checked against the raw jsonl read directly
(`pass@1,create_test`, scaled from fraction to percent) and agree exactly:
e.g. dqwen3.5-9b@50k 59.15, control@50k 37.80, AR Qwen3.5-9B 62.80.

## Related gotchas for anyone writing a harvest script

- **MMLU on DLMs lives under decode `mc-nelbo`**, not the headline
  `block32-static-s32`. Keying only on the block decode yields an empty MMLU
  column for every diffusion row.
- **Metric names are not uniform**: `humaneval` reports `pass@1`, `mbpp`
  reports `pass_at_1`. A preference list must contain both.
- **math500 carries both `exact_match` and `math_verify`**, and they disagree
  in OPPOSITE directions for AR and DLM rows (strict extraction breaks on big
  post-trained AR models; on the DLM rows strict reads higher). The
  2026-08-12 `math_verify` ruling was made for the AR column — applying it
  blindly to DLM rows understates them. Open question for the manuscript.
