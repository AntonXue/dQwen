# regrades — CPU regrades of saved generations (no GPU involved)

Produced by `evalplus_driver.py --from <file>`: same completions,
denser tests. Names: `<launch-stamp>__<model>@<revision>__<benchmark>-plus__from-<source>.jsonl`
with EvalPlus's `_eval_results.json` alongside. The `from-` suffix names the
generation store the completions came out of (`battery`/`battery4b` = the 2026-08-05 battery launch dirs holding the family
v3@50k samples; `grid_v1-ar` = AR counterpart cells).

## Launch 20260812-091609 — HumanEval+ over the full Table-4 HE column

Consistency gate PASSED: regrade base pass@1 matched lm-eval on 7/10 cells
exactly, ±1 problem (0.61pp) on the other three (0.8B family, 4B AR,
Qwen3-1.7B AR) — grader exec-env noise.

| model | base pass@1 (lm-eval) | plus pass@1 |
|---|---|---|
| dQwen3.5-0.8B-Base-v3 | 32.93 (33.54) | 30.49 |
| dQwen3.5-2B-Base-v3@50k | 42.68 (42.68) | 37.20 |
| dQwen3.5-4B-Base-v3@50k | 59.76 (59.76) | 56.10 |
| dQwen3.5-9B-Base-v3@50k | 65.85 (65.85) | 60.98 |
| Qwen3.5-0.8B | 22.56 (22.56) | 21.34 |
| Qwen3.5-2B | 39.02 (39.02) | 34.15 |
| Qwen3.5-4B | 57.93 (57.32) | 51.22 |
| Qwen3.5-9B | 70.73 (70.73) | 63.41 |
| Qwen3-1.7B | 39.63 (39.02) | 35.98 |
| Qwen3-0.6B | 19.51 (19.51) | 17.07 |

SEMANTICS CORRECTION (2026-08-12, later the same day): plus pass@1 here
is OFFICIAL EvalPlus semantics, base AND plus — the four Qwen3.5 AR rows
were first published from plus_status alone and ran 0.61–1.22pp high;
this table is corrected. `summarize.py` computes base∧plus.

Note: the family rows regrade PROBE-mode generations; if the HE column gets
its GATE restamp, HE+ regrades rerun from the restamped cells (~25s each).
The 0.8B source (the first battery) loaded repo `main`, which was the step-50000
final at grading time.
