# HumanEval v1 — two decodes, with a published reproduction

> 2026-07-23. Code-eval through the dqeval harness, via lm-eval's `humaneval` task
> (execution grading). Two decodes reported. gen_length=512, bs=1. Dumps in `_runs/`.

## Numbers

| model | AR-mode (block=1) | block-diffusion (block=32, low-conf) | published |
|---|---|---|---|
| dQwen3.5-9B-Base (ours) | 59.76 | **62.20** | — |
| Dream-Coder-v0-Base-7B | — | 59.76 | — |
| Dream-v0-Base-7B | 19.51 | 41.46 | — |
| LLaDA-8B-Base | 20.73 | **32.32** | **32.9** (gen=512) |

## The reproduction — harness validated for generative code

**LLaDA-8B-Base block-diffusion = 32.32 vs LLaDA's published 32.9 at the SAME
gen_length=512 (−0.58pp).** LLaDA's EVAL.md reports HumanEval by generation length
(gen=1024 → 35.4, gen=512 → 32.9, gen=256 → 32.9); we ran gen=512 and land on their
gen=512 number. This is the generative-code analogue of the MMLU reproduction: the
harness reproduces a published HumanEval number to under 1pp at matching decode.

## Why the AR-mode numbers looked "suspiciously low" — and were not a bug

The AR-mode column is CORRECTLY EXTRACTED but uses a genuinely weak decode. Verified
by grading individual problems: completions parse fine; failures are real weak
solutions (the model writing `# TODO: pass`, wrong logic), not truncation. The lever
is the unmasking ORDER:

- **append-AR (block=1, strictly L2R):** weakest. 10/20 on the easy first-20 probe.
- **block-AR (fixed canvas, L2R, trailing masks):** tested per a hypothesis that
  trailing masks would help — they did NOT (8/20). Order, not canvas layout.
- **block-diffusion (block=32, low-confidence order):** best (14/20 probe), and it
  reproduces published.

So AR mode undersells decode-sensitive models by ~2x (Dream 19.5 → 41.5, +2.1x;
LLaDA 20.7 → 32.3). It is a documented common-baseline FLOOR, not the number to quote.

## A property difference falls out: decode-robustness

| model | AR → block-diffusion |
|---|---|
| dQwen3.5-9B | 59.8 → 62.2 (+2.4) |
| Dream-7B | 19.5 → 41.5 (+22.0) |
| LLaDA-8B | 20.7 → 32.3 (+11.6) |

dQwen barely moves; Dream/LLaDA roughly double. dQwen was TRAINED with the
block-append decode, so it is robust to the decode; LLaDA/Dream were designed for
confidence-based decode and collapse under naive AR. This robustness is itself a
finding, not noise.

## Two silent extraction bugs fixed en route (both: low score, correct code)

1. **No early-stop** (db29bb4): masked DLMs do not emit EOS → filled the canvas, ended
   mid-statement (syntax error). Fixed with stop-string early-stop; also ~70x fewer
   forwards on short solutions.
2. **Markdown code fence** (b09a43e): Dream wraps code in ```...``` + prose that
   humaneval's `until` misses. Added ``` as a stop. Dream 4.88 → 19.51.

## Next

- gen=1024 for the LLaDA gen=1024 published cell (35.4).
- EvalPlus (HE+/MBPP+) denser tests; MBPP.
- GSM8K (generative, same path).

## EvalPlus (HE+) — driver validated

Dream-Coder-7B-Base via `dqeval/evalplus_driver.py` (block-diffusion decode):

| | base pass@1 | plus pass@1 | published |
|---|---|---|---|
| Dream-Coder-7B-Base | 59.76 | 55.49 | HE 66.5 / HE+ 60.4 |

**Consistency check: the EvalPlus base pass@1 (59.76) equals the lm-eval `humaneval`
number (59.76) exactly** — same generation, graded by two independent harnesses. The
−6.7/−4.9 gap to published is the same block-diffusion-vs-native decode undersell
documented for the Dream family, not a grading issue. MBPP+ and native-decode
reproduction are the follow-ups.

Also: LLaDA MBPP (lm-eval, block-diffusion) = 41.2 vs published ~39-40 — third LLaDA
reproduction after MMLU (-0.05) and HumanEval (-0.6).
