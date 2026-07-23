# HumanEval v1 — greedy AR-mode ballpark

> 2026-07-23. First code-eval numbers through the dqeval harness. **Greedy AR mode**
> (`block_length=1`, sequential, temperature=0), gen_length=512, bs=1, via lm-eval's
> `humaneval` task (built-in execution grading). Full dumps in gitignored `_runs/`.

## Numbers

| model | HumanEval pass@1 |
|---|---|
| **dQwen3.5-9B-Base (ours)** | **59.76** |
| dQwen3.5-2B-Base (ours) | 28.05 |
| LLaDA-8B-Base | 20.73 |
| Dream-v0-Base-7B | 19.51 |

## What these are — and are NOT

- **Greedy AR mode is a COMMON BASELINE decode, not each model's native decode.** Every
  family can decode strictly L2R one token at a time (`block_length=1`), so this is the
  code analogue of the cfg=0 common protocol we used for MMLU. It is a floor, not a
  ceiling: the native block-diffusion decode should do better per model.
- **Therefore these do NOT reproduce published HumanEval.** LLaDA's published ~35 was
  with its own block-diffusion decode (gen_length=1024); our AR-mode LLaDA is 20.7, and
  the gap is the decode, not a harness error. The block-diffusion reproduction is a
  separate run (next).
- **pass@1, greedy, single run.** A ballpark, not a locked number.

## The headline

**dQwen3.5-9B-Base reaches 59.76% under the naive AR-mode decode** — the strongest in
this set, and this is a code-focused diffusion-converted model showing real
execution-based capability before any decode tuning.

Contrast with MMLU makes the specialization visible: dQwen3.5-2B is at CHANCE on MMLU
(27.8, general knowledge forgotten) but a functional 28.0 on HumanEval. The
code-focused training preserved code ability across sizes while sacrificing general
knowledge at the small end.

## Two extraction bugs fixed to get here (both silent)

Both produced the *same* symptom — a low score with visibly CORRECT generated code —
from different causes:

1. **No early-stop** (commit db29bb4). A masked DLM does not emit EOS, so in append
   mode it filled the whole canvas and ended mid-statement (unterminated string →
   syntax error → fail). Fixed with stop-string early-stopping during decode; also a
   large speedup (Dream `def add`: 512 forwards → 7).
2. **Markdown code fence** (commit b09a43e). Base models with instruct-flavoured
   pretraining (Dream) wrap correct code in ```...``` + prose, which HumanEval's
   `until` strings do not catch → unparseable graded code. Fixed by adding ``` as a
   stop. Impact: Dream 4.88 → 19.51, dQwen3.5-2B 15.85 → 28.05. LLaDA unaffected
   (it emits <|endoftext|>, never fences) — an internal consistency check.

The lesson: eyeball generations, don't trust the score. Both bugs looked like "the
model is bad" and were actually "our extraction is wrong."

## Next

- Block-diffusion (champion) decode per model — the native-decode numbers; expected
  higher than AR mode. LLaDA there should approach its published ~35.
- EvalPlus (HumanEval+/MBPP+) for the denser-test headline.
- MBPP through the same path.
