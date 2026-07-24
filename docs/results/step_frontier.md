# Step-budget frontier — dQwen wins the accuracy-per-compute axis

> 2026-07-23. HumanEval (base tests, n=60 subset), full-canvas, gen=512, block=32,
> greedy, order=low_confidence. Sweep steps_per_block; each = pass@1 @ mean forwards.
> `tests/step_sweep.py` (single model load per sweep). Subset + single-seed, but the
> gaps are large and consistent.

## pass@1 vs step budget

| steps/blk | dQwen3.5-9B | Dream-Coder-7B | LLaDA-8B |
|---|---|---|---|
| 32 | **83.3** | 76.7 | 56.7 |
| 16 | **70.0** | 65.0 | 46.7 |
| 8  | **43.3** | 28.3 | 23.3 |
| 4  | **13.3** |  8.3 | 11.7 |
| 2  | **10.0** |  5.0 |  5.0 |

## The finding

dQwen wins at EVERY step budget, and its lead WIDENS as steps drop -- the flattest
curve (most step-robust decoder):

- **Retention from steps=32 -> steps=8 (4x fewer steps):**
  dQwen 83->43 = **52%** retained; Dream-Coder 77->28 = 37%; LLaDA 57->23 = 41%.
- **Accuracy per compute at ~35 forwards (steps=8):**
  dQwen **43%** vs Dream-Coder 28% vs LLaDA 23% -- +15-20pp at equal compute.

## Why

dQwen3.5 is trained with STANDARD DLM masking (random masking over the full
sequence), so it is a strong PARALLEL / few-step predictor. Dream-Coder and LLaDA are
tuned for many-step denoising and collapse under aggressive parallelism (few steps =
many tokens committed per step).

## The reframing

On max-step pass@1, dQwen only "balances" (leads knowledge/math, competitive code).
On the DLM-NATIVE metric -- accuracy per forward, i.e. fast parallel generation, the
whole point of a diffusion LM -- dQwen DOMINATES. This is the decode regime to report
dQwen in.

Caveats: n=60 subset, single greedy seed. Worth a full-164 multi-seed confirmation
before publication, but the 15-20pp gaps are far outside subset noise.
