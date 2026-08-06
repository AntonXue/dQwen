# v1-vs-v3 mixture A/B at 0.8B — and an MBPP prompt-format effect

> 2026-08-05, eval box. `EER6b/dQwen3.5-0.8B-Base` (v1) vs `EER6b/dQwen3.5-0.8B-Base-v3`
> (v3, `main` == `step50000-swa`). **Both 50k steps**, so this is budget-matched — the
> confound the v1 grid warns about for the 25k leg does not apply here.
> Decode identical on every arm: **full canvas**, block=32, steps_per_block=32,
> low-confidence, greedy, bs=1, cfg=0; gen=512 code / gen=1024 GSM8K. Dumps in `_runs/battery/`.

## 1. Headline — v3 wins every benchmark, and code went UP while code share went DOWN

| benchmark | v1 | v3 | delta | paired p |
|---|--:|--:|--:|--:|
| MMLU (5-shot, n=14,042) | 26.46 | **30.22** | +3.76 | ≪0.001 |
| HumanEval (n=164) | 29.27 | **33.54** | +4.27 | 0.23 |
| MBPP `[BEGIN]` (n=500) | 19.20 | **23.60** | +4.40 | 0.023 |
| MBPP fence (n=500) | 23.20 | **29.60** | +6.40 | 0.0010 |
| GSM8K strict (8-shot, n=1319) | 6.14 | **7.43** | +1.29 | 0.16 |
| GSM8K flex (8-shot, n=1319) | 6.67 | **7.81** | +1.14 | 0.23 |

**All six deltas are positive.** MMLU and MBPP carry the significance; HumanEval and GSM8K
are directionally consistent but underpowered (GSM8K especially — at 0.8B both models sit
near the floor on multi-step arithmetic, so its 74-vs-57 discordant split is mostly churn).
State this as "v3 dominates across the board, with MMLU and MBPP doing the statistical
work" — not "every benchmark significantly improved".

Code share fell **71% → 50%** between the mixtures and code performance **rose ~4.4pp**.
The v1 mixture was spending ~21 points of token budget on code capability it was not
getting back; the gain came from code *targeting* (12% `code_sft`, verified
`code_concepts` replacing the bare-text variant), not code *volume*.

Code evidence combined across the two code benchmarks: Fisher p=0.010, Stouffer p=0.007,
pooled McNemar p=0.0076. Prefer the stratified ~0.01 — pooling lets MBPP's 500 problems
outvote HumanEval's 164.

**Caveats.** Churn is high (MBPP flips 54 to v3, 32 back to v1; 350/500 fail in both) —
these are two different weak models, not one uniformly better. MMLU 30.22 is ~5pp over the
25% chance floor: un-flooring, not recovery. HumanEval alone is underpowered (p=0.23); it is
MBPP that carries the code claim.

## 2. Harness validation — the v1 arm reproduces July on all four benchmarks

v1 was re-run rather than quoted, because the July jsons record no decode config and the
`gen_length` default changed 512→1024 in between.

| benchmark | July | today | diff |
|---|--:|--:|--:|
| MMLU 5-shot | 26.36 | 26.46 | +0.10 |
| HumanEval | 29.3 | 29.27 | −0.03 |
| MBPP | 19.2 | 19.20 | 0.00 |
| GSM8K strict / flex | 6.14 / 6.67 | 6.14 / 6.67 | **0.00 / 0.00** |

Two are bitwise identical across a two-week gap and separate processes. Note this also
**refutes a cross-process-nondeterminism explanation** for eval drift on these tasks: an
earlier 26.36→24.73 MMLU gap was hypothesised to be the known cuBLAS bf16 GEMM roulette on
the 248k head, but pinning `CUBLAS_WORKSPACE_CONFIG=:4096:8` changed **nothing** (24.73
both ways). The real cause was `--num-fewshot` defaulting to the task default (0), not 5.

## 3. The MBPP prompt-format effect — stock MBPP undersells BOTH models

Stock lm-eval `mbpp` wraps solutions in `[BEGIN]` … `[DONE]`. Scanning the baked corpus
(8k rows × 15 subsets, token-share weighted) that scaffold is **absent from training**:

| marker | v1 mixture | v3 mixture |
|---|--:|--:|
| `[BEGIN]` | **0.000%** | **0.000%** |
| `[DONE]` | 0.002% | 0.001% |
| "You are an expert Python programmer" | 0.000% | 0.000% |
| ` ```python ` | 20.76% | 29.00% |
| ` ``` ` (any) | 42.86% | 39.25% |

So **no scaffold-familiarity/leakage story is available** for any MBPP result here — you
cannot memorise a harness that occurs 0% of the time. (`code_concepts_verified_mbpp` exists
on disk and is **not** in `MIXTURE_WEIGHTS_V3`.)

Swapping the scaffold for a ```` ```python ```` fence (`mbpp_ticks`, one variable changed;
few-shot problems imported from lm-eval's task, not copied) is worth **+4 to +6pp on both**:

| MBPP n=500 | `[BEGIN]` | fence | format effect | McNemar p |
|---|--:|--:|--:|--:|
| v1 | 19.20 | 23.20 | +4.00 | 0.0066 |
| v3 | 23.60 | 29.60 | +6.00 | 0.0002 |

**v1-fenced (23.20) ≈ v3-scaffolded (23.60): the prompt format is worth about as much as the
entire mixture overhaul.** The mechanism is not "fences" — v1 already had 42.9% fence
exposure — but the **task → fenced-solution PATTERN**, 0% → 17% of tokens in v3 via
`code_sft` (99.7% fenced) and `code_concepts_verified_instruct` (100% fenced). `code_cc`'s
fences sit inside web prose instead.

### The comparator matrix settles what the fence effect IS

All five models, same decode, n=500, stock `[BEGIN]` vs `mbpp_ticks` fence:

| model | stock | fence | delta | McNemar p | published |
|---|--:|--:|--:|--:|--:|
| dQwen3.5-0.8B-v1 | 19.20 | 23.20 | **+4.00** | 0.0066 | — |
| dQwen3.5-0.8B-v3 | 23.60 | 29.60 | **+6.00** | 0.0002 | — |
| Dream-Coder-7B-Base | 63.20 | 63.40 | +0.20 | 1.000 | 75.9 |
| LLaDA-8B-Base | 40.80 | 38.40 | −2.40 | 0.088 | 39–40 |
| Dream-v0-7B-Base | 56.00 | 53.00 | −3.00 | 0.072 | 56.2 |

**The fence gain is corpus-specific to dQwen, NOT a property of masked-diffusion decoding.**
Were it a decode artifact (cleaner stop string, less trailing garbage graded), every family
would gain. Instead the sign tracks code-instruction exposure monotonically: dQwen (17% of
tokens in task→fenced-solution shape) gains; **Dream-Coder, the code specialist, is exactly
indifferent**; the two general-purpose models degrade. Dream-Coder's zero is what makes the
mechanism credible rather than a just-so story.

⚠ **Therefore a fenced MBPP column is NOT a fair common protocol** — it is worth +6 to us and
−3 to them, a ~9pp swing that has nothing to do with coding ability. **Stock `[BEGIN]` stays
the headline for every cross-model table**; fenced numbers belong in a labelled
format-sensitivity section as a finding about DATA, not a benchmark result.

Residual caveat that cannot be closed: `[BEGIN]` is 0.000% of *our* corpus, but LLaDA's and
Dream's corpora are not public at row level, so part of the split could be *their* scaffold
familiarity rather than purely our fence familiarity. Hedge accordingly.

### Two comparator facts worth correcting elsewhere

1. **The "our protocol undersells the Dream family" caveat is too broad.** Dream-v0 reproduces
   its published MBPP to **0.2pp** (56.00 vs 56.2) and LLaDA to +0.8 (40.80 vs 39–40). Only
   **Dream-Coder** is off (63.20 vs 75.9, −12.7). That is one model, not a family or a harness
   property.
2. **LLaDA's MBPP 41.2 in the v1 reference grid is an APPEND-mode number.** Its source file is
   `_runs/mbpp_llada_blockdiff.json` and no `mbpp_llada_full.json` exists, yet the grid header
   describes itself as full-canvas. The full-canvas value measured here is **40.80**. (For
   reference, append-vs-full on LLaDA is within noise and opposite-signed across benchmarks:
   HumanEval full won by +0.61, MBPP full loses by 0.4 — unlike dQwen, where full won by +2.43.)

## 4. What to do next
- **Run this battery on 4B-v3 / 9B-v3.** 4B is where code pass@1 is not floored and MMLU
  already retains (v1 41.8), so a mixture win shows up most legibly — and it would inform
  the 50k forks still training.
- If the 4B/9B results agree, the campaign's assumed "sacrifice a little code for large
  general gains" trade **may not need to be made at all**.
