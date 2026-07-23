# MMLU v1 — first validated numbers through the dqeval harness

> 2026-07-23. All runs: MMLU 57-subject, **5-shot, mc_num=1 (exact single-token
> MASK-slot marginal), cfg=0, bs=1**, transformers 5.13. Comparators loaded via the
> ported adapters + compat shims. Full per-subject dumps in gitignored `_runs/`.

## Headline: the loglikelihood pipeline reproduces published DLM numbers

| model | MMLU | published | diff | role |
|---|---|---|---|---|
| **LLaDA-8B-Base** | **65.85** | 65.9 | **−0.05** | comparator (ported) |
| **Dream-v0-Base-7B** | **71.64** | ~70 | +1.6 | comparator (ported) |
| Qwen3.5-0.8B-Base (AR) | 48.36 | — (sensible) | — | AR-twin gate (stock lm-eval) |
| dQwen3.5-2B-Base | 27.80 | — | — | ours (diffusion) |
| dQwen3.5-0.8B-Base | 26.36 | — | — | ours (diffusion) |

## What this establishes

1. **LLaDA reproduces to 0.05pp.** LLaDA runs MMLU at cfg=0, which our common
   protocol matches exactly, so there is no decode difference to hide behind — and
   we land on the published number. This is the sharpest possible evidence the
   loglikelihood path (prompt + few-shot + MC-NELBO scoring + aggregation) is correct.
2. **Dream reproduces within ~2pp AND validates the shift.** Dream is the one family
   whose adapter applies the AR→position logit shift (`cat([l[:,:1], l[:,:-1]])`). A
   wrong shift would collapse it toward chance; 71.6 with correct subject structure
   proves `_canonicalize` end-to-end on a real benchmark.
3. **The AR-twin gate passes.** Stock Qwen3.5-0.8B-Base through lm-eval's native
   `hf` model (zero dqeval code) gives 48.4 with a sensible subject profile —
   proving the task/scoring machinery independent of our DLM wrapper. Two
   independent anchors (a DLM comparator + a stock AR model) both land.
4. **Our models measure at chance, correctly.** dQwen3.5-{0.8B,2B} sit at 26.4/27.8,
   FLAT across all four subgroups (0.8B: 25.8–27.1). The flatness is the tell: a
   genuinely at-chance model shows no subject discrimination, so the eval is reporting
   real behaviour, not a scoring artifact. This matches the documented forgetting of
   the small members at their pretraining LR.

## The twin gap (same Qwen3.5-0.8B substrate)

| form | MMLU |
|---|---|
| Qwen3.5-0.8B-Base (autoregressive, original) | 48.4 |
| dQwen3.5-0.8B-Base (diffusion-converted) | 26.4 (chance) |

~22pp of MMLU knowledge erased by the conversion + pretraining at this size, down to
chance — the retention story quantified on one clean pair rather than inferred.

## Protocol notes (pin these in any table)

- **cfg=0 (common disclosed protocol).** LLaDA tunes classifier-free guidance
  per-task (0.5–2.0 on 5 multiple-choice tasks); we drop it. MMLU is unaffected —
  LLaDA runs MMLU at cfg=0 too — which is why the LLaDA MMLU match is near-exact.
- **mc_num=1 exact.** MMLU continuations are single-token letters (" A".." D"),
  verified single-token in all four tokenizers, so the estimator is exact, not a
  Monte-Carlo bound.
- **Fast path.** One forward per question (shared masked canvas across the 4
  options), verified bitwise identical to per-option scoring. Full MMLU ~30 min/model
  at bs=1.
