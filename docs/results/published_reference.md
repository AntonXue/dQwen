# Published comparator numbers — reference targets

> 2026-07-23. Base-model published benchmarks for the comparator families, pulled
> from primary sources (papers via pdftotext, EVAL.md, model cards). These are the
> reproduction targets; our numbers live in `mmlu_v1.md` / `humaneval_v1.md`.

## LLaDA-8B-Base

| benchmark | published | source |
|---|---|---|
| MMLU (5-shot) | 65.9 | LLaDA paper; Dream paper col-2 |
| GSM8K (8-shot) | 70.9 | Dream paper col-2 |
| HumanEval (gen=1024) | 35.4 | LLaDA EVAL.md |
| HumanEval (gen=512) | **32.9** | LLaDA EVAL.md; Dream paper (0-shot) |
| MBPP | 39.0–40.0 | LLaDA EVAL.md / Dream paper |

## Dream-v0-Base-7B (arXiv:2508.15487, Table — Dream is col 1)

| benchmark | published | our reproduction | note |
|---|---|---|---|
| MMLU (5-shot) | 69.5 | 71.64 | +2.1 ✓ |
| GSM8K (8-shot) | 77.2 | (running) | — |
| HumanEval (0-shot) | 57.9 | 41.46 (block-diff) | native decode needed |
| MBPP (4-shot) | 56.2 | — | — |

## Dream-Coder-v0-Base-7B (arXiv:2509.01142, Table 1)

| benchmark | published | our reproduction | note |
|---|---|---|---|
| HumanEval | 66.5 | 59.76 (block-diff) | native decode needed |
| HumanEval+ (EvalPlus) | 60.4 | — | — |
| MBPP | 75.9 | — | — |
| MBPP+ | 61.6 | — | — |
| BigCodeBench (Full) | 38.5 | — | — |

Dream-Coder-7B-**Instruct** (Table 4): HumanEval 82.9, MBPP 79.6, EvalPlus 73.1.

## Qwen3 base (arXiv:2505.09388, Table 8)

| size | MMLU | MMLU-Redux | MMLU-Pro | our AR reproduction |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 52.81 | 51.26 | 24.74 | 52.51 (−0.30) ✓ |
| Qwen3-1.7B-Base | 62.63 | 61.66 | 36.76 | 62.57 (−0.06) ✓ |

Qwen3.5 publishes NO plain base MMLU (only MMLU-Pro 82.5 / Redux 91.1 for 9B instruct).

## The decode caveat — why generic block-diffusion undersells Dream but not LLaDA

Our common-protocol decode (block=32, low-confidence, gen=512) reproduces LLaDA's
HumanEval almost exactly (32.32 vs 32.9) because LLaDA's published HE uses a
comparable full-canvas low-confidence decode. It UNDERSELLS Dream (41.5 vs 57.9) and
Dream-Coder (59.8 vs 66.5) because their published numbers use their OWN
`diffusion_generate` sampler with different/longer settings. This is the native-vs-
unified split by design:

- **unified** = fair common protocol across families (undersells decode-tuned models);
- **native** = each model's own sampler = reproduces published.

We have Dream's native sampler (`families/dream/sampler_native.py`); pointing it at
HumanEval with Dream's config is the next step to reproduce 57.9 / 66.5.
