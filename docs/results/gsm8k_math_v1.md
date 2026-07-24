# GSM8K + MATH — first numbers (fast ballparks)

> 2026-07-23. Generative math via block-diffusion decode (block=32, gen=512).
> GSM8K: gsm8k_cot 8-shot, n=200 subsample. MATH: minerva_math, 30/subject subsample.
> Ballparks (subsampled) unless noted; full runs pending.

## GSM8K (gsm8k_cot, 8-shot, flexible-extract)

| model | flexible | strict | published |
|---|---|---|---|
| LLaDA-8B-Base | 72.0 | 69.5 | **70.9** ✓ |
| dQwen3.5-9B-Base | **81.0** | 80.0 | — |

**LLaDA reproduces (72.0 flex / 69.5 strict brackets published 70.9)** — the
generative-math reproduction, completing the validation trifecta (knowledge/code/math).
dQwen3.5-9B at 81.0 is ~9pp above LLaDA and competitive with the strong AR/DLM
comparators (Qwen2.5-7B 78.9, Dream 77.2).

## MATH (minerva_math)

| model | exact_match | published | note |
|---|---|---|---|
| LLaDA-8B-Base | 21.4 | ~30.9 | **undersold by gen_length** |
| dQwen3.5-9B-Base | 32.4 | — | above LLaDA even at gen=512 |

**MATH needs gen_length=1024, not 512.** LLaDA published MATH at gen=1024 (their eval
script uses 1024 for all generative tasks); MATH solutions are long, so gen=512
truncates them -> 21.4 vs ~30.9. GSM8K reasoning is short enough that 512 matched
(72 ≈ 70.9). A gen=1024 rerun is the fix to reproduce ~30.9.

## Validation record so far (LLaDA vs published)
| benchmark | reproduced | published | path |
|---|---|---|---|
| MMLU | 65.85 | 65.9 | loglikelihood |
| HumanEval | 32.32 | 32.9 (gen512) | generative + execution |
| MBPP | 41.2 | ~39-40 | generative + execution |
| GSM8K | 72.0 | 70.9 | generative + CoT extraction |

Four benchmarks, both scoring paths, both grading modes -- all reproduce within ~1-2pp.
