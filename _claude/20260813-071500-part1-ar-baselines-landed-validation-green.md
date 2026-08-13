# Part 1 (AR twins) LANDED on cluster — validation GREEN, tiled path proven

> 2026-08-13 morning. Job 908632 (-N 9, tiled): 72/72 cells, 3h09m wall
> vs 3.2h estimated worst lane, ZERO cell failures. Parts 2+3 (12 jobs)
> were queued behind it and are running as this is written.

## The GATE AR table (bare repos, canvas 1024 via _ARLM, cluster GH200)

| model | mmlu | gsm8k | math500 | he | he+ | mbpp | mbpp+ | fence | fence+ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Qwen2.5-7B | 74.13 | 79.61 | 40.80 | 55.49 | 48.78 | 64.40 | 69.84 | 62.40 | 70.11 |
| Qwen2.5-Coder-7B | 68.07 | 78.77 | 37.20 | 59.76 | 53.66 | 68.40 | 71.43 | 68.20 | 70.37 |
| Qwen3-0.6B | 47.33 | 43.06 | 27.40 | 19.51 | 17.68 | 23.00 | 33.07 | 25.00 | 32.28 |
| Qwen3-1.7B | 60.30 | 74.75 | 48.40 | 40.24 | 37.80 | 43.80 | 46.03 | 42.00 | 48.68 |
| Qwen3.5-0.8B | 50.31 | 34.95 | 17.00 | 22.56 | 20.73 | 17.40 | 24.07 | 17.80 | 24.60 |
| Qwen3.5-2B | 57.23 | 58.98 | 30.80 | 37.20 | 34.15 | 23.40 | 31.75 | 26.40 | 32.80 |
| Qwen3.5-4B | 69.82 | 85.29 | 49.60 | 59.15 | 53.66 | 40.20 | 48.41 | 24.20 | 32.80 |
| Qwen3.5-9B | 69.85 | 87.34 | 48.80 | 68.90 | 62.80 | 55.00 | 63.76 | 51.00 | 57.14 |

(gsm8k = strict; math500 = math_verify; code = pass@1/pass_at_1; mmlu =
micro acc. Raw: _runs/grid_v1/*/Qwen_*__ar__s0of1.jsonl.)

## Diff vs the 2026-08-12 workstation batch — the predicted pattern, exactly

- **MMLU essentially exact** (2B 57.23 vs 57.34; 1.7B 60.30 vs 60.12):
  loglikelihood is canvas-free, so this is the pure cross-hardware
  numerics check. PASS.
- **GSM8K within noise on four of five** (0.8B -0.08, 4B +0.00 bitwise?,
  9B -0.30, 1.7B +1.13) — and **2B +5.38 (58.98 vs 53.60), which is the
  _ARLM canvas fix landing exactly where predicted**: bare 2B is the
  documented verbosity pathological (FINDING: its flexible-extract sat
  BELOW strict because generations ran past the answer); at cap 256 its
  long CoTs were truncated before finishing, at 1024 they complete. The
  other four barely move because their generations fit 256.
- **math500 all above the full-MATH refs** (+0.1 to +12.9): expected —
  triple protocol change (500-subset, cap 256 -> 1024 which matters most
  on MATH's long solutions, same math_verify metric). These are
  re-measurements, not drift.
- Sanity anchors: MBPP+ > MBPP everywhere (sanitized-subset effect);
  Coder-7B > Qwen2.5-7B on code; Qwen2.5-7B MMLU 74.13 ~ published.

## One curiosity, parked

Qwen3.5-4B drops 16pp on mbpp-FENCE vs scaffold (24.20 vs 40.20) while
every other AR model is fence~=scaffold. Smells like fenced-prompt
output formatting (explanation-before-code breaking extraction?) — the
fence column is the format-sensitivity study, AR fence rows were
over-run additions, so this is a per-sample read for later, not a
blocker.

## Verdict

Tiled SLURM path validated end to end on real queue hardware: submit ->
9-node fan-out -> 72 offline cells -> sentinels -> zero failures, wall
within 3% of the cost model. Parts 2+3 proceed unchanged.
