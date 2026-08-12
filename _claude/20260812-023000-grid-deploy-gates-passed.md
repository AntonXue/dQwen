# grid_v1 deploy gates: PASSED — spot-check 10/10, shard merge lossless, battery reproduction verified

> 2026-08-12 ~02:30. Everything below ran through `run.py` on this box's
> GPUs 1/2. The runner is deploy-ready; remaining pre-launch item is the
> manifest row lists.

## Spot check (10/10 PASS, ~7 min wall)
All six decode families (block/standard × static/tau, ar, mc-nelbo) ×
all six benchmarks (he, mbpp, mbpp-fence, gsm8k, math, mmlu) × four model
classes (0.8B, 2B, Dream-7B comparator, bare Qwen3-1.7B AR) × three
revisions × six stripe shapes. Comparator forwards accounting works;
MC cells correctly have no sidecar; math_verify scores.

## Gate A — shard-merge + HumanEval reproduction (0.8B v3 @50k, b32/s32)
- 4 stripes merged: **164/164 docs, zero overlap**, 4 distinct fingerprints.
- pass@1 **35.37 vs battery 33.54** (+1.83pp = net 3 docs; flips 4 up /
  1 down, scattered) — cross-era numerics drift, NOT protocol: decode
  configs are identical (checked against the battery json's decode block),
  and a same-day kernel probe (math-sdpa pin ON vs OFF) produced **0**
  per-doc flips, i.e. same-environment runs are deterministic.
- CALIBRATION consequence: expect ~±2pp on HE-164 when PROBE-era numbers
  get their GATE restamp; paired same-environment comparisons stay exact.

## Gate B — GSM8K template fidelity (2B v3 @50k, stripe 0/8, n=165)
- **163/165 per-doc agreement** with the battery samples; stripe strict
  accuracy identical (27.27 = 27.27). The 8-shot CoT prompt + strict
  extraction survive the get_task_dict → stripe → evaluate() path intact.

## Smoke frontiers (context for the tau grids; full logs _runs/smoke_tau/)
Block-adaptive: 2B plateaus ≥τ0.85 (42.7%), 9B ≥τ0.75 (66.5% @ 39 fwd ≈
2.8× cheaper than its sequential anchor). Standard-adaptive: ~4× the
forwards at ≤ the accuracy — block Pareto-dominates standard for our
models on HE. τ grid for the campaign: {0.5, 0.6, 0.7, 0.8, 0.9}.

## Known accounting details for the merge tool
- gsm8k/math log TWO lm_eval_sample records per doc (one per extraction
  filter) under the same `exact_match` key; strict = per-doc min.
- Global doc id from a k/n stripe: `global = k + doc_id * n`.
