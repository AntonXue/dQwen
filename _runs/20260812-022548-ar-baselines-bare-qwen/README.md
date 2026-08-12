# AR baselines on the bare Qwen repos (overnight 2026-08-12, GPUs 1+2)

Retired the -Base wrong-source finding: all six bare conversion sources
(Qwen3.5-{0.8,2,4,9}B, Qwen3-{0.6,1.7}B) × all five benchmarks, decode=ar,
via `run.py`. **Data landed in `_runs/grid_v1/<benchmark>/Qwen_*__ar__s0of1.jsonl`**
— this directory holds only the lane script and console logs of that launch.
