# τ-threshold smoke, HumanEval (2026-08-11 23:14, GPUs 1+2)

Pre-grid smoke to pick adaptive-decode thresholds: dQwen3.5-{2B,9B}@50k,
block32 + standard, τ sweep. Findings (recorded in the manuscript-repo
eval-request doc): knee is size-dependent (2B ~0.85, 9B ~0.75); block
Pareto-dominates standard. Output jsonls here are **smoke diagnostics, not
manuscript-grade cells** — the real τ grid runs through `run.py` into
`_runs/grid_v1/` with the fixed grid {0.5, 0.6, 0.7, 0.8, 0.9}.
