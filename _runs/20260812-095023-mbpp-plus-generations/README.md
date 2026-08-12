# MBPP+ generations (launched 20260812-095023, GPUs 1+2)

MBPP+ = EvalPlus's MBPP-sanitized: 378 problems, edited prompts, own format
— generation-defining, so these are NEW runs, not regrades (unlike HE+).
Fills the Table-4 MBPP+ column: family @50k + control (standard full-canvas
s512, sdpa-pinned) and the six AR counterparts (greedy).

Stops are EvalPlus's base-model EOS set (MBPP_STOPS in the driver), NOT the
HumanEval stop set — MBPP+ solutions legitimately contain helpers/comments.
Completions are saved raw-truncated; EvalPlus sanitize can rerun from the
samples on CPU if raw grading looks pathological.

GPU1 lane: family 0.8B/2B/4B + control. GPU2 lane: family 9B + 6 AR models.
Grading runs inline after each model; `*_eval_results.json` lands alongside.

## ABORTED 2026-08-12 10:45

Killed at Anton's call (~160/378 and ~100/378 into the first two models;
no samples files were written -- the driver held completions in memory).
Nothing here is citable. Superseded by the SLURM manifest path: the driver
now writes incrementally to the `_runs/evalplus/` store with resume, so
MBPP+ generation queues on the cluster like everything else.
