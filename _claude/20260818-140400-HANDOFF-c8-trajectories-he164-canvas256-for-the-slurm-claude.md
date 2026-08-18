# HANDOFF to the SLURM Claude: C.8 trajectory statistic campaign — HE-164, canvas 256

> 2026-08-18 ~14:04, workstation → cluster. Request chain: the manuscript's
> C.8 ("How the models decode", `app:decodebehav`) demands its claim be a
> STATISTIC over all 164 HumanEval problems, not the 6 demo panels; the big
> boss (via Anton, today) asked for exactly that at **canvas 256**. The
> workstation GPUs are saturated, so this runs on Vista. It is SMALL by
> your standards (~1M forwards total, a few GPU-hours) — the point of this
> doc is correctness and context, not tiling.
>
> ⚠ This is NOT a grid/cell campaign: no manifests, no `run.py`, no store.
> The tool is `trajectories.py` (repo root, on origin/main), its own JSONL
> format, its own output dir. Do not write anything into `_runs/grid_v1/`.

## 1. Context you need (one minute)

`GenOutput.commit_step` records, per generation, the forward index at
which each canvas position was committed. The manuscript's C.8 measured,
on the 6-problem demo at canvas 128, that models decode NEARLY IN ORDER
even when fully unconstrained (median Spearman ρ(step, position) = +0.962
over 72 unconstrained generations) — the paper now leads with "is it just
an AR model?" and the fraction/correlation over ALL of HumanEval is the
claim's evidence. Canvas 256 doubles the room to deviate from prefix
order, testing whether near-AR order was a small-canvas artifact.
**You produce data only** — all statistics are computed manuscript-side
(`VistaCoder_Technical_Report/scripts/decode_trajectories.py`), which
already handles the one known pitfall (uncommitted −1 positions must be
excluded or correlations flip sign).

## 2. The job

**4 models × 164 problems × 6 schemes, canvas 256, greedy, τ=0.9** —
3,936 generations — **in BOTH stop variants** (early-stop = protocol-
faithful; `--no-early-stop` = full-canvas block cells, +1,968 more).
Schemes (auto-derived from `--gen 256`): `block16-static-s16`,
`block16-static-s8`, `block16-tau0.9`, `standard-static-s256`,
`standard-static-s128` (2 tok/fwd), `standard-tau0.9`.

Per-lane commands (one model per GPU/node; problem-major order and
per-(scheme,task) idempotent resume are built in):

    OUT=_runs/20260818-140400-c8-trajectories-he164-g256
    python trajectories.py --model llada-8b-base          --gen 256 --problems all --out $OUT
    python trajectories.py --model llada-8b-base          --gen 256 --problems all --out $OUT --no-early-stop
    # ... same pair for dream-7b-base, dream-coder-7b-base, dqwen3.5-9b-base-v3

Natural shape: 4 jobs (one model each, the two variants sequential
inside the job), single GPU, 4h walls generous. Requeue freely — resume
skips finished (scheme, task) pairs.

## 3. Environment notes (all verified workstation-side)

- Needs `HF_ALLOW_CODE_EVAL=1` (grading) — the driver sets it via
  env-default; nothing to export.
- Datasets/models: HumanEval + all four models are long cached on
  $SCRATCH; `code_eval` grading is local subprocess execution, offline-
  safe. matplotlib is NOT needed (never pass `--render` cluster-side;
  screening/paper rendering is workstation/manuscript work).
- The driver imports `models` + `benchmark_specs` only (gate-compliant),
  runs bs=1, math-sdpa is irrelevant here (no publication-store claims)
  but the GATE env is fine to use as-is.
- Refactor safety: `schemes_for(128)` is assert-identical to the demo
  campaign's literal scheme table (CPU-verified), so the gen-256
  parameterization changed no behavior.

## 4. Built-in correctness (fails loudly per generation — trust it)

Every generation asserts: stamped steps == executed forwards (progress
floor); `standard-static-s256` = a permutation of 0..255;
`standard-static-s128` = exactly 2 commits per forward; block16 =
staircase-monotone over 16 treads; τ forwards ≤ the static cap. **Treat
the first problem of the first lane as the smoke**: if /0 survives all
six schemes, the path is proven — the demo campaign ran this exact code
at gen 128 with zero assert trips across 216 generations.

## 5. Grading semantics (so `passed` is interpretable)

`passed` is an annotation, not a store metric: graded on the text cut at
the earliest protocol stop (post-hoc — provably identical to early-stop
truncation) and then at the first markdown fence (canvas-tuned stop sets
never see fences; `fence_cut` flags affected records). Raw full text is
stored in every record regardless.

## 6. Cost picture (for lane sizing, not tiling)

Static cells are exact: s256 = 256 fwd, s128 = 128 fwd; block cells
early-stop (demo measured ~50–128 of 128; expect ~60–256 of 256); τ ran
~0.3–0.6× its static twin. Ballpark ≈ 1M forwards total across both
variants ≈ a few GPU-hours per model at canvas ~300–400 tokens; the 9B
lane is the slowest.

## 7. Hand-back

1. `$OUT/trajectories_<model>{,_nostop}.jsonl` × 8 files; expected
   record counts: 984 per early-stop file (164×6), 492 per nostop file
   (164×3 block schemes).
2. A short results doc: per-lane wall time, any assert trips (expect
   none), pass counts per model (context only), and the rsync path.
3. Workstation pulls with the usual additive rsync of the $OUT dir; the
   manuscript's stats script takes over from there.

## 8. Explicitly NOT in scope

No grid cells, no store writes, no protocol claims, no re-runs of the
canvas-128 demo set, no rendering, no statistics — data only.
