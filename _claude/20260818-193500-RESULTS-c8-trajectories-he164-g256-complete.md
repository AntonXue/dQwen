# RESULTS: C.8 trajectory statistic campaign — COMPLETE (5,904/5,904)

> 2026-08-18 ~19:35, cluster → workstation. Answers the 14:04 handoff
> (`20260818-140400-HANDOFF-c8-trajectories-he164-canvas256-...`).
> 4 models × 164 HumanEval × 6 schemes, canvas 256, greedy, τ=0.9, both
> stop variants. Data only, as specified — no statistics computed here,
> nothing touched in `_runs/grid_v1/`, no `--render`.

## 1. Hand-back files (rsync these)

    _runs/20260818-140400-c8-trajectories-he164-g256/
      trajectories_llada-8b-base.jsonl              984   # 164 x 6 schemes
      trajectories_llada-8b-base_nostop.jsonl       492   # 164 x 3 block schemes
      trajectories_dream-7b-base.jsonl              984
      trajectories_dream-7b-base_nostop.jsonl       492
      trajectories_dream-coder-7b-base.jsonl        984
      trajectories_dream-coder-7b-base_nostop.jsonl 492
      trajectories_dqwen3.5-9b-base-v3.jsonl        984
      trajectories_dqwen3.5-9b-base-v3_nostop.jsonl 492
      shards/                                       # raw per-(model,scheme)
                                                    # shards; provenance only

Usual additive pull (the 8 top-level files are ~5–6 MB total; `shards/`
is byte-identical content, skip it unless you want provenance):

    rsync -av --ignore-existing --exclude 'shards/' \
      vista:~/foo/dQwen/_runs/20260818-140400-c8-trajectories-he164-g256/ \
      _runs/20260818-140400-c8-trajectories-he164-g256/

Canonical files were merged from per-scheme shards in the canonical
scheme order (block s16, block s8, block τ0.9, standard s256, standard
s128, standard τ0.9), each problem-ordered within its scheme block, so
record order is scheme-major (the single-writer original would have been
problem-major — key on (model, scheme, task_id), which is unique; the
merge asserted uniqueness, JSON-parse of every line, model/scheme/
gen_length/early_stop field consistency, and len(commit_step)==256).

## 2. Verification

- **Assert trips: zero.** All per-generation invariants (stamped==
  forwards, s256 permutation, s128 two-commits, block staircase, τ cap)
  passed on all 5,904 generations — grep of all six job logs found no
  AssertionError/Traceback/FAILED lines.
- All 36 shards landed at exactly 164 records; totals 984/492 per
  canonical pair, 5,904 grand total.
- SLURM: all six jobs COMPLETED (no wall clips, no requeues needed).

## 3. Lane walls (job = scheme, 4 model-ranks in parallel per job)

| job    | scheme (both stop variants where applicable) | elapsed  |
|--------|-----------------------------------------------|----------|
| 920110 | block16-static-s16 (+nostop)                  | 1:55:46  |
| 920112 | block16-static-s8 (+nostop)                   | 0:38:53  |
| 920113 | block16-tau0.9 (+nostop)                      | 0:44:49  |
| 920114 | standard-static-s256                          | 0:47:34  |
| 920115 | standard-static-s128                          | 0:25:02  |
| 920116 | standard-tau0.9                               | 0:25:01  |

~5.0 node-hours × 4 nodes ≈ 20 GPU-hours total. Shape was six 4-node
jobs (one per scheme, rank=model) after the 24-node single job queued
too slowly; per-shard files kept writers disjoint. Note 920110 finished
4 min under its 2h wall — resume would have made a clip free, but it
wasn't needed.

## 4. Pass counts (context only — grading semantics per handoff §5)

pass@1 / 164, early-stop variant:

| model               | blk-s16 | blk-s8 | blk-τ0.9 | std-s256 | std-s128 | std-τ0.9 |
|---------------------|---------|--------|----------|----------|----------|----------|
| llada-8b-base       | 45      | 33     | 45       | 45       | 40       | 45       |
| dream-7b-base       | 65      | 49     | 66       | 62       | 47       | 61       |
| dream-coder-7b-base | 92      | 58     | 92       | 92       | 65       | 92       |
| dqwen3.5-9b-base-v3 | 96      | 74     | 95       | 91       | 75       | 91       |

Familiar shape from the grid campaigns: τ0.9 ≈ the full-static twin at a
fraction of the forwards; the halved statics (s8, s128) pay real
accuracy for their speed. The C.8 statistic itself (order correlations,
excluding uncommitted −1 positions) is manuscript-side work from here.
