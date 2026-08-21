# RESULTS + pickup: C.8 extension COMPLETE (7,380/7,380) — five new model rows

> 2026-08-21 ~00:30, cluster → workstation. Answers
> `20260820-215727-HANDOFF-c8-extension-five-more-rows-for-the-slurm-claude.md`.
> 5 models × 164 HumanEval × 6 schemes × both stop variants, canvas 256,
> greedy, τ=0.9. Additive to the 20260818 campaign: the output dir now
> holds all NINE model rows (18 canonical files = 9 models × 2 stop
> variants). Data only, as specified — no statistics, no rendering,
> nothing in `_runs/grid_v1/`.

## 1. Verification

- **Assert trips: zero** over all 7,380 new generations (grep of all six
  job logs: no AssertionError/Traceback/RuntimeError/FAILED). Campaign
  total now 13,284 generations, zero trips.
- All six SLURM jobs COMPLETED, no wall clips, no requeues.
- Merge audit (same machinery as 20260818): every line JSON-parses;
  model/scheme/gen_length/early_stop consistent with filename;
  **revision field verified** (`step50000-swa` for family+control,
  `null` for CoDA); no duplicate (scheme, task_id); counts exactly
  984/492 per pair; existing files never opened for write (merge
  refuses to overwrite).
- CoDA's untied-head gate and rope-recompute shims: clean on every load.

## 2. Lane walls (job = scheme, 5 model-ranks in parallel)

| job    | scheme (+nostop where applicable) | elapsed |
|--------|-----------------------------------|---------|
| 926103 | block16-static-s16 (+nostop)      | 1:17:47 |
| 926104 | block16-static-s8 (+nostop)       | 0:40:27 |
| 926105 | block16-tau0.9 (+nostop)          | 0:43:24 |
| 926106 | standard-static-s256              | 0:46:40 |
| 926107 | standard-static-s128              | 0:25:44 |
| 926108 | standard-tau0.9                   | 0:25:13 |

~4.0 node-hours × 5 nodes ≈ 20 GPU-hours. Long pole as predicted: the
4B's full-canvas block16-s16 staircase (~17s/record).

## 3. The pickup (paper agent: same dir, 10 NEW files, ~25 MB)

The SAME rsync line from `20260818-233000-HANDOFF-c8-rsync-...` picks
up the new files (it is additive; `--ignore-existing` skips the eight
you already have, whose md5s are unchanged):

    rsync -av --ignore-existing --exclude 'shards/' \
      vista:~/foo/dQwen/_runs/20260818-140400-c8-trajectories-he164-g256/ \
      _runs/20260818-140400-c8-trajectories-he164-g256/

After pulling you should have **18 files, 13,284 lines total**
(9 models × 984 early-stop + 9 models × 492 nostop). New-file
verification:

    wc -l: 984 for each of trajectories_{coda-1.7b-base,
      dqwen3-1.7b-base-v3, dqwen3.5-0.8b-base-v3, dqwen3.5-2b-base-v3,
      dqwen3.5-4b-base-v3}.jsonl; 492 for each *_nostop.jsonl

    md5sum:
      78dc4dcb1ce416a995bc83953cd9d917  trajectories_coda-1.7b-base.jsonl
      988456b80855cab45faea2d26719e045  trajectories_coda-1.7b-base_nostop.jsonl
      3710d878e06c2ee660fc8ecd93122beb  trajectories_dqwen3-1.7b-base-v3.jsonl
      5822e093695b69d8685a9fd0dfa3a341  trajectories_dqwen3-1.7b-base-v3_nostop.jsonl
      842dd4d9695501ac31a6f3dc8f2f3b12  trajectories_dqwen3.5-0.8b-base-v3.jsonl
      6331b56f8a5c391b885a169d02f01057  trajectories_dqwen3.5-0.8b-base-v3_nostop.jsonl
      b4629e1f3c780c863eaeeb2e4329bb70  trajectories_dqwen3.5-2b-base-v3.jsonl
      e6e1a7017730c5231fdac5802ce6b845  trajectories_dqwen3.5-2b-base-v3_nostop.jsonl
      631837c1730348fdc0328e7e73c9c932  trajectories_dqwen3.5-4b-base-v3.jsonl
      97d4ff218477d847972e83c2e41cc308  trajectories_dqwen3.5-4b-base-v3_nostop.jsonl

All provenance conventions from the 20260818 docs carry over verbatim:
scheme-major record order (key on (model, scheme, task_id)), −1 =
uncommitted (exclude before correlating), `passed` is fence-cut
annotation only, shards stay in `shards/` (flat-glob double-count
hazard).

## 4. Pass counts (context only — NOT protocol numbers)

pass@1 / 164, early-stop variant:

| model                 | blk-s16 | blk-s8 | blk-τ0.9 | std-s256 | std-s128 | std-τ0.9 |
|-----------------------|---------|--------|----------|----------|----------|----------|
| coda-1.7b-base        | 24      | 18     | 24       | 23       | 16       | 24       |
| dqwen3-1.7b-base-v3   | 61      | 37     | 61       | 60       | 47       | 61       |
| dqwen3.5-0.8b-base-v3 | 42      | 31     | 40       | 43       | 35       | 42       |
| dqwen3.5-2b-base-v3   | 58      | 46     | 58       | 53       | 49       | 55       |
| dqwen3.5-4b-base-v3   | 84      | 78     | 84       | 87       | 75       | 87       |

CoDA is low as forecast (that is signal). The τ0.9 ≈ full-static
pattern holds at every scale, including 0.8B — a nice omen for the
boss's across-scale question, though the ORDER statistics (the actual
C.8 claim) are yours to compute.

## 5. Cluster state

Queue drained; no dQwen jobs running. Standing by for the next request
via the usual git-pulled `_claude/` docs.
