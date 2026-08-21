# C.8 dataset COMPLETE and verified workstation-side: nine models, 13,284 records

> 2026-08-21 ~02:14, workstation. Closes the pickup loop on the cluster's
> extension hand-back (`20260821-004500`). The C.8 decode-order dataset
> is now WHOLE on this machine — this doc is the verification record and
> the one-stop pointer for the paper agent.

## The dataset (the paper agent reads from HERE)

`~/foo/dQwen/_runs/20260818-140400-c8-trajectories-he164-g256/` —
**18 files, 13,284 records**, the full nine-model roster:

| group | models |
|---|---|
| family | dqwen3.5-{0.8b,2b,4b,9b}-base-v3 @ step50000-swa |
| control | dqwen3-1.7b-base-v3 @ step50000-swa |
| comparators | llada-8b-base, dream-7b-base, dream-coder-7b-base, coda-1.7b-base |

Per model: 984 early-stop records (HE-164 × 6 schemes) + 492 `_nostop`
(× the 3 block16 schemes; standard-mode trajectories are provably
stop-independent — union logic in `trajectories.py::render`). Canvas
256, greedy, τ=0.9, `schemes_for(256)`; scheme-major record order; key
on (model, scheme, task_id).

## Verification (this machine, after Anton's additive rsync)

- 18 `*.jsonl`, 13,284 lines total — exact.
- **All 10 extension files match the cluster's published md5s** bit-for-
  bit (the 8 files from the 08-18 pull were md5-verified then and
  re-attested unchanged cluster-side).
- Campaign-wide: 13,284 generations with ZERO in-driver invariant trips;
  merge audits clean (unique keys, consistent fields, exact counts).

## The arc this closes (for the next reader)

1. 08-17: canvas-128 six-problem demo (4 models, 216 records + screens)
   → C.8/§4.5 written; the near-in-order finding (ρ=+0.962) reversed the
   any-order framing and reached the abstract.
2. 08-18: boss ask #1 — HE-164 at canvas 256, 4 models, both stop
   variants (5,904 records) on Vista.
3. 08-20/21: boss ask #2 — five more rows (rest of the family, control,
   CoDA), same shape (7,380 records), same dataset dir. The three
   questions it exists to answer: does near-in-order decoding hold
   across SCALE, across BACKBONE (control vs 2B, width-matched), and
   across ADAPTATION LINEAGE (CoDA)?

## Standing notes for the statistics/plots (unchanged, restated once)

- Exclude `-1` (uncommitted) positions before any correlation.
- `passed` is a fence-cut annotation, never comparable to grid_v1.
- CoDA: low pass counts (16–24/164) are signal; add the canvas-256-vs-
  native-768 disclosure if its row is featured.

Cluster queue is drained; nothing pending on any machine for C.8.
