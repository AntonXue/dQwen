# HANDOFF to the paper-writing Claude: rsync the C.8 trajectory results

> 2026-08-18 ~23:30, cluster (Vista) → workstation. The C.8 trajectory
> statistic campaign requested in
> `20260818-140400-HANDOFF-c8-trajectories-he164-canvas256-for-the-slurm-claude.md`
> is COMPLETE: 5,904/5,904 records, zero assert trips, all six SLURM
> jobs COMPLETED with no wall clips. Full campaign report:
> `20260818-193500-RESULTS-c8-trajectories-he164-g256-complete.md`.
> This doc is just the pickup manual.

## 1. What you are pulling

One NEW directory (does not exist on your side yet), 8 files, ~20 MB:

    _runs/20260818-140400-c8-trajectories-he164-g256/

4 models × 164 HumanEval problems × 6 schemes at canvas 256, greedy,
τ=0.9, in both stop variants. Early-stop files carry all 6 schemes
(164×6 = 984 records); `_nostop` files carry only the 3 block16 schemes
(164×3 = 492) — standard-mode trajectories are provably identical with
or without stop strings, so the standard rows of a no-stop analysis
come from the early-stop file (same convention as the canvas-128 demo;
`trajectories.py::render` shows the union logic).

## 2. The rsync (connectivity-friendly: 8 files, ~20 MB)

    rsync -av --ignore-existing --exclude 'shards/' \
      vista:~/foo/dQwen/_runs/20260818-140400-c8-trajectories-he164-g256/ \
      _runs/20260818-140400-c8-trajectories-he164-g256/

Notes:
- `--exclude 'shards/'` matters: `shards/` holds the raw per-(model,
  scheme) shard files the six parallel jobs wrote (provenance only,
  another ~19 MB of byte-identical content). ⚠ If you ever DO pull
  shards, keep them in their subdirectory — any glob of
  `trajectories_*.jsonl` over a flat mix (e.g. `render()`'s) would
  double-count every record.
- No `--exclude '*.tmp*'` needed here (no in-flight writers; campaign
  is drained and the cluster queue is empty), but it's harmless.
- This is NOT under `grid_v1/`, so the usual `grid_v1 → grid_v1_vista`
  mapping does not apply. Land it at the same relative path.

## 3. Verify the pickup (30 seconds)

Expected line counts:

    wc -l _runs/20260818-140400-c8-trajectories-he164-g256/*.jsonl
      984  trajectories_dqwen3.5-9b-base-v3.jsonl
      492  trajectories_dqwen3.5-9b-base-v3_nostop.jsonl
      984  trajectories_dream-7b-base.jsonl
      492  trajectories_dream-7b-base_nostop.jsonl
      984  trajectories_dream-coder-7b-base.jsonl
      492  trajectories_dream-coder-7b-base_nostop.jsonl
      984  trajectories_llada-8b-base.jsonl
      492  trajectories_llada-8b-base_nostop.jsonl
     5904  total

Bit-level check (slow link → truncation is the realistic failure):

    md5sum *.jsonl
      40cc4487dc3e9b12615829b8d191f739  trajectories_dqwen3.5-9b-base-v3.jsonl
      0beec414ca737e0a90594de1c7f18508  trajectories_dqwen3.5-9b-base-v3_nostop.jsonl
      6516a5c8a84617d89ef6f9f7b791d5b8  trajectories_dream-7b-base.jsonl
      70daaa3d18890308e8852e8bdd219d2d  trajectories_dream-7b-base_nostop.jsonl
      663d08fdb57f2e9a189e7356a5720be3  trajectories_dream-coder-7b-base.jsonl
      a7d62dab8881a1b791ae0328785ddca2  trajectories_dream-coder-7b-base_nostop.jsonl
      dd835c587557f6799bf3fb51c2e5e18f  trajectories_llada-8b-base.jsonl
      04e4368c688f95c58e9dfd854c8c100f  trajectories_llada-8b-base_nostop.jsonl

## 4. Record format + provenance (what changed vs the demo files)

Records are the same JSONL schema as the canvas-128 demo campaign
(model, revision, scheme, task_id, gen_length, commit_step, forwards,
passed, early_stop, fence_cut, graded_prefix_chars, generation,
raw_generation, decode_config), with `gen_length: 256` and
`commit_step` of length 256. Scheme names scale with the canvas:
`standard-static-s256` / `standard-static-s128` here (vs s128/s64 in
the demo files).

Two provenance details that differ from a single-writer run:
- **Record order is scheme-major, not problem-major**: canonical files
  were merged from 36 per-scheme shards in canonical scheme order
  (block s16, s8, τ0.9; standard s256, s128, τ0.9), problem-ordered
  within each scheme block. Key on (model, scheme, task_id) — unique,
  asserted at merge time — and ignore file order.
- The merge also asserted: every line JSON-parses; model/scheme/
  gen_length/early_stop fields consistent with the filename; no
  duplicate (scheme, task_id). Cluster-side per-generation invariants
  (stamped==forwards, s256 permutation, s128 two-commits-per-forward,
  block staircase, τ ≤ static cap) tripped ZERO times over all 5,904
  generations.

## 5. Known pitfalls for the statistic (both already handled)

- Uncommitted positions are `-1` in `commit_step` (early-stopped block
  cells): your `scripts/decode_trajectories.py` already excludes them
  before correlating — keep doing that, correlations flip sign
  otherwise.
- `passed` is an annotation, not a store metric: graded on the
  earliest-protocol-stop + first-markdown-fence prefix (`fence_cut`
  flags fence-affected records; raw full text is always in
  `generation`). Don't quote it against grid_v1 numbers.

Pass counts per model×scheme (context only) and per-job wall times are
in the RESULTS doc if the manuscript wants cost framing.

## 6. Not in scope / cluster state

Nothing was written to `_runs/grid_v1/`; no rendering ran cluster-side;
no statistics were computed here — the C.8 numbers (order-correlation
over HE-164 at canvas 256) are yours to compute from these files. The
Vista queue is drained; the SLURM Claude is standing by for the next
request via the usual git-pulled `_claude/` docs.
