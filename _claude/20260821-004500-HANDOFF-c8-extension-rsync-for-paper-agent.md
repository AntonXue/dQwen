# HANDOFF to the paper-writing Claude: rsync the C.8 EXTENSION results

> 2026-08-21 ~00:45, cluster (Vista) → workstation. The five-model C.8
> extension requested in `20260820-215727-HANDOFF-c8-extension-...` is
> COMPLETE: 7,380/7,380 records, zero assert trips, all six SLURM jobs
> COMPLETED, no wall clips. Full campaign report (walls, pass counts,
> audit details): `20260821-003000-RESULTS-c8-extension-complete-...`.
> This doc is just the pickup manual, in the style of the one you used
> for the first four models (`20260818-233000-HANDOFF-c8-rsync-...`).

## 1. What you are pulling

**10 NEW files (~25 MB) into the SAME directory you already pulled** on
2026-08-18. Nothing you already have has changed — the merge machinery
asserts it never opens an existing canonical file for write, and the
old md5s still match (re-verified cluster-side today). After this pull
the dir is the complete nine-model C.8 dataset:

    _runs/20260818-140400-c8-trajectories-he164-g256/   # 18 files, 13,284 records
      trajectories_<model>.jsonl          984 records  (164 problems x 6 schemes)
      trajectories_<model>_nostop.jsonl   492 records  (164 x 3 block16 schemes)

for the nine models: llada-8b-base, dream-7b-base, dream-coder-7b-base
(pulled 08-18) + coda-1.7b-base, dqwen3-1.7b-base-v3,
dqwen3.5-0.8b-base-v3, dqwen3.5-2b-base-v3, dqwen3.5-4b-base-v3 (NEW)
+ dqwen3.5-9b-base-v3 (pulled 08-18).

Same campaign parameters throughout: HE-164, canvas 256, greedy, τ=0.9,
`schemes_for(256)`, both stop variants. Revisions: family + control at
`step50000-swa`, CoDA at `main` (revision field is `null` in its
records) — verified per-record at merge time.

## 2. The rsync (identical line to last time — it is additive)

    rsync -av --ignore-existing --exclude 'shards/' \
      vista:~/foo/dQwen/_runs/20260818-140400-c8-trajectories-he164-g256/ \
      _runs/20260818-140400-c8-trajectories-he164-g256/

Notes:
- `--ignore-existing` means only the 10 new files transfer (~25 MB);
  the 8 you hold are skipped, which is correct — the cluster has not
  modified them (md5-provable, §3).
- `--exclude 'shards/'` still matters: `shards/` now holds 66 raw
  per-(model, scheme) shard files (provenance only, byte-identical
  content to the canonical files). ⚠ If you ever pull them, keep them
  in their subdirectory: any flat glob of `trajectories_*.jsonl` (e.g.
  `trajectories.py::render`'s) over a mixed dir double-counts every
  record.
- Still NOT under `grid_v1/` — no `grid_v1 → grid_v1_vista` mapping;
  land at the same relative path.
- No in-flight writers (queue drained), so no `*.tmp` risk.

## 3. Verify the pickup (60 seconds)

Line counts — this exact output, 13,284 total:

    cd _runs/20260818-140400-c8-trajectories-he164-g256 && wc -l *.jsonl
      # 984 for each of the nine trajectories_<model>.jsonl
      # 492 for each of the nine trajectories_<model>_nostop.jsonl
      # 13284 total, 18 files

md5s of the 10 NEW files (slow link → truncation is the realistic
failure; a partial rsync leaves no .tmp visible with these flags, so
checksum, don't trust sizes):

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

md5s of the 8 files from the 08-18 pull, unchanged (re-check only if
you suspect local damage):

    dd835c587557f6799bf3fb51c2e5e18f  trajectories_llada-8b-base.jsonl
    04e4368c688f95c58e9dfd854c8c100f  trajectories_llada-8b-base_nostop.jsonl
    6516a5c8a84617d89ef6f9f7b791d5b8  trajectories_dream-7b-base.jsonl
    70daaa3d18890308e8852e8bdd219d2d  trajectories_dream-7b-base_nostop.jsonl
    663d08fdb57f2e9a189e7356a5720be3  trajectories_dream-coder-7b-base.jsonl
    a7d62dab8881a1b791ae0328785ddca2  trajectories_dream-coder-7b-base_nostop.jsonl
    40cc4487dc3e9b12615829b8d191f739  trajectories_dqwen3.5-9b-base-v3.jsonl
    0beec414ca737e0a90594de1c7f18508  trajectories_dqwen3.5-9b-base-v3_nostop.jsonl

## 4. Record format + provenance (all conventions carry over)

Same JSONL schema as the 08-18 files and the canvas-128 demo (model,
revision, scheme, task_id, gen_length, commit_step, forwards, passed,
early_stop, fence_cut, graded_prefix_chars, generation, raw_generation,
decode_config); `gen_length: 256`, `commit_step` length 256.

- **Record order is scheme-major** (merged from per-scheme shards in
  canonical order: block s16, s8, τ0.9; standard s256, s128, τ0.9),
  problem-ordered within each scheme. Key on (model, scheme, task_id) —
  uniqueness asserted at merge.
- Cluster-side audit at merge: every line JSON-parses; model/scheme/
  gen_length/early_stop/revision consistent; no duplicates; counts
  exact. In-driver per-generation invariants tripped ZERO times over
  the extension's 7,380 generations (13,284 campaign-wide).
- Standard-mode no-stop convention unchanged: `_nostop` files carry
  ONLY the 3 block16 schemes; standard-mode trajectories are provably
  stop-string-independent, so a no-stop analysis takes its standard
  rows from the early-stop file (union logic: `trajectories.py::render`).

## 5. Pitfalls for the statistics (same two as before + one CoDA note)

- Uncommitted positions are `-1` in `commit_step` — exclude before
  correlating (your `scripts/decode_trajectories.py` already does;
  correlations flip sign otherwise).
- `passed` is a fence-cut annotation, not a protocol metric — never
  quote against grid_v1 numbers. CoDA's pass counts are LOW
  (16–24/164) as the request doc forecast; that is signal, and its
  low-pass records are exactly as trajectory-valid as passing ones.
- CoDA ran at canvas 256 vs its native 768 — fine for this exhibit
  (off-protocol by design), but worth a disclosure footnote if its row
  is featured.

## 6. Cluster state

Queue drained, nothing running. The 08-18 four-model data, this
extension, and all docs are on origin/main. The SLURM Claude stands by
for the next request via the usual git-pulled `_claude/` docs.
