# SLURM deployment kit + the 16-cell spot-check battery (all green)

> 2026-08-12 afternoon. Anton: anticipate the cluster design, then spot-check
> models x decodes incl. sharding + saving. Cluster fingerprints read out of
> ADLMC's launch scripts: TACC Vista, -p gh -A ASC25023, single-GPU GH200
> (AARCH64) nodes, $SCRATCH/cache/huggingface, cluster-side qwen35 conda env
> (built for training), 48h walls, _slurm_out/%x_%j.out convention, OFFLINE
> compute nodes.

## The kit (all committed)

- sbatch_cells.sh — one array task = one cell; qwen35 activate; HF offline
  env; NVMe triton cache. Usage: sbatch --array=$(run.py --pending m.jsonl)%32.
- prewarm_caches.py m.jsonl — login-node cache warm of EXACTLY what the
  manifest references (models@revisions via the registry, datasets via the
  same task construction run.py uses, the code_eval module).
- run.py --pending — validates every manifest line login-side + prints
  not-complete indices (requeue economics).
- setup_env.sh default = harness layer only (NEVER touches the hand-built
  aarch64 torch); --full = x86 workstation set.
- summarize.py --merge — the coalesce step: complete disjoint unions only
  (global id = k + doc_id*n), multi-record docs reduced by min (gsm8k
  strict), merged aggregate printed; partial groups listed as INCOMPLETE.
- run.py: unique per-writer tmp (concurrent requeues can't interleave).

## Battery: 16 cells via the LITERAL SLURM interface (manifest + index)

Covered: all 4 families + 4 AR repos; decodes ar/mc-nelbo/block32-static-
{s8,s32}/standard-static-{s64,s512}/block32-tau{0.5,0.7}/standard-tau0.9;
all 9 benchmarks; revisions step25000-swa + step50000-swa + main; shards
of4/of40/of95/of125/of300/of400/of1000. 16/16 DONE, structure audit clean
(meta launched_at/provenance/code_eval knobs, DLM sidecars with n_forward,
summary sentinel last), idempotent SKIP verified, --pending tracked the
live lane correctly.

## Two real bugs found and fixed

1. LLaDA's remote __init__ calls enable_flash_sdp(True)
   (modeling_llada.py:1056), CLOBBERING the determinism pin — caught by
   the provenance field doing its job (sdpa_math_only=False on exactly the
   llada cell). Fix: run.py re-pins AFTER model construction; llada.py
   docstring carries the trap. Re-ran the cell: True.
2. doc_id leaked into lm_eval_sample.metrics (numeric filter) and thus
   into merge means. Excluded at source and skipped in merge for old
   records.

## Merge validation

Qwen3-0.6B gsm8k, 4 stripes: union complete (1319 docs), merged strict
41.55 vs whole-run cell 41.32 — 0.23pp = 3 docs, attributable to HFLM
bs=16 batch composition differing between striped and whole runs. DLM
cells are bs=1 by construction and immune. POLICY NOTE: AR cells are fast;
run them UNSHARDED (shards=1) so batching composition can't move numbers;
sharding guidance is for the slow DLM gsm8k/math cells. Bonus: the merge
tool independently reassembled the old Gate-A humaneval 4-stripe group
(164 docs, pass@1 35.37).
