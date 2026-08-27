# Throughput-sweep viz dump — file inventory

Everything here is DERIVED from the raw `../20260827-throughput-gh200*` JSONLs
by `make_dump.py` (rerun it from the repo root to regenerate; raw files are
never modified). Analysis conventions + pitfalls: the rsync handoff doc
`_claude/20260827-113000-HANDOFF-throughput-rsync-viz-dump-for-workstation.md`.

| file | contents |
|---|---|
| `flat_all_records.csv` | every record from v1, v2-i50, stability-a/b/c (2,277 rows). One row per (dataset, model, arm, mode, B, L). `arm` is pre-decoded from the (`sdpa_arm`,`causal_conv1d`) signature. Empty `median_ms` + `oom=True` = OOM cell; `unsupported=True` = no trunk accessor (LLaDA, CoDA). `node` is empty for records written before the provenance patch (all of v1 except late cells) — dataset↔node map in the handoff doc. |
| `tier1_ratios.csv` | hybrid-2B / control-1.7B tok/s ratio over the ENTIRE v2 deployment grid (full+trunk+head modes, incl. L16k/32k, crossover-refinement, batch-saturation cells). THE headline curves. |
| `arms_2x2.csv` | all four kernel arms × 4 hybrid models × slice points (full mode) — the kernel-robustness figure. |
| `stability.csv` | Tier-1 deployment slice medians at 5 occasions (v1@c608, v2@c639, repeats a/b/c) — the reproducibility figure. B=1 rows jitter 28–43%; B=16 rows agree to 0.3–0.8%. |
| `v1_v2_delta.csv` | per-cell median deltas v1 (i10/20) vs v2 (i50), 780 shared deployment cells — iters-sensitivity. |
| `oom_envelope.csv` | every OOM cell, by dataset/arm. |
| `identity_check.csv` | full vs trunk+head decomposition per model (0.99–1.01). |
| `model_summary.csv` | 9-model quick table: dims, vocab, headline tok/s. |
| `fa2_delta_errors.jsonl` | the flash-attn-2 spot-check's load failures (the check is INAPPLICABLE — remote-code configs don't map to AutoModelForCausalLM). |
| `MD5SUMS.txt` | checksums of every file under `../20260827-throughput-gh200*` (relative to `_runs/`). |
