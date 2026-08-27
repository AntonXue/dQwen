# HANDOFF to the SLURM Claude: forward-pass throughput sweep (the RNN-layer cost question)

> 2026-08-26 ~23:53, workstation → cluster. Boss ask via Anton: do the
> hybrid (Gated DeltaNet) layers cost throughput vs full attention? Tool:
> `throughput.py` (repo root, pushed) — models × {full, trunk, head} ×
> B × L, no decoding, one JSONL line per cell. The workstation prototype
> (H100, GPU 2, co-tenanted — numbers indicative only) already answered
> the shape; your job is the CLEAN measurement on an **exclusive GH200
> node** — co-tenancy corrupts timing, which is the whole reason this
> runs on Vista. Design discussion + prototype: this doc; prototype
> records: `_runs/20260821-throughput-prototype-h100/` (H100 NVL,
> committed for reference).

## 1. What the prototype found (your expected shape, not your numbers)

Width-matched pair (dq3.5-2B hybrid vs dq3-1.7B control), trunk mode,
deployment kernels: hybrid **0.60×** at B=1/L=1024 (launch-bound; FLA
barely helps there), crossover by ~L2048 or any batching, **1.2–1.4×
ahead** at B16 / L4096, much flatter memory. Full-mode adds the vocab
head (248k vs 151k — the reason trunk mode exists). If GH200 shows a
qualitatively different shape, that is a finding, not an error — but
sanity-check gross anomalies against the prototype JSONLs.

## 2. The three modes (all in `throughput.py`, self-checking)

- `full`: input_ids → logits via `adapter.raw_logits` (deployment number)
- `trunk`: random (B, L, d) bf16 through the INNER stack only — the
  accessor SELF-DISCOVERS per family and validates by output shape;
  LLaDA reports `unsupported` (its remote code takes no embeds kwarg) —
  that is correct behavior, record it, do not chase it.
- `head`: the (B·L, d)×(d, V) GEMM. Identity full ≈ embed+trunk+head is
  the harness sanity check; spot-verify it once per model.

## 3. Kernel arms = {env} × {--sdpa}; NO forcing code anywhere

| arm | env | --sdpa | who runs it |
|---|---|---|---|
| **deployment** (headline) | +causal-conv1d env | fast | ALL models, FULL grid |
| FLA+math (cross) | +causal-conv1d env | math | hybrids only, slice |
| fallback+flash (cross) | stock env | fast | hybrids only, slice |
| naive | stock env | math | all models, slice |

The GDN axis is package presence: transformers' fast path activates iff
BOTH `fla` AND `causal_conv1d` import. ⚠ Your stock env (like ours)
already has `fla` but NOT `causal_conv1d` — so stock env = fallback arm
as-is. **Env recipe (do NOT install into the frozen eval env):**

    $EVAL_PY -m venv --system-site-packages $WORK/tp-venv
    $WORK/tp-venv/bin/pip install causal-conv1d --no-build-isolation

`causal-conv1d` is a CUDA extension: source build, needs nvcc (worked
first try on our x86 box; aarch64 is your known territory — you built
torch itself). FLA is Triton-JIT, already present, no build. **HARD
GATE before any measured cell**: one hybrid forward in the venv must NOT
print "The fast path is not available…" and provenance must read
`fla: true, causal_conv1d: true`. A partial install is a mongrel state —
refuse it, report, don't measure it.

## 4. The grid

- **Roster (9)**: the `REVISIONS` dict in throughput.py — family
  0.8B/2B/4B/9B @50k, control @50k, llada, dream, dream-coder, coda.
  Tier 1 = the 2B/control pair (the controlled claim); rest = context.
- **Deployment arm, full grid**: B ∈ {1,2,4,8,16,32}, L ∈ {256,512,
  1024,2048,4096,8192} (driver defaults), all 3 modes. OOM cells are
  recorded-and-skipped (the envelope is data; GH200 96GB ≈ our H100 NVL,
  so the prototype's envelope predicts yours).
- **Other three arms, slice only**: `--batches 1,16 --lengths 1024,4096`
  (kernel ratios are smooth in B×L; nobody needs the naive arm's
  envelope). Hybrids get all four arms; pure-attention models get
  deployment + naive only (their GDN axis does not exist).
- Timing discipline is in-driver: 5 warmup + 20 timed, CUDA events,
  median + IQR + peak-mem. Run ONE process at a time on the node.

Cost: deployment ≈ 1.5–3h (loads + the 9B/llada top corner), each
slice arm ≈ 15 min. One node, well under a day.

## 5. Hand-back

1. `_runs/<stamp>-throughput-gh200/throughput_<model>.jsonl` — arms
   distinguished by the in-record provenance (`sdpa_arm`, `fla`,
   `causal_conv1d`); one dir, all arms.
2. Results doc: the Tier-1 2×2 table at the slice points, the
   deployment-arm crossover reading per model, the full≈trunk+head
   spot-checks, any OOM boundary of note, and node/clock provenance
   (exclusive job id, `nvidia-smi -q -d CLOCK` snapshot).
3. Usual additive rsync line + md5s.

## 6. Not in scope (each excluded WITH its reason in the design chat)

flash_attn2 package (marginal over torch-flash, aarch64 burden,
comparators pin their own paths), torch.compile (benchmarks the
compiler; per-shape recompiles across the grid), quantization/fp8
(different study), CUDA graphs (would flatter exactly the launch-bound
B=1 regime that is honestly part of the story), AR parents (causal
masking confound — revisit only if the boss asks), FLA mode variants
(chunked IS the full-sequence path; recurrent is for decode, which this
experiment deliberately has none of).
