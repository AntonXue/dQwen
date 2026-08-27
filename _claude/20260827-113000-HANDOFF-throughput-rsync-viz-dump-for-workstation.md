# HANDOFF to the workstation Claude: pull the throughput sweep + viz dump

> 2026-08-27 ~11:30, cluster (Vista) → workstation. The GH200 forward-pass
> throughput campaign (boss ask: do the GDN layers cost throughput?) is
> COMPLETE — verdict + full analysis in
> `20260827-110000-RESULTS-throughput-sweep-gh200-complete.md` (read that
> first; this doc is the pickup + plotting manual, in the C.8 rsync-manual
> style). You are pulling raw data AND a ready-made diagnostic dump so the
> figures need zero cluster round-trips.

## 1. What you are pulling (7 sibling dirs, 44 files, ~1.6 MB total)

    _runs/20260827-throughput-gh200-i50/          # THE RELEASE DATASET (uniform 5+50 forwards)
    _runs/20260827-throughput-gh200/              # v1 quick pass (mixed iters) — sensitivity/inter-node checks ONLY
    _runs/20260827-throughput-gh200-stability-{a,b,c}/   # Tier-1 slice repeats (start/mid/end of session)
    _runs/20260827-throughput-gh200-fa2/          # flash-attn-2 spot check (LOAD FAILURES — inapplicable, see RESULTS §6)
    _runs/20260827-throughput-gh200-dump/         # derived CSVs for plotting + MD5SUMS + make_dump.py + dump_README.md

Raw format: `throughput_<model>.jsonl`, one JSON line per (mode, B, L) cell;
multiple kernel arms APPEND to the same file and are distinguished by
per-record provenance (decode table in §4). The dump dir's
`flat_all_records.csv` has all 2,277 records already flattened with the arm
pre-decoded — start there; touch raw JSONLs only to audit.

## 2. The rsync (one additive line; remote shell expands the glob)

    rsync -av --ignore-existing \
      vista:'~/foo/dQwen/_runs/20260827-throughput-gh200*' \
      _runs/

Notes:
- Lands all seven dirs at the same relative paths. `--ignore-existing` makes
  re-pulls safe; nothing cluster-side will be rewritten (campaign closed,
  queue empty of throughput work).
- Tiny payload (~1.6 MB) — no truncation drama, but checksum anyway (§3).
- If you regenerate dump CSVs locally, rerun
  `python _runs/20260827-throughput-gh200-dump/make_dump.py` from the dQwen
  repo root — it rewrites ONLY the dump dir, raw files untouched.

## 3. Verify (30 seconds)

    cd _runs && find 20260827-throughput-gh200* -type f | wc -l   # 44
    md5sum -c 20260827-throughput-gh200-dump/MD5SUMS.txt          # all OK
    md5sum 20260827-throughput-gh200-dump/MD5SUMS.txt
      # 883c6100f74f5959f54365e1602342a5  (the manifest's own checksum)

`MD5SUMS.txt` covers every file in all seven dirs (paths relative to
`_runs/`), computed cluster-side at close-out.

## 4. Record schema + the arm signature (memorize this table)

Fields: model, revision, d, vocab, node, device, torch, sdpa_arm, flash_sdp,
mem_eff_sdp, fla, causal_conv1d, trunk_module, embeds_kw, warmup, mode,
batch, length, iters, median_ms, iqr_ms, min_ms, tokens_per_s, peak_mem_mb
(+ `oom: true` cells, + `unsupported: true` trunk stubs).

| (`sdpa_arm`, `causal_conv1d`) | arm |
|---|---|
| fast, true | **deployment** (headline: flash SDPA + FLA/conv1d GDN fast path) |
| math, true | fla+math (hybrids only) |
| fast, false | fallback+flash (hybrids only) |
| math, false | naive (reference kernels) |

- Modes: `full` = ids→logits (deployment number); `trunk` = inner stack only
  (removes the 248k-vs-151k vocab-head confound); `head` = the lone vocab
  GEMM. full ≈ trunk+head holds to ≤1% (`identity_check.csv`).
- `iters` is per-record (50 in v2; 20 or 10 in v1) — never assume.
- `node`: records written after the provenance patch carry it. For earlier
  ones: ALL of v2/stability/fa2 = c639-072; v1 = c608-072 EXCEPT
  llada's naive arm (c639). Both nodes are GH200-120GB, identical kernel
  (`5.14.0-611.16.1.el9_7.aarch64+64k`); clock snapshots + job ids in
  `20260827-throughput-gh200/node_provenance.txt`.
- Tier-1 pair: `dqwen3.5-2b-base-v3` (hybrid) vs `dqwen3-1.7b-base-v3`
  (width-matched full-attention control). Hybrids = the four dqwen3.5 rows;
  everything else is pure attention (no fla+math / fallback arms — by
  design, their GDN axis does not exist).

## 5. Pitfalls for stats + figures (each burned us once already)

1. **Use v2-i50 for every figure.** v1 exists to SHOW iters/node insensitivity
   (`v1_v2_delta.csv`: median |Δ| 0.3–1.3%), not to be averaged in. v1 has
   no 9B (its partial file was deliberately deleted mid-campaign).
2. **B=1 cells are a regime, not a number**: launch-bound latency jitters
   28–43% across occasions (`stability.csv`) while B≥4 cells reproduce to
   0.3–0.8%. Quote "≈0.66×, launch-bound" — never a third digit. The
   RATIO is stable because both models jitter together within one run.
3. **Never average across arms**: group by the `arm` column (pre-decoded in
   the flat CSV). A file-level mean mixes deployment with naive.
4. **OOM cells are envelope data** (`oom_envelope.csv`), and the two
   naive-arm OOMs (llada, dream-coder @ full B16/L4096) are a finding —
   unfused attention's transient kills 7–8B attention models at shapes
   hybrids handle.
5. **Trunk is absent for LLaDA and CoDA** (remote code takes no embeds
   kwarg; recorded as `unsupported`). Their layer-stack cost is full − head
   if you need it — say so in the caption.
6. **Memory**: `peak_mem_mb` is the torch allocator per-cell peak. Full-mode
   memory is vocab-head-dominated (the prototype's "flatter memory" claim is
   REFUTED — plot trunk-mode memory if you want the architecture story).
7. fa2 numbers do not exist and cannot ("Unrecognized configuration class"
   — see `fa2_delta_errors.jsonl`): torch-SDPA-flash IS the best available
   attention for every row. Do not present fa2 as "skipped".

## 6. The figures this data was shaped for (dump file → plot)

1. `tier1_ratios.csv` → **ratio vs L, one line per B, full + trunk panels**
   (0.66× launch-bound → 2.2×/2.4× @L32k; crossover cells at L1536–6144 and
   B3–24 make the transition a measured curve — B=1 jumps 0.85→1.19 between
   L3584 and L4096, a real kernel-regime step, worth an annotation).
2. `arms_2x2.csv` → grouped bars per hybrid: deployment vs the three
   degraded arms (kernel-robustness; deployment dominates everywhere).
3. `tier1_ratios.csv` naive rows are NOT there — compute the both-naive
   ratio from `flat_all_records.csv` (arm=naive) if the reviewer-defense
   figure is wanted: it GROWS to 2.7–3.3× (fast kernels flatter the control).
4. `stability.csv` → strip/dot plot of 5 occasions per point (the B1 vs B16
   contrast is the whole message).
5. `model_summary.csv` → 9-model context bars @B16/L4096.
6. `flat_all_records.csv` (mode=trunk, peak_mem_mb vs B·L) → memory scaling.

## 7. Cluster state at write time

Battery closed (42 stages, zero failures, 7h20m on c639). The dQwen working
tree carries the uncommitted throughput.py patches (llada math-repin + node
provenance) and the three 08-27 `_claude` docs incl. this one — they reach
you via the usual git route once Anton green-lights the commit; the DATA
route is this rsync and is live now. idev 945525 keeps c639 for ~1.5 more
days; nothing queued. The SLURM Claude stands by.
