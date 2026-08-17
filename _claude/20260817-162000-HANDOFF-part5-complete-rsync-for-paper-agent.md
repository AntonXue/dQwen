# HANDOFF (cluster → workstation): part5 at 2,899/2,900 — rsync NOW, one-cell pickup later

> 2026-08-17 ~16:20, the SLURM Claude. Anton's call: ship the data now so
> the paper agent proceeds; the single in-flight straggler follows. This
> doc is the complete pickup manual — rsync commands, verification,
> what the data contains, and the manuscript-relevant analysis.

## 1. State of the cluster store, exactly

- `_runs/grid_v1/`: **2,899 of 2,900 cells** (find-count on `*.jsonl`;
  raw `ls` shows ~49 orphaned `*.tmp*` writer files from the 08-16
  storms — junk, excluded below, scrubbed cluster-side after drain).
- Final merge (snapshot inside `_runs/20260817-161500-part5-merge/`):
  exit 0, **exactly ONE INCOMPLETE**, the expected straggler:
  `dqwen3-1.7b-base-v3@step25000-swa math500 standard-tau0.95` (stripe 1
  of 2 in flight, sibling ran 71 min, ETA ~17:02). Every other group in
  the entire store is a complete disjoint union.
- **Round 2 ran with ZERO cell failures** — the storm-wait hardening +
  calm weather; the whole part5 = gates(6) + 5b(405) + 5c(84) + 5e(280)
  + 5g(98) + 5f(418) = 1,291 cells over two rounds.

## 2. The rsync (workstation, from ~/foo/dQwen/_runs)

    # (a) the cells — additive, no --delete, tmp-excluded:
    rsync -avz --info=progress2 --exclude '*.tmp*' \
      antonxue@vista.tacc.utexas.edu:/home1/11079/antonxue/foo/dQwen/_runs/grid_v1/ \
      grid_v1_vista/
    # (b) the merge snapshot dir (sibling):
    rsync -avz \
      antonxue@vista.tacc.utexas.edu:/home1/11079/antonxue/foo/dQwen/_runs/20260817-161500-part5-merge \
      ./

    # verify (a): expect exactly 2,899 now / 2,900 after pickup
    find grid_v1_vista -name '*.jsonl' | wc -l
    # integrity: reproduce the snapshot (expect the SAME single INCOMPLETE)
    cd .. && python summarize.py --merge --store vista

Transfer size: ~1,290 new files (~2GB; s1024 cells carry full 1024-token
raw generations per doc). Everything already present no-ops.

### The one-cell pickup (after the cluster announces the drain)

Re-run rsync (a) verbatim — it moves exactly one new file
(`math500/dqwen3-1.7b-base-v3@step25000-swa__math500__standard-tau0.95__s1of2.jsonl`)
— then re-merge; the INCOMPLETE disappears. Affects only the control's
τ0.95 point on the math500 τ-curve; nothing else waits on it.

## 3. What part5 delivers (for the flip)

All 14 campaign rows now carry, at canvas 1024:
- **Full-canvas static ladder** `standard-static-s{32,64,128,256,512,1024}`
  on HumanEval + MBPP + MATH500 (5b) — both comparator conventions are
  ladder rungs (Dream@512, LLaDA@1024).
- **gsm8k @ s1024** (8-shot, 5c) — the flipped tables' gsm8k column.
- **Headline variants @ s1024** (5g): mbpp-plus / mbpp-fence /
  mbpp-plus-fence / humaneval-plus as real generations (the amendment).
- **Full-canvas τ sweep** `standard-tau{0.5..0.95}` on HE+MBPP+MATH500
  (5f) — measured n_forward on every sample record (the cost axis).
- **Block-side MATH500 backfill** (5e): the 10 non-s32 block decodes, so
  both regimes draw complete MATH500 curves.
- MMLU/AR/block grids: unchanged from the earlier campaigns.

## 4. Manuscript-relevant analysis (computed cluster-side, 08-17)

Full details in chat with Anton; headlines the paper agent should know:

1. **Standard-for-block swap: every ordering survives.** HE/gsm8k/mbpp/
   math500 rankings identical between regimes; comparators drift
   slightly UP in their native regime (llada HE 33.5 = published;
   coda +1.8). GSM8K concession unchanged (ours 70.1 vs 71-75).
2. **The τ claim strengthens dramatically**: under full canvas, τ0.9
   matches s1024 accuracy within ~±1pp across 19 measured rows at
   **0.13-0.32x cost (3-8x cheaper)** — vs the block-era "roughly half".
   llada τ0.9 HE/MBPP are IDENTICAL to s1024; llada τ0.95 math500 is
   -0.6 at 4.9x cheaper. One outlier: 4b@50k math500 τ0.9 BEATS s1024
   by +9.6.
3. **MATH500-as-decode-axis is regime-robust**: at s1024 ours leads all
   comparators (36.8 > 34.8 > 29.4 > 28.4), same ordering as block.
4. **⚠ Retention-figure carve-out**: under the standard ladder the
   matched-pair retention is 5-of-6 hybrid-favoring with HE@50k a 0.3pp
   near-tie (block was a clean 6/6), and the standard s32 edge is a
   universal collapse (0.6-1.8% for everyone) — the graceful-degradation
   texture lives in BLOCK. Recommendation delivered to Anton: flip
   tables + cost/frontier figures to standard; KEEP the retention figure
   on block.
5. Anton's s512-vs-1024 counterfactual: answer it from the ladder (s512
   vs s1024 columns) — if deltas are small outside MATH, it's a
   publishable "canvas beyond 512 changes little" finding.

## 5. Campaign ledger (for the record)

Grand total: **2,900 cells** across parts 1-5 (72 AR / 272 big table /
810 block acceleration / 455 decode-axis / 1,291 full-canvas). Round-2
part5: zero failures, zero wall-clips. History of note: the 08-16
scratch storms burned round 1 (recoverable by design — nothing lost),
the miniconda loss + rebuild (now on $WORK with a scratch symlink), and
the CoDA latent-test-gap fix (load_matrix now encodes both AR-aligned
families).

— the SLURM Claude. Rsync at will; the 2,900th cell announces itself.
