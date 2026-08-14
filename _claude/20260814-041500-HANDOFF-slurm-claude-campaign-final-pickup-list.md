# HANDOFF from the SLURM Claude: campaign final state + the workstation pickup list

> 2026-08-14 ~04:15, written by the CLUSTER-side session (the one that has
> been driving Vista since bring-up on 08-13: tiled launch, 14 jobs, the
> CoDA diagnosis + fix + rerun). Everything below is the state you can
> rely on when you pull + rsync; exact commands included. Parent docs:
> 20260813-005207 (bring-up + manifests), 20260813-034500 (tiled launch),
> 20260814-011801 (your DELEGATION brief — executed), 20260814-023800
> (root cause + fix + rerun, incl. the restamped row).

## 1. State of the cluster store — FINAL, verified

- `_runs/grid_v1/`: **1,154/1,154 cells**, final `summarize.py --merge`:
  **zero INCOMPLETE across all 264 groups** (exit 0).
- The **19 CoDA cells are REFRESHED** (deleted + rerun post-fix on
  2026-08-14 ~03:00-03:50): 10 code shards, 6 gsm8k stripes, 2 math500
  stripes, 1 mc-nelbo mmlu. Same filenames as the garbage versions —
  your rsync overwrites them in place.
- Merge snapshot for convenience + as a transfer-integrity reference:
  `_runs/20260814-040500-campaign-merge/merged-1154.txt` (4-line
  provenance header + 264 merged-group lines, stamped with repo rev).
- Cluster store totals 381M on $HOME (du-inflated by 64k blocks; ~278M
  actual bytes, as you saw).

## 2. Code you get on pull (both already on origin/main)

- **0c95a30 — the CoDA fix, two parts:**
  1. `models/coda.py`: the post-load inv_freq recompute is now
     UNCONDITIONAL. Root cause was the `isfinite` trigger: torch
     2.10/aarch64 materialises meta buffers as ZEROS (finite!), so the
     repair declined to repair and the model ran position-blind.
  2. `models/adapter.py::assert_finite_rope`: hardened with the
     `inv_freq[0] == theta**0 == 1.0` invariant. ⚠ NOTE FOR YOU: this
     gate now runs on EVERY family load INCLUDING on the workstation.
     It should be a no-op there (your loads are healthy) — but if any
     family trips it under torch 2.7/x86, that is a REAL find, not a
     false positive. The invariant is exact for every rope variant.
- **254f9c3 / this doc** — records. Also earlier tonight: 39ea4cb
  (plan_mopup.py), cbe3ad8 (sbatch_coda_probe.sh — the diagnostic job
  template; keep or delete at your discretion, it did its job).

## 3. The restamped CoDA row (cluster GATE, canvas 1024, block32-s32)

| mmlu | gsm8k | math500(mv) | he | he+ | mbpp | mbpp+ | fence | fence+ |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 26.21 | 3.18 | 3.20 | **23.17** | 17.07 | 35.00 | 42.33 | 36.80 | 44.18 |

Acceptance vs your brief: HE **exactly** the PROBE-era 23.17 (canvas-
invariant, same mechanism as LLaDA's 32.93); mbpp 35 in their published
range; nothing 0.00; generations coherent; goldens matched digit-for-
digit post-fix (inv_freq 5.150445, lm_head, logit stats, argmax).

⚠ **ONE FLAG NEEDING YOUR EYES: mmlu 26.21** — below your "expect
30s-40s" band (chance=25; the garbage row scored 24.15). The load is
goldens-exact and the generative row proves the loaded model works, so
this is NOT the silent class. ACTION: check whether ANY PROBE-era CoDA
mmlu measurement exists on the workstation to compare. If none, the
paper takes 26.21 as the re-measurement with a one-line disclosure (a
2B code-specialist near chance on 5-shot MMLU is believable; Salesforce
never evaluates knowledge benchmarks).

## 4. Your exact pickup steps

    cd ~/foo/dQwen && git pull
    cd _runs
    # (a) refreshed cells: cluster grid_v1/ -> YOUR grid_v1_vista/
    rsync -avz --info=progress2 \
      antonxue@vista.tacc.utexas.edu:/home1/11079/antonxue/foo/dQwen/_runs/grid_v1/ \
      grid_v1_vista/
    # (b) merge snapshot
    rsync -avz \
      antonxue@vista.tacc.utexas.edu:/home1/11079/antonxue/foo/dQwen/_runs/20260814-040500-campaign-merge \
      ./
    # (c) verify: 1154 files, and your merge reproduces the snapshot
    find grid_v1_vista -name '*.jsonl' | wc -l          # -> 1154
    cd .. && python summarize.py --merge --store vista  # == merged-1154.txt lines

Only the 19 CoDA files + the snapshot dir actually transfer; rsync
no-ops the other 1,135.

## 5. Open threads, ranked (my read from this side)

1. **CoDA mmlu prior check** (S3 flag) — cheap, gates the row's
   disclosure wording.
2. **Manuscript harvest** — your own listed next step: benchmark-grid
   refresh from vista (family/comparators/control/AR band incl. the new
   gsm8k denominators — NB 2B 53.60 -> 58.98 moves the S4.1 fraction),
   math500 column + disclosures, part-3 acceleration figures.
3. **CoDA native-protocol cell** — now that CoDA is actually FIXED on
   the cluster, the long-parked "their published 768@1-step HumanEval"
   comparison cell is finally MEANINGFUL to run if the paper wants the
   armor (needs a BENCH row, e.g. humaneval-g768; the engine supports
   standard-static at any canvas). Anton's call; one cell, cheap now.
4. **4B mbpp-fence anomaly** (−16pp vs scaffold, only model affected) —
   per-sample read in the vista store; extraction/formatting suspect.
5. **Decontamination scan** — still MANDATORY-not-started (paper side).
6. Protocol appendix collects tonight's disclosures: math500 subset +
   stderr ~2.2pp, MC-NELBO asymmetry, AR canvas fix direction, CoDA
   truncation note (canvas-junk removed at grading), hardware statement
   (all GATE cells one generation: GH200).

## 6. Cluster housekeeping (so nothing surprises you)

- The idev session (job 907357) that hosted bring-up/probes/diagnosis
  ends ~17:01 today; nothing depends on it surviving. All queue work is
  drained; user job count is zero besides it.
- The overnight Lustre stalls were THIS NODE's client wedging after
  heavy IO (batch nodes unaffected — the probe job proved it); healed
  ~02:00. If a future session hits mystery hangs: fresh env-python
  startups hanging with zero CPU is the signature; get a fresh node.
- HF cache (1.8T) on $SCRATCH is purge-exposed and disposable
  (`run.py --prewarm` rebuilds). The mop-up tiles files
  (`*.mopup.tiles.json`) are stale-but-harmless (their cells are
  complete; resubmission would no-op in seconds).
- Campaign accounting: 14 jobs + 2 rerun jobs + 1 probe job; ONE
  transient cell failure (node bus error), ONE wall-clip (CoDA, from
  the then-unknown canvas-filling), both recovered by design. Actual
  usage landed near the 542 node-h conservative plan.

— the SLURM Claude, signing off a clean store.
