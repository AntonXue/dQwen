# Part 4 COMPLETE: 455/455 decode-axis cells, zero failures — both §4 figures now have their material

> 2026-08-15 ~12:25, the SLURM Claude. Executes HANDOFF 20260815-024044's
> checklist §6 item by item. Store: `_runs/grid_v1/` = **1,609 cells**.
> Ready for the workstation's additive rsync; both `fig:decode-retention`
> and `fig:decode-frontier` are drawable the moment it lands.

## Checklist (handoff §6), all green

1. **4a verdict table**: recorded + pushed overnight
   (`20260815-034500-part4a-tau-gate-all-four-pass.md`) — **all four
   comparators PASS**, no special τ handling anywhere (the unified commit
   rule ran verbatim; the protocol appendix needs only that sentence).
   Headline: τ0.9 = 2.3–2.7× fewer forwards at ≤0.6pp accuracy change;
   LLaDA and Dream-Coder pass@1 IDENTICAL to their s32 anchors.
2. **`--pending` = 0** on all five manifests (4/179/207/60/5).
3. **Merge: zero INCOMPLETE, zero mixed-protocol** — 364 groups (+100
   new: +50 gsm8k, +50 mbpp stripe groups, exactly as predicted).
   Snapshot: `_runs/20260815-122000-part4-merge/merged-1609.txt`.
4. **File counts exact**: humaneval 173 (+55), gsm8k 932 (+300),
   mbpp 316 (+100) = 455 new cells.
5. This doc. 6. Anton: ready to pull (see §Handback).

## The run

- 5 jobs (13/16/14/7/5 nodes, 8h walls), submitted ~03:25, drained
  ~12:20; **zero cell failures, zero wall-clips** the entire night.
- Gate discipline held: the 4-cell τ-gate finished on the idev GPU at
  03:42 (sequential, one-job-per-GPU per Anton) BEFORE any queued cell
  ran — nothing executed on an unproven path.
- Calibration note for the next planner pass: measured-anchor tiling
  under-priced comparator τ-gsm8k lanes ~1.4× (jobs ran ~4.5h vs ~3.2h
  planned; the flat τ=0.6 factor is optimistic on gsm8k). Harmless under
  8h walls; if it ever matters, fit per-benchmark τ factors from the
  now-on-disk sweep.

## Handback (workstation)

    git pull        # this doc + the gate verdicts
    rsync -avz --info=progress2 --exclude '*.tmp*' \
      antonxue@vista.tacc.utexas.edu:/home1/11079/antonxue/foo/dQwen/_runs/grid_v1/ \
      _runs/grid_v1_vista/
    # verify: find _runs/grid_v1_vista -name '*.jsonl' | wc -l  -> 1609
    # then scripts/decode_retention.py + decode_frontier.py take over

Also worth pulling: `_runs/20260815-122000-part4-merge/merged-1609.txt`
(same rsync pattern, sibling dir) — reproducing it locally via
`summarize.py --merge --store vista` is the transfer-integrity check.

## Notes to relay to the manuscript side (from the handoff §7 + the run)

- τ grid draws 0.5–0.95 complete for ALL nine family rows + control@50k
  + all four comparators; fast-end saturation is a finding to narrate.
- HE+ under acceleration: CPU-side regrade of saved generations — no
  cells needed, ever.
- 4f ceilings landed for all five rows; LLaDA's s1024 cell doubles as
  their published-decode reproduction anchor.
- Deliberately not run (unchanged): math500/mmlu/fence acceleration,
  CoDA native-g768 (still parked), τ beyond 0.95, non-32 blocks.

— the SLURM Claude. Store whole at 1,609; queue empty; nothing pending
on the cluster side.
