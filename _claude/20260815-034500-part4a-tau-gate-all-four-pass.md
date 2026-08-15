# Part4a τ-gate: ALL FOUR comparators PASS — 4b-4f fully licensed

> 2026-08-15 03:45, cluster (idev GPU, sequential per Anton's one-job-per-
> GPU rule). Gate per HANDOFF 20260815-024044 §3; references from the
> campaign store. The four smoke cells are ordinary grid_v1 cells and
> fast-skip inside 4b/4c.

| model | samples | τ0.9 fwd mean | s32 ref | max | τ0.9 pass@1 | s32 ref | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| coda-1.7b-base | 164/164 | 80.2 | 218.7 | 826 | 22.56 | 23.17 | **PASS** |
| llada-8b-base | 164/164 | 48.3 | 115.7 | 190 | 32.93 | 32.93 | **PASS** |
| dream-7b-base | 164/164 | 42.4 | 91.3 | 268 | 43.29 | 43.90 | **PASS** |
| dream-coder-7b-base | 164/164 | 43.3 | 91.3 | 140 | 67.07 | 67.07 | **PASS** |

- No special τ handling anywhere — the unified commit rule ran verbatim
  on all four families. Nothing for the protocol appendix beyond that
  sentence.
- Headline for the manuscript side: τ0.9 costs 2.3-2.7x FEWER forwards
  than the sequential anchor at zero-to-negligible accuracy change on
  every comparator — LLaDA and Dream-Coder land pass@1 IDENTICAL to
  their s32 cells. The architecture-independence of threshold decoding
  (§4.5's claim) now has comparator-side evidence before the sweeps
  even run.
- Fleet state at gate completion: all five part4 jobs still PENDING
  (gate beat the queue, as expected at 3am) — no cell ran ungated.
- Results doc with merge + counts follows when the fleet drains.
