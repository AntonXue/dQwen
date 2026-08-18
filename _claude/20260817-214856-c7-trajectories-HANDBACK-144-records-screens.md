# C.7 trajectories HAND-BACK: 144/144 records + 6 screening pages, ready for picks

> 2026-08-17 ~21:49, workstation. Closes the generation half of the
> manuscript's `20260817-151600` request under the final design
> (`20260817-202121`). Everything below is on disk and pushed; Anton
> screens the PNGs, the manuscript agent renders the paper figure from
> the JSONL.

## Where (the request §9 reply)

`~/foo/dQwen/_runs/20260817-202121-c7-decode-trajectories/`
- `trajectories_<model>.jsonl` × 4 — one record per generation (the
  request's schema + `raw_generation`, `fence_cut`, `decode_config`).
- `screen_HumanEval_{0,3,8,31,50,56}.png` — screening pages, 4 model
  rows × 6 scheme columns, shared 0–128 axes, gray AR diagonal, block
  rules, per-panel forwards + red ✗ on fails.
- `log_*.txt` — per-lane run logs.
- τ = **0.9 fixed** (ruling; no calibration). **Stage 2 NOT run**
  (deferred per the design doc; same driver can add it any time).

## The run

144 generations = 4 models × 6 schemes × the whole 6-problem pool;
problem-major, one process per GPU (GPU 3 was memory-full; Dream-Coder
chained behind Dream on GPU 2). Zero failures, every in-driver invariant
held on every generation (s128 permutation, s64 exactly-2-per-forward,
block16 staircase monotone, τ progress floor). Wall ≈ 20 min.

## Grading note (the one mid-run fix — trajectories were NEVER affected)

At canvas 128, models close a correct solution with a markdown fence +
prose that the protocol's stop set (tuned on the 1024 canvas) never
sees; graded raw, correct solutions failed. Fix: grading-time
sanitization only — grade `prompt + text` truncated at the first fence
(` ``` `), full text stored, `fence_cut` flags the 23 affected records.
Decode behavior untouched (commit 2a50a1e; Dream was re-run under the
fixed grader for uniform records — greedy determinism reproduced
identical trajectories).

## Pass pattern (117/144)

| model | pass | note |
|---|--:|---|
| dream-coder-7b | 30/36 | all 6 fails are /0 (canvas-128 flip vs its 1024 receipt) |
| llada-8b | 29/36 | fails cluster in ×2-speed cells |
| dream-7b | 26/36 | /8 + /56 ×2-speed cells, as the receipts predicted |
| dqwen3.5-9b | 32/36 | scattered singles |

The canvas-1024 all-pass receipts flip for some cells at 128 — expected,
disclosed in the design doc, and honestly marked ✗ on the panels (a
failed generation's trajectory is still informative; nothing is hidden).

## What the screens already show (for the C.7 prose, illustrative only)

1. **The any-order claim is visible**: standard-mode panels scatter far
   off the AR diagonal; block panels are forced staircases (8 treads).
2. **Model personalities differ** — the free-order columns separate the
   families: LLaDA commits far-ahead positions early and jumps around;
   Dream/Dream-Coder stay strikingly NEAR-DIAGONAL even when fully
   unconstrained; the 9B sits between, with early low-position commits
   and moderate scatter. (If C.7 wants one comparative sentence, it is
   this — as illustration, not statistics.)
3. **τ panels show the adaptive texture**: vertical multi-commit runs,
   ~half the forwards of their static twins.
4. **The ×2 columns end at half-width** on the shared axis — the
   whitespace is the speedup, no extra ink.

## Next (manuscript side)

Anton picks page layout from the screens (all 6 problems fit 3 pages at
two per page); the manuscript agent renders `figures/decode-behaviour.tex`
from the JSONL in the style kit; C.7 body cites: canvas 128 + block 16
(legibility; measured protocol is 1024/block-32 — mechanism identical),
grading fence-cut, panels illustrative.
