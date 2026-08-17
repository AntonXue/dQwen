# Campaign store VERIFIED workstation-side at 2,900/2,900; C.7 trajectory work launched

> 2026-08-17 ~17:15, workstation. Closes the pickup loop on the cluster's
> `20260817-162000` handoff + `20260817-171500` completion doc, and records
> the C.7 kickoff (manuscript EVAL-REQUEST
> `VistaCoder_Technical_Report/_claude/20260817-151600`, Anton's go this
> session).

## 1. The pickup, verified end-to-end (this machine, independent of the cluster)

Two-stage rsync (Anton ran both; MFA'd): the 2,899-cell sync on 08-16,
then today's exact-two-file delta (the straggler stripe + the final merge
snapshot). Verification at each stage, final state:

- `_runs/grid_v1_vista/` = **2,900 jsonl cells exactly**, zero tmp files.
- `summarize.py --store grid_v1_vista --merge`: exit 0, **896 MERGED
  groups, zero INCOMPLETE, zero MIXED-PROTOCOL** — and the MERGED/
  INCOMPLETE line set is **line-for-line identical** to the cluster's own
  `merged-2900.txt` (and was identical to `merged-2899.txt` at the
  intermediate stage). Both snapshots live in
  `_runs/20260817-161500-part5-merge/`.
- Spot checks on the new cell classes, all clean:
  - ladder `standard-static-s512`: every sample exactly 512 forwards (the
    no-early-exit arithmetic-cost property, present in the shipped data);
  - `gsm8k standard-static-s1024`: `meta.bench.shots = 8` (the 8-shot
    ruling held through the whole campaign), 1024 forwards exactly;
  - full-canvas `standard-tau0.9` (9B): 127 distinct measured `n_forward`
    values (~55–68 typical) — the real early-exit signal behind the
    "τ0.9 ≈ s1024 accuracy at 3–8× cheaper" finding.
  - The 2,900th group merges sane: `control@25k math500 standard-tau0.95`
    = 16.20 exact_match over 500 docs; family column around it reads
    5.80 / 13.00 / 28.20 / 41.60 (0.8B→9B @25k).

**`grid_v1_vista` is the release dataset: complete, frozen (cluster-side
too, per their completion doc), and independently verified.** The
manuscript flip (tables + cost/frontier figures → standard; retention
figure stays on block, per the cluster's carve-out analysis) harvests
from here with nothing pending.

## 2. C.7 decode-trajectory work: GO, running locally in parallel

The plan as agreed with Anton (responding to the manuscript request; its
§7 defaults accepted, two amendments):

1. **Instrumentation**: `GenOutput` gains `commit_step[i]` (the step at
   which position i was first committed; −1 never) — one tensor write per
   step in `generate()`, recorded for both commit policies.
2. **Driver**: `trajectories.py` at repo root (records are not cells, so
   not a run.py verb): renders HumanEval prompts through the same task
   machinery, runs the 2×2 scheme grid at `gen_length=128`, grades
   `passed` via the in-repo code_eval helpers, emits the request's JSONL
   (+ raw canvas text and stop-cut index, so the renderer can shade
   post-stop commits) and screening PNGs into a stamped launch dir under
   `_runs/`.
3. **Pool**: ~12 HumanEval problems selected on the request's pre-stated
   criteria, with receipts from data already on disk (solved-by-most =
   GATE-store per-doc metrics at block32-s32; canonical-solution token
   lengths; structure tags).
4. **τ pick** (amendment): the "nearest 32 forwards" rule cannot be
   priced from canvas-1024 sweep data, so τ ∈ {0.8, 0.9, 0.95} is
   MEASURED on the 128-canvas pool and the pick made by the stated rule;
   only the pick is screened (no extra panels for Anton).
5. **Stage 2** (recommended, accepted as cheap): the leftmost-masked
   fraction over all 164 problems × full-canvas × 5 models (~105k
   forwards, local, ~1–2 h). NB the request's `eval.data.humaneval_50r()`
   fallback is ADLMC-era code that does not exist here; full-164 is cheap
   enough that no subset is needed.
6. **Where**: local GPUs (demonstration appendix; keeps Vista frozen;
   one-line hardware disclosure). Models: the request's five, incl. the
   control. CoDA excluded per the request — but its §8 note lands here:
   `models/coda.py`'s comment claiming the tie was "harmless under 4.x"
   is WRONG (measured: the tie fires on transformers 4.57.1) — comment
   fix queued with this work.

Status at writing: instrumentation in progress. Hand-back doc with the
JSONL path, chosen τ, and Stage 2 results follows when the data exists.
