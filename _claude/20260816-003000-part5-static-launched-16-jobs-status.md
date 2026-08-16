# Part5 STATIC LAUNCHED: 16 jobs / 867 cells in flight; gates green so far; τ verdict pending

> 2026-08-16 ~00:30 (times in this doc are cluster-side session clock),
> the SLURM Claude. Status broadcast for the workstation + manuscript
> Claudes: what is running, what was ruled, what changed in the specs,
> and what to expect next. Parent: HANDOFF 20260815-214527 (as amended);
> EVAL-REQUEST 20260815-211016; RULING 20260816-000500.

## 1. What is in the queue RIGHT NOW (verified, zero duplicates)

16 jobs = 5c×4 + 5g×4 + 5b×5 + 5e×3, sacct-audited one submission per
(manifest, job#). **867 static cells**: 5b ladder 405, 5c gsm8k-s1024
84, 5e math500 block backfill 280, 5g headline variants 98. ~1,537
node-h, 16h walls (Anton; worst lane est 10.0h), lanes merged to 10h
budgets to fit the 20-job QOS cap, **LPT everywhere** (heaviest lanes =
job 0; heaviest cells first within every lane; comparator-s1024 monsters
started first). The tiler now prices from measured store anchors and
takes a per-run budget arg.

## 2. Ruling recap (supersedes parts of the written request — final)

- **GSM8K is 8-shot, PERIOD** (RULING 20260816-000500): part5d/
  gsm8k-4shot (176 cells) DELETED after two reversals. Campaign B of the
  request does not exist. ⚠ Manuscript consequence: no per-comparator
  shot matching; uniform 8-shot with one disclosure line.
- Anton's compromise stands: MATH500 carries the decode axes; gsm8k
  keeps only the s1024 headline (5c).
- The §3b amendment + HE+-as-grid-cells resolution are both in flight
  as 5g (98 cells).

## 3. Gate verdicts so far (the request's §4)

| gate | verdict |
|---|---|
| 4a sN respects N | **PASS ×4** — 2B and LLaDA at s32/s128: n_forward EXACTLY N on every doc. s32 accuracy collapses (2B 0.61, LLaDA 1.83) but outputs are degraded-LEXICAL, s128 recovers (2B 7.93), s1024 matches known-good: a REAL capability cliff = the ladder's left edge, not wiring. |
| 4b standard-τ exists | ⏳ two cells queued on the idev GPU behind LLaDA's s128 anchor; verdict tonight. 5f (418 cells) held behind it, tiled only after its measured n_forward prices it. |
| 4c s1024 measured-vs-nominal | **ANSWERED**: stop check is block-granular, standard = one block ⇒ structurally exact-N (no early exit). The "block is 5–11x cheaper" denominator is SAFE. Empirically confirmed by 4a's exact-N at 32/128. |
| 4d shot coexistence | MOOT (Campaign B deleted). |

## 4. Extra assurance run tonight (Anton: "super make sure")

- Both bitwise parity gates RE-PASS on this hardware (sampler
  token-for-token vs LLaDA native; MC-NELBO exact vs their reference).
- All three previously-untested bench×decode combos smoked clean on
  0.8B stripes (junk scrubbed after): block-τ×math500 21s,
  standard-s1024×gsm8k 473s, standard-s1024×mbpp-plus 265s.
- Full spec validation (request → compromise → amendments → rulings)
  done requirement-by-requirement: everything queued + 5f-if-gated
  covers every obligation; the only deviations are the two
  Anton-authorized ones above.
- Disclosure: a ~17-min GPU co-residency race (insurance chain's
  GPU-free predicate fired between gate-chain cells) — bitwise/count
  verdicts unaffected; wall-clocks in that window polluted.

## 5. What the workstation should expect

- NOTHING to pull/rsync yet — cells land over the next ~10-16h; the
  heavy comparator-s1024 lanes were deliberately started first.
- At drain: the usual close-out (pending sweep, merge with zero
  INCOMPLETE expected, counts, results doc) + rsync line. Store target:
  1,612 current + 864 remaining static (+2 τ gates + 418 5f if gated
  in).
- Report-back §8 items land with the close-out; Dream-under-full-canvas
  qualitative watch is on the list.
- The dlm1b training jobs occasionally seen in squeue are Anton's
  separate ~/foo/DLM1B work, not eval.

— the SLURM Claude. Fingers crossed, per management.
