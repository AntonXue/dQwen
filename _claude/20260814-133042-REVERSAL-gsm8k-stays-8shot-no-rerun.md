# REVERSAL: GSM8K stays 8-shot (stock default). The 4-shot rerun is CANCELLED.

> 2026-08-14 ~13:30, workstation. Anton's ruling, reversing the same-day
> 4-shot plan after a full discussion of the trade: **"We'll keep it this
> way since it's the stock default. It's defendable."** The paper
> discloses; no cluster work exists.

## ⚠ SLURM Claude: STAND DOWN

If you arrived via `20260814-130859-HANDOFF-gsm8k-4shot-rerun-…`: **do not
execute it.** Do not archive your gsm8k cells, do not prewarm the -Base
anchors, do not run any phase. That handoff, the PLAN's §6 (20260814-121922),
and the phase manifests are all superseded by this doc. Your store is
final as it stands — the campaign closed at your `20260814-041500` handoff
and stays closed. There is NO pending cluster work.

## The ruling and its rationale

GSM8K remains `gsm8k_cot` at lm-eval's **stock 8-shot default** (the
canonical Wei et al. CoT exemplars — also Dream's exact published
convention). Qwen3 and LLaDA declare 4-shot; we are a disclosed deviation
rather than a re-measured alignment. What the decision weighs: stock-
default defensibility + zero compute + the campaign store staying whole,
against the outlier status on the normalized math axis and the loss of the
Phase-1 published-number calibration gate. Anton took the first side;
settled.

## What was reverted (commit-level: this commit; state = pre-episode)

- `run.py` BENCH gsm8k `shots=4` → **8** (authoritative value).
- `benchmark_specs.py` `num_fewshot` → 8, `metadata.version` → 3.0,
  section comment records the episode.
- `tests/task_freeze_gate.py` SHOTS → 8. Gate re-run: **PASSED**
  (byte-identical to pinned stock, same as before the episode).
- `manifests/`: the gsm8k-rerun-phase{1,2,3}.jsonl files DELETED and their
  generation removed from make_manifests.py; part1/2/3 regenerate
  byte-identical (72/272/810).
- `_runs/grid_v1_vista_gsm8k_8shot/` dissolved: all 632 gsm8k cells moved
  BACK into `_runs/grid_v1_vista/gsm8k/`; store re-verified whole —
  1,154 files, `--merge` = 264 MERGED / 0 INCOMPLETE / 0 MIXED-PROTOCOL.
- `_runs/README.md` stores table restored.

## What SURVIVES the reversal (do not re-litigate these)

1. **The MATH500 metric findings and rulings** (PLAN §2, §5.2): column
   reports stock `exact_match`; the MATH500 AR normalization stays
   DROPPED (broken denominators are a protocol mismatch, not metric
   choice). Unaffected by shot counts.
2. **The harness-credibility correction** (PLAN §3.1): only our MMLU
   agreement with LLaDA is a same-method reproduction; the generation-
   benchmark agreements are cross-grader. §4's paragraph must be fixed.
   Related upgrade from 20260814-125102: LLaDA's own lm-eval repro pins
   `lm_eval==0.4.8` (our exact pin), so for LLaDA there IS a published
   same-harness-version reference column.
3. **The `summarize.py` MIXED-PROTOCOL merge guard** — kept as general
   protection; `meta.bench.shots` is recorded in every cell either way.
4. **The protocol-alignment table** (20260814-125102) — now feeds the
   DISCLOSURE instead of a rerun. Dream = 8-shot (we match exactly);
   Qwen3/LLaDA = 4-shot (disclosed deviation); MBPP 3-shot and
   HumanEval 0-shot and MMLU 5-shot match everyone; MATH500 4-shot
   matches the convention on a disclosed subset.

## Manuscript obligations created/confirmed by this ruling

- Protocol appendix line: "GSM8K: 8-shot chain-of-thought (`gsm8k_cot`,
  the lm-eval default and Dream's convention); Qwen3 and LLaDA report
  4-shot. LLaDA's own lm-eval reproduction (5-shot stock `gsm8k`) lands
  within 0.4pp of their internal 4-shot number, so shot-count sensitivity
  is small for well-behaved models."
- The AR-counterpart GSM8K column is a same-protocol re-measurement, not
  reconcilable against Qwen-published numbers (those are 4-shot; and the
  bare post-trained repos have no published base numbers anyway).
- §4 credibility paragraph fix (item 2 above) still owed.

## Net state

The campaign store is exactly what it was at the cluster's 041500 handoff:
1,154 cells, final, publication-ready. Zero reruns pending anywhere. The
open queue returns to: manuscript harvest (now fully unblocked — nothing
is waiting on any eval), the three unported comparator rows (Anton's
port-or-cut call), v1-mixture prose decision, 4B mbpp-fence per-sample
read, CoDA native-protocol cell (optional), decontamination scan.
