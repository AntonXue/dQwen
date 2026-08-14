# Protocol alignment re-verified against primary sources; rerun manifests cut

> 2026-08-14, workstation. Anton asked for a from-scratch confirmation that
> our shots/grading align with Qwen3 and LLaDA BEFORE the 4-shot grid ships
> to the cluster. Every cell below was re-read from the primary source this
> session — `~/foo/LLaDA/{EVAL.md,eval_llada.sh}`, `~/foo/Dream/eval/
> eval_dream_gen.sh`, the Qwen3 report (arXiv 2505.09388 §3.3 + Table 8) —
> not from our earlier docs. Verdict: **aligned; grid is GO.** Three new
> facts strengthen the case beyond `20260814-121922` §3.

## The alignment table (all entries primary-source-verified today)

| bench | Qwen3 report | LLaDA paper/internal | LLaDA's own lm-eval repro | Dream script | ours |
|---|---|---|---|---|---|
| MMLU | 5-shot | 5-shot, likelihood | `mmlu` 5-shot, mc_num=1 | — | 5-shot ✅ |
| GSM8K | **4-shot CoT** | **4-shot** | stock `gsm8k` = 5-shot, "####" extraction | `gsm8k_cot` **8-shot** | 4-shot CoT ✅ |
| MATH | 4-shot CoT, full | 4-shot, full | `minerva_math` = 4-shot | 4-shot | 4-shot, MATH500 (disclosed) ✅ |
| MBPP | 3-shot | "4-shot" in paper | stock `mbpp` = **3-shot** | 3-shot | 3-shot ✅ |
| HumanEval | 0-shot (EvalPlus) | 0-shot | `humaneval` = 0-shot | 0-shot, temp 0.2 | 0-shot ✅ |

Qwen3 Table 8 anchors re-verified verbatim: 0.6B-Base 52.81/59.59/32.44/36.60
(MMLU/GSM8K/MATH/MBPP), 1.7B-Base 62.63/75.44/43.50/55.40. Phase-1 gate
unchanged (>3pp = STOP).

## Three new facts (beyond the PLAN doc)

1. **LLaDA's lm-eval reproduction pins `lm_eval==0.4.8` — OUR exact pin**
   (`eval_llada.sh` line 1). Their published lm-eval column (GSM8K 70.3,
   Math 31.4, HE 35.4, MBPP 40.0 at gen=steps=block=1024) shares our
   grader/extraction code path verbatim. The credibility paragraph can say
   something stronger than "different graders agree": for LLaDA there IS a
   published same-harness-version reference column.
2. **The MBPP keep-3 ruling is stronger than "no consensus":** LLaDA's own
   lm-eval commands pass no `--num_fewshot`, so their repro ran stock
   3-shot — matching us and Qwen3. Only their internal number says 4.
3. **Dream is the 8-shot house** (`gsm8k_cot`, 8-shot, temp 0) — so our
   archived 8-shot record IS Dream's convention, making the protocol-
   sensitivity framing concrete: the archive measures every model under
   Dream's shot count, the new grid under Qwen3/LLaDA's. Also: Dream's
   code tasks sample at temp 0.2 / top-p 0.95 (not greedy) — reinforces
   the existing "unified protocol undersells native samplers" tripwire.

Residual, unchanged from §5.3 of the PLAN: 4-shot aligns the COUNT, not the
byte-level prompt (Qwen3's exemplars are internal; LLaDA's library is
unreleased; their lm-eval repro even used a different task at 5-shot and
landed within 0.4pp of internal — encouraging for prompt-insensitivity).
The Phase-1 anchors are the end-to-end test of prompt+extraction+grading.

## Manifests cut (deterministic; `make_manifests.py` regenerates all six)

| file | cells | what |
|---|--:|---|
| `gsm8k-rerun-phase1.jsonl` | 16 | 2 anchors + 8 bare AR at gsm8k; anchor extras (mmlu/mbpp/humaneval ×2) checkable against Table 8 |
| `gsm8k-rerun-phase2.jsonl` | 84 | 14 big-table rows × 6 stripes, block32-static-s32 |
| `gsm8k-rerun-phase3.jsonl` | 540 | 9 accel rows × 10 remaining decodes × 6 stripes |

634 gsm8k cells = the 632 archived + 2 anchors. part1/2/3 regenerate
byte-identical (checked). Shards stay 6; re-sharding 4B/9B to 12 remains
the cluster's call (regenerate with a DLM_SHARDS override if taken).
Cluster sequence: pull → archive its `grid_v1/gsm8k/` → `--prewarm`
phase1 (the two -Base anchors are new downloads) → phase1 + gate →
phase2 (manuscript unblocks) → phase3.
