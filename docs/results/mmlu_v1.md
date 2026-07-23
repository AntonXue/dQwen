# MMLU v1 — validated numbers through the dqeval harness

> 2026-07-23. All runs: MMLU 57-subject, **5-shot, cfg=0, bs=1**, transformers 5.13.
> DLM/diffusion models: mc_num=1 (exact single-token MASK-slot marginal) via the
> dqeval wrapper + ported adapters. AR baselines: **stock lm-eval `hf`** (zero dqeval
> code). Per-subject dumps in gitignored `_runs/`.

## 1. Reproduction gate — the harness matches published DLM numbers

| model | reproduced | published | diff |
|---|---|---|---|
| LLaDA-8B-Base | **65.85** | 65.9 | **−0.05** |
| Dream-v0-Base-7B | **71.64** | ~70 | +1.6 |

LLaDA runs MMLU at cfg=0, which our common protocol matches exactly — no decode
difference to hide behind, and we land on the number. Dream additionally validates
the logit **shift** end-to-end (its adapter applies `cat([l[:,:1], l[:,:-1]])`; a
wrong shift would collapse it toward chance). Both are published *plain-MMLU base*
numbers, so both carry the "reproduced ≈ reported" claim.

## 2. AR-vs-diffusion twin sweep (same Qwen3.5 substrate, all four sizes)

| size | Qwen3.5-Base (AR) | dQwen3.5-Base (diffusion) | gap | diffusion state |
|---|---|---|---|---|
| 0.8B | 48.36 | 26.36 | 22.00 | flat / chance |
| 2B | 53.97 | 27.80 | 26.17 | flat / chance |
| 4B | 73.02 | 41.81 | 31.21 | structured / retained |
| 9B | 71.93 | **71.87** | **0.06** | **matches AR twin** |

**Headline: at 9B the diffusion conversion is essentially lossless on MMLU** —
71.87 vs its AR twin's 71.93 (gap 0.06pp), and competitive with the best DLM
comparators (Dream 71.6, LLaDA 65.9).

**The gap is NON-monotonic.** It *widens* through the mid sizes (22 → 26 → 31) as the
AR baseline sprints ahead, then collapses to ~0 at 9B. So the story is a sharp
retention threshold between 4B and 9B, not a smooth catch-up. The small members
(0.8B/2B) sit at chance, FLAT across all four subgroups — the eval correctly reports
genuinely-at-chance behaviour (a scoring bug would scatter, not flatten). 4B is the
first size with real subject discrimination; 9B recovers full AR-level knowledge.

### Caveat to verify before publication
The AR baseline is itself non-monotonic: **Qwen3.5-4B (73.02) ≥ Qwen3.5-9B (71.93)**.
A 9B scoring at/below a 4B on MMLU is unusual. The 9B twin's subject profile is
coherent (STEM 67.9 / hum 69.9 / social 76.7 / other 74.4 — not a broken load), and
Qwen3.5-4B is reportedly a standout size, so it may be real — but re-run the 9B AR
baseline to confirm before it goes in a table.

## 3. On a "reported" column for Qwen3.5 — there isn't one

Qwen3.5 does **not** publish plain 5-shot MMLU for its base models. What the release
reports (checked 2026-07-23):
- Qwen3.5-9B **instruct**: MMLU-**Pro** 82.5, MMLU-**Redux** 91.1 — no plain MMLU.
- Base model cards carry no benchmark tables; no per-size base numbers are published.

MMLU-Pro/-Redux are harder variants and instruct-only, so they are not comparable to
our plain-MMLU base reproductions — forcing them into a "reported" column would be a
protocol mismatch. **For Qwen3.5, our reproduced values ARE the reference.** The
"reproduced ≈ reported" validation lives on the DLM comparators (§1), which do have
published plain-MMLU base numbers; Qwen3.5 cannot supply it.
Sources: huggingface.co/Qwen/Qwen3.5-9B (MMLU-Pro 82.5 / Redux 91.1, instruct);
huggingface.co/Qwen/Qwen3.5-9B-Base (no table).

## Protocol notes
- **cfg=0 (common disclosed protocol)** — LLaDA tunes per-task CFG (0.5–2.0 on 5 MC
  tasks); we drop it. MMLU is unaffected (LLaDA runs MMLU at cfg=0 too), which is why
  the LLaDA MMLU match is near-exact.
- **mc_num=1 exact** — MMLU continuations are single-token letters, verified
  single-token in all four tokenizers.
- **Fast path** — one forward per question (shared masked canvas across the 4
  options), verified bitwise identical to per-option scoring. Full MMLU ~30 min/model
  at bs=1 for the DLMs; AR baselines ~5 min (KV cache + batch 16).

## 4. Qwen3-adapted family (dQwen3) — twin sweep WITH a published reference

Unlike Qwen3.5, the **Qwen3 tech report (arXiv:2505.09388, Table 8) publishes plain
5-shot base MMLU**, so this family gets a real reported column.

| size | reported (Qwen3 report) | Qwen3-Base AR (reproduced) | dQwen3-Base (diffusion) | repro − reported |
|---|---|---|---|---|
| 0.6B | 52.81 | 52.51 | 26.64 (chance) | **−0.30** |
| 1.7B | 62.63 | 62.57 | 36.55 (retained) | **−0.06** |

- **AR reproductions match the official numbers to ≤0.30pp** — a proper
  reproduced-vs-reported validation (the Qwen3.5 side couldn't supply one). Third
  published anchor overall, alongside LLaDA (−0.05) and Dream (~+2).
- **dQwen3-1.7B retained (36.6, structured); dQwen3-0.6B at chance (26.6, flat).**
  Note the Qwen3-based 1.7B retains where the Qwen3.5-based 2B did not (27.8) —
  earlier retention onset on the Qwen3 substrate.
- Qwen3 report also gives MMLU-Pro / MMLU-Redux for these sizes (0.6B: 24.74 / 51.26;
  1.7B: 36.76 / 61.66) — the reported targets if those tasks are added later.
