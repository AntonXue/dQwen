# PLAN: re-run GSM8K at 4-shot — and the metric/protocol audit that produced the decision

> 2026-08-14, workstation. Written for the cluster agent to pick up. §1–§4
> are the audit (findings, with the evidence); §5 is the decision set; §6 is
> the executable plan with cell counts and acceptance gates. If you only read
> one section before launching, read §6 — but §5.3's caveat about what
> "4-shot" does and does not buy should travel with any claim about the
> resulting numbers.
>
> Prior art: `20260814-041500-HANDOFF-…` (campaign final state),
> `20260814-025900-…` (the summarize.py column bug, already fixed).

---

## 0. TL;DR

- **GSM8K moves 8-shot → 4-shot and gets re-run: 632 cells.** Both of our
  reference protocols (Qwen3 tech report, LLaDA) use 4-shot; we were the
  outlier, on what is about to become the manuscript's ONLY normalized math
  axis.
- **MBPP stays 3-shot** (Anton). Qwen3 says 3, LLaDA says 4 — no consensus to
  violate, and 3 matches the family we adapted from.
- **MATH500 stays** (not full MATH). Cost decision, already disclosed.
- **MATH500's metric is settled as stock `exact_match`**, and its *AR
  normalization* is dropped — the AR denominators are broken by a protocol
  mismatch, not by metric choice (§2).
- Phase 1 of the rerun is a **validation gate against published numbers** that
  we have never had for a generation benchmark. If it fails, stop.

---

## 1. What started this

Harvesting the campaign into the manuscript's benchmark grid required picking
one metric for the MATH500 column. The 2026-08-12 ruling was `math_verify`,
made for the AR column. Applying it to the DLM rows costs them ~2–5pp, which
looked wrong, so it got audited.

**Two bugs and one design flaw were found. The design flaw is the important
one.**

---

## 2. MATH500: both metrics are broken, in OPPOSITE directions

### 2.1 The numbers

Per-doc agreement, `dqwen3.5-9b-base-v3@step50000-swa`, 500 docs:

    agree 467   strict=1/verify=0: 28   strict=0/verify=1: 5

Net −23 docs = −4.6pp, which is exactly the 42.00 → 37.40 gap. Same shape on
every DLM row.

### 2.2 Why verify loses on DLM rows: unparseable answers

The 28 docs, verbatim tails:

    $-2+7i$        $x^3+3x-6$      $\cot x$        $2k + 2$
    $\textbf{(B)}$ $Navin$         $\frac{\sqrt{3}}{3}}$   $\frac{20000}{\pi}$

`math_verify` is sympy-backed: it must PARSE both sides. A person's name, a
multiple-choice label, an unbalanced brace, a symbolic expression with a free
variable — these fail to parse and it **fails closed**, scoring 0 on a string
that exactly equals gold. `exact_match` normalizes and compares, so it is
immune.

### 2.3 Why strict loses on AR rows: template deviation

`Qwen_Qwen3.5-9B`, 500 docs: **157 docs strict=0 / verify=1** (vs 15 the other
way). The answers are present and correct; the Minerva template frame is not:

    "I hope it is $\frac{14}{3}$."      (instead of "I hope it is correct.")
    "I hope it is square root of 196."
    "I hope it is with correct."
    "I hope it is"                       (truncated)

Minerva's strict extractor keys on the exact closing frame. Post-trained
models answer in their own voice, extraction returns nothing, score 0.
**31% of the set.**

### 2.4 Coverage: how much of ground truth each metric captures

`strict / union`, where union = (strict OR verify) doc-wise:

| DLM rows | | AR rows | |
|---|--:|---|--:|
| dQwen3.5-9B | 97.7% | AR Qwen3.5-0.8B | 91.0% |
| dQwen3.5-4B | 97.9% | AR Qwen3.5-2B | 97.2% |
| CONTROL 1.7B | 96.7% | AR Qwen3-1.7B | 89.4% |
| LLaDA-8B | 97.5% | **AR Qwen3.5-4B** | **37.6%** |
| Dream-v0-7B | 96.6% | **AR Qwen3.5-9B** | **39.4%** |
| Dream-Coder-7B | 95.4% | | |

**The defect is confined to two models.** Both land on exactly **20.40**,
below the 2B's 34.80 — non-monotonic in size and suspiciously identical, which
is a broken measurement a reviewer spots immediately.

### 2.5 Why this mattered: the metric choice FLIPPED a manuscript claim

§4.1's MATH500 architecture comparison (hybrid dQwen3.5-2B vs control
dQwen3-1.7B, each normalized by its AR counterpart):

| metric | hybrid | control | leads |
|---|--:|--:|---|
| strict | 0.322 | 0.359 | **control** +0.037 |
| verify | 0.338 | 0.326 | **hybrid** +0.011 |
| union | 0.330 | 0.332 | tie |

A claim whose SIGN depends on the metric cannot be published either way.

### 2.6 GSM8K is clean — and that is why it can carry the axis

    DLM  dqwen3.5-9b@50k   strict 73.64   flexible 73.64   (IDENTICAL)
    AR   Qwen_Qwen3.5-9B   strict 87.34   flexible 88.40
    AR   Qwen_Qwen3.5-2B   strict 58.98   flexible 38.51   (the known anomaly)

Our DLM rows emit one well-formed answer and both extractors agree exactly.
`flexible-extract` is NOT a robustness check — it grabs the first number-ish
token out of a verbose response and is differently broken. **Do not present it
as the second metric.**

---

## 3. Reference protocols — what LLaDA and Qwen actually do

### 3.1 LLaDA does NOT use lm-eval for generation benchmarks

From the paper's evaluation appendix (arXiv 2502.09992), verbatim:

> "For datasets evaluated with conditional likelihood estimation, we adopt the
> widely used lm-evaluation-harness (Gao et al., 2024) framework. **However,
> for datasets evaluated with conditional generation, we rely on an internal
> library for evaluation**, as lm-evaluation-harness lacks implementations for
> certain key metrics (e.g., HumanEval-FIM)."

> "For the base model, we use conditional likelihood estimation for MMLU,
> CMMLU, C-Eval, ARC-C, Hellaswag, TruthfulQA, WinoGrande, PIQA, and GPQA,
> while the remaining benchmarks are evaluated using conditional generation."

So **GSM8K, MATH, HumanEval, MBPP all come from unreleased internal tooling.**
Their decode: *"we apply the lowest confidence remasking strategy to the base
model, setting both the response length and the number of sampling steps to
1024"* — one token per step.

Published base numbers (shots in parens): MMLU 65.9 (5) · GSM8K 70.7 (4) ·
Math 27.3 (4) · HumanEval 33.5 (0) · MBPP 38.2 (4).

⚠ **Manuscript consequence.** §4's harness-credibility paragraph lists
"LLaDA MMLU 65.85 vs 65.9 · LLaDA HumanEval 32.93 vs 32.9 · Dream MBPP 56.00
vs 56.2" as one set. Only the **MMLU** one is apples-to-apples (both lm-eval,
both likelihood). The generation agreements are across different graders AND
different decoders — encouraging, but categorically weaker. Flagged for the
paper agent.

### 3.2 Qwen3.5 publishes nothing we can use

No arXiv tech report exists for the base Qwen3.5 series (verified 2026-08-03,
re-verified today). The 9B model card publishes **MMLU-Pro 82.5, MMLU-Redux
91.1, GPQA-Diamond 81.7, HMMT, LiveCodeBench v6, OJBench, LongBench v2** —
none of our benchmarks — under chat templating, temperature 1.0 / top-p 0.95,
up to 32k–80k output tokens. There is nothing to reproduce and no grader
choice that would make our AR numbers match theirs.

### 3.3 Qwen3 DOES publish base numbers, under a protocol close to ours

Qwen3 tech report (arXiv 2505.09388), base-model eval:

> "MMLU (5-shot) … GSM8K (4-shot, CoT), MATH (4-shot, CoT) … EvalPlus
> (0-shot) (Average of HumanEval…), MBPP-3shot"

Table 8:

| | MMLU | GSM8K | MATH | MBPP | EvalPlus |
|---|--:|--:|--:|--:|--:|
| Qwen3-0.6B-**Base** | 52.81 | 59.59 | 32.44 | 36.60 | 36.23 |
| Qwen3-1.7B-**Base** | 62.63 | 75.44 | 43.50 | 55.40 | 52.70 |

Incidental: **SDAR's "75.4" for Qwen3-1.7B is this table's 75.44** — they were
quoting the `-Base` figure. Our own measurement of the bare repo is 74.75 /
73.62.

### 3.4 The three-way protocol comparison

| | Qwen3 | LLaDA | ours | verdict |
|---|---|---|---|---|
| MMLU | 5-shot | 5-shot | 5-shot | ✅ match |
| HumanEval | 0-shot | 0-shot | 0-shot | ✅ match |
| MBPP | 3-shot | 4-shot | 3-shot | ⚠️ no consensus; keep 3 |
| **GSM8K** | **4-shot** | **4-shot** | **8-shot** | ❌ **we are the outlier** |
| MATH | 4-shot, full | 4-shot, full | 4-shot, MATH500 | ⚠️ set differs, disclosed |

Our 8-shot is lm-eval's documented default for `gsm8k_cot` (from the original
CoT paper's 8 exemplars), so it is not arbitrary — but both reference points
chose the other convention.

---

## 4. Already fixed this session

`summarize.py` could not read the HumanEval+ column at all: `HEAD_TASK`
mapped `humaneval-plus` → `humaneval_plus` but the cells carry
`humaneval_plus_sound`, so 22 cells rendered `?`. Fixed in **6c728a9** (exact
lookup, then a UNIQUE prefix match, else `?`). Verified strictly additive: 22
`?` → 0, and no previously-readable cell changed. Details in
`20260814-025900-…`.

---

## 5. Decisions

### 5.1 Taken (Anton)

1. **GSM8K → 4-shot, re-run.** Rationale below.
2. **MBPP stays 3-shot.**
3. **MATH500 stays** (not full MATH).

### 5.2 Recommended, for the paper agent (not cluster work)

4. **MATH500 column reports stock `exact_match`.** It is the lm-eval default
   and captures 95–98% of ground truth on every DLM row.
5. **Drop the MATH500 AR normalization from §4.1.** The AR denominators are
   broken by protocol mismatch (post-trained models under a base-style 4-shot
   prompt), not by metric choice. A withheld number is defensible; a
   sign-flipping one is not. GSM8K then carries the normalized math axis
   alone — which is precisely why its protocol must not be the outlier.
6. **Do not report the union.** It is the most accurate measure but has
   unverified false-positive exposure (the cells carry no gold, so only false
   NEGATIVES were provable here) and no comparability. Keep it as a
   diagnostic.

### 5.3 ⚠ What the 4-shot rerun does and does not buy

"4-shot" does not fully specify the prompt. lm-eval's `gsm8k_cot` at
`num_fewshot=4` takes the first 4 of its 8 exemplars; Qwen3 and LLaDA used
their own prompts, LLaDA's from an unreleased library. **We get closer, not
identical.** The win is removing an outlier choice and enabling a meaningful
comparison — NOT exact reproduction. Any claim written on these numbers must
say so.

### 5.4 Proposed, undecided — the grading contract

Worth adopting because we store all raw generations, so grading is a CPU-only
pass and a number never requires re-generation:

- generation frozen and raw; grading a separate versioned pass
- table reports the **stock** lm-eval metric
- every cell ALSO stores a **robust** metric (template-agnostic extraction,
  then string-equality-OR-symbolic-equivalence) — one definition applied to
  all benchmarks, NOT whichever secondary metric each task happens to emit
  (see §2.6: `flexible-extract` fails this test)
- **publication gate:** if |stock − robust| exceeds that benchmark's stderr
  (MATH500's is ~2.2pp, already in the spec), the row is disclosed or withheld
- per-doc logging of extracted candidate, gold, and which rule fired
- release raw generations + grader

Applied today this passes every DLM MATH500 row and fails AR Qwen3.5-4B/9B
automatically — i.e. it reaches §5.2's conclusion by rule rather than by our
discretion. **Not yet decided; needs Anton.**

---

## 6. THE PLAN — executable

**Scope: 632 GSM8K cells.** Confirmed by counting the store:

| group | models | cells |
|---|--:|--:|
| acceleration set (11 decode configs × 6 shards) | 9 | 594 |
| headline-only (block32-static-s32 × 6 shards) | 5 | 30 |
| AR (unsharded) | 8 | 8 |

Decode configs per accelerated model: `block32-static-s{2,4,8,16,32}` +
`block32-tau{0.5,0.6,0.7,0.8,0.9,0.95}`.

**Re-run ALL of it, not just the headline 84.** `static-s32` appears in both
the benchmark table and the step-frontier figures; splitting shot counts
between them leaves the figure's anchor disagreeing with the table, which is
harder to explain than the compute costs.

### Phase 0 — freeze the change

- `benchmark_specs.py`: `gsm8k_cot` → `num_fewshot=4`.
- **Bump a protocol/grader version** so 8-shot and 4-shot cells can never be
  silently merged. 632 old and 632 new cells will coexist.
- Re-run `tests/task_freeze_gate.py`.
- **Do NOT delete the 8-shot cells.** Archive them (suggest
  `_runs/grid_v1_vista_gsm8k_8shot/` on the workstation, and the cluster
  equivalent). Having both shot counts on identical models converts our
  deviation into a free protocol-sensitivity result.

### Phase 1 — validation gate (~10 cells, minutes) — RUN THIS FIRST

Purpose: the first generation-side reproduction anchor this project has ever
had. Everything else is gated on it.

- **NEW models, anchors only (not counterparts):** `Qwen/Qwen3-1.7B-Base`,
  `Qwen/Qwen3-0.6B-Base`.
- Re-run the 8 existing AR cells at 4-shot.

**Acceptance:** `Qwen3-1.7B-Base` GSM8K near **75.44**, `Qwen3-0.6B-Base` near
**59.59**. Treat >3pp as a FAIL and stop — that would indicate a real problem
in the AR extraction path, and it is far cheaper to find here than after 632
cells.

⚠ These two `-Base` models are **calibration anchors, NOT AR counterparts.**
The counterparts remain the bare (post-trained) repos, because that is what we
adapted; §3.1 of the manuscript names them explicitly. Two model sets, two
roles, no conflict. Do not swap them in the normalization.

While here, the same two models at MMLU 5-shot / MBPP 3-shot / HumanEval
0-shot cost little and are directly checkable against 62.63/55.40 and
52.81/36.60 — MMLU is the purest harness check since it is likelihood-based
and involves no extraction.

### Phase 2 — headline cells (84)

14 models × 6 shards at `block32-static-s32`. **This is what the manuscript is
blocked on** — the benchmark table and §4.1's fractions. After Phase 2 the
paper side can proceed; Phase 3 is not blocking.

### Phase 3 — acceleration sweep (540)

9 models × 10 remaining configs × 6 shards. Feeds the §4.3 step-frontier
figures.

### Sharding

Anton wants aggressive sharding. Suggestion, not a prescription — you own
`plan_tiles.py` and your cost model came within 3% last time:

- Shard **by cost, not uniformly.** 6 → 12 shards for 4B/9B where generation
  dominates; keep 6 for 0.8B/2B where model-load overhead is a larger
  fraction of cell time. Finer shards double load overhead.
- The 4-shot prompt is shorter than 8-shot, so prefill drops slightly across
  the board; generation length is unchanged.
- Same QOS discipline as the campaign (20 running / 40 submitted, ≤16-node
  tiles, 8h walls) — it never bound last time.

---

## 7. Open threads inherited / created

1. **CoDA native-protocol cell** — meaningful now that CoDA is fixed; needs a
   BENCH row (e.g. `humaneval-g768`). Anton's call, one cell.
2. **4B mbpp-fence anomaly** (−16pp, only model affected) — per-sample read.
   Note this is exactly the shape §5.4's gate would catch automatically.
3. **Decontamination scan** — still MANDATORY-not-started, paper side.
4. **§4 harness-credibility paragraph** needs the §3.1 correction (only MMLU
   is a true reproduction).
5. **Protocol appendix** collects: GSM8K 8-vs-4-shot with the measured delta,
   MATH500 subset + stderr ~2.2pp, MATH500 metric choice + the withheld
   normalization, MC-NELBO asymmetry, AR canvas fix, CoDA truncation note,
   hardware statement (all GATE cells on GH200).
