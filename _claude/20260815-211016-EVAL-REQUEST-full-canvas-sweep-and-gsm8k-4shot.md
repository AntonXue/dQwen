# EVAL REQUEST: the FULL-CANVAS sweep, and GSM8K at 4-shot alongside 8-shot

> **2026-08-15 ~21:10, workstation → cluster.** Two campaigns, one brief.
> Anton, verbatim: *"I don't give a shit how much compute we need to spend
> on the SLURM cluster, but I wanna fucking make sure that we have the FULL
> CANVAS with non-block clocked in."* Compute is explicitly not a constraint
> on this one. Do it properly rather than cheaply.
>
> This is the largest request we have sent. Read §1 before scoping anything:
> the reason it exists is a protocol mismatch we shipped without noticing,
> not a curiosity.

---

## 1. WHY. Every comparator publishes FULL-CANVAS numbers; we measured them in blocks

Our entire evaluation reports `block32-*`. That was never checked against how
the comparators actually generate. It should have been. What their own
sources say:

**LLaDA** (arXiv 2502.09992v2, appendix, verbatim):

> "For benchmarks evaluated using conditional generation, we apply the lowest
> confidence remasking strategy to **the base model**, setting both the
> response length and the number of sampling steps to **1024**."

No blocks. Block-32 appears in that paper only for the *other* model:

> "For the LLaDA 8B **Instruct** model, we adopt semi-autoregressive sampling
> with a block length of 32."

**Dream and Dream-Coder.** Their official `generation_utils.py` — both repos,
checked today — contains **zero occurrences of the string "block"**. The
sampler exposes `steps` (default 512) and `alg` (`origin`, `maskgit_plus`,
`topk_margin`, `entropy`) over the whole canvas. Their sampler *cannot* do
block decoding, so their published numbers are full-canvas by construction.

**The confirmation.** Our one full-canvas cell for LLaDA,
`standard-static-s1024` on HumanEval, scores **33.54**. LLaDA's published
number is **33.5**. Our block-32 measurement of the same model is 32.93 —
close, but by luck rather than by matching.

⚠ **Two honest qualifications on that agreement.** First, per
`20260814-121922`, LLaDA uses lm-eval only for likelihood tasks and an
**unreleased internal library** for GSM8K / MATH / HumanEval / MBPP. So even a
protocol-matched full-canvas run is a *cross-grader* agreement, not a
same-method reproduction. It is strong evidence that we are in the right
regime, not proof of identity. Second — and this matters for how this request
is read — **none of §1 is new**. `20260814-121922` already recorded LLaDA's
"steps=length=1024" decoding and already stated that only our MMLU
reproduction is apples-to-apples. That was written a day before this request
and was not acted on. This doc is that finding finally being resourced, not a
discovery.

So the cross-family tables currently compare our models at *our* operating
point against theirs at a point they never use. For Dream that point is not
merely unusual, it is unimplemented upstream. This is very likely the whole
explanation for the 14-point gap between our Dream-v0 HumanEval (43.90) and
their published 57.9.

**Consequence for the paper:** full-canvas is expected to become the main
tables' protocol and block-32 to move to the appendix — the inverse of the
current manuscript. Do not assume any current number survives.

---

**This is also the repo delivering on its own stated goal.** `_claude/README`
opens by defining the project as: call each model's *own shipped sampler*, then
run *ours* on the same weights, because "model x sampler is a cross-product,
not one number per model." We have only ever built the second half of that
cross-product. Campaign A is the first half.

---

## 2. WHAT EXISTS TODAY (checked, `_runs/grid_v1_vista`, 1,609 cells)

```
1,236  block32-*                the whole acceleration campaign
   11  standard-static-s1024    HumanEval ONLY, one budget, 11 models
   72  ar        14  mc-nelbo        10  regrade
```

`make_manifests.py:152` hardcodes the ceiling cell to `humaneval`. That was a
deliberate scoping call on the paper side (the full-canvas configs were listed
under "Explicitly NOT requested" in `20260815-023130`), and it is the call
being reversed here. **Nothing was dropped or lost — it was never asked for.**

⚠ `standard-static-sN` has only ever been run at **N=1024**. `standard-tau*`
has **never been run at all**. Both need gating (§4).

---

## 3. CAMPAIGN A — the full-canvas sweep

### Rows (14)

```
dqwen3.5-{0.8b,2b,4b,9b}-base-v3 @ step25000-swa   (4)
dqwen3.5-{0.8b,2b,4b,9b}-base-v3 @ step50000-swa   (4)
dqwen3-1.7b-base-v3 @ step25000-swa, step50000-swa (2)
llada-8b-base, dream-7b-base, dream-coder-7b-base,
coda-1.7b-base  @ main                             (4)
```

### Benchmarks (4) and their shard counts

`humaneval` 1, `mbpp` 2, `gsm8k` 6, `math500` 2 — the existing `DLM_SHARDS`.
**Include math500**: if full-canvas becomes the main protocol, the MATH column
restamps with everything else. `humaneval-plus` / `mbpp-plus` need NO cells —
they are CPU-side regrades of saved generations, per your own part-4 note.
MMLU is `mc-nelbo` and is decode-policy-independent; do not re-run it.

### Configs

**Static ladder (6), the priority:**
`standard-static-s{32,64,128,256,512,1024}`

Both comparator conventions must be inside the ladder — LLaDA reports at
**1024**, Dream at **512** — so neither endpoint may be trimmed. `s1024` is
half the ladder's cost on its own; run it last if you want early results, but
run it.

**Threshold sweep (6), gated:** `standard-tau{0.5,0.6,0.7,0.8,0.9,0.95}`,
**only if §4b passes.** Never executed in any mode.

### Cell count

Per row: 11 shard-units × 6 static = 66 cells. **14 × 66 = 924 cells.**
With the tau half, **1,848**.

### Cost, stated honestly

Full-canvas costs **one forward per step** — `standard-static-sN` is N
forwards per problem, with no early exit (our existing s1024 cells report
exactly 1024 for every model, which is itself worth confirming, see §4c).
The static ladder sums to **2,016 forwards per problem** where the entire
block ladder is roughly 250. Per model-benchmark this is **~8× the block
sweep**, so in block-equivalents Campaign A alone is on the order of 7,000
cells. GSM8K at 1,319 problems × 1,024 forwards on a 9B is the heavy tail.
Anton has explicitly accepted this cost. Shard aggressively.

---

## 4. GATES — run these before queueing anything

### 4a. Does `standard-static-sN` work below N=1024?

Only 1024 has ever run. **4 cells:** `standard-static-s32` and
`standard-static-s128` on `humaneval`, for `dqwen3.5-2b-base-v3@step50000-swa`
and `llada-8b-base@main`.

Pass criteria:
1. **Measured `n_forward` ≈ N.** If s32 reports ~1024, the step count is being
   ignored and the whole ladder is one config run six times.
2. Accuracy degrades smoothly rather than collapsing to 0.
3. Samples logged.

### 4b. Does `standard-tau` exist at all?

**2 cells:** `standard-tau0.9` on `humaneval` for the same two models. If the
sampler rejects the config, **report and skip the tau half entirely** — the
static ladder is what answers the protocol question and the tau half is an
extension. Do not implement a new sampler to satisfy this request.

### 4c. Is the s1024 forward count measured or nominal?

Every existing s1024 cell reports exactly 1024.0 forwards. Confirm whether
full-canvas decoding has no early-exit path, or whether `n_forward` is being
recorded nominally. **This matters to the paper**: our published "block-32 is
5–11× cheaper than sequential" comparison rests on that denominator.

### 4d. Can 4-shot and 8-shot GSM8K coexist in one store?

`summarize.py`'s `merge_groups` refuses stripes whose `meta.bench.shots`
disagree — it prints `MIXED-PROTOCOL … refusing to merge`. That guard was
built for the cancelled *replacement* rerun. This request is **additive**, so
both must live together. See §5.

---

## 5. CAMPAIGN B — GSM8K at 4-shot, ADDITIONALLY

⚠ **This is not the cancelled rerun.** `20260814-133042-REVERSAL` cancelled a
plan to *replace* 8-shot with 4-shot. Anton now wants **both**: *"why don't we
also rerun all the GSM8k 8-shot mode with a 4-shot mode as well, while we're
at it."* 8-shot stays exactly as it is. Nothing is archived, nothing moves.

### Why both is the right shape

Our comparators do not agree with each other. Dream publishes 8-shot; LLaDA
and the Qwen3 report declare 4-shot. Holding both lets the paper meet each
comparator on its own convention instead of picking a side and disclosing a
deviation — which is what the reversal settled for, reasonably, when a rerun
meant losing the 8-shot cells. It no longer does.

### The coexistence problem, and the fix we prefer

Do **not** flip `BENCH["gsm8k"].shots`. Add a **distinct benchmark id** —
suggested `gsm8k-4shot` — pointing at the same task with `shots=4`. That way:

- the mixed-protocol guard never trips, because the two never share a group;
- both live in `grid_v1` with no archiving and no store surgery;
- every downstream consumer selects by benchmark id, which is what the
  manuscript's generators already do.

The `20260814-123751-gsm8k-4shot-phase0-frozen` work is the reference for the
prompt machinery (first 4 of the 8 canonical CoT exemplars via lm-eval
`first_n`, renders byte-identical otherwise). Reuse the finding, not the
replacement plumbing.

### Scope

Everything that currently has an 8-shot GSM8K cell gets a 4-shot twin:

- **block32 grid**: 14 rows × 11 configs × 6 shards = **924 cells**
- **AR rows**: 8 AR models × 6 shards = **48 cells**
- **full-canvas** (Campaign A's gsm8k half, at 4-shot): 14 × 6 × 6 = **504**,
  or 1,008 with tau

**≈1,476 cells**, or ~1,980 with tau.

---

## 6. TOTALS AND ORDERING

| | cells | notes |
|---|---|---|
| Gates 4a–4c | 6 | minutes; everything waits |
| A: full-canvas static | 924 | the priority |
| B: gsm8k 4-shot, block + AR | 972 | independent of A |
| A: full-canvas tau | 924 | only if 4b passes |
| B: gsm8k 4-shot, full-canvas | 504 (+504) | after A's ladder validates |

**≈2,800 cells minimum, ~3,800 with both tau halves**, and the per-cell cost
is far above the block campaign's.

**Order:** gates → A-static → B-block/AR → the two tau/extension blocks.
If anything has to be cut, cut tau. **The static ladder is the request.**

---

## 7. PROTOCOL LOCKS — unchanged, these are publication cells

- **GATE path only**: `sdpa=math`, global canvas, `bs=1`.
- Same store as the campaign; the one-hardware rule still applies.
- **Always log samples.** No exceptions.
- **`n_forward` on every per-sample record.** The manuscript's cost axis is
  this measurement; a nominal value silently destroys the figures.
- Pre-2026-08-09 n=80 stepsweep numbers remain ~19pp inflated. Never merge.

---

## 8. WHAT TO REPORT BACK

1. **Gate verdicts, per gate**, especially 4a's `n_forward ≈ N` check and 4c's
   measured-vs-nominal answer. Both change what the paper can say.
2. Whether `standard-tau` exists, and the tau half's fate.
3. The benchmark id you chose for 4-shot GSM8K, so the manuscript generators
   can select on it.
4. `summarize.py --merge` output, with confirmation of **zero MIXED-PROTOCOL**
   lines — that is the check that 4-shot and 8-shot are coexisting cleanly.
5. Anything where full-canvas behaves qualitatively differently from block for
   a specific model family. Dream is the one to watch: if it needs different
   handling, that is a protocol disclosure the paper must carry.

---

## 9. EXPLICITLY NOT REQUESTED

- Re-running MMLU. `mc-nelbo` does not depend on decode policy.
- Re-running the existing block32 grid at 8-shot. It stays exactly as it is.
- `humaneval-plus` / `mbpp-plus` cells. CPU-side regrades of saved
  generations, as your part-4 report established.
- Any change to the MATH500 metric ruling (`math_verify`), which is
  orthogonal.
- Append-mode anything. Retired and staying retired.
