# CoDA ported into dqeval (5 shims) + the first same-backbone head-to-head

> 2026-08-07, eval box. Adds a fifth family to the harness. Motivation is not
> completeness: **CoDA-v0-Base is adapted from `Qwen/Qwen3-1.7B`, the same backbone as
> our `dQwen3-1.7B-Base` control**, so it is the only comparator that differs from us in
> data + budget but NOT in starting weights — the closest thing available to a
> recipe-isolating comparison. Every other comparator differs in all three.

## 1. The numbers (unified protocol, both models identical)

Full canvas, gen=512, block=32, steps=32, low-confidence, greedy, bs=1, 0-shot.

| | HumanEval (n=164) | MBPP (n=500) | MMLU 5-shot |
|---|--:|--:|--:|
| **dQwen3-1.7B-Base** (ours, v1-era recipe, 50k) | **42.68** | **35.00** | **36.55** |
| **CoDA-v0-Base** (Salesforce) | 23.17 | 34.40 | 26.22 |
| delta | **+19.51** | **+0.60** | **+10.33** |

**The win is HumanEval and knowledge — MBPP is a TIE (+0.60).** Do not summarise this as
"~1.8x better"; that is true of HumanEval only. CoDA at 26.22 MMLU sits essentially at the
25% chance floor, i.e. it traded away general knowledge entirely — the same pathology v1 showed
at 0.8B/2B. Our 36.55 exactly reproduces the documented `dQwen3-1.7B` value. All of this is
with our *superseded* v1-era mixture.

⚠ **CoDA's published HumanEval is 29.3, not 23.17, and the gap is protocol.** From their
arXiv source (2510.03270, `sections/experiments.tex`): they cap generation at **768 tokens**
and "align the diffusion schedule to the sequence length, i.e. **a single token is produced
at each diffusion step**" (768 forwards/problem) with Dream-style entropy confidence
sampling. Ours is 512 tokens / 32 steps-per-block = <=512 forwards with up to 32 tokens
committed per step. Their own Fig. `speed_accuracy` shows accuracy still climbing to ~512
steps, so a coarse schedule costs them real points. **Our protocol undersells CoDA by ~6pp**
— the same disclosed effect we document for Dream. Before any CoDA number is published,
run the native-ish cell: `--gen-length 768 --block-length 768 --steps-per-block 768
--order entropy` (closest expressible point; `DecodeConfig` has entropy ORDERING but not
their per-token entropy REWEIGHTING, so it is a re-measurement, not a reproduction).

⚠ **"Same backbone" is not "same parameter count".** CoDA UNTIED the head — verified
against their raw checkpoint, not inferred: `lm_head.weight` vs `model.embed_tokens.weight`
differ by max 4.41, cosine 0.48 on row 0. That is ~311M extra parameters (2048 x 151936),
so CoDA is **2.032B** against Qwen3-1.7B's tied ~1.7B. Their pretraining budget is
undisclosed. State both caveats wherever the comparison appears.

## 2. Their benchmark table is mislabelled — read the source, not the README

`tables/model_evaluation.tex` (arXiv source) is authoritative; columns are
**HumanEval / HumanEval+ / MBPP / MBPP+ / EvalPlus**. The repo README renders the first as
"Humaneval Instruct" even for base rows — ignore it.

The real oddity: **MBPP+ (46.0) > MBPP (35.2)** for CoDA-Base, which is backwards for a
plus-variant. It is NOT a column swap — recomputing their stated definition ("Evalplus =
average of pass@1 on plus-enhanced variants") gives `mean(23.8, 46.0) = 34.9`, exactly the
printed EvalPlus, and this checks out on 9 of 10 rows. So 46.0 really is their MBPP+, and
the likely reading is that their "MBPP" is the full test split while "MBPP+" is EvalPlus's
378-problem *sanitized* subset — a different, easier problem set, not a superset.
**Consequence: their MBPP numbers are not comparable to ours** without matching that subset.
HumanEval (one standard 164-problem set) is the only clean replication target.

## 3. The port: five shims, all restorations, three of them NEW

`dqeval/families/coda/{adapter,compat}.py`. Prior art was
`ADLMC/coda/old/environment.md` (verified on transformers **4.57**); shims 2-4 below break
only on **5.x** and are not in that doc.

| # | shim | why |
|---|---|---|
| 1 | `CoDAModel._supports_sdpa = True` | tf PR #39423 (v4.54, breaking) made `_sdpa_can_dispatch` strict. CoDA's OUTER class declares the flag, the INNER `CoDAModel` does not, so init explodes. Documented prior art. |
| 2 | `rope_scaling` -> `None` | tf5 synthesises a rope dict with `rope_type`; the shipped config.json has `"rope_scaling": null`, so this branch was DEAD in 4.x. Provably right: `CoDARotaryEmbedding` raises `NotImplementedError` for ANY non-None scaling. |
| 3 | `tie_word_embeddings` True -> **False** | **THE DANGEROUS ONE.** config.json OMITS the key; tf5 defaults it True. CoDA is untied (§1). In 4.x this was harmless because `CoDALanguageModel.__init__` never calls `post_init()` (where tying happens) — only the inner class does. Unrestored, tf5 overwrites the trained head with the embedding matrix: silent, no exception, every logit wrong. |
| 4 | `all_tied_weights_keys = {}` | tf5's `_finalize_model_loading` reads it; the base class sets it in `post_init()`, which this class never calls. Empty mapping is the exact restoration (it registers no tied weights). We do NOT call `post_init()` — that would additionally run tie_weights + gradient-checkpointing setup 4.x never ran here. |
| 5 | post-load `inv_freq` recompute | The Dream/SDAR meta-device trap: `inv_freq` non-persistent AND `_init_weights` overridden. CoDA's rotary has no `reset_parameters`, so we recompute with THEIR `default_rope_frequencies`. |

**Gates in `build()`** (fail loudly, because shim 3's failure mode is silent): `lm_head` must
not share storage with `embed_tokens` AND must not be bitwise equal to it; config
`mask_token_id` must match the registry; `assert_finite_rope`.

**Verified on load:** 2.032B params, mask=151669, pad=151643, `lm_head` matches the
checkpoint bitwise (bf16 cast), `_canonicalize` == their verbatim shift, bidirectional probe
16-22 logits of movement at earlier positions.

## 4. CoDA is the SECOND AR-aligned family

Their `modeling_coda.py` returns `lm_head(hidden)` with the comment *"we shift logits at
inference time"*, and `generation_utils.py:457` applies
`torch.cat([logits[:, :1], logits[:, :-1]], dim=1)` after every forward — **byte-identical
to Dream's expression**. `_canonicalize` applies it; native samplers must use `raw_logits`
or they double-shift. `adapter.py`'s module docstring updated: it previously asserted Dream
was the only such family.

Two further facts read out of their source rather than assumed: CoDA is genuinely
**bidirectional** (`CoDAModel.forward` builds a `torch.triu(-inf)` mask and threads it down,
but `AttentionModule` never passes it to SDPA and hardcodes `is_causal=False` — the mask is
dead code); and consequently CoDA **ignores padding masks entirely**, so bs=1 is required
for correctness — free for us, since dqeval is bs=1 by construction.

## 5. Operational note

CoDA's HumanEval first ran 3x slower than dQwen3-1.7B's and I attributed it to stop-string
behaviour. Wrong: **GPU0 was contended** (another user's 74.4 GB job at 100%). Moved to the
idle GPU2 and it ran at normal speed. **Check `nvidia-smi` before theorising about a model's
decode.** (Second time this class of error occurred today; the first was blaming cuBLAS
nondeterminism for what turned out to be a `--num-fewshot` default.)

## 6. Next
- Native-ish CoDA cell (768/768/entropy) — REQUIRED before publishing any CoDA number.
- Re-run the head-to-head against a **v3** 1.7B when/if `1.7Bq3` is trained; today's win uses
  the superseded v1-era recipe. See the decision doc in `ADLMC/_claude/`.
- `run_rev.sh` has `OUT` hardcoded to `_runs/battery4b`; CoDA results pooled there despite
  being launched from `_runs/coda`. Cosmetic (tags are correct), but parameterise it.
