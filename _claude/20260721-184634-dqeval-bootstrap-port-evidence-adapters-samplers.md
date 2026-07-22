# dqeval bootstrap: the tf-5.13 port evidence, the adapter contract, first samplers

> 2026-07-21. Session that created this repo. Covers **why** the architecture is
> shaped this way (it is all empirical — every design choice below was forced by a
> measurement, several of which refuted my prior), what is **built and verified**,
> and what is **next**. Self-contained: a fresh Claude on another machine should be
> able to continue from this file alone.

## 0. What this repo is

The **evaluation** package for the dQwen3.5 / dQwen3 release, plus the comparator
DLMs (LLaDA, Dream/Dream-Coder, SDAR), all under **one environment**. Goals, in
Anton's framing:

1. Load each model and call **its own shipped sampler** — so we can report "this is
   the number with the sampler shipped with LLaDA".
2. Run **our own sampler** on the same weights — so we can also report "here's how it
   did with our sampler". Model x sampler is a cross-product, not one number each.
3. Reproduce the numbers those papers claim, via **lm-eval-harness** (Phase 3, not
   yet started).

`train/` will eventually be a copy-paste dump of ADLMC `diffuqwen35/`; it needs no
design attention.

## 1. THE decisive finding: the port is bitwise-exact

Our models need **transformers 5.13** (Qwen3.5 requires >= 5.x). Every comparator
pins 4.4x–4.5x. Measured, they are perfectly disjoint:

| model | tf 4.57 | tf 5.13 (unshimmed) |
|---|---|---|
| LLaDA-8B-Base | loads | `AttributeError: all_tied_weights_keys` |
| Dream-v0-Instruct-7B | loads | `KeyError: 'default'` |
| dQwen3.5-0.8B-Base | `ImportError: Qwen3_5ForConditionalGeneration` | loads |

So "one venv for everything" is impossible **unless the comparators are ported**.
They were. Decomposing the drift with torch held fixed:

| comparison | max\|d\| | mean\|d\| | argmax |
|---|---|---|---|
| torch effect (2.5.1+cu121 -> 2.7.1+cu128) | 4.69e-01 | 4.28e-02 | 100% |
| **transformers effect (4.57 -> 5.13, +shims)** | **0.00e+00** | **0.00e+00** | 100% |

**100% of drift is torch; 0% is transformers.** Confirmed independently on LLaDA
(this session) and on Dream (via a third env, `adlmc` = tf4.57 + torch2.7.1).

⇒ **The port acceptance gate is BITWISE equality, not a tolerance.** An earlier
recommendation of "argmax agreement + a recorded epsilon" was WRONG — it came from a
confounded comparison where transformers and torch varied together. Do not
reintroduce a tolerance policy.

⇒ **torch is part of the reproducibility contract.** The reference env is not a
conda env, it is `python -m venv --system-site-packages` + `pip install
transformers==4.57.1` on top of the target env (~30 seconds).

## 2. Shims are archaeology, not invention

Every shim RESTORES a behaviour tf 4.x had, verified by reading it out of the native
environment. Nothing is guessed. Example: `cfg.use_cache = False` for LLaDA was not
assumed — the config was loaded under tf 4.57 and the value read.

Totals: **LLaDA 3, Dream 4, SDAR 3 — ~90 lines, zero invented.**

### 2a. The SILENT failure class (hit 2 of 3 families)

tf>=5 materialises models under a bare `torch.device("meta")`. Rotary `inv_freq` is a
**non-persistent** buffer, so it is absent from the checkpoint and comes back as
uninitialised memory. tf5 repairs this in **base** `PreTrainedModel._init_weights` —
but Dream and SDAR both **override** `_init_weights`, so the repair branch never runs.

Unrepaired: the model loads, forwards, raises **nothing**, and returns noise.
Measured argmax agreement vs native **20–40%**; `inv_freq.sum()` = `nan` /
`-279837.44` where native gives `5.150445`.

- Dream repair: post-load `reset_parameters()`.
- SDAR repair: no `reset_parameters()`, so re-run its own `rope_init_fn`.
- Hooking `_init_weights` does **not** work — tf5 calls it while still on meta.
- `dqeval/hf.py::assert_finite_rope()` makes this non-silent forever.

**This is why golden fixtures are mandatory.** Nothing else catches it.

## 3. Two logit surfaces (the shift)

Verified per family:

| family | raw output | shift applied by |
|---|---|---|
| **Dream / Dream-Coder** | AR-aligned | **consumer** — `generation_utils.py:421-422` AND `eval/eval.py:354`, same expression |
| LLaDA | position-aligned | nobody |
| SDAR | position-aligned | nobody |
| **dQwen3.5 / dQwen3** | position-aligned | **model** (prepends BOS, returns `lm_head(hidden[:, :-1, :])`) |

Hence `raw_logits()` (native samplers — they shift themselves) vs `logits()`
(portable samplers — canonicalised once). Feeding canonicalised logits to Dream's
native sampler **double-shifts silently**. This is structure, not a `logit_shift`
config flag — that flag has already caused a bug once on a sibling model.

## 4. Sampler landscape (my two priors were WRONG)

- **Prior: Dream's sampler is deeply HF-coupled and will be hard.** False.
  `DreamGenerationMixin` is a *plain class*, not a `GenerationMixin` subclass
  (`isinstance == False` in both envs). It reimplements the loop and only borrows
  `GenerationConfig`. 6/7 configs bit-identical cross-env; the 7th differs only at
  temperature>0 from torch's RNG stream.
- **Prior: SDAR ships no sampler.** False. `JetAstra/SDAR/generate.py` has
  `block_diffusion_generate` (317 lines, pure torch + `DynamicCache`). It needs a
  custom `store_kv=` forward kwarg, and **the HF-shipped `modeling_sdar.py` supports
  it** (lines 269, 273, 324, 342, 467, 534). It is in their GitHub, not the HF repo.
- LLaDA's `generate.py` is 130 lines of pure torch; verified to produce **identical
  ids and text** in native and ported envs.

### 4a. SDAR is a reproducibility minefield — read before quoting any SDAR number

Four disagreeing implementations, three different thresholds:

| implementation | strategies | greedy? | threshold |
|---|---|---|---|
| `SDAR/generate.py` | 4 | **no** | 0.85 |
| JetEngine (`Labman42/JetEngine`) | ~15 | yes | 0.75 (`opt28: 0.9->0.75`) |
| **LMDeploy — produced their published table** | — | yes | **0.9** |

Their own `generate.py` **cannot** reproduce their published numbers: the published
config is greedy and that script calls `torch.multinomial` unconditionally after
`logits / temperature`, so `temperature=0` divides by zero. **Any SDAR number we
produce is a re-measurement, not a reproduction**, unless LMDeploy is stood up.

Also: their HF repos are **missing** `fused_linear_diffusion_cross_entropy.py`, which
`modeling_sdar.py` imports. Fails under tf 4.57 too — a repo defect, not our port.

### 4b. Which harness produced which table

- **LLaDA** — lm-eval, `@register_model("llada_dist")`. Reproducible.
- **Dream** — lm-eval, but they **vendor a modified copy** (`eval_instruct/lm_eval`,
  v0.4.8 — same version we use). Modifications still need diffing.
- **SDAR** — OpenCompass + LMDeploy. Not lm-eval at all.

LLaDA's own `EVAL.md` shows their numbers moving **45.0 -> 50.4** purely from
`gen_length`/`steps`/`block_length`. Decode spec must be pinned or a config
difference reads as a failed reproduction.

## 5. What is built and VERIFIED (commit `9fba7ac` + follow-ups)

```
dqeval/
  adapter.py    ModelAdapter (raw_logits/logits), ModelSpec, MODELS registry, load()
  config.py     DecodeConfig (the one knob space) + PRESETS with provenance
  hf.py         dynamic-module resolution + assert_finite_rope()
  samplers/
    unified.py  portable bs=1 block-diffusion engine
  families/{llada,dream,sdar,dqwen}/{adapter,compat}.py
envs/requirements-eval.txt   third_party/LOCKFILE.md
```

**All four families load through one contract** (tf 5.13, GPU-verified):

| model | shims | canonicalize | note |
|---|---|---|---|
| llada-8b-base | 3 | identity | |
| dream-7b-instruct | 4 | **shift**, == Dream's own expression | |
| sdar-1.7b-chat | 3 | identity | + repo-defect workaround |
| dqwen3.5-0.8b-base @`step30000-swa` | 0 | identity | **HF revisions work** |

**The unified sampler runs on all four and produces coherent code.** And the key
result: on LLaDA at matched config it is **token-for-token identical to LLaDA's own
`generate.py`**:

```
' return a + b\n\ndef sub(a, b):\n     return a - b\n\ndef mul(a, b):\n     return a * b'
```

⇒ the central design claim — *native decode is a preset in one parameter space, not a
separate code path* — is empirically supported, not assumed.

## 6. Decisions worth not relitigating

- **bs=1 everywhere.** Not just simpler — *more faithful*: LLaDA's and SDAR's
  reference samplers both hardcode `torch.full((1, ...))`. Batching them would be a
  divergence with no upstream counterpart to parity-test. It also sidesteps our
  champion decode's batch non-invariance (measured 2–5pp swings; HE read 30.5 <-> 50).
  Parallelism goes ABOVE the sampler: map over prompts, shard over GPUs.
- **`temperature == 0` means argmax**, never `logits / 0`.
- **We own the harness.** Upstream repos are reference + test fixtures, pinned in
  `third_party/LOCKFILE.md`. Never runtime dependencies.
- **Provenance per result row** — implementation, upstream commit, exact parameter
  values. Given SDAR's four implementations, a config without a citation is unusable.

## 7. Next

1. `families/*/sampler_native.py` — vendor each upstream sampler with documented
   touchups (SDAR needs a greedy branch it does not have upstream).
2. `families/*/sampler_upstream.py` — ~10-line bridge to the pinned `third_party/`
   checkout. The escape hatch; also mints the parity fixtures. Ceiling: it reaches
   `SDAR/generate.py`, NOT the LMDeploy path behind their table.
3. `tests/parity/` — four gates per family: shim (bitwise, torch fixed), shift
   identity, native vs upstream, preset vs native.
4. Regression gate: unified@`dqwen_champion` == ADLMC `block_append_generate` bitwise.
5. **Then** Phase 3, the lm-eval harness (ADLMC's `lm_eval_probe/dllm_lmeval.py` is
   ~70% done: needs rank/world-size, chat template, `model_args`, MASK suppression).

## 8. Environment / handles

- Target env: conda `qwen35` — tf 5.13.0, torch 2.7.1+cu128, flash_attn 2.8.3,
  einops 0.8.1, evalplus 0.3.1. `lm_eval` NOT installed there (Phase 3).
- `HF_HOME=/ssd1/ayx98/cache/huggingface`, always `HF_HUB_DISABLE_XET=1`.
- **All 14 registered models are cached and load-tested** (`tests/load_matrix.py`,
  14/14 PASS): dQwen3.5 {0.8B,2B,4B,9B} with all `step-*` revisions, dQwen3
  {0.6B,1.7B}, LLaDA Base+Instruct, Dream-v0 {Base,Instruct}-7B, Dream-Coder-v0
  {Base,Instruct}-7B, SDAR {1.7B,4B}-Chat. ~50GB was pulled this session.
- Generation verified on all families. Note instruct/chat models given a *base-style*
  completion prompt terminate immediately (LLaDA-Instruct emits `<|eot_id|>`, SDAR-Chat
  `<|endoftext|>`) -- correct behaviour, but they need `adapter.chat()` formatting for
  any real eval.
- ADLMC assets still to rescue: `ablations/paper_evals/mc_nelbo.py` (bit-exact LLaDA
  MC-NELBO port, reproduces their reference at diff 0.00e+00) and
  `ablations/paper_evals/lm_eval_probe/` — **both git-UNTRACKED in ADLMC.**
