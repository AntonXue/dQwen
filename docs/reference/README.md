# `docs/reference/` — preserved prior art, not live code

Nothing here is imported by `dqeval`. These are working artifacts kept verbatim
because they encode hard-won detail that would be expensive to rediscover.

## `lm_eval_probe/`

The lm-evaluation-harness proof-of-concept from the ADLMC research repo
(`ablations/paper_evals/lm_eval_probe/`), copied unchanged. **It was git-untracked
in ADLMC** — it existed in exactly one place on disk, which is why it is here.

It is roughly **70% of a working lm-eval wrapper** and it ran end-to-end on
`arc_challenge`. Phase 3 will rewrite it against our `ModelAdapter` contract rather
than extend it, but it is the best available starting point and it already answers
several questions correctly.

**What it gets right:**
- `@register_model("dllm")` with `generate_until` and `loglikelihood` implemented.
- `loglikelihood` wired to the MC-NELBO estimator, using `mc_num=1` for single-token
  continuations (exact) and full MC otherwise.
- Returns `(ll, False)` — `is_greedy` hardcoded False, matching LLaDA's convention.
- `generate_until` translates lm-eval's `until` stop-strings into post-hoc
  truncation, since a diffusion decode stops on an all-PAD block, not a stop string.

**Known gaps (all still open):**
- No `Accelerator` `_rank`/`_world_size` → single process only, no request sharding.
- No `apply_chat_template` / `tokenizer_name` → blocks instruct models and
  `mmlu_generative`, which is on LLaDA's published task list.
- `block` / `steps_per_block` not surfaced through `model_args` — hardcoded in the
  `__init__` signature, so the decode spec cannot be swept from the CLI.
- MASK-id suppression is applied in the decode path but **not** in the loglikelihood
  path.
- `loglikelihood_rolling` raises `NotImplementedError` (fine for gsm8k/mmlu/humaneval,
  not fine for perplexity tasks).

**Environment note:** it ran in a venv layered on the `qwen35` conda env with
`lm_eval==0.4.8` plus **two manual source patches** (`hf_vlms.py` →
`AutoModelForImageTextToText`, for transformers 5.x). Those patches are not captured
by any requirements file anywhere; reproducing this needs them re-applied. Turning
that into a documented install step is part of Phase 3.

## Where the estimator went

`ablations/paper_evals/mc_nelbo.py` was **not** merely preserved — it was ported into
the package proper as `dqeval/nelbo.py`, keeping `forward_process` byte-faithful, and
is now covered by `tests/parity/test_nelbo_vs_llada.py`, which reproduces LLaDA's
reference at `diff = 0.000e+00`.
