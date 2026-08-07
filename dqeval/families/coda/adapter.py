"""CoDA adapter -- Salesforce's masked DLM adapted from Qwen3-1.7B.

WHY THIS FAMILY MATTERS: CoDA-v0-Base starts from the SAME backbone as our
dQwen3-1.7B control (`Qwen/Qwen3-1.7B`, hidden 2048 / 28 layers / vocab 151936,
all confirmed against its config). Every other comparator differs from us in
backbone AND data AND budget; CoDA differs in data and budget only, so it is the
closest thing available to a recipe-isolating comparison.

CoDA is the SECOND family whose raw output is AR-aligned (Dream is the other):
`modeling_coda.py` returns `lm_head(hidden)` with the comment "we shift logits at
inference time", and their own sampler does exactly

    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)   # generation_utils.py

immediately after every forward -- byte-identical to Dream's expression. So
`_canonicalize` applies that shift and NATIVE samplers must use `raw_logits`;
routing a native CoDA sampler through `logits()` would double-shift silently.

Two further facts worth recording, both read out of their source rather than
assumed:
  * CoDA is genuinely BIDIRECTIONAL. `CoDAModel.forward` builds a
    `torch.triu(-inf)` causal mask and threads it down, but `AttentionModule`
    never passes it to `scaled_dot_product_attention` and hardcodes
    `is_causal=False` on both backends -- the mask is dead code.
  * Consequently CoDA IGNORES padding masks entirely, so batching would silently
    attend across padding. bs=1 is required for correctness here; dqeval is bs=1
    by construction, so this costs us nothing.

`forward` returns a plain `(logits, loss)` tuple, not a ModelOutput, and only
takes the inference branch when the module is in eval mode (`hf.materialize`
calls `.eval()`); in train mode it expects labels and does its own masking.
"""

from __future__ import annotations

from typing import Optional

import torch

from dqeval import hf
from dqeval.adapter import ModelAdapter, ModelSpec
from dqeval.families.coda import compat


class CoDAAdapter(ModelAdapter):
    family = "coda"

    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, **kw)
        return out[0] if isinstance(out, tuple) else out.logits

    def _canonicalize(self, logits: torch.Tensor) -> torch.Tensor:
        """Their inference-time shift, verbatim (generation_utils.py)."""
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> CoDAAdapter:
    cfg, klass = hf.resolve(spec.repo, revision=revision, auto_class=spec.auto_class)
    shims = compat.apply(cfg, klass)          # MUST precede materialisation
    tok = hf.tokenizer(spec.repo, revision=revision)
    model = hf.materialize(klass, spec.repo, cfg, revision=revision,
                           dtype=dtype, device=device)
    hf.assert_finite_rope(model)

    declared = getattr(cfg, "mask_token_id", None)
    if declared is not None and declared != spec.mask_id:
        raise ValueError(
            f"{spec.repo}: config mask_token_id={declared} != registry {spec.mask_id}")

    # UNTIED-HEAD GATE. CoDA ships a trained `lm_head.weight`, but its config omits
    # `tie_word_embeddings` and tf5 defaults that to True (see compat shim 3). If the
    # restoration ever regresses, the head becomes a view of the embedding table and
    # every logit is wrong while nothing raises. Same storage OR equal values both
    # indicate tying, so we reject both.
    head, emb = model.lm_head.weight, model.model.embed_tokens.weight
    if head.data_ptr() == emb.data_ptr():
        raise RuntimeError(f"{spec.repo}: lm_head is TIED to embed_tokens (shares storage); "
                           "CoDA is untied -- compat shim 3 has regressed")
    if torch.equal(head, emb):
        raise RuntimeError(f"{spec.repo}: lm_head equals embed_tokens bitwise; "
                           "the trained head was overwritten by tying")
    return CoDAAdapter(model, tok, spec, revision=revision, shims=shims)
