"""SDAR adapter (JetLM/SDAR-*-Chat).

SDAR is a block-diffusion model built on a Qwen3 fork. Its head predicts the token
AT each position (it was retrained with a block-diffusion objective), so raw output
is position-aligned and `_canonicalize` is the identity -- confirmed by reading
their `block_diffusion_generate`, which samples `model(cur_x).logits` directly with
no reindexing.

Two SDAR-specific wrinkles handled in compat.py:
  * their HF repo is missing a module it imports (fails under tf4 too);
  * `auto_map["AutoModel"]` points at SDARModel, which has NO lm_head -- so we must
    resolve AutoModelForCausalLM or we would silently be reading hidden states
    instead of logits.
"""

from __future__ import annotations

from typing import Optional

import torch

from dqeval import hf
from dqeval.adapter import ModelAdapter, ModelSpec

from . import compat


class SDARAdapter(ModelAdapter):
    family = "sdar"

    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        with torch.no_grad():
            return self.model(input_ids=input_ids, **kw).logits

    # _canonicalize: identity (inherited) -- SDAR needs no shift.


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> SDARAdapter:
    cfg, klass = compat.resolve(spec.repo, revision=revision, auto_class=spec.auto_class)
    shims = compat.apply(cfg, klass)
    tok = hf.tokenizer(spec.repo, revision=revision)
    model = hf.materialize(klass, spec.repo, cfg, revision=revision,
                           dtype=dtype, device=device)
    shims.append(compat.repair_rope(model))
    hf.assert_finite_rope(model)

    if not hasattr(model, "lm_head"):
        raise RuntimeError(
            f"{spec.repo}: resolved class {klass.__name__} has no lm_head -- "
            "auto_class must be AutoModelForCausalLM, not AutoModel"
        )
    return SDARAdapter(model, tok, spec, revision=revision, shims=shims)
