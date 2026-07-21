"""Dream / Dream-Coder adapter -- the one family that needs an eval-side logit shift.

Dream's raw output is AR-aligned: `logits[:, i]` predicts position i+1. Every
consumer in their own codebase corrects for this immediately after each forward,
with the identical expression, in two independent places:

    generation_utils.py:421-422   (their sampler)
    eval/eval.py:354              (their lm-eval wrapper)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

So the shift belongs to the CONSUMER, and we put it in `_canonicalize`. The
consequence to keep in mind:

    portable samplers -> adapter.logits()      shift applied here, once
    Dream's native sampler -> adapter.raw_logits()   it shifts internally, itself

Feeding canonicalised logits to Dream's native sampler would apply the shift twice:
a silent off-by-one with no exception. That is the whole reason the two surfaces are
separate methods.
"""

from __future__ import annotations

from typing import Optional

import torch

from dqeval import hf
from dqeval.adapter import ModelAdapter, ModelSpec

from . import compat


class DreamAdapter(ModelAdapter):
    family = "dream"

    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, **kw)
        return out.logits if hasattr(out, "logits") else out[0]

    def _canonicalize(self, logits: torch.Tensor) -> torch.Tensor:
        """AR-aligned -> position-aligned. Verbatim Dream's own expression."""
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> DreamAdapter:
    cfg, klass = hf.resolve(spec.repo, revision=revision, auto_class=spec.auto_class)
    shims = compat.apply(cfg, klass)
    tok = hf.tokenizer(spec.repo, revision=revision)
    model = hf.materialize(klass, spec.repo, cfg, revision=revision,
                           dtype=dtype, device=device)
    hf.assert_finite_rope(model)          # shim (3) is silent if it regresses

    cfg_mask = getattr(cfg, "mask_token_id", None)
    if cfg_mask is not None and cfg_mask != spec.mask_id:
        raise ValueError(
            f"{spec.repo}: registry mask_id={spec.mask_id} but config says {cfg_mask}"
        )
    return DreamAdapter(model, tok, spec, revision=revision, shims=shims)
