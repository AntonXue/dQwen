"""LLaDA adapter.

LLaDA is a pure masked diffusion LM: its raw output is already position-aligned
(`logits[:, i]` is the distribution for position i), so `_canonicalize` is the
inherited identity and there is no eval-side shift. Confirmed by reading their
`generate.py`, which consumes `model(x).logits` directly with no reindexing.
"""

from __future__ import annotations

from typing import Optional

import torch

from dqeval import hf
from dqeval.adapter import ModelAdapter, ModelSpec

from . import compat


class LLaDAAdapter(ModelAdapter):
    family = "llada"

    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        with torch.no_grad():
            return self.model(input_ids=input_ids, **kw).logits

    # _canonicalize: identity (inherited) -- LLaDA needs no shift.


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> LLaDAAdapter:
    cfg, klass = hf.resolve(spec.repo, revision=revision, auto_class=spec.auto_class)
    shims = compat.apply(cfg, klass)
    tok = hf.tokenizer(spec.repo, revision=revision)
    model = hf.materialize(klass, spec.repo, cfg, revision=revision,
                           dtype=dtype, device=device)
    hf.assert_finite_rope(model)
    return LLaDAAdapter(model, tok, spec, revision=revision, shims=shims)
