"""dQwen adapter -- our own family (dQwen3.5-* and dQwen3-*).

No compat shims: these models are built against transformers 5.x and load natively.

The BOS/logit shift is INTERNALISED by the modeling code -- it prepends BOS and
returns `lm_head(hidden[:, :-1, :])` -- so the raw output is already position-aligned
and `_canonicalize` is the identity. Applying an eval-side shift here would be a
silent off-by-one; that mistake has been made once already on a sibling model, which
is why this is structure rather than a `logit_shift` config flag.

Step checkpoints are published as HF revisions (`step30000-swa`, `step50000-swa`, ...)
so `revision=` is a first-class argument rather than a separate repo per checkpoint.
"""

from typing import Optional
import torch
from dqeval import models as hf
from dqeval.models import ModelAdapter, ModelSpec


class DQwenAdapter(ModelAdapter):
    family = "dqwen"

    def raw_logits(self, input_ids: torch.Tensor, **kw) -> torch.Tensor:
        with torch.no_grad():
            return self.model(input_ids=input_ids, **kw).logits

    # _canonicalize: identity (inherited) -- the shift is internal to the model.


def build(spec: ModelSpec, revision: Optional[str] = None,
          dtype=torch.bfloat16, device: str = "cuda") -> DQwenAdapter:
    cfg, klass = hf.resolve(spec.repo, revision=revision, auto_class=spec.auto_class)
    tok = hf.tokenizer(spec.repo, revision=revision)
    model = hf.materialize(klass, spec.repo, cfg, revision=revision,
                           dtype=dtype, device=device)
    hf.assert_finite_rope(model)

    # Our own ids are pinned in the registry; assert rather than trust.
    vocab = model.config.vocab_size
    if not spec.mask_id < vocab:
        raise ValueError(f"{spec.repo}: mask_id {spec.mask_id} outside vocab {vocab}")
    return DQwenAdapter(model, tok, spec, revision=revision, shims=[])
