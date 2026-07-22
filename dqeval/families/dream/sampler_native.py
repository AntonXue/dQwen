"""Dream's own sampler: `model.diffusion_generate`.

UNUSUAL AMONG OUR FAMILIES: Dream's sampler ships INSIDE the HF repo, loaded with
the weights via `trust_remote_code` (`generation_utils.py::DreamGenerationMixin`).
So for Dream, native IS upstream -- there is nothing to vendor and nothing to drift.
That is why there is no `sampler_upstream.py` here: it would be the same object.
The pinned `third_party/Dream` checkout is still useful as a reading reference and
for their lm-eval wrapper, but the sampler itself is whatever the model revision
carries, which is the honest thing to report anyway.

It also means the compat shims are doing real work at generation time: shim 4
(`DreamGenerationConfig.from_model_config`) exists purely for this path.

THE SHIFT: Dream's sampler applies `cat([l[:, :1], l[:, :-1]])` itself, immediately
after every forward (generation_utils.py:421-422). We therefore hand it the raw
model and never `adapter.logits` -- canonicalising first would shift twice, silently.

Their unmasking strategies (`alg`) and our `DecodeConfig.order` are the same ideas
under different names; the mapping is explicit below rather than guessed.
"""

from __future__ import annotations

import torch

from dqeval.adapter import GenOutput, ModelAdapter
from dqeval.config import DecodeConfig

# our vocabulary -> theirs. 'origin' is their default: sample per position, unmask
# by raw confidence. 'maskgit_plus' is the low-confidence-first ordering.
ORDER_TO_ALG = {
    "low_confidence": "maskgit_plus",
    "entropy": "entropy",
    "topk_margin": "topk_margin",
    "random": "origin",
}


@torch.no_grad()
def generate(adapter: ModelAdapter, prompt_ids: torch.Tensor, cfg: DecodeConfig, *,
             alg: str | None = None) -> GenOutput:
    """Dream `diffusion_generate`. `alg` overrides the ORDER_TO_ALG mapping."""
    if cfg.commit != "static":
        raise ValueError("Dream's sampler has no dynamic-commit policy (commit='static')")
    if alg is None:
        if cfg.order not in ORDER_TO_ALG:
            raise ValueError(f"no Dream alg for order={cfg.order!r}; pass alg= explicitly")
        alg = ORDER_TO_ALG[cfg.order]
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    # Dream decodes the whole canvas at once -- it has no block structure, so the
    # step budget is per-generation, not per-block.
    steps = cfg.steps_per_block * cfg.num_blocks

    out = adapter.model.diffusion_generate(
        prompt_ids,
        max_new_tokens=cfg.gen_length,
        steps=steps,
        temperature=cfg.temperature,
        top_p=cfg.top_p if cfg.top_p < 1.0 else None,
        top_k=cfg.top_k if cfg.top_k > 0 else None,
        alg=alg,
        alg_temp=0.0,
        mask_token_id=adapter.mask_id,
        return_dict_in_generate=True,
    )
    seq = out.sequences if hasattr(out, "sequences") else out
    gen_ids = seq[0, prompt_ids.shape[1]:]
    return GenOutput(prompt_ids=prompt_ids[0], gen_ids=gen_ids,
                     text=adapter.decode(gen_ids), n_forward=steps)
