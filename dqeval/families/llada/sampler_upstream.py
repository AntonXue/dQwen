"""Escape hatch: run LLaDA's `generate.py` verbatim from the pinned checkout.

Not used for normal evals -- `sampler_native.py` is the vendored equivalent, and
`tests/parity/` asserts the two agree token-for-token. This exists so that claim is
checkable on demand, and so a touched-up path can be bracketed against real upstream
when it has no parity counterpart.

Requires `third_party/fetch.sh`.
"""

from __future__ import annotations

import torch

from dqeval import upstream
from dqeval.adapter import GenOutput, ModelAdapter
from dqeval.config import DecodeConfig


@torch.no_grad()
def generate(adapter: ModelAdapter, prompt_ids: torch.Tensor,
             cfg: DecodeConfig) -> GenOutput:
    with upstream.on_path("LLaDA"):
        from generate import generate as _upstream_generate

        # NOTE the units differ, and this is exactly the sort of off-by-N a parity
        # test exists to catch: LLaDA's `steps` is the TOTAL across all blocks
        # (it does `steps = steps // num_blocks` internally), whereas DecodeConfig
        # carries steps PER BLOCK. Convert, do not pass through.
        total_steps = cfg.steps_per_block * cfg.num_blocks

        out = _upstream_generate(
            adapter.model, prompt_ids,
            steps=total_steps,
            gen_length=cfg.gen_length,
            block_length=cfg.block_length,
            temperature=cfg.temperature,
            cfg_scale=cfg.cfg_scale,
            remasking=cfg.order,          # 'low_confidence' | 'random'
            mask_id=adapter.mask_id,
        )

    gen_ids = out[0, prompt_ids.shape[1]:]
    return GenOutput(prompt_ids=prompt_ids[0], gen_ids=gen_ids,
                     text=adapter.decode(gen_ids), n_forward=total_steps)
