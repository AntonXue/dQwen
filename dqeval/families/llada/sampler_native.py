"""LLaDA's own sampler, vendored from `generate.py` (ML-GSAI/LLaDA @ 96441d4).

This is the "sampler shipped with LLaDA" path: the algorithm is reproduced exactly,
so a number produced here is theirs, not ours. `tests/parity/test_llada_sampler.py`
runs this against the unmodified upstream file and requires token-identical output.

TOUCHUPS -- the complete list, each one behaviour-preserving:

  1. `-np.inf` -> `float("-inf")`. Identical value (np.inf is a plain Python float);
     drops the numpy dependency so dqeval/ stays torch-only.
  2. Signature adapted to the dqeval sampler contract
     `generate(adapter, prompt_ids, cfg) -> GenOutput`. The body is untouched.
  3. Logits come from `adapter.raw_logits` rather than `model(x).logits`. For LLaDA
     these are the same thing -- it is position-aligned and needs no shift -- but
     native samplers use raw_logits by convention, since a family that shifts
     internally (Dream) must not receive canonicalised logits.

DELIBERATELY NOT CHANGED, though our unified engine does it differently:

  * LLaDA does NOT suppress the MASK id before the argmax. Our engine does. Left
    faithful here; if the two ever diverge on a prompt, this is the first suspect.
  * Confidence is read from softmax of the RAW logits, before temperature. Faithful.
  * `add_gumbel_noise` returns logits unchanged at temperature 0, so greedy works
    without a division -- unlike SDAR's script, which cannot express greedy at all.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dqeval.adapter import GenOutput, ModelAdapter
from dqeval.config import DecodeConfig

NEG_INF = float("-inf")


def add_gumbel_noise(logits, temperature):
    """Verbatim LLaDA generate.py:8-19.

    float64 is deliberate on their side (arXiv:2409.02908: low-precision Gumbel-max
    helps perplexity but hurts generation quality). Do not downcast.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Verbatim LLaDA generate.py:22-40. Linear noise schedule => even split."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(adapter: ModelAdapter, prompt_ids: torch.Tensor,
             cfg: DecodeConfig) -> GenOutput:
    """LLaDA `generate()`, body verbatim. `prompt_ids` is [1, L]."""
    if cfg.order not in ("low_confidence", "random"):
        raise ValueError(
            f"LLaDA's sampler implements 'low_confidence' and 'random' only; "
            f"got order={cfg.order!r}. Use the unified sampler for other orders."
        )
    if cfg.commit != "static":
        raise ValueError("LLaDA's sampler has no dynamic-commit policy (commit='static' only)")
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    mask_id = adapter.mask_id
    prompt = prompt_ids
    gen_length, block_length = cfg.gen_length, cfg.block_length
    steps, temperature, cfg_scale = cfg.steps_per_block, cfg.temperature, cfg.cfg_scale
    remasking = cfg.order

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long).to(adapter.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    n_forward = 0

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length:
                 prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = adapter.raw_logits(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = adapter.raw_logits(x)
            n_forward += 1

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            else:  # "random"
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = NEG_INF

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, NEG_INF)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    gen_ids = x[0, prompt.shape[1]:]
    return GenOutput(prompt_ids=prompt_ids[0], gen_ids=gen_ids,
                     text=adapter.decode(gen_ids), n_forward=n_forward)
